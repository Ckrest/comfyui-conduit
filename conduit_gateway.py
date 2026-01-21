"""
Conduit Gateway - External API for running workflows

Endpoint: POST /conduit/run/{workflow_name}

Accepts input values, injects them into the workflow, runs it,
and returns the captured outputs.
"""

import json
import copy
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web
import server

from .conduit_workflows import get_workflow_path, WORKFLOWS_DIR
from .conduit_validation import validate_workflow_inputs
from .conduit_mimicry import apply_workflow_mimicry

# ComfyUI API base URL (assumes running on same host)
COMFY_API_URL = "http://127.0.0.1:8188"

# Timeout constants (seconds)
QUEUE_TIMEOUT = 10
HISTORY_POLL_TIMEOUT = 5
HISTORY_POLL_INTERVAL = 0.5
DEFAULT_EXECUTION_TIMEOUT = 120.0
MIN_EXECUTION_TIMEOUT = 5.0
MAX_EXECUTION_TIMEOUT = 600.0
from .conduit_outputs import (
    get_pending_outputs, clear_pending_outputs, get_output_folder,
    run_handlers, get_default_handlers,
    register_execution_context, update_execution_context, clear_execution_context,
    get_always_run_handler
)


def is_socket_active(socket: dict) -> bool:
    """
    Check if a socket is active (not disabled).

    Sockets are active unless they have explicit `active: false`.
    This supports backwards compatibility - older schemas without
    the active field are treated as all-active.
    """
    return socket.get("active") is not False


def filter_active_sockets(sockets: List[dict]) -> List[dict]:
    """Filter a list of sockets to only include active ones."""
    return [s for s in sockets if is_socket_active(s)]


def generate_prompt_id() -> str:
    """Generate a unique prompt ID."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    random_suffix = hex(int.from_bytes(__import__('os').urandom(2), 'big'))[2:]
    return f"{timestamp}_{random_suffix}"


def inject_conduit_outputs(workflow: dict, output_sockets: List[dict], prompt_id: str) -> dict:
    """
    Inject ConduitOutput nodes for tagged output sockets.
    Mirrors the frontend's injectConduitOutputs() function.
    """
    modified = copy.deepcopy(workflow)

    for socket in output_sockets:
        node_id = str(socket["nodeId"])
        slot_index = socket["slotIndex"]
        tag_name = socket["tagName"]
        data_type = socket["dataType"]

        # Generate unique ID for injected node
        injected_id = f"conduit_{node_id}_{slot_index}"

        # Convert type to safe node name
        safe_type = "".join(c if c.isalnum() else "_" for c in data_type)
        node_class = f"ConduitOutput_{safe_type}"

        modified[injected_id] = {
            "class_type": node_class,
            "inputs": {
                "data": [node_id, slot_index],
                "tag_name": tag_name,
                "prompt_id": prompt_id,
            }
        }

    return modified


# =============================================================================
# Input Processing
# =============================================================================

class InputResult:
    """Result of processing inputs - tracks what was applied, defaulted, and warnings."""
    def __init__(self):
        self.applied: Dict[str, dict] = {}      # tag -> {value, type, coerced_from?}
        self.defaulted: List[dict] = []          # [{tagName, type, reason}]
        self.warnings: List[dict] = []           # [{type, message, ...}]

    def to_dict(self) -> dict:
        return {
            "inputs_applied": self.applied,
            "inputs_defaulted": self.defaulted,
            "warnings": self.warnings,
        }


def coerce_value(value: Any, target_type: str, tag_name: str) -> tuple[Any, Optional[str]]:
    """
    Attempt to coerce a value to the target type.
    Returns (coerced_value, warning_message) or (None, error_message) if impossible.
    """
    if target_type == "STRING":
        # Everything can become a string
        if isinstance(value, str):
            return value, None
        return str(value), f"Coerced {type(value).__name__} to string"

    elif target_type == "INT":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, float):
            return int(value), f"Truncated float {value} to int {int(value)}"
        if isinstance(value, str):
            # Try to parse
            try:
                if '.' in value:
                    return int(float(value)), f"Parsed string '{value}' to int"
                return int(value), f"Parsed string '{value}' to int"
            except ValueError:
                return None, f"Cannot parse '{value}' as INT"
        return None, f"Cannot coerce {type(value).__name__} to INT"

    elif target_type == "FLOAT":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value), f"Parsed string '{value}' to float"
            except ValueError:
                return None, f"Cannot parse '{value}' as FLOAT"
        return None, f"Cannot coerce {type(value).__name__} to FLOAT"

    elif target_type == "BOOLEAN":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True, f"Parsed string '{value}' to True"
            if value.lower() in ('false', '0', 'no', 'off'):
                return False, f"Parsed string '{value}' to False"
            return None, f"Cannot parse '{value}' as BOOLEAN"
        if isinstance(value, (int, float)):
            return bool(value), f"Coerced {value} to {bool(value)}"
        return None, f"Cannot coerce {type(value).__name__} to BOOLEAN"

    elif target_type == "IMAGE":
        # IMAGE must be a non-empty string (filename)
        if isinstance(value, str) and value.strip():
            return value, None
        if isinstance(value, str) and not value.strip():
            return None, "IMAGE requires a filename, got empty string"
        return None, f"IMAGE requires a filename string, got {type(value).__name__}"

    else:
        # Unknown type - pass through as-is
        return value, f"Unknown type {target_type}, passing value as-is"


def process_inputs(
    workflow: dict,
    input_sockets: List[dict],
    user_inputs: Dict[str, Any]
) -> tuple[dict, InputResult]:
    """
    Process and inject input values with full validation and coercion.

    Handles:
    - null values: Skip injection, use workflow default
    - Empty strings: Valid for STRING, invalid for INT/FLOAT/IMAGE
    - Type coercion: Attempt reasonable conversions
    - Unknown inputs: Warn about inputs that don't match any tag

    Returns (modified_workflow, InputResult)
    """
    modified = copy.deepcopy(workflow)
    result = InputResult()

    # Build lookup of valid tag names
    valid_tags = {socket["tagName"] for socket in input_sockets}

    # Check for unknown inputs (possible typos)
    for key in user_inputs.keys():
        if key not in valid_tags:
            # Try to find similar tag (simple suggestion)
            similar = [t for t in valid_tags if t.lower() == key.lower()]
            if similar:
                result.warnings.append({
                    "type": "unknown_input",
                    "key": key,
                    "message": f"Unknown input '{key}'",
                    "suggestion": similar[0]
                })
            else:
                result.warnings.append({
                    "type": "unknown_input",
                    "key": key,
                    "message": f"Unknown input '{key}' - no matching tagged socket"
                })

    # Process each tagged input socket
    for socket in input_sockets:
        tag_name = socket["tagName"]
        node_id = str(socket["nodeId"])
        slot_name = socket["slotName"]
        data_type = socket["dataType"]

        # Check if node exists in workflow
        if node_id not in modified:
            result.warnings.append({
                "type": "missing_node",
                "tagName": tag_name,
                "message": f"Node {node_id} not found in workflow"
            })
            continue

        # Case 1: Input not provided - use default
        if tag_name not in user_inputs:
            result.defaulted.append({
                "tagName": tag_name,
                "type": data_type,
                "reason": "not_provided"
            })
            continue

        value = user_inputs[tag_name]

        # Case 2: Explicit null - use default
        if value is None:
            result.defaulted.append({
                "tagName": tag_name,
                "type": data_type,
                "reason": "explicit_null"
            })
            continue

        # Case 3: Empty string handling
        if isinstance(value, str) and value == "":
            if data_type == "STRING":
                # Empty string is valid for STRING - inject via node
                injected_id = f"conduit_input_{node_id}_{slot_name}"
                modified[injected_id] = {
                    "class_type": "PrimitiveString",
                    "inputs": {"value": ""},
                }
                modified[node_id]["inputs"][slot_name] = [injected_id, 0]
                result.applied[tag_name] = {"value": "", "type": data_type}
                continue
            else:
                # Empty string invalid for other types
                result.defaulted.append({
                    "tagName": tag_name,
                    "type": data_type,
                    "reason": "empty_string_invalid"
                })
                result.warnings.append({
                    "type": "invalid_empty",
                    "tagName": tag_name,
                    "message": f"Empty string invalid for {data_type}, using default"
                })
                continue

        # Case 4: Coerce and inject
        coerced, warning = coerce_value(value, data_type, tag_name)

        if coerced is None:
            # Coercion failed - use default
            result.defaulted.append({
                "tagName": tag_name,
                "type": data_type,
                "reason": "coercion_failed"
            })
            result.warnings.append({
                "type": "coercion_failed",
                "tagName": tag_name,
                "message": warning
            })
            continue

        if warning:
            result.warnings.append({
                "type": "coerced",
                "tagName": tag_name,
                "message": warning
            })

        # Inject via node - ALWAYS use node injection for consistency
        # This ensures linked inputs work correctly (replaces the link reference)
        injected_id = f"conduit_input_{node_id}_{slot_name}"

        if data_type == "IMAGE":
            # IMAGE: LoadImage node
            modified[injected_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": coerced},
            }
            modified[node_id]["inputs"][slot_name] = [injected_id, 0]

        elif data_type == "STRING":
            # STRING: PrimitiveString node
            modified[injected_id] = {
                "class_type": "PrimitiveString",
                "inputs": {"value": coerced},
            }
            modified[node_id]["inputs"][slot_name] = [injected_id, 0]

        elif data_type == "INT":
            # INT: PrimitiveInt node
            modified[injected_id] = {
                "class_type": "PrimitiveInt",
                "inputs": {"value": coerced},
            }
            modified[node_id]["inputs"][slot_name] = [injected_id, 0]

        elif data_type == "FLOAT":
            # FLOAT: PrimitiveFloat node
            modified[injected_id] = {
                "class_type": "PrimitiveFloat",
                "inputs": {"value": coerced},
            }
            modified[node_id]["inputs"][slot_name] = [injected_id, 0]

        elif data_type == "BOOLEAN":
            # BOOLEAN: PrimitiveBoolean node
            modified[injected_id] = {
                "class_type": "PrimitiveBoolean",
                "inputs": {"value": coerced},
            }
            modified[node_id]["inputs"][slot_name] = [injected_id, 0]

        else:
            # Unknown type: try direct injection as fallback
            # This handles custom types that don't have primitive nodes
            modified[node_id]["inputs"][slot_name] = coerced
            result.warnings.append({
                "type": "direct_injection",
                "tagName": tag_name,
                "message": f"No primitive node for {data_type}, using direct injection"
            })

        # Record what was applied
        applied_entry = {"value": coerced, "type": data_type}
        if warning and "coerced" in warning.lower():
            applied_entry["coerced_from"] = type(value).__name__
        result.applied[tag_name] = applied_entry

    return modified, result


async def queue_prompt(workflow: dict) -> Optional[str]:
    """
    Queue a prompt via ComfyUI's /prompt endpoint.
    Returns the prompt_id if successful.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{COMFY_API_URL}/prompt",
                json={"prompt": workflow},
                timeout=aiohttp.ClientTimeout(total=QUEUE_TIMEOUT)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("prompt_id")
                else:
                    text = await resp.text()
                    print(f"[Conduit Gateway] Prompt failed: {resp.status} - {text[:200]}")
                    return None
    except Exception as e:
        print(f"[Conduit Gateway] Error queuing prompt: {e}")
        return None


async def wait_for_completion(comfy_prompt_id: str, timeout: float = 120.0) -> dict:
    """
    Wait for a prompt to complete by polling /history.

    Returns dict with:
        - completed: True if finished (success or error)
        - success: True if completed without errors
        - error: Error message if failed
        - status: Full status object from ComfyUI
    """
    start = asyncio.get_event_loop().time()

    while True:
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed > timeout:
            print(f"[Conduit Gateway] Timeout waiting for {comfy_prompt_id}")
            return {"completed": False, "success": False, "error": f"Timeout after {timeout}s"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{COMFY_API_URL}/history/{comfy_prompt_id}",
                    timeout=aiohttp.ClientTimeout(total=HISTORY_POLL_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if comfy_prompt_id in history:
                            entry = history[comfy_prompt_id]
                            status = entry.get("status", {})

                            # Check if execution completed (success or error)
                            if status.get("completed"):
                                status_str = status.get("status_str", "unknown")

                                if status_str == "success":
                                    return {"completed": True, "success": True, "status": status}
                                else:
                                    # Extract error message from messages
                                    error_msg = None
                                    for msg_type, msg_data in status.get("messages", []):
                                        if msg_type == "execution_error":
                                            error_msg = msg_data.get("exception_message", str(msg_data))
                                            break

                                    print(f"[Conduit Gateway] Execution failed: {error_msg or status_str}")
                                    return {
                                        "completed": True,
                                        "success": False,
                                        "error": error_msg or f"Execution failed: {status_str}",
                                        "status": status
                                    }
        except Exception as e:
            print(f"[Conduit Gateway] Error checking history: {e}")

        await asyncio.sleep(HISTORY_POLL_INTERVAL)


# =============================================================================
# API Endpoint
# =============================================================================

@server.PromptServer.instance.routes.post("/conduit/run/{workflow_name}")
async def run_workflow(request):
    """
    Run a saved workflow with provided inputs.

    POST /conduit/run/{workflow_name}
    Body: {
        "inputs": {
            "tagName1": "value1",      // Values for tagged inputs
            "tagName2": 123,
            ...
        },
        "handlers": ["handler1", ...], // optional, explicit handler override
                                       // omit = use workflow/defaults
                                       // [] = run no handlers
        "mimicry": true | false | "default",
                                       // optional, frontend mimicry mode
                                       // true = enable transforms (seed randomization, etc.)
                                       // false = disable transforms
                                       // "default" = ignore workflow setting, use default (true)
                                       // omit = use workflow setting or default to true
        "wait": true,                  // optional, default true - wait for completion
        "timeout": 120,                // optional, seconds to wait (5-600, default 120)
        "context": {                   // optional, metadata passed to handlers
            "source": "my_app",
            ...
        }
    }

    Returns: {
        "status": "success" | "error" | "timeout" | "queued",
        "prompt_id": "...",
        "comfy_prompt_id": "...",
        "outputs": [...],              // if wait=true and success
        "handler": {...},              // handler execution results
        "values_used": {               // actual values used for each tagged input
            "tagName": {
                "value": <final_value>,
                "source": "user_input" | "mimicry" | "workflow",
                "original": <pre-transform>,  // if different from value
                "transform": "...",           // which transform was applied
                "control_mode": "...",        // if control widget was involved
                "note": "...",                // additional context
                "warning": "..."              // if any issues
            }
        },
        "error": "...",                // if error or timeout
        "inputs_applied": {...},       // what was injected
        "inputs_defaulted": [...],     // what used defaults
        "warnings": [...]              // coercion warnings, etc.
    }
    """
    workflow_name = request.match_info["workflow_name"]

    # Load the workflow
    file_path = get_workflow_path(workflow_name)
    if not file_path.exists():
        return web.json_response(
            {"status": "error", "message": f"Workflow not found: {workflow_name}"},
            status=404
        )

    try:
        saved = json.loads(file_path.read_text())
    except Exception as e:
        return web.json_response(
            {"status": "error", "message": f"Failed to load workflow: {e}"},
            status=500
        )

    # Parse request body
    try:
        body = await request.json()
    except:
        body = {}

    user_inputs = body.get("inputs", {})
    wait_for_result = body.get("wait", True)
    timeout_seconds = body.get("timeout", DEFAULT_EXECUTION_TIMEOUT)
    timeout_seconds = min(max(float(timeout_seconds), MIN_EXECUTION_TIMEOUT), MAX_EXECUTION_TIMEOUT)
    user_context = body.get("context", {})

    # Handler selection:
    # - If "handlers" in body: use exactly those ([] = none)
    # - Else if workflow has saved handlers: use those
    # - Else: use defaults
    request_handlers = body.get("handlers")  # None means "not specified"

    workflow = saved["workflow"]
    schema = saved.get("schema", {})
    # Filter to only active sockets (disabled sockets should be invisible)
    input_sockets = filter_active_sockets(schema.get("inputs", []))
    output_sockets = filter_active_sockets(schema.get("outputs", []))

    # Generate our prompt ID
    prompt_id = generate_prompt_id()

    print(f"[Conduit Gateway] Running workflow: {workflow_name}")
    print(f"[Conduit Gateway]   Prompt ID: {prompt_id}")
    print(f"[Conduit Gateway]   Inputs provided: {list(user_inputs.keys())}")

    # Validate inputs against registry (warn only, never blocks)
    validation_result = None
    if input_sockets:
        validated_inputs, validation_result = await validate_workflow_inputs(
            workflow, input_sockets, user_inputs
        )

        # Log validation warnings
        if validation_result.issues:
            print(f"[Conduit Gateway]   Validation: {len(validation_result.issues)} issues")
            for issue in validation_result.issues:
                print(f"[Conduit Gateway]     {issue.severity.value}: {issue.message}")

        # Use validated/transformed inputs (with clamped values)
        user_inputs = validated_inputs

    # Determine mimicry mode (cascade: request → workflow → default)
    # - "default" in request = use True regardless of workflow setting
    # - True/False in request = use that value
    # - Not specified = use workflow setting or default to True
    workflow_changes = {}
    request_mimicry = body.get("mimicry")  # None = not specified
    if request_mimicry is not None:
        if request_mimicry == "default":
            mimicry_enabled = True  # Escape hatch: ignore workflow setting
        else:
            mimicry_enabled = bool(request_mimicry)
    elif "mimicry" in schema:
        mimicry_enabled = bool(schema["mimicry"])
    else:
        mimicry_enabled = True  # Default: enabled

    # Build lookup of tagged sockets by tag name
    socket_by_tag = {s["tagName"]: s for s in input_sockets} if input_sockets else {}

    # Apply user inputs directly to workflow BEFORE mimicry
    # Track which inputs were user-provided (these skip control widget transforms)
    user_provided_inputs = set()
    for tag_name, value in user_inputs.items():
        if value is None:
            continue  # Explicit null = use workflow default
        socket = socket_by_tag.get(tag_name)
        if not socket:
            continue  # Unknown tag, will be warned later
        node_id = str(socket["nodeId"])
        slot_name = socket["slotName"]
        if node_id in workflow and isinstance(workflow[node_id], dict):
            inputs_dict = workflow[node_id].get("inputs", {})
            # Only overwrite if not a link reference (don't break connections here)
            current_value = inputs_dict.get(slot_name)
            if not (isinstance(current_value, list) and len(current_value) == 2):
                inputs_dict[slot_name] = value
                user_provided_inputs.add((node_id, slot_name))

    # Apply frontend mimicry transforms (UNIFIED - all nodes, all inputs)
    if mimicry_enabled:
        workflow_changes = await apply_workflow_mimicry(
            workflow,
            user_provided_inputs=user_provided_inputs,
        )

        if workflow_changes:
            change_count = sum(len(v) for v in workflow_changes.values())
            print(f"[Conduit Gateway]   Mimicry: {change_count} changes across {len(workflow_changes)} nodes")
            # Log user-provided inputs that were preserved
            if user_provided_inputs:
                preserved = [f"{n}:{i}" for n, i in user_provided_inputs]
                print(f"[Conduit Gateway]   User values preserved (no control transforms): {preserved}")

    # Process and inject input values
    input_result = InputResult()
    if input_sockets:
        workflow, input_result = process_inputs(workflow, input_sockets, user_inputs)

        # Log input processing summary
        if input_result.applied:
            print(f"[Conduit Gateway]   Inputs applied: {list(input_result.applied.keys())}")
        if input_result.defaulted:
            tags = [d["tagName"] for d in input_result.defaulted]
            print(f"[Conduit Gateway]   Inputs defaulted: {tags}")
        if input_result.warnings:
            for w in input_result.warnings:
                print(f"[Conduit Gateway]   Warning: {w['message']}")

    # Inject ConduitOutput nodes
    if output_sockets:
        workflow = inject_conduit_outputs(workflow, output_sockets, prompt_id)

    # Register execution context BEFORE queuing
    # Source priority: user_context.source > "unknown"
    source = user_context.get("source", "unknown")

    register_execution_context(prompt_id, {
        "source": source,
        "workflow_name": workflow_name,
        "tagged_inputs": input_sockets,
        "tagged_outputs": output_sockets,
        **input_result.to_dict(),  # inputs_applied, inputs_defaulted, warnings
        # Pass through any additional user context (tags, custom data, etc.)
        "caller_context": {k: v for k, v in user_context.items() if k != "source"},
    })

    # Log the exact workflow being sent to ComfyUI for debugging
    print(f"[Conduit Gateway] ═══════════════════════════════════════════════════════")
    print(f"[Conduit Gateway] WORKFLOW PAYLOAD SENT TO COMFYUI:")
    print(f"[Conduit Gateway] ───────────────────────────────────────────────────────")
    print(json.dumps(workflow, indent=2, default=str))
    print(f"[Conduit Gateway] ═══════════════════════════════════════════════════════")

    # Queue the prompt
    comfy_prompt_id = await queue_prompt(workflow)
    if not comfy_prompt_id:
        clear_execution_context(prompt_id)  # Cleanup on failure
        return web.json_response(
            {"status": "error", "message": "Failed to queue prompt"},
            status=500
        )

    # Update context with comfy_prompt_id
    update_execution_context(prompt_id, {"comfy_prompt_id": comfy_prompt_id})

    print(f"[Conduit Gateway]   ComfyUI Prompt ID: {comfy_prompt_id}")

    if not wait_for_result:
        return web.json_response({
            "status": "queued",
            "prompt_id": prompt_id,
            "comfy_prompt_id": comfy_prompt_id,
            "validation": validation_result.to_dict() if validation_result else None,
            **input_result.to_dict(),
        })

    # Wait for completion
    result = await wait_for_completion(comfy_prompt_id, timeout=timeout_seconds)

    # Handle timeout
    if not result.get("completed"):
        clear_execution_context(prompt_id)
        return web.json_response({
            "status": "timeout",
            "prompt_id": prompt_id,
            "comfy_prompt_id": comfy_prompt_id,
            "error": result.get("error", "Timeout waiting for completion"),
            "validation": validation_result.to_dict() if validation_result else None,
            **input_result.to_dict(),
        })

    # Handle execution error
    if not result.get("success"):
        clear_pending_outputs(prompt_id)
        clear_execution_context(prompt_id)
        print(f"[Conduit Gateway]   Execution error: {result.get('error')}")
        return web.json_response({
            "status": "error",
            "prompt_id": prompt_id,
            "comfy_prompt_id": comfy_prompt_id,
            "error": result.get("error", "Unknown execution error"),
            "validation": validation_result.to_dict() if validation_result else None,
            **input_result.to_dict(),
        })

    # Get captured outputs
    outputs = get_pending_outputs(prompt_id)
    output_folder = get_output_folder() / prompt_id

    # Determine which handlers to run:
    # 1. Explicit request override (including empty list = no handlers)
    # 2. Workflow-saved handlers (if sparse storage present)
    # 3. System defaults
    if request_handlers is not None:
        # Explicit override from API request
        handler_ids = request_handlers
    elif "handlers" in schema:
        # Workflow has saved handler config (sparse - only if different from defaults)
        handler_ids = schema["handlers"]
    else:
        # Use system defaults
        handler_ids = None  # run_handlers will use defaults

    # Run handlers if we have outputs, OR if always_run_handler is enabled
    handler_result = {"ran": False, "reason": "no_outputs"}
    if outputs or get_always_run_handler():
        handler_result = run_handlers(prompt_id, outputs, output_folder, handler_ids)

    # Clear pending outputs and context
    clear_pending_outputs(prompt_id)
    clear_execution_context(prompt_id)

    print(f"[Conduit Gateway]   Complete! {len(outputs)} outputs captured")
    if handler_result.get("ran"):
        handlers_run = list(handler_result.get("handlers", {}).keys())
        status_msg = "success" if handler_result.get("success") else "partial"
        print(f"[Conduit Gateway]   Handlers ({status_msg}): {handlers_run}")

    # Map workflow changes to tagged inputs for visibility in response
    # This shows seed randomization, dynamic prompts, and other auto-transforms
    tagged_workflow_changes = {}
    if workflow_changes and input_sockets:
        socket_by_node_input = {
            (str(s["nodeId"]), s["slotName"]): s["tagName"]
            for s in input_sockets
        }
        for node_id, node_changes in workflow_changes.items():
            for input_name, change_info in node_changes.items():
                tag_name = socket_by_node_input.get((node_id, input_name))
                if tag_name:
                    tagged_workflow_changes[tag_name] = {
                        "original_value": change_info["old"],
                        "transformed_value": change_info["new"],
                        "reason": change_info["reason"],
                    }

    response = {
        "status": "success",
        "prompt_id": prompt_id,
        "comfy_prompt_id": comfy_prompt_id,
        "outputs": outputs,
        "handler": handler_result,
        "validation": validation_result.to_dict() if validation_result else None,
        "workflow_transforms": tagged_workflow_changes if tagged_workflow_changes else None,
        **input_result.to_dict(),
    }

    return web.json_response(response)


print("[Conduit Gateway] Endpoint: POST /conduit/run/{workflow_name}")
