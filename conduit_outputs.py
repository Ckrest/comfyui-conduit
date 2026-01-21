"""
Conduit Output Nodes - Capture tagged outputs and run handlers

Architecture:
1. ConduitOutput_{TYPE} nodes save files immediately during execution
2. Output info is accumulated per prompt_id
3. After execution completes, JS calls /conduit/flush/{prompt_id}
4. Flush runs configured handler scripts with all outputs as JSON

Config file: conduit_config.json
- output_folder: where to save files (empty = default output/conduit)
- handlers: registry of handler scripts with enable/disable state
- always_run_handler: run handlers even with no outputs
"""

import os
import sys
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import folder_paths
import nodes  # ComfyUI's node registry
from aiohttp import web
import server

# Optional dependencies
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = Path(__file__).parent / "conduit_config.json"

def load_config() -> dict:
    """Load configuration from conduit_config.json"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Conduit] Error loading config: {e}")
    return {}

def save_config(config: dict):
    """Save configuration to conduit_config.json"""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def get_output_folder() -> Path:
    """Get the output folder from config or use default."""
    config = load_config()
    folder = config.get("output_folder", "").strip()
    if folder:
        return Path(folder)
    # Default: output/conduit relative to ComfyUI output directory
    return Path(folder_paths.get_output_directory()) / "conduit"

def get_always_run_handler() -> bool:
    """Check if handler should run even with no outputs."""
    config = load_config()
    return config.get("always_run_handler", False)


# =============================================================================
# Handler Registry
# =============================================================================

def get_handlers_registry() -> Dict[str, dict]:
    """Get all registered handlers from config."""
    config = load_config()
    return config.get("handlers", {})


def get_handler_info(handler_id: str) -> Optional[dict]:
    """Get info for a specific handler."""
    handlers = get_handlers_registry()
    if handler_id in handlers:
        return {"id": handler_id, **handlers[handler_id]}
    return None


def get_default_handlers() -> List[str]:
    """Get list of handler IDs that are enabled by default."""
    handlers = get_handlers_registry()
    return [hid for hid, info in handlers.items() if info.get("enabled", False)]


def resolve_handler_command(handler_id: str) -> Optional[str]:
    """
    Get the full command path for a handler.
    Resolves relative paths against the config file's directory.
    """
    handlers = get_handlers_registry()
    if handler_id not in handlers:
        return None

    command = handlers[handler_id].get("command", "").strip()
    if not command:
        return None

    # Resolve relative paths against config directory (conduit/)
    if not os.path.isabs(command):
        command = str(CONFIG_PATH.parent / command)

    return command


def set_handler_enabled(handler_id: str, enabled: bool) -> bool:
    """Set whether a handler is enabled by default."""
    config = load_config()
    handlers = config.get("handlers", {})

    if handler_id not in handlers:
        return False

    handlers[handler_id]["enabled"] = enabled
    config["handlers"] = handlers
    save_config(config)
    return True


# =============================================================================
# Per-Prompt Output Accumulator
# =============================================================================

# Global storage: prompt_id -> {"outputs": [...], "created_at": timestamp}
_pending_outputs: Dict[str, dict] = {}

# Cleanup thresholds (seconds)
REGISTRY_TTL_SECONDS = 3600       # Remove stale entries after 1 hour
CLEANUP_INTERVAL_SECONDS = 300    # Run cleanup every 5 minutes
HANDLER_TIMEOUT_SECONDS = 30      # Max time for handler to run

def accumulate_output(prompt_id: str, output_info: dict):
    """Add an output to the pending list for this prompt."""
    if prompt_id not in _pending_outputs:
        _pending_outputs[prompt_id] = {
            "outputs": [],
            "created_at": time.time()
        }
    _pending_outputs[prompt_id]["outputs"].append(output_info)

def get_pending_outputs(prompt_id: str) -> List[dict]:
    """Get all pending outputs for a prompt."""
    entry = _pending_outputs.get(prompt_id)
    return entry["outputs"] if entry else []

def clear_pending_outputs(prompt_id: str):
    """Clear pending outputs for a prompt."""
    if prompt_id in _pending_outputs:
        del _pending_outputs[prompt_id]


# =============================================================================
# Execution Context Registry
# =============================================================================

# Global storage: prompt_id -> execution context (includes created_at)
_execution_contexts: Dict[str, dict] = {}

def register_execution_context(prompt_id: str, context: dict):
    """
    Register execution context before queuing a prompt.

    Context should include:
    - source: "frontend" | "gateway"
    - workflow_name: str or None
    - comfy_prompt_id: str or None (set after queuing)
    - inputs_applied: dict
    - inputs_defaulted: list
    - tagged_outputs: list (the output schema)
    - tagged_inputs: list (the input schema)
    """
    _execution_contexts[prompt_id] = {
        "prompt_id": prompt_id,
        "registered_at": datetime.now().isoformat(),
        "created_at": time.time(),  # For TTL cleanup
        **context
    }

def update_execution_context(prompt_id: str, updates: dict):
    """Update an existing execution context (e.g., add comfy_prompt_id after queuing)."""
    if prompt_id in _execution_contexts:
        _execution_contexts[prompt_id].update(updates)

def get_execution_context(prompt_id: str) -> Optional[dict]:
    """Get execution context for a prompt."""
    return _execution_contexts.get(prompt_id)

def clear_execution_context(prompt_id: str):
    """Clear execution context for a prompt."""
    if prompt_id in _execution_contexts:
        del _execution_contexts[prompt_id]


# =============================================================================
# Registry Cleanup
# =============================================================================

def cleanup_stale_entries():
    """
    Remove entries older than REGISTRY_TTL_SECONDS from both registries.
    Called periodically to prevent memory leaks from failed/abandoned prompts.
    """
    now = time.time()
    threshold = now - REGISTRY_TTL_SECONDS

    # Cleanup pending outputs
    stale_outputs = [
        pid for pid, entry in _pending_outputs.items()
        if entry.get("created_at", 0) < threshold
    ]
    for pid in stale_outputs:
        del _pending_outputs[pid]

    # Cleanup execution contexts
    stale_contexts = [
        pid for pid, ctx in _execution_contexts.items()
        if ctx.get("created_at", 0) < threshold
    ]
    for pid in stale_contexts:
        del _execution_contexts[pid]

    if stale_outputs or stale_contexts:
        print(f"[Conduit] Cleaned up {len(stale_outputs)} stale outputs, "
              f"{len(stale_contexts)} stale contexts")

    return len(stale_outputs) + len(stale_contexts)


def get_registry_stats() -> dict:
    """Get statistics about the registries for debugging."""
    return {
        "pending_outputs": len(_pending_outputs),
        "execution_contexts": len(_execution_contexts),
        "ttl_seconds": REGISTRY_TTL_SECONDS,
    }


# Periodic cleanup thread
def _cleanup_loop():
    """Background thread that periodically cleans up stale entries."""
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            cleanup_stale_entries()
        except Exception as e:
            print(f"[Conduit] Cleanup error: {e}")

_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()
print("[Conduit] Registry cleanup thread started (TTL: 1 hour)")


def build_handler_event(prompt_id: str, outputs: List[dict], output_folder: Path) -> dict:
    """
    Build the unified event structure for the handler.
    Merges execution context with captured outputs.
    """
    context = get_execution_context(prompt_id) or {}

    return {
        # Identity
        "prompt_id": prompt_id,
        "comfy_prompt_id": context.get("comfy_prompt_id"),

        # Source info
        "source": context.get("source", "unknown"),
        "workflow_name": context.get("workflow_name"),

        # Input processing results
        "inputs_applied": context.get("inputs_applied", {}),
        "inputs_defaulted": context.get("inputs_defaulted", []),
        "warnings": context.get("warnings", []),

        # Schema (what was tagged)
        "tagged_inputs": context.get("tagged_inputs", []),
        "tagged_outputs": context.get("tagged_outputs", []),

        # Caller-provided context (tags, custom data, etc.)
        # Handlers can use this for extra metadata, or ignore it
        "caller_context": context.get("caller_context", {}),

        # Outputs
        "output_folder": str(output_folder),
        "outputs": outputs,
    }


# =============================================================================
# Type Detection and Saving
# =============================================================================

def detect_type(obj: Any) -> str:
    """Detect the type of an object for saving purposes."""
    if isinstance(obj, str):
        return "string"

    if Image and isinstance(obj, Image.Image):
        return "image"

    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.ndim == 4:
            if obj.shape[1] in [1, 3, 4]:  # B, C, H, W
                return "image_tensor"
            if obj.shape[3] in [1, 3, 4]:  # B, H, W, C (ComfyUI format)
                return "image_tensor"
        return "tensor"

    if np is not None and isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[2] in [3, 4]:
            return "image"
        elif obj.ndim == 1:
            return "audio"
        return "tensor"

    if isinstance(obj, dict):
        if "samples" in obj:
            return "latent"
        return "dict"

    if isinstance(obj, (int, float)):
        return "number"

    if isinstance(obj, (list, tuple)):
        return "list"

    class_name = obj.__class__.__name__.lower()
    if "conditioning" in class_name:
        return "conditioning"
    if "model" in class_name:
        return "model_ref"
    if "vae" in class_name:
        return "vae_ref"
    if "clip" in class_name:
        return "clip_ref"

    return "unknown"


def save_output(obj: Any, folder: Path, tag_name: str) -> tuple[Path, str]:
    """Save an output object to file. Returns (file_path, file_type)."""
    folder.mkdir(parents=True, exist_ok=True)
    detected_type = detect_type(obj)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag_name)

    if detected_type == "image" and Image:
        file_path = folder / f"{safe_name}.png"
        if isinstance(obj, Image.Image):
            obj.save(file_path)
        elif np is not None and isinstance(obj, np.ndarray):
            Image.fromarray(obj).save(file_path)
        return file_path, "image"

    elif detected_type == "image_tensor" and torch is not None:
        file_path = folder / f"{safe_name}.png"
        if obj.shape[-1] in [1, 3, 4]:  # B, H, W, C
            img_array = (obj[0].cpu().numpy() * 255).astype(np.uint8)
        else:  # B, C, H, W
            img_array = (obj[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if Image:
            Image.fromarray(img_array).save(file_path)
        return file_path, "image"

    elif detected_type == "latent":
        file_path = folder / f"{safe_name}_latent.pt"
        if torch is not None:
            torch.save(obj, file_path)
        return file_path, "latent"

    elif detected_type == "tensor" and torch is not None:
        file_path = folder / f"{safe_name}.pt"
        torch.save(obj, file_path)
        return file_path, "tensor"

    elif detected_type in ("number", "string"):
        file_path = folder / f"{safe_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(obj))
        return file_path, "text"

    elif detected_type in ("dict", "list"):
        file_path = folder / f"{safe_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return file_path, "json"

    elif detected_type in ("model_ref", "vae_ref", "clip_ref", "conditioning"):
        file_path = folder / f"{safe_name}_ref.json"
        metadata = {
            "type": detected_type,
            "class": obj.__class__.__name__,
            "repr": repr(obj)[:500]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return file_path, "reference"

    else:
        file_path = folder / f"{safe_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Type: {detected_type}\n")
            f.write(f"Class: {obj.__class__.__name__}\n")
            f.write(f"Repr: {repr(obj)[:1000]}\n")
        return file_path, "unknown"


# =============================================================================
# Handler Execution
# =============================================================================

def run_single_handler(handler_id: str, event: dict) -> dict:
    """
    Run a single handler with the event data.

    Returns:
        dict with:
            - ran: bool - whether handler was attempted
            - success: bool - whether handler completed successfully
            - exit_code: int - handler exit code (if ran)
            - stdout: str - handler stdout (if ran)
            - stderr: str - handler stderr (if ran)
            - error: str - error message (if failed)
    """
    handler_cmd = resolve_handler_command(handler_id)
    if not handler_cmd:
        return {"ran": False, "reason": f"handler_not_found: {handler_id}"}

    try:
        # Run handler, passing JSON via stdin
        result = subprocess.run(
            handler_cmd,
            shell=True,
            input=json.dumps(event),
            capture_output=True,
            text=True,
            timeout=HANDLER_TIMEOUT_SECONDS,
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"[Conduit:{handler_id}] {line}")

        if result.returncode != 0:
            print(f"[Conduit] Handler '{handler_id}' exited with code {result.returncode}")
            if result.stderr:
                print(f"[Conduit:{handler_id} Error] {result.stderr}")
            return {
                "ran": True,
                "success": False,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        return {
            "ran": True,
            "success": True,
            "exit_code": 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        print(f"[Conduit] Handler '{handler_id}' timed out after {HANDLER_TIMEOUT_SECONDS}s")
        return {"ran": True, "success": False, "error": f"timeout_{HANDLER_TIMEOUT_SECONDS}s"}
    except Exception as e:
        print(f"[Conduit] Failed to run handler '{handler_id}': {e}")
        return {"ran": True, "success": False, "error": str(e)}


def run_handlers(
    prompt_id: str,
    outputs: List[dict],
    output_folder: Path,
    handler_ids: Optional[List[str]] = None
) -> dict:
    """
    Run multiple handlers with the event data.

    Args:
        prompt_id: The prompt ID for this execution
        outputs: List of captured outputs
        output_folder: Where outputs were saved
        handler_ids: Specific handlers to run (None = use defaults)

    Returns:
        dict with:
            - ran: bool - whether any handlers were attempted
            - handlers: dict of handler_id -> result
            - success: bool - True if all handlers succeeded
    """
    # Determine which handlers to run
    if handler_ids is None:
        handler_ids = get_default_handlers()

    if not handler_ids:
        print(f"[Conduit] No handlers to run")
        return {"ran": False, "reason": "no_handlers", "handlers": {}}

    # Build the unified event (merges context + outputs)
    event = build_handler_event(prompt_id, outputs, output_folder)

    # Run each handler
    results = {}
    all_success = True

    print(f"[Conduit] Running {len(handler_ids)} handler(s): {handler_ids}")

    for handler_id in handler_ids:
        result = run_single_handler(handler_id, event)
        results[handler_id] = result

        if result.get("ran") and not result.get("success"):
            all_success = False

    return {
        "ran": True,
        "success": all_success,
        "handlers": results,
    }


# Legacy wrapper for backwards compatibility
def run_handler(prompt_id: str, outputs: List[dict], output_folder: Path) -> dict:
    """Legacy function - runs default handlers."""
    return run_handlers(prompt_id, outputs, output_folder)


# =============================================================================
# API Endpoints
# =============================================================================

@server.PromptServer.instance.routes.post("/conduit/context/{prompt_id}")
async def register_context(request):
    """
    Register execution context before queuing a prompt.
    Called by frontend before queuePrompt to store context for the handler.

    POST /conduit/context/{prompt_id}
    Body: {
        "source": "frontend",
        "workflow_name": "...",
        "comfy_prompt_id": "...",  // set after queuing if known
        "tagged_inputs": [...],
        "tagged_outputs": [...],
        "inputs_applied": {...},  // for frontend, usually empty
        "inputs_defaulted": [...],
    }
    """
    prompt_id = request.match_info["prompt_id"]

    try:
        context = await request.json()
    except:
        context = {}

    register_execution_context(prompt_id, context)

    return web.json_response({
        "status": "registered",
        "prompt_id": prompt_id
    })


@server.PromptServer.instance.routes.patch("/conduit/context/{prompt_id}")
async def update_context(request):
    """
    Update execution context (e.g., add comfy_prompt_id after queuing).
    """
    prompt_id = request.match_info["prompt_id"]

    try:
        updates = await request.json()
    except:
        updates = {}

    update_execution_context(prompt_id, updates)

    return web.json_response({
        "status": "updated",
        "prompt_id": prompt_id
    })


@server.PromptServer.instance.routes.post("/conduit/flush/{prompt_id}")
async def flush_outputs(request):
    """Called by JS after execution completes to run the handler."""
    prompt_id = request.match_info["prompt_id"]
    outputs = get_pending_outputs(prompt_id)
    context = get_execution_context(prompt_id)
    output_folder = get_output_folder() / prompt_id

    if not outputs:
        # No outputs captured - check if we should still run handler
        if get_always_run_handler() and context:
            # Run handler with empty outputs for consistency
            run_handler(prompt_id, [], output_folder)
            clear_execution_context(prompt_id)
            return web.json_response({
                "status": "flushed_empty",
                "prompt_id": prompt_id,
                "output_count": 0
            })
        else:
            # Clear context and skip handler
            clear_execution_context(prompt_id)
            return web.json_response({"status": "no_outputs", "prompt_id": prompt_id})

    # Run handler (uses unified event builder internally)
    run_handler(prompt_id, outputs, output_folder)

    # Clear pending outputs AND execution context
    clear_pending_outputs(prompt_id)
    clear_execution_context(prompt_id)

    return web.json_response({
        "status": "flushed",
        "prompt_id": prompt_id,
        "output_count": len(outputs)
    })


@server.PromptServer.instance.routes.get("/conduit/config")
async def get_config(request):
    """
    Get current Conduit configuration.

    Returns all settings with their current values and defaults.
    """
    config = load_config()
    default_output = str(Path(folder_paths.get_output_directory()) / "conduit")

    return web.json_response({
        "settings": {
            "output_folder": {
                "value": config.get("output_folder", ""),
                "default": default_output,
                "description": "Where Conduit saves outputs (empty = default)",
            },
            "handler_command": {
                "value": config.get("handler_command", ""),
                "default": "",
                "description": "Script/command to run after each workflow completion",
            },
            "always_run_handler": {
                "value": config.get("always_run_handler", False),
                "default": False,
                "description": "Run handler even when no outputs are tagged",
            },
        },
        # Flattened for easy access
        "output_folder": config.get("output_folder", ""),
        "handler_command": config.get("handler_command", ""),
        "always_run_handler": config.get("always_run_handler", False),
        "default_output_folder": default_output,
    })


def validate_config(data: dict) -> tuple[dict, List[str]]:
    """
    Validate configuration values.

    Returns:
        (validated_config, errors) - validated dict and list of error messages
    """
    validated = {}
    errors = []

    if "output_folder" in data:
        folder = str(data["output_folder"]).strip()
        if folder:
            # Check if path is valid and parent exists (or path exists)
            folder_path = Path(folder)
            if folder_path.exists():
                if not folder_path.is_dir():
                    errors.append(f"output_folder '{folder}' exists but is not a directory")
                elif not os.access(folder_path, os.W_OK):
                    errors.append(f"output_folder '{folder}' is not writable")
                else:
                    validated["output_folder"] = folder
            elif folder_path.parent.exists():
                # Parent exists - we can create the folder
                if not os.access(folder_path.parent, os.W_OK):
                    errors.append(f"output_folder parent '{folder_path.parent}' is not writable")
                else:
                    validated["output_folder"] = folder
            else:
                errors.append(f"output_folder path '{folder}' has no valid parent directory")
        else:
            # Empty string = use default (valid)
            validated["output_folder"] = ""

    if "handler_command" in data:
        cmd = str(data["handler_command"]).strip()
        # Basic validation - must be a string, no special checks
        # (we can't validate if the command exists since it might be a full path
        # or use shell features like pipes)
        validated["handler_command"] = cmd

    if "always_run_handler" in data:
        val = data["always_run_handler"]
        if isinstance(val, bool):
            validated["always_run_handler"] = val
        elif isinstance(val, str):
            if val.lower() in ('true', '1', 'yes', 'on'):
                validated["always_run_handler"] = True
            elif val.lower() in ('false', '0', 'no', 'off'):
                validated["always_run_handler"] = False
            else:
                errors.append(f"always_run_handler must be boolean, got string '{val}'")
        elif isinstance(val, (int, float)):
            validated["always_run_handler"] = bool(val)
        else:
            errors.append(f"always_run_handler must be boolean, got {type(val).__name__}")

    return validated, errors


@server.PromptServer.instance.routes.post("/conduit/config")
async def set_config(request):
    """
    Update Conduit configuration.

    Accepts any of:
        - output_folder: string (path or empty for default)
        - handler_command: string (shell command)
        - always_run_handler: boolean

    Returns validation errors if any values are invalid.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response(
            {"status": "error", "message": "Invalid JSON body"},
            status=400
        )

    # Validate inputs
    validated, errors = validate_config(data)

    if errors:
        return web.json_response({
            "status": "error",
            "message": "Validation failed",
            "errors": errors,
        }, status=400)

    # Apply validated changes
    config = load_config()
    changed = []

    for key in ["output_folder", "handler_command", "always_run_handler"]:
        if key in validated:
            config[key] = validated[key]
            changed.append(key)

    if changed:
        save_config(config)
        print(f"[Conduit] Config updated: {changed}")

    return web.json_response({
        "status": "saved",
        "changed": changed,
        "config": config,
    })


@server.PromptServer.instance.routes.get("/conduit/status")
async def get_status(request):
    """
    Get a unified overview of Conduit system status.

    Returns config, saved workflows, and system info in one call.
    """
    from .conduit_workflows import WORKFLOWS_DIR

    config = load_config()
    default_output = str(Path(folder_paths.get_output_directory()) / "conduit")

    # Get saved workflows
    workflows = []
    for file_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            schema = data.get("schema", {})
            workflows.append({
                "name": file_path.stem,
                "inputs": len(schema.get("inputs", [])),
                "outputs": len(schema.get("outputs", [])),
                "updated_at": data.get("updated_at", data.get("created_at")),
            })
        except Exception:
            workflows.append({"name": file_path.stem, "error": "Failed to load"})

    # Get handler info
    registry = get_handlers_registry()
    handlers = {
        hid: {
            "name": info.get("name", hid),
            "description": info.get("description", ""),
            "enabled": info.get("enabled", False),
        }
        for hid, info in registry.items()
    }

    return web.json_response({
        "config": {
            "output_folder": config.get("output_folder", "") or default_output,
            "always_run_handler": config.get("always_run_handler", False),
        },
        "handlers": {
            "registered": handlers,
            "defaults": get_default_handlers(),
        },
        "workflows": {
            "count": len(workflows),
            "list": workflows,
            "directory": str(WORKFLOWS_DIR),
        },
        "endpoints": {
            "run": "POST /conduit/run/{workflow_name}",
            "save": "POST /conduit/save",
            "list": "GET /conduit/workflows",
            "config": "GET|POST /conduit/config",
            "handlers": "GET /conduit/handlers",
            "status": "GET /conduit/status",
        },
    })


# =============================================================================
# Handler API Endpoints
# =============================================================================

@server.PromptServer.instance.routes.get("/conduit/handlers")
async def list_handlers(request):
    """
    List all registered handlers with their metadata.

    Returns:
        {
            "handlers": {
                "handler_id": {
                    "id": "handler_id",
                    "name": "Display Name",
                    "description": "What this handler does",
                    "enabled": true/false
                },
                ...
            },
            "defaults": ["handler_id", ...]  // Currently enabled handlers
        }
    """
    registry = get_handlers_registry()

    handlers = {}
    for handler_id, info in registry.items():
        handlers[handler_id] = {
            "id": handler_id,
            "name": info.get("name", handler_id),
            "description": info.get("description", ""),
            "enabled": info.get("enabled", False),
        }

    return web.json_response({
        "handlers": handlers,
        "defaults": get_default_handlers(),
    })


@server.PromptServer.instance.routes.post("/conduit/handlers/{handler_id}/enable")
async def enable_handler(request):
    """Enable a handler by default."""
    handler_id = request.match_info["handler_id"]

    if set_handler_enabled(handler_id, True):
        print(f"[Conduit] Enabled handler: {handler_id}")
        return web.json_response({
            "status": "enabled",
            "handler_id": handler_id,
            "defaults": get_default_handlers(),
        })
    else:
        return web.json_response(
            {"status": "error", "message": f"Handler not found: {handler_id}"},
            status=404
        )


@server.PromptServer.instance.routes.post("/conduit/handlers/{handler_id}/disable")
async def disable_handler(request):
    """Disable a handler by default."""
    handler_id = request.match_info["handler_id"]

    if set_handler_enabled(handler_id, False):
        print(f"[Conduit] Disabled handler: {handler_id}")
        return web.json_response({
            "status": "disabled",
            "handler_id": handler_id,
            "defaults": get_default_handlers(),
        })
    else:
        return web.json_response(
            {"status": "error", "message": f"Handler not found: {handler_id}"},
            status=404
        )


@server.PromptServer.instance.routes.post("/conduit/handlers/defaults")
async def set_default_handlers(request):
    """
    Set which handlers are enabled by default.

    Body: { "handlers": ["handler_id1", "handler_id2", ...] }
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response(
            {"status": "error", "message": "Invalid JSON body"},
            status=400
        )

    requested = data.get("handlers", [])
    if not isinstance(requested, list):
        return web.json_response(
            {"status": "error", "message": "handlers must be an array"},
            status=400
        )

    registry = get_handlers_registry()
    config = load_config()

    # Validate all requested handlers exist
    unknown = [h for h in requested if h not in registry]
    if unknown:
        return web.json_response(
            {"status": "error", "message": f"Unknown handlers: {unknown}"},
            status=400
        )

    # Update all handlers
    for handler_id in registry:
        config["handlers"][handler_id]["enabled"] = (handler_id in requested)

    save_config(config)

    print(f"[Conduit] Set default handlers: {requested}")
    return web.json_response({
        "status": "updated",
        "defaults": get_default_handlers(),
    })


# =============================================================================
# Dynamic Node Creation
# =============================================================================

def create_conduit_output_node(data_type: str):
    """Create a ConduitOutput node class for a specific data type."""

    class ConduitOutputNode:
        """Captures output data, saves to file, accumulates for handler."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "data": (data_type,),
                    "tag_name": ("STRING", {"default": "output"}),
                },
                "optional": {
                    "prompt_id": ("STRING", {"default": ""}),
                }
            }

        RETURN_TYPES = (data_type,)
        RETURN_NAMES = ("data",)
        FUNCTION = "capture"
        CATEGORY = "Conduit/Outputs"
        OUTPUT_NODE = True

        def capture(self, data, tag_name: str, prompt_id: str = ""):
            if not prompt_id:
                prompt_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + os.urandom(2).hex()

            # Get output folder from config
            output_folder = get_output_folder()
            exec_folder = output_folder / prompt_id

            # Save the data
            file_path, file_type = save_output(data, exec_folder, tag_name)
            print(f"[Conduit] Captured '{tag_name}' -> {file_path}")

            # Accumulate output info (handler called later via /conduit/flush)
            accumulate_output(prompt_id, {
                "file_path": str(file_path),
                "tag_name": tag_name,
                "file_type": file_type,
                "data_type": data_type,
            })

            return (data,)

    return ConduitOutputNode


# =============================================================================
# Node Registration
# =============================================================================

CONDUIT_OUTPUT_MAPPINGS = {}
CONDUIT_OUTPUT_DISPLAY_NAMES = {}

# Create per-type nodes
all_data_types = set()

if hasattr(nodes, 'NODE_CLASS_MAPPINGS'):
    for node_cls in nodes.NODE_CLASS_MAPPINGS.values():
        if hasattr(node_cls, 'RETURN_TYPES'):
            try:
                for rt in node_cls.RETURN_TYPES:
                    if isinstance(rt, str):
                        all_data_types.add(rt)
            except Exception:
                pass

for data_type in sorted(all_data_types):
    safe_type = "".join(c if c.isalnum() else "_" for c in data_type)
    node_name = f"ConduitOutput_{safe_type}"
    node_cls = create_conduit_output_node(data_type)
    node_cls.__name__ = node_name
    node_cls.__qualname__ = node_name
    CONDUIT_OUTPUT_MAPPINGS[node_name] = node_cls
    CONDUIT_OUTPUT_DISPLAY_NAMES[node_name] = f"Conduit Output ({data_type})"

print(f"[Conduit] Created {len(CONDUIT_OUTPUT_MAPPINGS)} output nodes")
print(f"[Conduit] Config: {CONFIG_PATH}")
print(f"[Conduit] Output folder: {get_output_folder()}")
print(f"[Conduit] Handlers registered: {list(get_handlers_registry().keys())}")
print(f"[Conduit] Default handlers: {get_default_handlers()}")
