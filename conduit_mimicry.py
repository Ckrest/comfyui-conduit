"""
Conduit Mimicry - Frontend execution emulation for backend API calls.

This module replicates frontend widget transformations (serializeValue, beforeQueued,
afterQueued) so that workflows executed via the Conduit API behave similarly to
workflows executed from the ComfyUI frontend.

Key features:
- Pattern-first: Auto-detects and handles common patterns (control widgets, dynamic prompts)
- Explicit overrides: Allows registration of Python equivalents for complex nodes
- Transparent: Returns detailed results showing what transforms were applied

Usage:
    from .conduit_mimicry import apply_mimicry

    results = await apply_mimicry(workflow, user_inputs, input_sockets)
    for tag, result in results.items():
        user_inputs[tag] = result.value
"""

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .conduit_registry import (
    get_node_def,
    get_input_spec,
    ensure_registry_loaded,
    InputSpec,
    InputKind,
    MimicryCategory,
    ControlWidgetPair,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MimicryResult:
    """Result of applying mimicry to a single input value"""
    value: Any                          # Transformed (or original) value
    status: str                         # "applied", "passthrough", "warn"
    confidence: float = 1.0             # 0.0 to 1.0
    warning: Optional[str] = None       # Warning message if any
    transform_applied: Optional[str] = None  # Which transform was used
    original_value: Optional[Any] = None     # For debugging
    control_mode: Optional[str] = None  # If control widget was involved
    note: Optional[str] = None          # Additional context (e.g., for increment warning)
    value_source: str = "user"          # "user" or "workflow" - where the input value came from

    def to_dict(self) -> dict:
        result = {
            "value": self.value,
            "status": self.status,
            "confidence": self.confidence,
        }
        if self.warning:
            result["warning"] = self.warning
        if self.transform_applied:
            result["transform"] = self.transform_applied
        if self.original_value is not None and self.original_value != self.value:
            result["original"] = self.original_value
        if self.control_mode:
            result["control_mode"] = self.control_mode
        if self.note:
            result["note"] = self.note
        if self.value_source:
            result["value_source"] = self.value_source
        return result


@dataclass
class MimicryContext:
    """Context passed to override functions"""
    workflow: dict                      # Full workflow dict
    node_id: str                        # Current node ID
    node_data: dict                     # Current node's data from workflow
    class_type: str                     # Node class type
    input_name: Optional[str] = None    # Current input name
    input_spec: Optional[InputSpec] = None  # From registry
    other_inputs: Dict[str, Any] = field(default_factory=dict)  # Other inputs on this node
    prompt_id: str = ""                 # Conduit prompt ID
    batch_index: int = 0                # For batch execution


@dataclass
class NodeOverrideResult:
    """
    Result from a node-level override.

    Allows overrides to:
    - Modify node data (inputs, etc.)
    - Specify which inputs were handled (skip default mimicry for these)
    - Optionally skip the entire node from execution

    Usage:
        @register_node_override("KSampler")
        async def override_ksampler(node_data: dict, ctx: MimicryContext) -> NodeOverrideResult:
            inputs = node_data.get("inputs", {})

            # Custom seed handling
            if "seed" in inputs:
                inputs["seed"] = my_custom_logic(inputs["seed"])

            return NodeOverrideResult(
                node_data=node_data,
                handled_inputs={"seed"},  # Skip default mimicry for seed
                # steps, cfg, etc. will still get default mimicry
            )
    """
    node_data: dict                     # Modified node data
    handled_inputs: set = field(default_factory=set)  # Inputs that were specially handled
    skip_node: bool = False             # If True, remove node from workflow execution
    changes: Dict[str, Any] = field(default_factory=dict)  # Track what changed for logging


# =============================================================================
# Override Registry
# =============================================================================

# Dict of class_type -> async node-level transformer
_node_override_registry: Dict[str, Callable] = {}


def register_node_override(class_type: str):
    """
    Decorator to register a Python override for an entire node.

    When a node override is registered, it takes FULL CONTROL of that node's
    processing during mimicry. The override can:
    - Modify any/all inputs on the node
    - Mark specific inputs as "handled" to skip default transforms
    - Skip the node entirely from execution

    Usage:
        @register_node_override("LoraManager")
        async def override_lora_manager(node_data: dict, ctx: MimicryContext) -> NodeOverrideResult:
            inputs = node_data.get("inputs", {})

            # Custom processing for specific inputs
            if "loras" in inputs:
                inputs["loras"] = transform_loras(inputs["loras"])

            return NodeOverrideResult(
                node_data=node_data,
                handled_inputs={"loras", "trigger_words"},  # Skip default mimicry for these
            )
    """
    def decorator(func: Callable):
        _node_override_registry[class_type] = func
        return func
    return decorator


def get_node_override(class_type: str) -> Optional[Callable]:
    """Get registered node-level override"""
    return _node_override_registry.get(class_type)


# =============================================================================
# Built-in Transformers
# =============================================================================

def apply_control_mode(value: Any, mode: str, spec: Optional[InputSpec]) -> MimicryResult:
    """
    Apply control widget transformation to a value.

    For randomize: generates a new random value (matches frontend formula with step support)
    For increment/decrement: uses the current value (state changes are reported separately)
    For fixed: no change

    Args:
        value: Current value
        mode: Control mode ("fixed", "randomize", "increment", "decrement")
        spec: Input specification (for min/max/step)

    Returns:
        MimicryResult with transformed value
    """
    original = value

    if mode == "fixed":
        return MimicryResult(
            value=value,
            status="passthrough",
            control_mode="fixed",
            original_value=original,
        )

    if mode == "randomize":
        if spec and spec.type_name == "INT":
            min_val = int(spec.min_value or 0)
            # Match frontend: limit to what JavaScript can handle
            max_val = min(int(spec.max_value or 2**31 - 1), 1125899906842624)
            min_val = max(min_val, -1125899906842624)
            step = int(spec.step or 1)

            # Match frontend formula: Math.floor(Math.random() * range) * step + min
            # where range = (max - min) / step
            range_steps = (max_val - min_val) // step
            new_value = random.randint(0, range_steps) * step + min_val

            return MimicryResult(
                value=new_value,
                status="applied",
                confidence=1.0,
                transform_applied="control_randomize",
                original_value=original,
                control_mode="randomize",
            )
        elif spec and spec.type_name == "FLOAT":
            min_val = float(spec.min_value or 0.0)
            max_val = float(spec.max_value or 1.0)
            step = float(spec.step or 0.01)

            # Match frontend formula for floats
            range_steps = int((max_val - min_val) / step)
            new_value = random.randint(0, range_steps) * step + min_val

            return MimicryResult(
                value=new_value,
                status="applied",
                confidence=1.0,
                transform_applied="control_randomize",
                original_value=original,
                control_mode="randomize",
            )
        else:
            # Unknown type, try random int with default range
            new_value = random.randint(0, 2**31 - 1)
            return MimicryResult(
                value=new_value,
                status="applied",
                confidence=0.7,
                transform_applied="control_randomize",
                original_value=original,
                control_mode="randomize",
                warning="Unknown type for randomize, using random int",
            )

    if mode in ("increment", "decrement"):
        # In stateless API: use current value for execution
        # State progression (next_value) is calculated separately and returned in response
        return MimicryResult(
            value=value,
            status="passthrough",
            confidence=1.0,
            original_value=original,
            control_mode=mode,
            note=f"Executed with current value; next_value reported in state_progression",
        )

    # Unknown mode - pass through
    return MimicryResult(
        value=value,
        status="passthrough",
        control_mode=mode,
        original_value=original,
        warning=f"Unknown control mode: {mode}",
    )


def compute_next_value(value: Any, mode: str, spec: Optional[InputSpec]) -> Optional[Any]:
    """
    Compute what the value would be AFTER the workflow runs.

    This matches frontend behavior:
    - randomize: the randomized value (what was used)
    - increment: value + step, clamped to max
    - decrement: value - step, clamped to min
    - fixed: same value (no change)

    Args:
        value: The value that was used for execution
        mode: Control mode
        spec: Input specification (for min/max/step)

    Returns:
        The next value, or None if not applicable
    """
    if mode == "fixed":
        return value  # No change

    if mode == "randomize":
        return value  # For randomize, "next" is the value we just used

    if mode == "increment":
        if spec and spec.type_name == "INT":
            step = int(spec.step or 1)
            max_val = min(int(spec.max_value or 2**31 - 1), 1125899906842624)
            next_val = int(value) + step
            # Clamp to max (frontend behavior - no wrap)
            return min(next_val, max_val)
        elif spec and spec.type_name == "FLOAT":
            step = float(spec.step or 0.01)
            max_val = float(spec.max_value or 1e10)
            next_val = float(value) + step
            return min(next_val, max_val)
        else:
            # Best effort for unknown type
            return value + 1 if isinstance(value, (int, float)) else value

    if mode == "decrement":
        if spec and spec.type_name == "INT":
            step = int(spec.step or 1)
            min_val = max(int(spec.min_value or 0), -1125899906842624)
            next_val = int(value) - step
            # Clamp to min (frontend behavior - no wrap)
            return max(next_val, min_val)
        elif spec and spec.type_name == "FLOAT":
            step = float(spec.step or 0.01)
            min_val = float(spec.min_value or 0.0)
            next_val = float(value) - step
            return max(next_val, min_val)
        else:
            # Best effort for unknown type
            return value - 1 if isinstance(value, (int, float)) else value

    return None  # Unknown mode


def resolve_dynamic_prompt(text: str) -> MimicryResult:
    """
    Resolve {option1|option2|...} patterns in text.

    Args:
        text: String potentially containing dynamic prompt patterns

    Returns:
        MimicryResult with resolved text
    """
    if not isinstance(text, str):
        return MimicryResult(value=text, status="passthrough")

    pattern = r'\{([^{}]+)\}'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return MimicryResult(value=text, status="passthrough")

    original = text
    resolved = text

    def replace_match(match):
        options = match.group(1).split('|')
        return random.choice(options)

    resolved = re.sub(pattern, replace_match, text)

    return MimicryResult(
        value=resolved,
        status="applied",
        confidence=1.0,
        transform_applied="dynamic_prompt",
        original_value=original,
    )


def wrap_array(value: list) -> MimicryResult:
    """
    Wrap array values as {__value__: [...]} for ComfyUI backend.

    This distinguishes array widget values from node connection references.
    """
    if not isinstance(value, list):
        return MimicryResult(value=value, status="passthrough")

    wrapped = {"__value__": value}
    return MimicryResult(
        value=wrapped,
        status="applied",
        confidence=1.0,
        transform_applied="array_wrap",
        original_value=value,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

async def apply_workflow_mimicry(
    workflow: dict,
    user_provided_inputs: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply frontend mimicry transforms to the ENTIRE workflow.

    This is the UNIFIED mimicry system - it processes ALL nodes in the workflow,
    treating tagged and untagged inputs exactly the same. This matches frontend
    behavior where beforeQueued callbacks run on every node in the graph.

    Processing order:
    1. Node-level overrides (if registered) - can handle specific inputs and mark them as handled
    2. Default transforms for non-handled inputs:
       - Control widgets (seed randomization via control_after_generate)
       - Dynamic prompts ({option1|option2} resolution)
       - Array wrapping (for backend compatibility)

    User-provided inputs (via API) skip control widget transforms but still get
    dynamic prompt resolution and array wrapping. This ensures explicit user values
    are respected while still applying necessary structural transforms.

    Args:
        workflow: The workflow dict (will be modified in place)
        user_provided_inputs: Optional set of (node_id, input_name) tuples that
            were explicitly provided by the user via API. These skip control
            widget transforms (randomize/increment/decrement) but still get
            other transforms applied.

    Returns:
        Dict mapping node_id -> {input_name -> change_info} for all changes made
    """
    await ensure_registry_loaded()

    changes: Dict[str, Dict[str, Any]] = {}
    nodes_to_remove = []
    user_provided = user_provided_inputs or set()

    for node_id, node_data in list(workflow.items()):
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        if not class_type:
            continue

        node_def = get_node_def(class_type)
        inputs = node_data.get("inputs", {})

        # Track which inputs have been handled (skip default mimicry for these)
        handled_inputs: set = set()

        # =================================================================
        # Phase 1: Node-level overrides (highest priority)
        # =================================================================
        node_override = get_node_override(class_type)
        if node_override:
            try:
                ctx = MimicryContext(
                    workflow=workflow,
                    node_id=node_id,
                    node_data=node_data,
                    class_type=class_type,
                    other_inputs=inputs,
                )
                result = await node_override(node_data, ctx)

                if isinstance(result, NodeOverrideResult):
                    # Handle skip_node
                    if result.skip_node:
                        nodes_to_remove.append(node_id)
                        if result.changes:
                            changes[node_id] = result.changes
                        continue

                    # Apply modified node_data
                    workflow[node_id] = result.node_data
                    node_data = result.node_data
                    inputs = node_data.get("inputs", {})

                    # Mark handled inputs
                    handled_inputs = result.handled_inputs

                    # Record changes
                    if result.changes:
                        if node_id not in changes:
                            changes[node_id] = {}
                        changes[node_id].update(result.changes)

                elif isinstance(result, dict):
                    # Legacy support: just return modified node_data
                    workflow[node_id] = result
                    node_data = result
                    inputs = node_data.get("inputs", {})

            except Exception as e:
                print(f"[Conduit Mimicry] Node override failed for {class_type}: {e}")

        # =================================================================
        # Phase 2: Default transforms for non-handled inputs
        # =================================================================

        # Process each input on this node
        for input_name, value in list(inputs.items()):
            # Skip if already handled by node override
            if input_name in handled_inputs:
                continue

            # Skip linked inputs (node references)
            if isinstance(value, list) and len(value) == 2:
                continue

            # Get input spec for type info
            input_spec = get_input_spec(class_type, input_name) if node_def else None

            # Check if this is a user-provided input (skip control transforms)
            is_user_provided = (node_id, input_name) in user_provided

            # Control widget handling via control_after_generate flag
            # This flag comes from object_info and indicates the frontend adds a
            # randomize/increment/decrement control widget. Default behavior is "randomize".
            if input_spec and input_spec.control_after_generate and not is_user_provided:
                result = apply_control_mode(value, "randomize", input_spec)
                if result.status == "applied":
                    inputs[input_name] = result.value
                    if node_id not in changes:
                        changes[node_id] = {}
                    changes[node_id][input_name] = {
                        "old": value,
                        "new": result.value,
                        "reason": "control_after_generate:randomize"
                    }
                continue

            # Dynamic prompts in strings (applies to ALL inputs, including user-provided)
            if isinstance(value, str) and "{" in value and "|" in value:
                result = resolve_dynamic_prompt(value)
                if result.status == "applied":
                    inputs[input_name] = result.value
                    if node_id not in changes:
                        changes[node_id] = {}
                    changes[node_id][input_name] = {
                        "old": value,
                        "new": result.value,
                        "reason": "dynamic_prompt"
                    }
                continue

            # Array wrapping (applies to ALL inputs, including user-provided)
            # ComfyUI backend needs arrays wrapped as {"__value__": [...]}
            if isinstance(value, list) and not (len(value) == 2 and isinstance(value[0], str)):
                # Skip link references [node_id, slot_index] - those are handled above
                result = wrap_array(value)
                if result.status == "applied":
                    inputs[input_name] = result.value
                    if node_id not in changes:
                        changes[node_id] = {}
                    changes[node_id][input_name] = {
                        "old": value,
                        "new": result.value,
                        "reason": "array_wrap"
                    }

    # Remove skipped nodes
    for node_id in nodes_to_remove:
        del workflow[node_id]
        if node_id not in changes:
            changes[node_id] = {}
        changes[node_id]["__skipped__"] = True

    return changes


# =============================================================================
# Module Initialization
# =============================================================================

print("[Conduit Mimicry] Module loaded")
print("[Conduit Mimicry] Built-in transforms: control_mode, dynamic_prompt, array_wrap")
