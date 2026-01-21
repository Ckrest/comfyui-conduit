"""
Conduit Overrides - Explicit Python handlers for complex frontend transformations.

This module provides a registration system for custom nodes that have frontend logic
too complex for pattern detection to handle automatically.

Most nodes should work automatically via pattern detection. Only use overrides when:
- The node has complex serializeValue logic
- The transformation involves multiple widget interactions
- Pattern detection would produce incorrect results

NODE-LEVEL OVERRIDES - Run on ALL nodes of this type during workflow mimicry:

    from conduit_overrides import register_node_override, MimicryContext, NodeOverrideResult

    @register_node_override("LoraManager")
    async def override_lora_manager(node_data: dict, ctx: MimicryContext) -> NodeOverrideResult:
        inputs = node_data.get("inputs", {})

        # Custom processing for specific inputs
        if "loras" in inputs:
            inputs["loras"] = transform_loras(inputs["loras"])

        return NodeOverrideResult(
            node_data=node_data,
            handled_inputs={"loras"},  # Skip default mimicry for these
            changes={"loras": {"old": ..., "new": ..., "reason": "custom"}},
        )

When a node override is registered, it takes FULL CONTROL of that node's processing.
The override can:
- Modify any/all inputs on the node
- Mark specific inputs as "handled" to skip default transforms
- Skip the node entirely from execution (skip_node=True)

Inputs NOT in handled_inputs still get default mimicry (seed randomization, etc.)

This module auto-discovers and loads all override files in this directory.
"""

import sys
import importlib
import importlib.util
from pathlib import Path


def load_overrides():
    """
    Auto-discover and load all override modules in this directory.

    Any .py file (except __init__.py) will be imported, which triggers
    the @register_node_override decorators to register their handlers.
    """
    # Import mimicry module to get the symbols we need
    from ..conduit_mimicry import (
        register_node_override,
        MimicryContext,
        NodeOverrideResult,
    )

    # Create a virtual module that override files can import from
    # This avoids the "No module named 'conduit'" issue
    import types
    conduit_overrides = types.ModuleType("conduit_overrides")
    conduit_overrides.register_node_override = register_node_override
    conduit_overrides.MimicryContext = MimicryContext
    conduit_overrides.NodeOverrideResult = NodeOverrideResult
    sys.modules["conduit_overrides"] = conduit_overrides

    overrides_dir = Path(__file__).parent
    loaded = []

    for py_file in overrides_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem

        try:
            spec = importlib.util.spec_from_file_location(
                f"conduit_overrides.{module_name}",
                py_file,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded.append(module_name)
        except Exception as e:
            print(f"[Conduit Overrides] Failed to load {module_name}: {e}")

    if loaded:
        print(f"[Conduit Overrides] Loaded: {', '.join(loaded)}")
    else:
        print("[Conduit Overrides] No override modules found (this is fine)")
