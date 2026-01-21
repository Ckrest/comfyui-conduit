"""
Conduit - Socket Tagging System for ComfyUI

Capture, route, and handle workflow data through a tagging interface.

Features:
- Socket Tagging: Right-click any socket to tag it for capture or input
- Workflow Export: Save tagged socket configurations with your workflow
- Handler System: Process captured outputs with custom handlers
- Queue Hook: Automatically inject capture nodes when executing workflows

https://github.com/PLACEHOLDER/conduit
"""

# ==============================================================================
# Backend: Diagnostic logging endpoint
# ==============================================================================
from . import conduit_diagnostic

# ==============================================================================
# Backend: Output capture system (creates ConduitOutput_* nodes dynamically)
# ==============================================================================
from .conduit_outputs import CONDUIT_OUTPUT_MAPPINGS, CONDUIT_OUTPUT_DISPLAY_NAMES

# ==============================================================================
# Backend: Workflow registry (save/load workflows with tagged schemas)
# ==============================================================================
from . import conduit_workflows

# ==============================================================================
# Backend: Gateway API for external workflow execution
# ==============================================================================
from . import conduit_gateway

# ==============================================================================
# Backend: Node Registry with frontend-parity validation
# ==============================================================================
from . import conduit_registry

# ==============================================================================
# Backend: Frontend Mimicry System for API parity
# ==============================================================================
from . import conduit_mimicry

# ==============================================================================
# Backend: Introspection API for graph/socket discovery
# ==============================================================================
from . import conduit_introspection

# ==============================================================================
# Backend: Explicit Overrides for complex frontend transforms
# ==============================================================================
from .overrides import load_overrides
load_overrides()

# ==============================================================================
# Frontend: Makes the 'js' folder available to ComfyUI frontend
# ==============================================================================
WEB_DIRECTORY = "js"

# ==============================================================================
# Node Registration
# ==============================================================================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge dynamically-created ConduitOutput nodes
NODE_CLASS_MAPPINGS.update(CONDUIT_OUTPUT_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONDUIT_OUTPUT_DISPLAY_NAMES)

# Required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("------------------------------------------")
print("### Conduit loaded ###")
print(f"  > Output nodes: {len(CONDUIT_OUTPUT_MAPPINGS)} types")
print(f"  > Workflow registry: Active")
print(f"  > Gateway API: Active")
print(f"  > Node Registry: Active (lazy-load)")
print(f"  > Frontend Mimicry: Active")
print(f"  > Introspection API: Active")
print(f"  > Frontend: js/")
print("------------------------------------------")
