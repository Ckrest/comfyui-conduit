# Notes - ComfyUI Conduit

Brief context for agents working with this package.

## Build / Run

**Installation:**
```bash
# Symlink to ComfyUI custom_nodes
ln -s /home/nick/Systems/ai/comfyui-conduit /home/nick/Systems/ai/comfyui/custom_nodes/comfyui-conduit
systemctl --user restart comfyui
```

**Development:**
Changes to Python files require ComfyUI restart.
Changes to JS files require browser refresh.

## Path Dependencies

| Path | Purpose |
|------|---------|
| `js/` | Web UI extensions |
| `handlers/` | Custom output handlers |
| `overrides/` | Patches for other nodes |
| `conduit_workflows/` | Example workflow files |

## Key Files

- `conduit_gateway.py` - Main API entry point
- `conduit_registry.py` - Socket tag management
- `conduit_introspection.py` - Workflow analysis
- `conduit_outputs.py` - Output capture system

## Integration Points

- Used by MCP tools for workflow execution
- Integrates with comfy-viewer for workflow management
