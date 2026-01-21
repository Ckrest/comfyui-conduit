# Conduit

Socket tagging system for ComfyUI that enables external workflow execution with typed inputs and captured outputs.

## Quick Start

**1. Tag sockets** - Right-click any socket to tag it as an input or output

**2. Save workflow** - Open Conduit sidebar, click "Save Workflow"

**3. Call externally**:
```bash
# Minimal - all inputs are optional, uses workflow defaults
curl -X POST http://localhost:8188/conduit/run/MyWorkflow

# With overrides
curl -X POST http://localhost:8188/conduit/run/MyWorkflow \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"prompt": "a sunset over mountains", "steps": 30}}'
```

## Tagging Sockets

| Action | Effect |
|--------|--------|
| Right-click socket | Toggle tag on/off |
| Shift+Right-click | Tag (if needed) + rename |
| Hover tagged socket | Shows tag name tooltip |

**Tag names** default to `slotName (nodeTitle #nodeId)` format, ensuring uniqueness. Custom names must also be unique across the workflow.

**Copy/paste**: Tags are cleared when nodes are pasted - each socket must be tagged manually.

## API Reference

### Execute Workflow

```
POST /conduit/run/{workflow_name}
```

**Request body** (all fields optional):
```json
{
  "inputs": {
    "tagName": "value"
  },
  "wait": true,
  "timeout": 120,
  "handlers": ["redis_publish"],
  "mimicry": true,
  "context": {"source": "my_app"}
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `inputs` | `{}` | Values for tagged inputs (all optional) |
| `wait` | `true` | Wait for completion vs return immediately |
| `timeout` | `120` | Max seconds to wait (5-600) |
| `handlers` | workflow default | Output handler override |
| `mimicry` | `true` | Apply frontend transforms (seed randomization) |
| `context` | `{}` | Metadata passed to handlers |

**Input handling**:
- Missing inputs use workflow defaults
- `null` values explicitly use defaults
- Type coercion (string "25" → int 25)
- Unknown inputs generate warnings but don't fail

**Response**:
```json
{
  "status": "success",
  "prompt_id": "20260105_abc123",
  "outputs": [...],
  "inputs_applied": {"steps": {"value": 30, "type": "INT"}},
  "inputs_defaulted": [{"tagName": "seed", "reason": "not_provided"}],
  "warnings": []
}
```

### Workflow Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conduit/workflows` | GET | List all saved workflows |
| `/conduit/workflows` | POST | Save workflow with schema |
| `/conduit/workflows/{name}` | GET | Get workflow + schema |
| `/conduit/workflows/{name}` | DELETE | Delete workflow |

### Introspection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conduit/workflows/{name}/schema` | GET | Input schema for external callers |
| `/conduit/workflows/{name}/sockets` | GET | All sockets (tagged + untagged) |
| `/conduit/workflows/{name}/graph` | GET | Node/edge topology |
| `/conduit/workflows/{name}/socket/{node_id}/{side}/{slot}` | GET | Socket detail |
| `/conduit/sockets` | GET | All tagged sockets across all workflows |
| `/conduit/workflows/diff?a=X&b=Y` | GET | Compare two workflows |

### Registry (Node Definitions)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conduit/registry/node/{class_type}` | GET | Node input/output spec |
| `/conduit/registry/widget/{class_type}/{input}` | GET | Widget constraints |
| `/conduit/workflows/{name}/inputs` | GET | Enriched input info |

## Schema Format

**`/conduit/workflows/{name}/schema`** - What external callers need:
```json
{
  "inputs": {
    "prompt": {
      "type": "STRING",
      "required": false,
      "default": "a photo",
      "constraints": {"multiline": true}
    },
    "steps": {
      "type": "INT",
      "required": false,
      "default": 20,
      "constraints": {"min": 1, "max": 100}
    }
  },
  "outputs": ["final_image"],
  "example_request": {
    "inputs": {"prompt": "a sunset", "steps": 30}
  }
}
```

## Handler System

Handlers process workflow outputs. Configure in `conduit_config.json`:

```json
{
  "output_folder": "output/conduit",
  "always_run_handler": false,
  "handlers": {
    "redis_publish": {
      "name": "Redis Publisher",
      "command": "handlers/redis_handler.py",
      "enabled": true
    }
  }
}
```

Handlers receive JSON on stdin:
```json
{
  "prompt_id": "20260105_abc123",
  "workflow_name": "MyWorkflow",
  "outputs": {
    "final_image": {
      "images": [{"filename": "output.png", "subfolder": "", "type": "output"}]
    }
  },
  "context": {"source": "my_app"}
}
```

## Frontend Mimicry

Conduit replicates frontend behavior when executing workflows via API:

- **Control widgets**: Inputs with `control_after_generate` widget are processed according to their mode (randomize/increment/decrement/fixed)
- **Dynamic prompts**: `{red|blue|green}` syntax is resolved randomly
- **Array wrapping**: Array values are wrapped for backend compatibility

**User-provided values**: When you explicitly provide a value via the API, control widget transforms are skipped for that input. Your value is used exactly as provided. Dynamic prompts and array wrapping still apply.

```bash
# Workflow seed has control_after_generate="randomize"
# But user provides explicit seed - it will NOT be randomized
curl -X POST http://localhost:8188/conduit/run/MyWorkflow \
  -d '{"inputs": {"seed": 12345}}'  # Uses exactly 12345
```

Disable all mimicry per-request with `"mimicry": false` or per-workflow in its saved schema.

## Node Overrides

Some custom nodes have complex frontend logic that default mimicry can't handle. Register a node override in `overrides/` to provide custom handling:

```python
# overrides/my_node.py
from conduit_overrides import register_node_override, MimicryContext, NodeOverrideResult

@register_node_override("MyComplexNode")
async def override_my_node(node_data: dict, ctx: MimicryContext) -> NodeOverrideResult:
    inputs = node_data.get("inputs", {})

    # Custom transform
    if "special_input" in inputs:
        inputs["special_input"] = transform(inputs["special_input"])

    return NodeOverrideResult(
        node_data=node_data,
        handled_inputs={"special_input"},  # Skip default mimicry for these
    )
```

Inputs not in `handled_inputs` still get default mimicry transforms.

## Files

```
conduit/
├── __init__.py                 # Entry point, node registration
├── conduit_gateway.py          # /conduit/run endpoint
├── conduit_workflows.py        # Workflow save/load/list
├── conduit_outputs.py          # Output capture + handlers
├── conduit_introspection.py    # Schema/socket/graph endpoints
├── conduit_registry.py         # Node definition cache
├── conduit_mimicry.py          # Frontend transform replication
├── conduit_validation.py       # Input validation + coercion
├── conduit_config.json         # Handler configuration
├── conduit_workflows/          # Saved workflows (gitignored)
├── handlers/                   # Output handler scripts
├── overrides/                  # Node-specific transform overrides
│   └── lora_manager.py         # Example: LoraManager override
└── js/                         # Frontend
    ├── conduit.js              # Extension entry point
    └── conduit/
        ├── core.js             # Tag data management
        ├── panel.js            # Sidebar UI
        ├── tooltip.js          # Hover tooltips
        ├── click-handlers.js   # Right-click tagging
        ├── queue-hook.js       # Output node injection
        └── save-hook.js        # Auto-save detection
```

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PLACEHOLDER/conduit
# Restart ComfyUI
```

## Troubleshooting

**Workflow not found**: Save via Conduit sidebar, not regular ComfyUI save.

**Inputs not applying**: Tag names are case-sensitive. Check `/conduit/workflows/{name}/schema` for exact names.

**Handlers not running**: Set `always_run_handler: true` and `enabled: true` in config.

**Stale frontend**: Hard refresh browser (Ctrl+Shift+R) after backend updates.

## License

MIT License
