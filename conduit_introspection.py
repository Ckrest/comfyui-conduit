"""
Conduit Introspection Module
API endpoints for inspecting workflow graphs, sockets, and schemas.

Endpoints:
- GET /conduit/workflows/{name}/sockets - All sockets (tagged + untagged)
- GET /conduit/workflows/{name}/schema - Minimal schema for external callers
- GET /conduit/sockets - All tagged sockets across all workflows
- GET /conduit/workflows/{name}/socket/{node_id}/{side}/{slot_name} - Socket detail
- GET /conduit/workflows/{name}/graph - Graph topology (nodes + edges)
- GET /conduit/workflows/diff - Compare two workflows
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web
import server

from .conduit_workflows import get_workflow_path, WORKFLOWS_DIR
from .conduit_registry import (
    ensure_registry_loaded,
    get_node_def,
    get_all_node_defs,
    InputKind,
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_workflow(name: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load a saved workflow by name.
    Returns (workflow_data, error_message). One will be None.
    """
    file_path = get_workflow_path(name)
    if not file_path.exists():
        return None, f"Workflow not found: {name}"

    try:
        data = json.loads(file_path.read_text())
        return data, None
    except Exception as e:
        return None, f"Failed to load workflow: {e}"


def get_node_title(workflow: dict, node_id: str) -> str:
    """Get node title from workflow metadata if available."""
    node_data = workflow.get(node_id, {})
    meta = node_data.get("_meta", {})
    return meta.get("title") or node_data.get("class_type", "Unknown")


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


def extract_all_sockets(
    workflow: dict,
    schema: dict,
    include_registry: bool = True
) -> Tuple[List[dict], List[dict]]:
    """
    Extract all input and output sockets from a workflow.
    Cross-references with schema for tag info and registry for type info.

    Returns (input_sockets, output_sockets)
    """
    # Build lookup for tagged sockets (only active ones)
    tagged_inputs = {
        (str(s.get("nodeId")), s.get("slotName")): s
        for s in schema.get("inputs", [])
        if is_socket_active(s)
    }
    tagged_outputs = {
        (str(s.get("nodeId")), s.get("slotIndex", s.get("slotName"))): s
        for s in schema.get("outputs", [])
        if is_socket_active(s)
    }

    all_inputs = []
    all_outputs = []

    for node_id, node_data in workflow.items():
        # Skip non-node entries (like injected conduit nodes)
        if not isinstance(node_data, dict):
            continue
        if node_id.startswith("conduit_"):
            continue

        class_type = node_data.get("class_type", "")
        if not class_type:
            continue

        node_title = get_node_title(workflow, node_id)
        node_inputs = node_data.get("inputs", {})

        # Get node definition from registry
        node_def = get_node_def(class_type) if include_registry else None

        # Process inputs
        if node_def and node_def.inputs:
            for slot_index, (input_name, input_spec) in enumerate(node_def.inputs.items()):
                workflow_value = node_inputs.get(input_name)
                is_connected = isinstance(workflow_value, list) and len(workflow_value) == 2

                tag_info = tagged_inputs.get((node_id, input_name))

                socket_info = {
                    "node_id": node_id,
                    "node_type": class_type,
                    "node_title": node_title,
                    "slot_index": slot_index,
                    "slot_name": input_name,
                    "data_type": input_spec.type_name,
                    "kind": input_spec.kind.name.lower() if input_spec.kind else "unknown",
                    "tagged": tag_info is not None,
                    "tag_name": tag_info.get("tagName") if tag_info else None,
                    "connected": is_connected,
                    "workflow_value": None if is_connected else workflow_value,
                    "connection": {
                        "source_node": str(workflow_value[0]),
                        "source_slot": workflow_value[1]
                    } if is_connected else None,
                }

                # Add constraints if available
                if input_spec.kind == InputKind.PRIMITIVE:
                    socket_info["constraints"] = {
                        "min": input_spec.min_value,
                        "max": input_spec.max_value,
                        "step": input_spec.step,
                        "default": input_spec.default,
                    }
                elif input_spec.kind == InputKind.COMBO:
                    socket_info["constraints"] = {
                        "options": input_spec.options,
                        "default": input_spec.default,
                    }

                all_inputs.append(socket_info)
        else:
            # Fallback: use workflow inputs directly without registry
            for input_name, value in node_inputs.items():
                is_connected = isinstance(value, list) and len(value) == 2
                tag_info = tagged_inputs.get((node_id, input_name))

                socket_info = {
                    "node_id": node_id,
                    "node_type": class_type,
                    "node_title": node_title,
                    "slot_index": None,
                    "slot_name": input_name,
                    "data_type": "UNKNOWN",
                    "kind": "unknown",
                    "tagged": tag_info is not None,
                    "tag_name": tag_info.get("tagName") if tag_info else None,
                    "connected": is_connected,
                    "workflow_value": None if is_connected else value,
                    "connection": {
                        "source_node": str(value[0]),
                        "source_slot": value[1]
                    } if is_connected else None,
                }
                all_inputs.append(socket_info)

        # Process outputs
        if node_def and node_def.outputs:
            for slot_index, output_type in enumerate(node_def.outputs):
                output_name = node_def.output_names[slot_index] if slot_index < len(node_def.output_names) else output_type

                # Check if tagged (by slot index for outputs)
                tag_info = tagged_outputs.get((node_id, slot_index))

                socket_info = {
                    "node_id": node_id,
                    "node_type": class_type,
                    "node_title": node_title,
                    "slot_index": slot_index,
                    "slot_name": output_name,
                    "data_type": output_type,
                    "tagged": tag_info is not None,
                    "tag_name": tag_info.get("tagName") if tag_info else None,
                }
                all_outputs.append(socket_info)

    return all_inputs, all_outputs


def build_edge_list(workflow: dict) -> List[dict]:
    """
    Build list of edges (connections) from workflow.
    An edge exists when an input value is [node_id, output_slot_index].
    """
    edges = []

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        if node_id.startswith("conduit_"):
            continue

        class_type = node_data.get("class_type", "")
        node_inputs = node_data.get("inputs", {})

        for input_name, value in node_inputs.items():
            if isinstance(value, list) and len(value) == 2:
                source_node = str(value[0])
                source_slot = value[1]

                # Get source node info
                source_data = workflow.get(source_node, {})
                source_type = source_data.get("class_type", "")
                source_def = get_node_def(source_type) if source_type else None

                # Resolve output name and type
                output_name = None
                data_type = "UNKNOWN"
                if source_def and source_def.outputs:
                    if source_slot < len(source_def.outputs):
                        data_type = source_def.outputs[source_slot]
                    if source_def.output_names and source_slot < len(source_def.output_names):
                        output_name = source_def.output_names[source_slot]

                edges.append({
                    "from_node": source_node,
                    "from_slot": source_slot,
                    "from_name": output_name or f"output_{source_slot}",
                    "to_node": node_id,
                    "to_slot": input_name,
                    "data_type": data_type,
                })

    return edges


def diff_schemas(schema_a: dict, schema_b: dict) -> dict:
    """Compare two workflow schemas and return differences."""
    inputs_a = {s.get("tagName"): s for s in schema_a.get("inputs", [])}
    inputs_b = {s.get("tagName"): s for s in schema_b.get("inputs", [])}
    outputs_a = {s.get("tagName"): s for s in schema_a.get("outputs", [])}
    outputs_b = {s.get("tagName"): s for s in schema_b.get("outputs", [])}

    # Find differences
    inputs_only_a = [k for k in inputs_a if k not in inputs_b]
    inputs_only_b = [k for k in inputs_b if k not in inputs_a]
    inputs_both = [k for k in inputs_a if k in inputs_b]

    outputs_only_a = [k for k in outputs_a if k not in outputs_b]
    outputs_only_b = [k for k in outputs_b if k not in outputs_a]
    outputs_both = [k for k in outputs_a if k in outputs_b]

    # Check for changes in common inputs
    input_changes = []
    for tag in inputs_both:
        a = inputs_a[tag]
        b = inputs_b[tag]
        changes = {}

        if str(a.get("nodeId")) != str(b.get("nodeId")):
            changes["node_moved"] = {
                "a": str(a.get("nodeId")),
                "b": str(b.get("nodeId"))
            }
        if a.get("dataType") != b.get("dataType"):
            changes["type_changed"] = {
                "a": a.get("dataType"),
                "b": b.get("dataType")
            }
        if a.get("slotName") != b.get("slotName"):
            changes["slot_changed"] = {
                "a": a.get("slotName"),
                "b": b.get("slotName")
            }

        if changes:
            input_changes.append({"tag": tag, "changes": changes})

    return {
        "inputs_only_in_a": inputs_only_a,
        "inputs_only_in_b": inputs_only_b,
        "inputs_in_both": inputs_both,
        "input_changes": input_changes,
        "outputs_only_in_a": outputs_only_a,
        "outputs_only_in_b": outputs_only_b,
        "outputs_in_both": outputs_both,
    }


# ============================================================================
# Endpoint 1: Socket Discovery
# ============================================================================

@server.PromptServer.instance.routes.get("/conduit/workflows/{name}/sockets")
async def workflow_sockets(request):
    """
    GET /conduit/workflows/{name}/sockets
    List ALL sockets in a workflow (tagged and untagged).
    """
    await ensure_registry_loaded()

    workflow_name = request.match_info["name"]
    data, error = load_workflow(workflow_name)

    if error:
        return web.json_response({"status": "error", "message": error}, status=404)

    workflow = data.get("workflow", {})
    schema = data.get("schema", {})

    inputs, outputs = extract_all_sockets(workflow, schema)

    # Build summary
    tagged_inputs = [s for s in inputs if s["tagged"]]
    tagged_outputs = [s for s in outputs if s["tagged"]]

    return web.json_response({
        "workflow_name": workflow_name,
        "inputs": inputs,
        "outputs": outputs,
        "summary": {
            "total_inputs": len(inputs),
            "tagged_inputs": len(tagged_inputs),
            "untagged_inputs": len(inputs) - len(tagged_inputs),
            "total_outputs": len(outputs),
            "tagged_outputs": len(tagged_outputs),
            "untagged_outputs": len(outputs) - len(tagged_outputs),
        }
    })


# ============================================================================
# Endpoint 2: Schema (For External Callers)
# ============================================================================

@server.PromptServer.instance.routes.get("/conduit/workflows/{name}/schema")
async def workflow_schema(request):
    """
    GET /conduit/workflows/{name}/schema
    Minimal schema for external callers - what they need to construct requests.
    """
    await ensure_registry_loaded()

    workflow_name = request.match_info["name"]
    data, error = load_workflow(workflow_name)

    if error:
        return web.json_response({"status": "error", "message": error}, status=404)

    workflow = data.get("workflow", {})
    schema = data.get("schema", {})

    # Build input schema
    input_schema = {}
    example_inputs = {}

    for socket in filter_active_sockets(schema.get("inputs", [])):
        tag_name = socket.get("tagName", "")
        node_id = str(socket.get("nodeId", ""))
        slot_name = socket.get("slotName", "")
        data_type = socket.get("dataType", "")

        # Get node info
        node_data = workflow.get(node_id, {})
        class_type = node_data.get("class_type", "")
        node_inputs = node_data.get("inputs", {})
        workflow_value = node_inputs.get(slot_name)

        # Get registry info
        node_def = get_node_def(class_type) if class_type else None
        input_spec = node_def.inputs.get(slot_name) if node_def else None

        # Build constraints
        constraints = {}
        default_value = None

        if input_spec:
            if input_spec.kind == InputKind.PRIMITIVE:
                if input_spec.min_value is not None:
                    constraints["min"] = input_spec.min_value
                if input_spec.max_value is not None:
                    constraints["max"] = input_spec.max_value
                if input_spec.step is not None:
                    constraints["step"] = input_spec.step
                default_value = input_spec.default
            elif input_spec.kind == InputKind.COMBO:
                constraints["options"] = input_spec.options
                default_value = input_spec.default

            if input_spec.multiline:
                constraints["multiline"] = True

        # Use workflow value as default if no registry default
        if default_value is None and not isinstance(workflow_value, list):
            default_value = workflow_value

        input_schema[tag_name] = {
            "type": data_type,
            "required": False,  # Conduit never requires inputs
            "default": default_value,
        }
        if constraints:
            input_schema[tag_name]["constraints"] = constraints

        # Build example value
        if default_value is not None:
            example_inputs[tag_name] = default_value
        elif data_type == "STRING":
            example_inputs[tag_name] = "example text"
        elif data_type == "INT":
            example_inputs[tag_name] = 42
        elif data_type == "FLOAT":
            example_inputs[tag_name] = 1.0
        elif data_type == "BOOLEAN":
            example_inputs[tag_name] = True

    # Get output tag names (only active sockets)
    output_tags = [s.get("tagName") for s in filter_active_sockets(schema.get("outputs", []))]

    return web.json_response({
        "workflow_name": workflow_name,
        "inputs": input_schema,
        "outputs": output_tags,
        "example_request": {
            "inputs": example_inputs
        }
    })


# ============================================================================
# Endpoint 3: Socket Detail
# ============================================================================

@server.PromptServer.instance.routes.get("/conduit/workflows/{name}/socket/{node_id}/{side}/{slot_name}")
async def socket_detail(request):
    """
    GET /conduit/workflows/{name}/socket/{node_id}/{side}/{slot_name}
    Get all information about a specific socket.
    """
    await ensure_registry_loaded()

    workflow_name = request.match_info["name"]
    node_id = request.match_info["node_id"]
    side = request.match_info["side"]
    slot_name = request.match_info["slot_name"]

    if side not in ("input", "output"):
        return web.json_response(
            {"status": "error", "message": "side must be 'input' or 'output'"},
            status=400
        )

    data, error = load_workflow(workflow_name)
    if error:
        return web.json_response({"status": "error", "message": error}, status=404)

    workflow = data.get("workflow", {})
    schema = data.get("schema", {})

    # Find the node
    node_data = workflow.get(node_id)
    if not node_data:
        return web.json_response(
            {"status": "error", "message": f"Node {node_id} not found"},
            status=404
        )

    class_type = node_data.get("class_type", "")
    node_title = get_node_title(workflow, node_id)
    node_def = get_node_def(class_type) if class_type else None

    # Find tag info
    tag_info = None
    sockets = schema.get("inputs" if side == "input" else "outputs", [])
    for s in sockets:
        if str(s.get("nodeId")) == node_id and s.get("slotName") == slot_name:
            tag_info = s
            break

    if side == "input":
        # Get input spec
        input_spec = node_def.inputs.get(slot_name) if node_def else None
        node_inputs = node_data.get("inputs", {})
        workflow_value = node_inputs.get(slot_name)
        is_connected = isinstance(workflow_value, list) and len(workflow_value) == 2

        # Find slot index
        slot_index = None
        if node_def:
            for idx, name in enumerate(node_def.inputs.keys()):
                if name == slot_name:
                    slot_index = idx
                    break

        response = {
            "identity": {
                "node_id": node_id,
                "node_type": class_type,
                "node_title": node_title,
                "side": "input",
                "slot_index": slot_index,
                "slot_name": slot_name,
            },
            "tag": {
                "tagged": tag_info is not None and is_socket_active(tag_info),
                "tag_name": tag_info.get("tagName") if tag_info else None,
                "active": is_socket_active(tag_info) if tag_info else None,
            },
            "type": {
                "data_type": input_spec.type_name if input_spec else tag_info.get("dataType") if tag_info else "UNKNOWN",
                "kind": input_spec.kind.name.lower() if input_spec and input_spec.kind else "unknown",
            },
            "connection": {
                "connected": is_connected,
                "source_node": str(workflow_value[0]) if is_connected else None,
                "source_slot": workflow_value[1] if is_connected else None,
            },
            "workflow_value": None if is_connected else workflow_value,
        }

        # Add constraints
        if input_spec:
            constraints = {}
            if input_spec.kind == InputKind.PRIMITIVE:
                constraints = {
                    "min": input_spec.min_value,
                    "max": input_spec.max_value,
                    "step": input_spec.step,
                    "default": input_spec.default,
                }
            elif input_spec.kind == InputKind.COMBO:
                constraints = {
                    "options": input_spec.options,
                    "default": input_spec.default,
                }
            response["constraints"] = constraints

            # Check for control widget
            if node_def and input_spec.kind == InputKind.COMBO:
                if input_spec.options and set(input_spec.options) >= {"fixed", "randomize"}:
                    # This looks like a control widget
                    # Find what it controls (usually the previous input)
                    input_names = list(node_def.inputs.keys())
                    if slot_index and slot_index > 0:
                        controlled_input = input_names[slot_index - 1]
                        response["control_widget"] = {
                            "is_control": True,
                            "controls": controlled_input,
                            "current_mode": workflow_value if not is_connected else None,
                        }

    else:
        # Output socket
        slot_index = None
        data_type = "UNKNOWN"
        output_name = slot_name

        if node_def and node_def.outputs:
            for idx, out_type in enumerate(node_def.outputs):
                name = node_def.output_names[idx] if idx < len(node_def.output_names) else out_type
                if name == slot_name or str(idx) == slot_name:
                    slot_index = idx
                    data_type = out_type
                    output_name = name
                    break

        response = {
            "identity": {
                "node_id": node_id,
                "node_type": class_type,
                "node_title": node_title,
                "side": "output",
                "slot_index": slot_index,
                "slot_name": output_name,
            },
            "tag": {
                "tagged": tag_info is not None and is_socket_active(tag_info),
                "tag_name": tag_info.get("tagName") if tag_info else None,
                "active": is_socket_active(tag_info) if tag_info else None,
            },
            "type": {
                "data_type": data_type,
            },
        }

    return web.json_response(response)


# ============================================================================
# Endpoint 5: Graph Topology
# ============================================================================

@server.PromptServer.instance.routes.get("/conduit/workflows/{name}/graph")
async def workflow_graph(request):
    """
    GET /conduit/workflows/{name}/graph
    Get workflow graph topology: nodes and their connections.
    """
    await ensure_registry_loaded()

    workflow_name = request.match_info["name"]
    data, error = load_workflow(workflow_name)

    if error:
        return web.json_response({"status": "error", "message": error}, status=404)

    workflow = data.get("workflow", {})

    # Build node list
    nodes = []
    node_ids = set()

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        if node_id.startswith("conduit_"):
            continue

        class_type = node_data.get("class_type", "")
        if not class_type:
            continue

        node_ids.add(node_id)
        node_title = get_node_title(workflow, node_id)
        node_def = get_node_def(class_type)

        node_info = {
            "id": node_id,
            "type": class_type,
            "title": node_title,
        }

        # Add position if available in metadata
        meta = node_data.get("_meta", {})
        if "position" in meta:
            node_info["position"] = meta["position"]

        # Add input/output names from registry
        if node_def:
            node_info["inputs"] = list(node_def.inputs.keys())
            node_info["outputs"] = node_def.output_names or node_def.outputs
        else:
            node_info["inputs"] = list(node_data.get("inputs", {}).keys())
            node_info["outputs"] = []

        nodes.append(node_info)

    # Build edge list
    edges = build_edge_list(workflow)

    # Find orphan nodes (no incoming or outgoing edges)
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge["from_node"])
        nodes_with_edges.add(edge["to_node"])

    orphan_nodes = [n["id"] for n in nodes if n["id"] not in nodes_with_edges]

    return web.json_response({
        "workflow_name": workflow_name,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "orphan_nodes": orphan_nodes,
        }
    })


# ============================================================================
# Endpoint 6: Workflow Diff
# ============================================================================

@server.PromptServer.instance.routes.get("/conduit/workflows/diff")
async def workflow_diff(request):
    """
    GET /conduit/workflows/diff?a={name1}&b={name2}
    Compare two workflows' structure and tagged sockets.
    """
    workflow_a = request.query.get("a")
    workflow_b = request.query.get("b")

    if not workflow_a or not workflow_b:
        return web.json_response(
            {"status": "error", "message": "Both 'a' and 'b' query params required"},
            status=400
        )

    data_a, error_a = load_workflow(workflow_a)
    if error_a:
        return web.json_response(
            {"status": "error", "message": f"Workflow A: {error_a}"},
            status=404
        )

    data_b, error_b = load_workflow(workflow_b)
    if error_b:
        return web.json_response(
            {"status": "error", "message": f"Workflow B: {error_b}"},
            status=404
        )

    schema_a = data_a.get("schema", {})
    schema_b = data_b.get("schema", {})
    workflow_a_data = data_a.get("workflow", {})
    workflow_b_data = data_b.get("workflow", {})

    # Compare schemas
    schema_diff = diff_schemas(schema_a, schema_b)

    # Compare graph structure
    nodes_a = {
        node_id for node_id, node_data in workflow_a_data.items()
        if isinstance(node_data, dict) and node_data.get("class_type")
    }
    nodes_b = {
        node_id for node_id, node_data in workflow_b_data.items()
        if isinstance(node_data, dict) and node_data.get("class_type")
    }

    # Count edges
    edges_a = build_edge_list(workflow_a_data)
    edges_b = build_edge_list(workflow_b_data)

    # Determine compatibility (same tagged inputs = compatible)
    inputs_match = (
        set(schema_diff["inputs_only_in_a"]) == set() and
        set(schema_diff["inputs_only_in_b"]) == set() and
        len(schema_diff["input_changes"]) == 0
    )

    return web.json_response({
        "workflow_a": workflow_a,
        "workflow_b": workflow_b,
        "schema_diff": schema_diff,
        "graph_diff": {
            "nodes_only_in_a": list(nodes_a - nodes_b),
            "nodes_only_in_b": list(nodes_b - nodes_a),
            "nodes_in_both": len(nodes_a & nodes_b),
            "edges_in_a": len(edges_a),
            "edges_in_b": len(edges_b),
        },
        "compatible": inputs_match,
    })


# ============================================================================
# Module Registration
# ============================================================================

print("[Conduit] Introspection endpoints registered")
