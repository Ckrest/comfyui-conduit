"""
Conduit Workflow Registry

Saves workflows with their tagged socket schemas for external execution.
Workflows are stored as JSON files in conduit_workflows/ directory.
"""

import json
from datetime import datetime
from pathlib import Path

from aiohttp import web
import server

# Storage directory
WORKFLOWS_DIR = Path(__file__).parent / "conduit_workflows"
WORKFLOWS_DIR.mkdir(exist_ok=True)


def sanitize_name(name: str) -> str:
    """Convert workflow name to safe filename."""
    # Replace unsafe characters with underscores
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    # Collapse multiple underscores/spaces
    while "  " in safe or "__" in safe:
        safe = safe.replace("  ", " ").replace("__", "_")
    return safe.strip().strip("_") or "untitled"


def get_workflow_path(name: str) -> Path:
    """Get the file path for a workflow by name."""
    safe_name = sanitize_name(name)
    return WORKFLOWS_DIR / f"{safe_name}.json"


# =============================================================================
# API Endpoints
# =============================================================================

@server.PromptServer.instance.routes.post("/conduit/workflows")
async def save_workflow(request):
    """
    Save a workflow with its tagged socket schema.

    Expected JSON body:
    {
        "name": "My Workflow",
        "workflow": { ... ComfyUI workflow JSON ... },
        "sockets": [ ... canonical tagged socket array ... ],
        "handlers": ["handler1", ...]  // optional, sparse storage - only if different from defaults
    }
    """
    try:
        data = await request.json()

        name = data.get("name", "").strip()
        if not name:
            return web.json_response(
                {"status": "error", "message": "Workflow name is required"},
                status=400
            )

        workflow = data.get("workflow")
        if not workflow:
            return web.json_response(
                {"status": "error", "message": "Workflow data is required"},
                status=400
            )

        sockets = data.get("sockets", [])
        handlers = data.get("handlers")  # None = use defaults, list = specific handlers

        # Build the saved workflow structure
        saved = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "schema": {
                "inputs": [s for s in sockets if s.get("side") == "input"],
                "outputs": [s for s in sockets if s.get("side") == "output"],
            },
            "workflow": workflow,
        }

        # Sparse handler storage: only include if explicitly provided (different from defaults)
        if handlers is not None:
            saved["schema"]["handlers"] = handlers

        # Check if updating existing
        file_path = get_workflow_path(name)
        is_update = file_path.exists()

        if is_update:
            # Preserve original created_at
            try:
                existing = json.loads(file_path.read_text())
                saved["created_at"] = existing.get("created_at", saved["created_at"])
            except Exception:
                pass

        # Save to file
        file_path.write_text(json.dumps(saved, indent=2))

        print(f"[Conduit] {'Updated' if is_update else 'Saved'} workflow: {name} ({file_path.name})")
        print(f"[Conduit]   Inputs: {len(saved['schema']['inputs'])}, Outputs: {len(saved['schema']['outputs'])}")
        if handlers is not None:
            print(f"[Conduit]   Handlers (custom): {handlers}")

        return web.json_response({
            "status": "saved",
            "name": name,
            "file": file_path.name,
            "is_update": is_update,
            "inputs": len(saved["schema"]["inputs"]),
            "outputs": len(saved["schema"]["outputs"]),
            "handlers": handlers,  # Echo back what was saved
        })

    except Exception as e:
        print(f"[Conduit] Error saving workflow: {e}")
        return web.json_response(
            {"status": "error", "message": str(e)},
            status=500
        )


@server.PromptServer.instance.routes.get("/conduit/workflows")
async def list_workflows(request):
    """List all saved workflows."""
    workflows = []

    for file_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            data = json.loads(file_path.read_text())
            workflows.append({
                "name": data.get("name", file_path.stem),
                "file": file_path.name,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "inputs": len(data.get("schema", {}).get("inputs", [])),
                "outputs": len(data.get("schema", {}).get("outputs", [])),
            })
        except Exception as e:
            print(f"[Conduit] Error reading {file_path}: {e}")

    return web.json_response({"workflows": workflows})


@server.PromptServer.instance.routes.get("/conduit/workflows/{name}")
async def get_workflow(request):
    """Get a specific workflow by name."""
    name = request.match_info["name"]
    file_path = get_workflow_path(name)

    if not file_path.exists():
        return web.json_response(
            {"status": "error", "message": f"Workflow not found: {name}"},
            status=404
        )

    try:
        data = json.loads(file_path.read_text())
        return web.json_response(data)
    except Exception as e:
        return web.json_response(
            {"status": "error", "message": str(e)},
            status=500
        )


@server.PromptServer.instance.routes.delete("/conduit/workflows/{name}")
async def delete_workflow(request):
    """Delete a workflow by name."""
    name = request.match_info["name"]
    file_path = get_workflow_path(name)

    if not file_path.exists():
        return web.json_response(
            {"status": "error", "message": f"Workflow not found: {name}"},
            status=404
        )

    try:
        file_path.unlink()
        print(f"[Conduit] Deleted workflow: {name}")
        return web.json_response({"status": "deleted", "name": name})
    except Exception as e:
        return web.json_response(
            {"status": "error", "message": str(e)},
            status=500
        )


@server.PromptServer.instance.routes.post("/conduit/workflows/open-folder")
async def open_workflows_folder(request):
    """Open the conduit_workflows folder in the system file manager."""
    import subprocess
    import sys

    try:
        folder = str(WORKFLOWS_DIR)

        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", folder])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        else:
            return web.json_response(
                {"status": "error", "message": f"Unsupported platform: {sys.platform}"},
                status=500
            )

        print(f"[Conduit] Opened workflows folder: {folder}")
        return web.json_response({"status": "opened", "path": folder})

    except Exception as e:
        print(f"[Conduit] Error opening folder: {e}")
        return web.json_response(
            {"status": "error", "message": str(e)},
            status=500
        )


print(f"[Conduit] Workflow registry: {WORKFLOWS_DIR}")
print(f"[Conduit] Existing workflows: {len(list(WORKFLOWS_DIR.glob('*.json')))}")
