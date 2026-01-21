"""
Conduit Diagnostic Logging

Endpoints for frontend logging - appends events to a JSON Lines file.
Use /conduit/clear to reset the log file.
"""

import json
from datetime import datetime
from pathlib import Path

from aiohttp import web
from server import PromptServer

LOG_FILE = Path(__file__).parent / "diagnostic_logs" / "latest.json"
LOG_FILE.parent.mkdir(exist_ok=True)


@PromptServer.instance.routes.post("/conduit/log")
async def receive_log(request):
    """Receive and append event."""
    try:
        payload = await request.json()
        payload["server_time"] = datetime.now().isoformat()

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")

        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


@PromptServer.instance.routes.post("/conduit/clear")
async def clear_log(request):
    """Clear log file."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    return web.json_response({"status": "cleared"})


print(f"[Conduit] Log endpoint: POST /conduit/log -> {LOG_FILE}")
