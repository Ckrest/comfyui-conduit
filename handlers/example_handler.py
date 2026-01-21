#!/usr/bin/env python3
"""
Conduit Example Handler
=======================

This is a template showing how to create a Conduit handler.
Handlers receive workflow execution results and can do anything with them:
send notifications, log to databases, trigger other systems, etc.

HOW IT WORKS
------------
1. Conduit executes a workflow
2. After completion, Conduit runs each enabled handler
3. The handler receives a JSON event via stdin
4. The handler does its work and exits (0 = success, non-zero = failure)

EVENT STRUCTURE
---------------
{
    "prompt_id": "20251222060547_o904",
    "comfy_prompt_id": "abc123-...",
    "source": "frontend" | "gateway",
    "workflow_name": "MyWorkflow",

    "inputs_applied": {
        "prompt": {"value": "a cat", "type": "STRING"},
        ...
    },
    "inputs_defaulted": [
        {"tagName": "seed", "type": "INT", "reason": "not_provided"}
    ],

    "output_folder": "/path/to/output/conduit/20251222060547_o904",
    "outputs": [
        {
            "tag_name": "GeneratedImage",
            "file_path": "/path/to/output/conduit/.../image.png",
            "file_type": "image",
            "data_type": "IMAGE"
        },
        ...
    ],

    "caller_context": { ... },  # Custom data passed by API caller
    "tagged_inputs": [ ... ],   # Input schema
    "tagged_outputs": [ ... ]   # Output schema
}

CREATING YOUR OWN HANDLER
-------------------------
1. Copy this file to a new name (e.g., my_handler.py)
2. Implement your logic in main()
3. Register it in conduit_config.json:

   "handlers": {
       "my_handler": {
           "name": "My Handler",
           "description": "Does something cool with outputs",
           "command": "/path/to/my_handler.py",
           "enabled": true
       }
   }

4. Make it executable: chmod +x my_handler.py

TIPS
----
- Keep handlers fast (< 30 second timeout)
- For slow operations, queue work and return immediately
- Exit 0 for success, non-zero for failure
- Print to stdout for logging (shown in Conduit logs)
- Print to stderr for errors
"""
import sys
import json


def main():
    # Read the event from stdin
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[Example Handler] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract useful fields
    prompt_id = event.get('prompt_id', 'unknown')
    workflow_name = event.get('workflow_name', 'unknown')
    source = event.get('source', 'unknown')
    outputs = event.get('outputs', [])

    # Log what we received (this appears in Conduit's output)
    print(f"[Example Handler] Received event from {source}")
    print(f"[Example Handler]   Workflow: {workflow_name}")
    print(f"[Example Handler]   Prompt ID: {prompt_id}")
    print(f"[Example Handler]   Outputs: {len(outputs)}")

    # Show each output
    for output in outputs:
        tag = output.get('tag_name', 'unknown')
        file_type = output.get('file_type', 'unknown')
        file_path = output.get('file_path', '')
        print(f"[Example Handler]     - {tag}: {file_type} -> {file_path}")

    # Your handler logic would go here!
    # Examples:
    #   - POST to a webhook
    #   - Insert into a database
    #   - Send a notification
    #   - Trigger another workflow
    #   - Copy files somewhere

    print(f"[Example Handler] Done! (This handler doesn't do anything else)")

    # Exit 0 = success
    sys.exit(0)


if __name__ == '__main__':
    main()
