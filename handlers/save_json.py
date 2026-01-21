#!/usr/bin/env python3
"""Simple script to save JSON input to a text file."""

import sys
import json
from pathlib import Path

def main():
    # Read JSON from stdin
    data = json.load(sys.stdin)

    # Save to file in same directory as script
    script_dir = Path(__file__).parent
    output_file = script_dir / "output.txt"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
