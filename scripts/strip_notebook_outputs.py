#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_notebook(path: Path) -> tuple[bool, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        metadata = cell.get("metadata")
        if isinstance(metadata, dict):
            for key in ["execution", "collapsed"]:
                if key in metadata:
                    metadata.pop(key, None)
                    changed = True
    return changed, obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Strip Jupyter notebook outputs for git-friendly diffs.")
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="Fail if outputs would be stripped; do not modify files.")
    args = parser.parse_args()

    failed = False
    for path in args.notebooks:
        changed, stripped = strip_notebook(path)
        if args.check:
            if changed:
                print(f"needs stripping: {path}")
                failed = True
            else:
                print(f"clean: {path}")
            continue
        if changed:
            path.write_text(json.dumps(stripped, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"stripped: {path}")
        else:
            print(f"already clean: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
