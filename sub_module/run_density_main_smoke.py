from __future__ import annotations

import sys

from .run_density_smoke import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "main", *sys.argv[1:]]))
