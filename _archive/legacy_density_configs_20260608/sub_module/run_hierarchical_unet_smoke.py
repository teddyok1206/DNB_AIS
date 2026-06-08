from __future__ import annotations

import sys
from pathlib import Path

from .run_density_smoke import ROOT, main


if __name__ == "__main__":
    default_config = ROOT / "configs" / "dnb_density_unet_occupancy_spatial.json"
    raise SystemExit(main(["--config", str(default_config), "--model", "main", *sys.argv[1:]]))
