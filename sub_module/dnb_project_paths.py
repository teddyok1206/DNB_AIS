from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEP3 = ROOT / "[3]_DNB_AIS - (STEP 3)"


def _resolve_env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


OUTPUT_ROOT = _resolve_env_path("DNB_AIS_OUTPUT_ROOT", ROOT / "outputs")
DENSITY_OUTPUT_ROOT = _resolve_env_path("DNB_DENSITY_OUTPUT_ROOT", OUTPUT_ROOT / "dnb_density")


def project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (ROOT / value).resolve()


def density_output_path(*parts: str | Path) -> Path:
    out = DENSITY_OUTPUT_ROOT
    for part in parts:
        out = out / Path(part)
    return out.resolve()
