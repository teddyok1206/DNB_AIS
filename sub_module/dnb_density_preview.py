from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def preview_cmap_for_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if "error" in normalized:
        return "inferno"
    if "brightness" in normalized or "radiance" in normalized or "image" in normalized:
        return "magma"
    if any(token in normalized for token in ("mask", "seed", "owner", "valid")):
        return "gray"
    if any(token in normalized for token in ("persistence", "attention", "lifetime")):
        return "viridis"
    if "target" in normalized or "pred" in normalized or "density" in normalized:
        return "viridis"
    return "viridis"


def preview_limits_for_name(name: str, arr: Any) -> tuple[float | None, float | None]:
    normalized = str(name).strip().lower().replace("-", "_")
    if any(token in normalized for token in ("mask", "seed", "owner", "valid")):
        return 0.0, 1.0
    if "attention" in normalized or "persistence" in normalized:
        return 0.0, 1.0
    if "lifetime" in normalized:
        values = np.asarray(arr, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        vmax = max(float(np.nanmax(finite)), 1.0e-8)
        return 0.0, vmax
    return None, None


def robust_vmax(arrays: list[Any], *, percentile: float = 99.5, eps: float = 1.0e-8) -> float:
    values = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float32).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size:
            values.append(finite)
    if not values:
        return float(eps)
    merged = np.concatenate(values)
    if merged.size == 0:
        return float(eps)
    return max(float(np.nanpercentile(merged, float(percentile))), float(eps))


def save_panel_grid(
    path: Path,
    panels: list[tuple[str, Any, str, float | None, float | None]],
    *,
    title: str,
    cols: int = 4,
    dpi: int = 150,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not panels:
        raise ValueError("save_panel_grid requires at least one panel")
    cols = max(int(cols), 1)
    rows = int(ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (panel_title, arr, cmap, vmin, vmax) in zip(axes_arr, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_arr[len(panels) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return str(path)


def save_density_patch_previews(
    patches: list[Any],
    *,
    scene_key: str,
    output_dir: Path,
    limit: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for patch in patches[: max(int(limit), 0)]:
        panels = [
            ("brightness", patch.image, "magma"),
            ("parent PH mask", patch.parent_mask, "gray"),
            ("child PH union", patch.child_union_mask, "gray"),
            ("seed map", patch.seed_map, "gray"),
            ("persistence map", patch.persistence_map, "viridis"),
            ("soft attention", patch.soft_attention, "viridis"),
            ("raw GT count", patch.raw_count, "viridis"),
            ("crop-level density target", patch.target_density, "viridis"),
            ("loss weight", patch.loss_weight, "viridis"),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
        for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
            im = ax.imshow(arr, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            (
                f"{scene_key} | partition={patch.cluster_id} | children={len(patch.child_ids)} | "
                f"raw_sum={patch.raw_count_sum:.1f} | target_sum={patch.target_sum:.1f}"
            ),
            fontsize=13,
        )
        path = output_dir / f"{scene_key}_partition_{int(patch.cluster_id)}_density_patch_preview.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths
