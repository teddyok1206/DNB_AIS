from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
