from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


NOTEBOOK_CELLS = [
    markdown_cell(
        """# DNB_GAT_v1

## Pipeline Blocks
- Block 1. Runtime, paths, and MPS/PyG setup
- Block 2. Scene loading and GT density map preparation
- Block 3. DRUID `area_limit` sweep for patch-size diagnostics
- Block 4. DRUID-based irregular contour patch extraction
- Block 5. Graph receptive-field sweep over radius and layer count
- Block 6. Representative graph visualization for one DRUID cluster
- Block 7. Loss weighting comparison for count-sensitive supervision
- Block 8. Patch-to-graph conversion with PyG `radius_graph`
- Block 9. GATv2Conv density regression and minimal training loop
- Block 10. Lifetime-weighted patch merge to geocoded heatmap GeoTIFF

## Notes
- Default mode is `batch_demo` because it is the validated end-to-end path in the current workspace.
- Switch `SCENE_MODE` to `kr_full_scene` to run the same pipeline on the larger pseudo full-scene TIFF.
- Ground truth uses ship-center pixels. If the requested geojson is missing, the notebook can regenerate it from `ships.db` and `metadata_JPSS-2.csv`.
- `max_catalogue_clusters` is disabled by default so DRUID candidate selection is driven by `area_limit`, not a top-k lifetime cap.
- Each execution writes to a fresh `RUN_TAG` subdirectory so Desktop/iCloud overwrite stalls do not block reruns.
"""
    ),
    code_cell(
        """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from sub_module.dnb_gat_pipeline import (
    DruidClusterStore,
    DruidConfig,
    GATDensityRegressor,
    GraphBuilder,
    GraphConfig,
    GroundTruthResolver,
    SceneAssembler,
    SceneRaster,
    TrainingConfig,
    area_limit_sweep,
    choose_representative_cluster,
    count_model_parameters,
    graph_receptive_field_sweep,
    load_model_checkpoint,
    loss_weighting_sweep,
    make_overlay_rgb,
    predict_graphs,
    resolve_device,
    save_model_checkpoint,
    train_gat,
    visualize_graph_cluster,
)

ROOT = Path.cwd()
STEP3 = ROOT / "[3]_DNB_AIS - (STEP 3)"
OUTPUT_ROOT = ROOT / "outputs" / "DNB_GAT_v1"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SEED = 1
RUN_TAG = datetime.now().strftime("run_%m%d_%H%M%S")

np.random.seed(SEED)
torch.manual_seed(SEED)

SCENES = {
    "batch_demo": {
        "scene_tif": STEP3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif",
        "gt_geojson": STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson",
        "druid": DruidConfig(
            cutup=False,
            area_limit=12,
            max_catalogue_clusters=None,
            min_nodes=16,
            max_nodes=2500,
        ),
        "graph": GraphConfig(radius_pixels=4.0),
        "training": TrainingConfig(
            hidden_channels=32,
            heads=4,
            num_layers=3,
            epochs=4,
            batch_size=4,
            dropout=0.05,
            positive_weight=12.0,
            count_weight_alpha=6.0,
        ),
    },
    "kr_full_scene": {
        "scene_tif": STEP3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_KR.tif",
        "gt_geojson": STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson",
        "druid": DruidConfig(
            cutup=True,
            cutup_size=512,
            cutup_buffer=64,
            area_limit=12,
            max_catalogue_clusters=None,
            min_nodes=32,
            max_nodes=2500,
        ),
        "graph": GraphConfig(radius_pixels=4.0),
        "training": TrainingConfig(
            hidden_channels=32,
            heads=4,
            num_layers=3,
            epochs=4,
            batch_size=4,
            dropout=0.05,
            positive_weight=12.0,
            count_weight_alpha=6.0,
        ),
    },
}

SCENE_MODE = "batch_demo"
ACTIVE = SCENES[SCENE_MODE]
DEVICE = resolve_device("mps")

print(f"SCENE_MODE={SCENE_MODE}")
print(f"DEVICE={DEVICE}")
print(f"SEED={SEED}")
print(f"RUN_TAG={RUN_TAG}")
print(f"MPS available={torch.backends.mps.is_available()}")
print(f"Scene path={ACTIVE['scene_tif']}")
"""
    ),
    markdown_cell(
        """## Block 2. Scene Loading and GT Density Map Preparation

Load the target GeoTIFF, resolve the GT geojson path, and rasterize ship-center counts onto the scene grid. The count map is the node-level regression target before DRUID patching.
"""
    ),
    code_cell(
        """scene = SceneRaster.load(ACTIVE["scene_tif"])
scene_output_dir = OUTPUT_ROOT / scene.key / SCENE_MODE / RUN_TAG
scene_output_dir.mkdir(parents=True, exist_ok=True)

gt_resolver = GroundTruthResolver(
    metadata_csv=STEP3 / "metadata_JPSS-2.csv",
    ships_db_path=Path("/Users/jungtaeuk/ships/ships.db"),
    default_geojson_dir=STEP3 / "bboxes_JPSS-2",
)

gt_geojson_path = gt_resolver.resolve_geojson(scene, ACTIVE["gt_geojson"])
gt_points = gt_resolver.load_points(gt_geojson_path)
gt_count_map = gt_resolver.rasterize_counts(scene, gt_points)

print(f"scene.key={scene.key}")
print(f"scene.shape={scene.shape}")
print(f"gt_geojson_path={gt_geojson_path}")
print(f"gt_points={len(gt_points)}")
print(f"gt_nonzero_pixels={(gt_count_map > 0).sum()}")
print(f"gt_total_count={gt_count_map.sum():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(scene.image, cmap="cividis", vmin=0, vmax=1)
axes[0].set_title(f"Scene Radiance: {scene.path.name}")
axes[0].set_axis_off()

axes[1].imshow(gt_count_map, cmap="inferno")
axes[1].set_title("GT Count Map (ship-center raster)")
axes[1].set_axis_off()
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Block 3. DRUID `area_limit` Sweep for Patch-Size Diagnostics

Run a sweep over candidate `area_limit` values with `min_nodes=1` and no top-k cap, so the report is driven by DRUID patch size rather than the later graph-size threshold. This block is only enabled for `batch_demo`.
"""
    ),
    code_cell(
        """if SCENE_MODE == "batch_demo":
    area_limit_table = area_limit_sweep(
        scene=scene,
        gt_count_map=gt_count_map,
        druid_root=STEP3 / "DRUID",
        area_limits=[4, 8, 12, 16],
        base_config=ACTIVE["druid"],
        min_nodes_override=1,
    )
    area_limit_table.to_csv(scene_output_dir / "area_limit_sweep.csv", index=False)
    display(area_limit_table)
else:
    area_limit_table = pd.DataFrame()
    print("area_limit sweep skipped outside batch_demo mode.")
"""
    ),
    markdown_cell(
        """## Block 4. DRUID-Based Irregular Contour Patch Extraction

Run DRUID, keep cluster lifetime values, reconstruct contour masks, filter nested clusters, and summarize patch sizes. This is the state-carrying class layer that will also support later DRUID refinement work.
"""
    ),
    code_cell(
        """cluster_store = DruidClusterStore.from_scene(
    scene=scene,
    gt_count_map=gt_count_map,
    druid_root=STEP3 / "DRUID",
    config=ACTIVE["druid"],
)

cluster_summary = cluster_store.summary_frame()
cluster_summary.to_csv(scene_output_dir / "cluster_summary.csv", index=False)

print(f"clusters_after_filter={len(cluster_store.clusters)}")
print("patch_size_suggestions=", cluster_store.patch_size_suggestions())
display(cluster_summary.head(10))

sample_cluster = cluster_store.clusters[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(sample_cluster.patch_image, cmap="cividis", vmin=0, vmax=1)
axes[0].set_title(f"Cluster {sample_cluster.cluster_id} patch")
axes[0].set_axis_off()

axes[1].imshow(sample_cluster.mask, cmap="gray")
axes[1].set_title("Irregular contour mask")
axes[1].set_axis_off()

axes[2].imshow(sample_cluster.patch_gt, cmap="magma")
axes[2].set_title("Patch GT count")
axes[2].set_axis_off()
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Block 5. Graph Receptive-Field Sweep

Compare `radius_graph` radius and GAT layer count on the same DRUID patch set. This helps decide whether wider context should come from larger local neighborhoods, deeper message passing, or both.
"""
    ),
    code_cell(
        """if SCENE_MODE == "batch_demo":
    graph_sweep_table = graph_receptive_field_sweep(
        scene=scene,
        gt_count_map=gt_count_map,
        clusters=cluster_store.clusters,
        radius_values=[2.0, 4.0, 6.0],
        layer_values=[2, 3],
        base_training_config=ACTIVE["training"],
        device=DEVICE,
        base_graph_config=ACTIVE["graph"],
        seed=SEED,
    )
    graph_sweep_table.to_csv(scene_output_dir / "graph_receptive_field_sweep.csv", index=False)
    display(graph_sweep_table)
else:
    graph_sweep_table = pd.DataFrame()
    print("graph receptive-field sweep skipped outside batch_demo mode.")
"""
    ),
    markdown_cell(
        """## Block 6. Representative Cluster Graph Visualization

Build the active graph set, select one cluster with non-zero GT support if available, and render the actual pixel-node graph so the neighborhood structure can be inspected directly.
"""
    ),
    code_cell(
        """graph_builder = GraphBuilder(ACTIVE["graph"])
graphs = graph_builder.build(cluster_store.clusters)
graph_map = {
    int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0]): graph
    for graph in graphs
}

representative_cluster = choose_representative_cluster(cluster_store.clusters)
representative_graph = graph_map[representative_cluster.cluster_id]

print(f"graph_count={len(graphs)}")
print(f"total_nodes={sum(int(graph.num_nodes) for graph in graphs)}")
print(f"first_graph_nodes={graphs[0].num_nodes}")
print(f"first_graph_edges={graphs[0].edge_index.shape[1]}")
print(
    f"representative_cluster_id={representative_cluster.cluster_id}, "
    f"gt_sum={representative_cluster.gt_sum:.1f}, "
    f"node_count={representative_cluster.node_count}"
)

fig = visualize_graph_cluster(representative_cluster, representative_graph)
graph_viz_path = scene_output_dir / f"{scene.key}_cluster_{representative_cluster.cluster_id}_graph.png"
fig.savefig(graph_viz_path, dpi=180, bbox_inches="tight")
plt.show()
print(f"graph_viz_path={graph_viz_path}")
"""
    ),
    markdown_cell(
        """## Block 7. Loss Weighting Comparison

Compare several supervision variants on the same graph set: the current positive-pixel baseline, count-aware weighting, count-aware weighting with a patch-sum constraint, and a reference-only target scaling run. The last option is included only to show what happens when the target unit itself is stretched.
"""
    ),
    code_cell(
        """if SCENE_MODE == "batch_demo":
    loss_weighting_table = loss_weighting_sweep(
        scene=scene,
        gt_count_map=gt_count_map,
        clusters=cluster_store.clusters,
        graph_config=ACTIVE["graph"],
        base_training_config=ACTIVE["training"],
        device=DEVICE,
        seed=SEED,
    )
    loss_weighting_table.to_csv(scene_output_dir / "loss_weighting_sweep.csv", index=False)
    display(loss_weighting_table)
else:
    loss_weighting_table = pd.DataFrame()
    print("loss weighting sweep skipped outside batch_demo mode.")
"""
    ),
    markdown_cell(
        """## Block 8. Patch-to-Graph Conversion, GAT Training, and Checkpoint Export

Each pixel inside a DRUID contour mask becomes a node. Node features are `[brightness, local_x, local_y]`, edges come from the configured `radius_graph` radius, and the target is the per-pixel ship count. The model output is a non-negative density estimate via ReLU.
"""
    ),
    code_cell(
        """model = GATDensityRegressor(
    in_channels=3,
    hidden_channels=ACTIVE["training"].hidden_channels,
    heads=ACTIVE["training"].heads,
    num_layers=ACTIVE["training"].num_layers,
    dropout=ACTIVE["training"].dropout,
)

history = train_gat(model, graphs, DEVICE, ACTIVE["training"])
history.to_csv(scene_output_dir / "training_history.csv", index=False)
checkpoint_info = save_model_checkpoint(
    scene_output_dir / f"{scene.key}_gat_checkpoint.pt",
    model,
    graph_config=ACTIVE["graph"],
    training_config=ACTIVE["training"],
    scene=scene,
    metadata={
        "scene_mode": SCENE_MODE,
        "run_tag": RUN_TAG,
        "seed": SEED,
        "graph_count": len(graphs),
        "cluster_count": len(cluster_store.clusters),
        "total_nodes": int(sum(int(graph.num_nodes) for graph in graphs)),
    },
)
reloaded_model, reloaded_bundle = load_model_checkpoint(checkpoint_info["checkpoint_path"], map_location="cpu")

display(history)
print(f"parameter_count={count_model_parameters(model)}")
print(f"trainable_parameter_count={count_model_parameters(model, trainable_only=True)}")
print(f"checkpoint_path={checkpoint_info['checkpoint_path']}")
print(f"checkpoint_summary_json={checkpoint_info['summary_json_path']}")
print(f"checkpoint_size_bytes={checkpoint_info['file_size_bytes']}")
print(f"checkpoint_size_mb={checkpoint_info['file_size_mb']:.4f}")
print(f"reloaded_model_class={type(reloaded_model).__name__}")
print(f"reloaded_num_layers={reloaded_bundle['architecture']['num_layers']}")
"""
    ),
    markdown_cell(
        """## Block 9. Cluster Inference and Representative Prediction View

Predict each cluster graph, then revisit the representative cluster to compare node-level predictions against the graph structure before the full-scene merge.
"""
    ),
    code_cell(
        """predictions = predict_graphs(model, graphs, DEVICE)
representative_pred = predictions[representative_cluster.cluster_id]
fig = visualize_graph_cluster(
    representative_cluster,
    representative_graph,
    pred_values=representative_pred,
)
pred_viz_path = scene_output_dir / f"{scene.key}_cluster_{representative_cluster.cluster_id}_prediction.png"
fig.savefig(pred_viz_path, dpi=180, bbox_inches="tight")
plt.show()
print(f"pred_viz_path={pred_viz_path}")
"""
    ),
    markdown_cell(
        """## Block 10. Lifetime-Weighted Scene Merge and GeoTIFF Export

Map each cluster prediction back to the original pixel grid, combine overlaps by lifetime-weighted averaging, and save the merged result as a geocoded density heatmap GeoTIFF.
"""
    ),
    code_cell(
        """assembler = SceneAssembler(scene)
for cluster in cluster_store.clusters:
    assembler.accumulate(cluster, predictions[cluster.cluster_id])

heatmap = assembler.finalize()
heatmap_path = assembler.save_geotiff(
    scene_output_dir / f"{scene.key}_gat_density_heatmap.tif",
    heatmap,
)

overlay_rgb = make_overlay_rgb(scene.image, heatmap)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].imshow(scene.image, cmap="cividis", vmin=0, vmax=1)
axes[0, 0].set_title("Scene radiance")
axes[0, 0].set_axis_off()

axes[0, 1].imshow(gt_count_map, cmap="inferno")
axes[0, 1].set_title("GT count map")
axes[0, 1].set_axis_off()

axes[1, 0].imshow(heatmap, cmap="magma")
axes[1, 0].set_title("Predicted density heatmap")
axes[1, 0].set_axis_off()

axes[1, 1].imshow(overlay_rgb)
axes[1, 1].set_title("Radiance + predicted heatmap overlay")
axes[1, 1].set_axis_off()
plt.tight_layout()
overview_viz_path = scene_output_dir / f"{scene.key}_scene_overview.png"
fig.savefig(overview_viz_path, dpi=180, bbox_inches="tight")
plt.show()

print(f"heatmap_path={heatmap_path}")
print(f"overview_viz_path={overview_viz_path}")
print(f"heatmap_nonzero_pixels={(heatmap > 0).sum()}")
print(f"heatmap_min={heatmap.min():.6f}")
print(f"heatmap_max={heatmap.max():.6f}")
print(f"heatmap_mean={heatmap.mean():.6f}")
"""
    ),
]


NOTEBOOK = {
    "cells": NOTEBOOK_CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (DNB_AIS)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "DNB_GAT_v1.ipynb"
    notebook_path.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=2))
    print(notebook_path)


if __name__ == "__main__":
    main()
