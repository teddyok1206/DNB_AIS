from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    predict_graphs,
    resolve_device,
    train_gat,
)


def main() -> None:
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = Path(__file__).resolve().parents[1]
    step3 = root / "[3]_DNB_AIS - (STEP 3)"
    output_dir = root / "outputs" / "DNB_GAT_v1" / "A2025001_1754_021" / "batch_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = SceneRaster.load(step3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif")
    resolver = GroundTruthResolver(
        metadata_csv=step3 / "metadata_JPSS-2.csv",
        ships_db_path=Path("/Users/jungtaeuk/ships/ships.db"),
        default_geojson_dir=step3 / "bboxes_JPSS-2",
    )
    gt_points = resolver.load_points(
        resolver.resolve_geojson(scene, step3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson")
    )
    gt_count_map = resolver.rasterize_counts(scene, gt_points)
    cluster_store = DruidClusterStore.from_scene(
        scene=scene,
        gt_count_map=gt_count_map,
        druid_root=step3 / "DRUID",
        config=DruidConfig(
            cutup=False,
            area_limit=12,
            max_catalogue_clusters=None,
            min_nodes=16,
            max_nodes=2500,
        ),
    )
    graphs = GraphBuilder(
        GraphConfig(
            radius_pixels=4.0,
            gt_smoothing_hop_weights=(1.0, 0.6, 0.2),
            gt_smoothing_preserve_mass=True,
        )
    ).build(cluster_store.clusters)
    cluster_by_id = {cluster.cluster_id: cluster for cluster in cluster_store.clusters}

    device = resolve_device("mps")
    if device.type != "mps":
        raise RuntimeError(f"MPS required, got {device}")

    base = TrainingConfig(
        hidden_channels=32,
        heads=4,
        num_layers=3,
        epochs=4,
        batch_size=4,
        dropout=0.05,
        output_activation="softplus",
        loss_name="poisson_nll",
        positive_weight=0.0,
        target_field="y_edge_decay",
    )
    raw_graph_sum = float(sum(float(graph.y.sum()) for graph in graphs))
    raw_scene_sum = float(gt_count_map.sum())
    scene_positive_mask = gt_count_map > 0
    scene_positive = int(scene_positive_mask.sum())

    rows = []
    for lam in [0.0, 1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = replace(base, count_sum_lambda=float(lam))
        model = GATDensityRegressor(
            in_channels=int(graphs[0].x.shape[1]),
            hidden_channels=cfg.hidden_channels,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_activation=cfg.output_activation,
        )
        history = train_gat(model, graphs, device, cfg)
        predictions = predict_graphs(model, graphs, device)
        assembler = SceneAssembler(scene)
        for cluster_id, pred in predictions.items():
            assembler.accumulate(cluster_by_id[cluster_id], pred)
        heatmap = assembler.finalize()
        pred_graph_sum = float(sum(float(pred.sum()) for pred in predictions.values()))
        heatmap_sum = float(heatmap.sum())
        heatmap_max = float(heatmap.max())
        heatmap_mean = float(heatmap.mean())
        topk_hit = None
        if scene_positive > 0 and np.any(heatmap > 0):
            flat_heat = heatmap.reshape(-1)
            flat_gt = scene_positive_mask.reshape(-1)
            top_idx = np.argpartition(flat_heat, -scene_positive)[-scene_positive:]
            topk_hit = float(flat_gt[top_idx].mean())
        rows.append(
            {
                "count_sum_lambda": float(lam),
                "seed": seed,
                "final_train_loss": float(history["train_loss"].iloc[-1]),
                "pred_graph_sum": pred_graph_sum,
                "pred_graph_to_raw_ratio": pred_graph_sum / raw_graph_sum if raw_graph_sum else None,
                "heatmap_sum": heatmap_sum,
                "heatmap_to_raw_ratio": heatmap_sum / raw_scene_sum if raw_scene_sum else None,
                "heatmap_max": heatmap_max,
                "heatmap_mean": heatmap_mean,
                "scene_topk_hit_rate": topk_hit,
                "heatmap_gt_mean": float(heatmap[scene_positive_mask].mean()) if scene_positive else 0.0,
                "heatmap_bg_mean": float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    out_path = output_dir / "sum_preserving_count_sum_lambda_sweep.csv"
    result.to_csv(out_path, index=False)
    print(result.to_csv(index=False))
    print(str(out_path))


if __name__ == "__main__":
    main()
