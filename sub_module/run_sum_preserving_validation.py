from __future__ import annotations

import json
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
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
    make_overlay_rgb,
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
    scene_path = step3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif"
    geojson_path = step3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson"
    output_dir = root / "outputs" / "DNB_GAT_v1" / "A2025001_1754_021" / "batch_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("run_%m%d_%H%M%S")
    run_output_dir = output_dir / run_tag
    run_output_dir.mkdir(parents=True, exist_ok=True)

    scene = SceneRaster.load(scene_path)
    resolver = GroundTruthResolver(
        metadata_csv=step3 / "metadata_JPSS-2.csv",
        ships_db_path=Path("/Users/jungtaeuk/ships/ships.db"),
        default_geojson_dir=step3 / "bboxes_JPSS-2",
    )
    gt_geojson = resolver.resolve_geojson(scene, geojson_path)
    gt_points = resolver.load_points(gt_geojson)
    gt_count_map = resolver.rasterize_counts(scene, gt_points)

    druid_config = DruidConfig(
        cutup=False,
        area_limit=12,
        max_catalogue_clusters=None,
        min_nodes=16,
        max_nodes=2500,
    )
    graph_config = GraphConfig(
        radius_pixels=4.0,
        gt_smoothing_hop_weights=(1.0, 0.6, 0.2),
        gt_smoothing_preserve_mass=True,
    )
    training_config = TrainingConfig(
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

    cluster_store = DruidClusterStore.from_scene(
        scene=scene,
        gt_count_map=gt_count_map,
        druid_root=step3 / "DRUID",
        config=druid_config,
    )
    graphs = GraphBuilder(graph_config).build(cluster_store.clusters)

    raw_scene_sum = float(gt_count_map.sum())
    raw_graph_sum = float(sum(float(graph.y.sum()) for graph in graphs))
    edge_graph_sum = float(sum(float(graph.y_edge_decay.sum()) for graph in graphs))
    per_graph_diff = [float(graph.y_edge_decay.sum() - graph.y.sum()) for graph in graphs]
    max_abs_graph_diff = float(max(abs(value) for value in per_graph_diff)) if per_graph_diff else 0.0

    positive_graph_diffs = [
        {
            "cluster_id": int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0]),
            "raw_sum": float(graph.y.sum()),
            "edge_sum": float(graph.y_edge_decay.sum()),
            "diff": float(graph.y_edge_decay.sum() - graph.y.sum()),
        }
        for graph in graphs
        if float(graph.y.sum()) > 0
    ]
    positive_graph_diffs = sorted(positive_graph_diffs, key=lambda row: abs(row["diff"]), reverse=True)
    device = resolve_device("mps")
    pred_graph_sum = None
    heatmap_sum = None
    heatmap_max = None
    heatmap_mean = None
    history_path = None
    heatmap_path = None
    overlay_path = None
    cluster_pred_summary_path = None
    pred_positive_pixel_count = None
    pred_topk_hit_rate = None
    pred_gt_mean = None
    pred_bg_mean = None
    if device.type == "mps":
        model = GATDensityRegressor(
            in_channels=int(graphs[0].x.shape[1]),
            hidden_channels=training_config.hidden_channels,
            heads=training_config.heads,
            num_layers=training_config.num_layers,
            dropout=training_config.dropout,
            output_activation=training_config.output_activation,
        )
        history = train_gat(model, graphs, device, training_config)
        predictions = predict_graphs(model, graphs, device)
        assembler = SceneAssembler(scene)
        cluster_by_id = {cluster.cluster_id: cluster for cluster in cluster_store.clusters}
        cluster_pred_rows = []
        for cluster_id, pred in predictions.items():
            assembler.accumulate(cluster_by_id[cluster_id], pred)
            graph = next(graph for graph in graphs if int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0]) == cluster_id)
            cluster_pred_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "raw_target_sum": float(graph.y.sum()),
                    "edge_target_sum": float(graph.y_edge_decay.sum()),
                    "pred_sum": float(pred.sum()),
                    "pred_max": float(pred.max()) if pred.size else 0.0,
                    "pred_to_edge_ratio": float(pred.sum() / float(graph.y_edge_decay.sum())) if float(graph.y_edge_decay.sum()) > 0 else None,
                }
            )
        heatmap = assembler.finalize()

        pred_graph_sum = float(sum(float(pred.sum()) for pred in predictions.values()))
        heatmap_sum = float(heatmap.sum())
        heatmap_max = float(heatmap.max())
        heatmap_mean = float(heatmap.mean())
        pred_positive_pixel_count = int((heatmap > 0).sum())
        scene_positive_mask = gt_count_map > 0
        gt_scene_positive = int(scene_positive_mask.sum())
        pred_gt_mean = float(heatmap[scene_positive_mask].mean()) if gt_scene_positive else 0.0
        pred_bg_mean = float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0
        if gt_scene_positive > 0 and pred_positive_pixel_count > 0:
            flat_heatmap = heatmap.reshape(-1)
            flat_gt = scene_positive_mask.reshape(-1)
            top_scene_idx = flat_heatmap.argpartition(-gt_scene_positive)[-gt_scene_positive:]
            pred_topk_hit_rate = float(flat_gt[top_scene_idx].mean())

        history_path = run_output_dir / "sum_preserving_training_history.csv"
        history.to_csv(history_path, index=False)
        heatmap_path = assembler.save_geotiff(run_output_dir / f"{scene.key}_sum_preserving_heatmap.tif", heatmap)
        overlay_path = run_output_dir / f"{scene.key}_sum_preserving_overlay.png"
        overlay_rgb = make_overlay_rgb(scene.image, heatmap)
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay_rgb)
        plt.title("Sum-preserving heatmap overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=180, bbox_inches="tight")
        plt.close()
        cluster_pred_summary = pd.DataFrame(cluster_pred_rows).sort_values("pred_sum", ascending=False).reset_index(drop=True)
        cluster_pred_summary_path = run_output_dir / "sum_preserving_cluster_prediction_summary.csv"
        cluster_pred_summary.to_csv(cluster_pred_summary_path, index=False)

    report = {
        "scene_key": scene.key,
        "seed": seed,
        "run_tag": run_tag,
        "device": str(device),
        "graph_count": len(graphs),
        "raw_scene_sum": raw_scene_sum,
        "raw_graph_sum": raw_graph_sum,
        "edge_graph_sum": edge_graph_sum,
        "max_abs_graph_diff": max_abs_graph_diff,
        "pred_graph_sum": pred_graph_sum,
        "heatmap_sum": heatmap_sum,
        "heatmap_max": heatmap_max,
        "heatmap_mean": heatmap_mean,
        "heatmap_nonzero_pixels": pred_positive_pixel_count,
        "heatmap_gt_mean": pred_gt_mean,
        "heatmap_bg_mean": pred_bg_mean,
        "scene_topk_hit_rate": pred_topk_hit_rate,
        "pred_graph_to_raw_ratio": (pred_graph_sum / raw_graph_sum) if (pred_graph_sum is not None and raw_graph_sum) else None,
        "heatmap_to_raw_ratio": (heatmap_sum / raw_scene_sum) if (heatmap_sum is not None and raw_scene_sum) else None,
        "training_skipped": device.type != "mps",
        "heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "cluster_pred_summary_path": str(cluster_pred_summary_path) if cluster_pred_summary_path is not None else None,
        "graph_config": asdict(graph_config),
        "training_config": asdict(training_config),
        "largest_positive_graph_diffs": positive_graph_diffs[:10],
    }

    report_path = output_dir / "sum_preserving_validation.json"
    run_report_path = run_output_dir / "sum_preserving_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    run_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(report_path))
    print(str(run_report_path))
    if history_path is not None:
        print(str(history_path))


if __name__ == "__main__":
    main()
