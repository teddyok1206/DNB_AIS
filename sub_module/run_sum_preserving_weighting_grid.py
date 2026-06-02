from __future__ import annotations

from pathlib import Path

from sub_module.dnb_gat_pipeline import (
    DruidClusterStore,
    DruidConfig,
    GraphConfig,
    GroundTruthResolver,
    SceneRaster,
    TrainingConfig,
    resolve_device,
    weighting_grid_sweep,
)


def main() -> None:
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

    device = resolve_device("mps")
    if device.type != "mps":
        raise RuntimeError(f"MPS required, got {device}")

    table = weighting_grid_sweep(
        scene=scene,
        gt_count_map=gt_count_map,
        clusters=cluster_store.clusters,
        graph_config=GraphConfig(
            radius_pixels=4.0,
            gt_smoothing_hop_weights=(1.0, 0.6, 0.2),
            gt_smoothing_preserve_mass=True,
        ),
        base_training_config=TrainingConfig(
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
        ),
        device=device,
        positive_weights=[0.0, 4.0, 8.0, 12.0],
        seed=1,
    )
    out_path = output_dir / "sum_preserving_weighting_grid.csv"
    table.to_csv(out_path, index=False)
    print(
        table[
            [
                "positive_weight",
                "scene_topk_hit_rate",
                "graph_topk_hit_rate",
                "pred_sum_ratio",
                "heatmap_sum",
                "heatmap_max",
                "final_train_loss",
            ]
        ].to_csv(index=False)
    )
    print(str(out_path))


if __name__ == "__main__":
    main()
