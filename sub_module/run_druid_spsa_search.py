from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sub_module.dnb_gat_pipeline import (
    DruidClusterStore,
    DruidConfig,
    GroundTruthResolver,
    SceneRaster,
)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lower: float
    upper: float
    kind: str  # "int" or "float"

    def clip(self, value: float) -> float:
        return float(min(max(value, self.lower), self.upper))

    def to_value(self, unit_value: float) -> float:
        unit_clipped = float(min(max(unit_value, 0.0), 1.0))
        raw = self.lower + unit_clipped * (self.upper - self.lower)
        if self.kind == "int":
            return float(int(round(raw)))
        return float(raw)

    def to_unit(self, raw_value: float) -> float:
        if self.upper <= self.lower:
            return 0.0
        return float(min(max((raw_value - self.lower) / (self.upper - self.lower), 0.0), 1.0))


@dataclass(frozen=True)
class SceneContext:
    scene_id: str
    scene: SceneRaster
    gt_count_map: np.ndarray
    gt_geojson_path: Path
    metadata_csv: Path


@dataclass(frozen=True)
class SceneBudget:
    cluster_budget: int
    node_budget: int


def _safe_quantile(values: np.ndarray, q: float) -> int:
    if values.size == 0:
        return 0
    return int(np.quantile(values, q))


def summarize_clusters(scene: SceneRaster, gt_count_map: np.ndarray, cluster_store: DruidClusterStore) -> dict[str, float | int]:
    covered = np.zeros(scene.shape, dtype=np.uint8)
    overlap = np.zeros(scene.shape, dtype=np.uint16)
    node_counts = np.array([cluster.node_count for cluster in cluster_store.clusters], dtype=np.int32)
    clusters_with_gt = int(sum(cluster.gt_sum > 0 for cluster in cluster_store.clusters))

    for cluster in cluster_store.clusters:
        rows = cluster.global_rc[:, 0]
        cols = cluster.global_rc[:, 1]
        covered[rows, cols] = 1
        overlap[rows, cols] += 1

    gt_mask = gt_count_map > 0
    covered_mask = covered > 0
    gt_total_pixels = int(gt_mask.sum())
    gt_total_mass = float(gt_count_map.sum())
    covered_gt_pixels = int((gt_mask & covered_mask).sum())
    covered_gt_mass = float(gt_count_map[covered_mask].sum()) if covered_mask.any() else 0.0

    overlap_mean_on_covered = float(overlap[covered_mask].mean()) if covered_mask.any() else 0.0
    overlap_mean_on_gt = float(overlap[gt_mask].mean()) if gt_total_pixels > 0 else 0.0
    overlap_p95_on_covered = float(np.quantile(overlap[covered_mask], 0.95)) if covered_mask.any() else 0.0

    return {
        "catalogue_count": int(len(cluster_store.catalogue)),
        "cluster_count": int(len(cluster_store.clusters)),
        "total_nodes": int(node_counts.sum()) if node_counts.size else 0,
        "node_count_min": int(node_counts.min()) if node_counts.size else 0,
        "node_count_p25": _safe_quantile(node_counts, 0.25),
        "node_count_median": _safe_quantile(node_counts, 0.50),
        "node_count_p75": _safe_quantile(node_counts, 0.75),
        "node_count_p90": _safe_quantile(node_counts, 0.90),
        "node_count_max": int(node_counts.max()) if node_counts.size else 0,
        "clusters_lt16": int((node_counts < 16).sum()) if node_counts.size else 0,
        "clusters_lt32": int((node_counts < 32).sum()) if node_counts.size else 0,
        "clusters_with_gt": clusters_with_gt,
        "gt_total_pixels": gt_total_pixels,
        "gt_total_mass": gt_total_mass,
        "covered_gt_pixels": covered_gt_pixels,
        "covered_gt_mass": covered_gt_mass,
        "coverage_pixel_ratio": float(covered_gt_pixels / max(gt_total_pixels, 1)),
        "coverage_mass_ratio": float(covered_gt_mass / max(gt_total_mass, 1.0e-6)),
        "overlap_mean_on_covered": overlap_mean_on_covered,
        "overlap_mean_on_gt": overlap_mean_on_gt,
        "overlap_p95_on_covered": overlap_p95_on_covered,
    }


def score_summary(
    summary: dict[str, float | int],
    *,
    cluster_budget: int,
    node_budget: int,
    overlap_target: float,
) -> tuple[float, dict[str, float]]:
    cluster_count = int(summary["cluster_count"])
    total_nodes = int(summary["total_nodes"])
    clusters_with_gt = int(summary["clusters_with_gt"])

    coverage_mass_ratio = float(summary["coverage_mass_ratio"])
    coverage_pixel_ratio = float(summary["coverage_pixel_ratio"])
    gt_cluster_ratio = float(clusters_with_gt / max(cluster_count, 1))
    coverage_efficiency = float(summary["covered_gt_mass"]) / max(total_nodes, 1)
    tiny_cluster_ratio = float(summary["clusters_lt16"]) / max(cluster_count, 1)

    cluster_budget_penalty = max(cluster_count - int(cluster_budget), 0) / max(int(cluster_budget), 1)
    node_budget_penalty = max(total_nodes - int(node_budget), 0) / max(int(node_budget), 1)
    overlap_penalty = max(float(summary["overlap_mean_on_covered"]) - float(overlap_target), 0.0)

    score = (
        6.0 * coverage_mass_ratio
        + 2.0 * coverage_pixel_ratio
        + 0.6 * gt_cluster_ratio
        + 14.0 * coverage_efficiency
        - 0.35 * tiny_cluster_ratio
        - 0.80 * cluster_budget_penalty
        - 0.80 * node_budget_penalty
        - 0.50 * overlap_penalty
    )
    components = {
        "score_coverage_mass": 6.0 * coverage_mass_ratio,
        "score_coverage_pixel": 2.0 * coverage_pixel_ratio,
        "score_gt_cluster_ratio": 0.6 * gt_cluster_ratio,
        "score_efficiency": 14.0 * coverage_efficiency,
        "penalty_tiny_cluster": 0.35 * tiny_cluster_ratio,
        "penalty_cluster_budget": 0.80 * cluster_budget_penalty,
        "penalty_node_budget": 0.80 * node_budget_penalty,
        "penalty_overlap": 0.50 * overlap_penalty,
    }
    return float(score), components


def vector_to_params(vector: np.ndarray, specs: list[ParamSpec]) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for index, spec in enumerate(specs):
        value = spec.to_value(float(vector[index]))
        if spec.kind == "int":
            params[spec.name] = int(round(value))
        else:
            params[spec.name] = float(value)

    detection = float(params["detection_threshold"])
    analysis = float(params["analysis_threshold"])
    if detection > analysis:
        params["detection_threshold"], params["analysis_threshold"] = analysis, detection
    return params


def params_to_vector(params: dict[str, float | int], specs: list[ParamSpec]) -> np.ndarray:
    values = [spec.to_unit(float(params[spec.name])) for spec in specs]
    return np.asarray(values, dtype=np.float64)


def params_to_config(base_config: DruidConfig, params: dict[str, float | int]) -> DruidConfig:
    return replace(
        base_config,
        area_limit=int(params["area_limit"]),
        min_nodes=int(params["min_nodes"]),
        detection_threshold=float(params["detection_threshold"]),
        analysis_threshold=float(params["analysis_threshold"]),
        lifetime_limit_fraction=float(params.get("lifetime_limit_fraction", base_config.lifetime_limit_fraction)),
        smooth_sigma=float(params.get("smooth_sigma", base_config.smooth_sigma)),
    )


def key_from_params(params: dict[str, float | int], specs: list[ParamSpec]) -> tuple[Any, ...]:
    items: list[Any] = []
    for spec in specs:
        value = params[spec.name]
        if spec.kind == "int":
            items.append(int(value))
        else:
            items.append(round(float(value), 6))
    return tuple(items)


def evaluate_druid_config(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    druid_root: Path,
    config: DruidConfig,
) -> DruidClusterStore:
    return DruidClusterStore.from_scene(
        scene=scene,
        gt_count_map=gt_count_map,
        druid_root=druid_root,
        config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient-like DRUID parameter search with SPSA (DRUID-only, no GNN).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--spsa-a", type=float, default=0.18)
    parser.add_argument("--spsa-c", type=float, default=0.12)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)

    parser.add_argument("--area-min", type=int, default=4)
    parser.add_argument("--area-max", type=int, default=20)
    parser.add_argument("--min-nodes-min", type=int, default=8)
    parser.add_argument("--min-nodes-max", type=int, default=40)

    parser.add_argument("--cluster-budget", type=int, default=0, help="If 0, infer per-scene from baseline.")
    parser.add_argument("--node-budget", type=int, default=0, help="If 0, infer per-scene from baseline.")
    parser.add_argument("--cluster-budget-scale", type=float, default=1.35)
    parser.add_argument("--node-budget-scale", type=float, default=1.35)
    parser.add_argument("--cluster-budget-pad", type=int, default=40)
    parser.add_argument("--node-budget-pad", type=int, default=2000)
    parser.add_argument("--overlap-target", type=float, default=1.35)
    parser.add_argument("--max-nodes", type=int, default=2500)

    parser.add_argument("--agg-median-weight", type=float, default=0.55)
    parser.add_argument("--agg-worst-weight", type=float, default=0.30)
    parser.add_argument("--agg-mean-weight", type=float, default=0.15)

    parser.add_argument("--scene-list-json", type=str, default="")
    parser.add_argument("--scene-glob", type=str, default="")
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--scene-path", type=str, default="")
    parser.add_argument("--geojson-path", type=str, default="")
    parser.add_argument("--metadata-csv", type=str, default="")

    parser.add_argument("--ships-db", type=str, default="/Users/jungtaeuk/ships/ships.db")
    parser.add_argument("--druid-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def load_scene_specs(args: argparse.Namespace, step3: Path) -> list[dict[str, Any]]:
    if args.scene_list_json:
        payload = json.loads(Path(args.scene_list_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("scenes", [])
        if not isinstance(payload, list):
            raise ValueError("--scene-list-json must be a list or an object containing a 'scenes' list.")
        specs: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict) or "scene_path" not in item:
                raise ValueError("Each scene-list item must contain at least 'scene_path'.")
            specs.append(dict(item))
        return specs

    if args.scene_glob:
        paths = sorted(glob.glob(args.scene_glob))
        if not paths:
            dir_part, sep, file_part = args.scene_glob.rpartition("/")
            if sep:
                escaped_pattern = f"{glob.escape(dir_part)}/{file_part}"
            else:
                escaped_pattern = glob.escape(args.scene_glob)
            paths = sorted(glob.glob(escaped_pattern))
        return [{"scene_path": str(Path(path).resolve())} for path in paths]

    default_scene = step3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif"
    scene_path = Path(args.scene_path) if args.scene_path else default_scene
    spec: dict[str, Any] = {"scene_path": str(scene_path.resolve())}
    if args.geojson_path:
        spec["geojson_path"] = str(Path(args.geojson_path).resolve())
    if args.metadata_csv:
        spec["metadata_csv"] = str(Path(args.metadata_csv).resolve())
    return [spec]


def load_scene_contexts(
    scene_specs: list[dict[str, Any]],
    *,
    step3: Path,
    global_metadata_csv: Path,
    global_geojson_path: Path | None,
    ships_db_path: Path,
) -> list[SceneContext]:
    contexts: list[SceneContext] = []
    default_geojson_dir = step3 / "bboxes_JPSS-2"
    id_counts: dict[str, int] = {}

    for spec in scene_specs:
        scene_path = Path(spec["scene_path"]).resolve()
        metadata_csv = Path(spec.get("metadata_csv", global_metadata_csv)).resolve()
        requested_geojson_raw = spec.get("geojson_path")
        requested_geojson = Path(requested_geojson_raw).resolve() if requested_geojson_raw else global_geojson_path

        scene = SceneRaster.load(scene_path)
        resolver = GroundTruthResolver(
            metadata_csv=metadata_csv,
            ships_db_path=ships_db_path,
            default_geojson_dir=default_geojson_dir,
        )
        gt_geojson = resolver.resolve_geojson(scene, requested_geojson if requested_geojson is not None else None)
        gt_points = resolver.load_points(gt_geojson)
        gt_count_map = resolver.rasterize_counts(scene, gt_points)

        base_id = f"{scene.key}__{scene.path.stem}"
        serial = id_counts.get(base_id, 0) + 1
        id_counts[base_id] = serial
        scene_id = base_id if serial == 1 else f"{base_id}_{serial}"
        contexts.append(
            SceneContext(
                scene_id=scene_id,
                scene=scene,
                gt_count_map=gt_count_map,
                gt_geojson_path=gt_geojson,
                metadata_csv=metadata_csv,
            )
        )
    return contexts


def aggregate_scene_scores(
    scene_scores: list[float],
    *,
    weight_median: float,
    weight_worst: float,
    weight_mean: float,
) -> tuple[float, dict[str, float]]:
    values = np.asarray(scene_scores, dtype=np.float64)
    score_median = float(np.median(values))
    score_worst = float(values.min())
    score_mean = float(values.mean())
    aggregate = (
        float(weight_median) * score_median
        + float(weight_worst) * score_worst
        + float(weight_mean) * score_mean
    )
    return aggregate, {
        "score_median": score_median,
        "score_worst": score_worst,
        "score_mean": score_mean,
        "score_std": float(values.std()),
    }


def eval_id_from_key(key: tuple[Any, ...]) -> str:
    digest = hashlib.md5(repr(key).encode("utf-8")).hexdigest()[:12]
    return f"eval_{digest}"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    root = Path(__file__).resolve().parents[1]
    step3 = root / "[3]_DNB_AIS - (STEP 3)"
    global_metadata_csv = Path(args.metadata_csv).resolve() if args.metadata_csv else (step3 / "metadata_JPSS-2.csv").resolve()
    global_geojson_path = Path(args.geojson_path).resolve() if args.geojson_path else None
    ships_db_path = Path(args.ships_db).resolve()
    druid_root = Path(args.druid_root).resolve() if args.druid_root else (step3 / "DRUID").resolve()

    raw_specs = load_scene_specs(args, step3)
    if args.scene_limit > 0:
        raw_specs = raw_specs[: int(args.scene_limit)]
    if not raw_specs:
        raise ValueError("No scenes matched. Check --scene-glob/--scene-list-json/--scene-path.")

    contexts = load_scene_contexts(
        raw_specs,
        step3=step3,
        global_metadata_csv=global_metadata_csv,
        global_geojson_path=global_geojson_path,
        ships_db_path=ships_db_path,
    )

    weight_sum = float(args.agg_median_weight + args.agg_worst_weight + args.agg_mean_weight)
    if weight_sum <= 0.0:
        raise ValueError("Aggregate weights must sum to a positive value.")
    weight_median = float(args.agg_median_weight / weight_sum)
    weight_worst = float(args.agg_worst_weight / weight_sum)
    weight_mean = float(args.agg_mean_weight / weight_sum)

    base_config = DruidConfig(
        cutup=False,
        area_limit=12,
        max_catalogue_clusters=None,
        min_nodes=16,
        max_nodes=int(args.max_nodes),
        detection_threshold=1.0,
        analysis_threshold=1.0,
        lifetime_limit_fraction=1.001,
        smooth_sigma=0.0,
    )

    specs = [
        ParamSpec("area_limit", float(args.area_min), float(args.area_max), "int"),
        ParamSpec("min_nodes", float(args.min_nodes_min), float(args.min_nodes_max), "int"),
        ParamSpec("detection_threshold", 0.70, 1.80, "float"),
        ParamSpec("analysis_threshold", 0.80, 2.40, "float"),
    ]

    baseline_params: dict[str, float | int] = {
        "area_limit": int(base_config.area_limit),
        "min_nodes": int(base_config.min_nodes),
        "detection_threshold": float(base_config.detection_threshold),
        "analysis_threshold": float(base_config.analysis_threshold),
    }

    scene_budgets: dict[str, SceneBudget] = {}
    baseline_scene_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        baseline_store = evaluate_druid_config(ctx.scene, ctx.gt_count_map, druid_root, base_config)
        summary = summarize_clusters(ctx.scene, ctx.gt_count_map, baseline_store)
        cluster_budget = int(args.cluster_budget) if args.cluster_budget > 0 else max(
            int(summary["cluster_count"]) + int(args.cluster_budget_pad),
            int(round(int(summary["cluster_count"]) * float(args.cluster_budget_scale))),
        )
        node_budget = int(args.node_budget) if args.node_budget > 0 else max(
            int(summary["total_nodes"]) + int(args.node_budget_pad),
            int(round(int(summary["total_nodes"]) * float(args.node_budget_scale))),
        )
        scene_budgets[ctx.scene_id] = SceneBudget(cluster_budget=cluster_budget, node_budget=node_budget)
        baseline_scene_rows.append(
            {
                "scene_id": ctx.scene_id,
                "scene_key": ctx.scene.key,
                "scene_path": str(ctx.scene.path),
                "cluster_budget": cluster_budget,
                "node_budget": node_budget,
                "baseline_cluster_count": int(summary["cluster_count"]),
                "baseline_total_nodes": int(summary["total_nodes"]),
                "baseline_coverage_mass_ratio": float(summary["coverage_mass_ratio"]),
                "baseline_coverage_pixel_ratio": float(summary["coverage_pixel_ratio"]),
            }
        )

    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def evaluate(params: dict[str, float | int]) -> dict[str, Any]:
        key = key_from_params(params, specs)
        if key in cache:
            return dict(cache[key]["row"])

        eval_id = eval_id_from_key(key)
        config = params_to_config(base_config, params)
        scene_rows: list[dict[str, Any]] = []
        scene_scores: list[float] = []
        coverage_mass_list: list[float] = []
        coverage_pixel_list: list[float] = []
        cluster_count_list: list[int] = []
        total_nodes_list: list[int] = []
        failed = 0
        error_messages: list[str] = []

        for ctx in contexts:
            budget = scene_budgets[ctx.scene_id]
            scene_row: dict[str, Any] = {
                "eval_id": eval_id,
                "scene_id": ctx.scene_id,
                "scene_key": ctx.scene.key,
                "scene_path": str(ctx.scene.path),
                "cluster_budget": int(budget.cluster_budget),
                "node_budget": int(budget.node_budget),
            }
            try:
                store = evaluate_druid_config(ctx.scene, ctx.gt_count_map, druid_root, config)
                summary = summarize_clusters(ctx.scene, ctx.gt_count_map, store)
                score, components = score_summary(
                    summary,
                    cluster_budget=int(budget.cluster_budget),
                    node_budget=int(budget.node_budget),
                    overlap_target=float(args.overlap_target),
                )
                scene_row.update(summary)
                scene_row.update(components)
                scene_row["scene_score"] = float(score)
                scene_row["error"] = ""

                scene_scores.append(float(score))
                coverage_mass_list.append(float(summary["coverage_mass_ratio"]))
                coverage_pixel_list.append(float(summary["coverage_pixel_ratio"]))
                cluster_count_list.append(int(summary["cluster_count"]))
                total_nodes_list.append(int(summary["total_nodes"]))
            except Exception as exc:  # pragma: no cover - runtime-dependent DRUID failures
                failed += 1
                error_messages.append(f"{ctx.scene_id}: {exc}")
                scene_row.update(
                    {
                        "catalogue_count": 0,
                        "cluster_count": 0,
                        "total_nodes": 0,
                        "coverage_pixel_ratio": 0.0,
                        "coverage_mass_ratio": 0.0,
                        "clusters_with_gt": 0,
                        "overlap_mean_on_covered": 0.0,
                        "scene_score": -1.0e9,
                        "error": str(exc),
                    }
                )
                scene_scores.append(-1.0e9)
                coverage_mass_list.append(0.0)
                coverage_pixel_list.append(0.0)
                cluster_count_list.append(0)
                total_nodes_list.append(0)

            scene_rows.append(scene_row)

        aggregate_score, aggregate_parts = aggregate_scene_scores(
            scene_scores,
            weight_median=weight_median,
            weight_worst=weight_worst,
            weight_mean=weight_mean,
        )

        row: dict[str, Any] = {spec.name: params[spec.name] for spec in specs}
        row.update(
            {
                "eval_id": eval_id,
                "num_scenes": len(contexts),
                "num_failed_scenes": failed,
                "coverage_mass_ratio_mean": float(np.mean(coverage_mass_list)),
                "coverage_mass_ratio_median": float(np.median(coverage_mass_list)),
                "coverage_mass_ratio_min": float(np.min(coverage_mass_list)),
                "coverage_pixel_ratio_mean": float(np.mean(coverage_pixel_list)),
                "coverage_pixel_ratio_median": float(np.median(coverage_pixel_list)),
                "coverage_pixel_ratio_min": float(np.min(coverage_pixel_list)),
                "cluster_count_mean": float(np.mean(cluster_count_list)),
                "cluster_count_median": float(np.median(cluster_count_list)),
                "cluster_count_max": int(np.max(cluster_count_list)),
                "total_nodes_mean": float(np.mean(total_nodes_list)),
                "total_nodes_median": float(np.median(total_nodes_list)),
                "total_nodes_max": int(np.max(total_nodes_list)),
                **aggregate_parts,
                "score": float(aggregate_score),
                "error": " | ".join(error_messages[:3]),
            }
        )

        cache[key] = {"row": dict(row), "scene_rows": [dict(item) for item in scene_rows]}
        return row

    baseline_eval = evaluate(baseline_params)
    best_eval = dict(baseline_eval)
    best_params = dict(baseline_params)

    history_rows: list[dict[str, Any]] = []
    best_vector = params_to_vector(best_params, specs)
    vector_size = len(specs)
    a = float(args.spsa_a)
    c = float(args.spsa_c)
    alpha = float(args.spsa_alpha)
    gamma = float(args.spsa_gamma)
    A = max(float(args.iterations) * 0.1, 1.0)

    for restart in range(int(args.restarts)):
        if restart == 0:
            x = params_to_vector(baseline_params, specs)
        else:
            x = np.clip(best_vector + rng.normal(0.0, 0.12, size=vector_size), 0.0, 1.0)

        for step in range(1, int(args.iterations) + 1):
            global_step = restart * int(args.iterations) + step
            ck = c / (global_step ** gamma)
            ak = a / ((global_step + A) ** alpha)
            delta = rng.choice(np.array([-1.0, 1.0]), size=vector_size)

            x_plus = np.clip(x + ck * delta, 0.0, 1.0)
            x_minus = np.clip(x - ck * delta, 0.0, 1.0)

            plus_eval = evaluate(vector_to_params(x_plus, specs))
            minus_eval = evaluate(vector_to_params(x_minus, specs))

            grad = ((float(plus_eval["score"]) - float(minus_eval["score"])) / max(2.0 * ck, 1.0e-8)) * delta
            x = np.clip(x + ak * grad, 0.0, 1.0)

            current_params = vector_to_params(x, specs)
            current_eval = evaluate(current_params)
            if float(current_eval["score"]) > float(best_eval["score"]):
                best_eval = dict(current_eval)
                best_params = dict(current_params)
                best_vector = x.copy()

            history_rows.append(
                {
                    "restart": restart,
                    "step": step,
                    "global_step": global_step,
                    "ak": float(ak),
                    "ck": float(ck),
                    "score_plus": float(plus_eval["score"]),
                    "score_minus": float(minus_eval["score"]),
                    "score_current": float(current_eval["score"]),
                    "score_best": float(best_eval["score"]),
                    **{name: current_params[name] for name in [spec.name for spec in specs]},
                }
            )

    improved = True
    integer_names = [spec.name for spec in specs if spec.kind == "int"]
    spec_by_name = {spec.name: spec for spec in specs}
    while improved:
        improved = False
        current_eval = evaluate(best_params)
        for name in integer_names:
            spec = spec_by_name[name]
            base_value = int(best_params[name])
            for step in (-2, -1, 1, 2):
                candidate = dict(best_params)
                candidate[name] = int(round(spec.clip(base_value + step)))
                candidate_eval = evaluate(candidate)
                if float(candidate_eval["score"]) > float(current_eval["score"]):
                    best_params = candidate
                    best_eval = dict(candidate_eval)
                    best_vector = params_to_vector(best_params, specs)
                    improved = True
                    current_eval = dict(candidate_eval)

    run_tag = datetime.now().strftime("run_%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif len(contexts) == 1:
        output_dir = root / "outputs" / "DNB_GAT_v1" / contexts[0].scene.key / "druid_spsa"
    else:
        output_dir = root / "outputs" / "DNB_GAT_v1" / "multi_scene" / "druid_spsa"
    run_dir = output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history_rows)
    eval_df = pd.DataFrame([entry["row"] for entry in cache.values()]).sort_values("score", ascending=False).reset_index(drop=True)
    scene_eval_df = pd.DataFrame(
        [scene_row for entry in cache.values() for scene_row in entry["scene_rows"]]
    ).sort_values(["eval_id", "scene_id"]).reset_index(drop=True)

    history_path = run_dir / "druid_spsa_history.csv"
    eval_path = run_dir / "druid_spsa_evaluations.csv"
    scene_eval_path = run_dir / "druid_spsa_scene_evaluations.csv"
    best_scene_path = run_dir / "druid_spsa_best_per_scene.csv"
    baseline_scene_path = run_dir / "druid_spsa_baseline_per_scene.csv"
    budget_path = run_dir / "druid_spsa_scene_budgets.csv"
    best_path = run_dir / "druid_spsa_best.json"
    latest_best_path = output_dir / "druid_spsa_best.json"

    history_df.to_csv(history_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    scene_eval_df.to_csv(scene_eval_path, index=False)

    baseline_key = key_from_params(baseline_params, specs)
    best_key = key_from_params(best_params, specs)
    pd.DataFrame(cache[baseline_key]["scene_rows"]).to_csv(baseline_scene_path, index=False)
    pd.DataFrame(cache[best_key]["scene_rows"]).to_csv(best_scene_path, index=False)
    pd.DataFrame(baseline_scene_rows).to_csv(budget_path, index=False)

    best_report = {
        "seed": int(args.seed),
        "run_tag": run_tag,
        "iterations": int(args.iterations),
        "restarts": int(args.restarts),
        "scene_count": len(contexts),
        "scenes": [
            {
                "scene_id": ctx.scene_id,
                "scene_key": ctx.scene.key,
                "scene_path": str(ctx.scene.path),
                "geojson_path": str(ctx.gt_geojson_path),
                "metadata_csv": str(ctx.metadata_csv),
            }
            for ctx in contexts
        ],
        "aggregate_weights": {
            "median": weight_median,
            "worst": weight_worst,
            "mean": weight_mean,
        },
        "spsa": {
            "a": a,
            "c": c,
            "alpha": alpha,
            "gamma": gamma,
        },
        "search_params": [asdict(spec) for spec in specs],
        "overlap_target": float(args.overlap_target),
        "baseline": baseline_eval,
        "best": best_eval,
        "base_druid_config": asdict(base_config),
        "scene_budget_path": str(budget_path),
        "baseline_scene_path": str(baseline_scene_path),
        "best_scene_path": str(best_scene_path),
        "history_path": str(history_path),
        "evaluations_path": str(eval_path),
        "scene_evaluations_path": str(scene_eval_path),
    }
    best_path.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_best_path.parent.mkdir(parents=True, exist_ok=True)
    latest_best_path.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(best_report, ensure_ascii=False, indent=2))
    print(str(best_path))
    print(str(history_path))
    print(str(eval_path))
    print(str(scene_eval_path))


if __name__ == "__main__":
    main()
