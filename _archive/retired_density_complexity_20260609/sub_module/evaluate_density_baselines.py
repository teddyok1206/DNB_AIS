from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .render_density_full_scene_predictions import build_all_scene_patches
from .dnb_project_paths import ROOT, STEP3
from .run_density_split_smoke_train import SceneSplitRecord, read_json, read_scene_split
from .dnb_pipeline_core import GroundTruthResolver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate train-calibrated heuristic density baselines on a scene split.")
    parser.add_argument("--scene-split-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dnb_density_unet_occupancy_spatial.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dnb_density" / "baseline_evaluations")
    parser.add_argument("--calibration-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit-calibration-scenes", type=int, default=0)
    parser.add_argument("--limit-eval-scenes", type=int, default=0)
    return parser


def _records_for_split(records: list[SceneSplitRecord], split: str, limit: int) -> list[SceneSplitRecord]:
    selected = [record for record in records if record.split == split]
    if int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def _score_uniform(patch) -> np.ndarray:
    return np.asarray(patch.valid_mask, dtype=np.float32)


def _score_brightness(patch) -> np.ndarray:
    return np.clip(np.asarray(patch.image, dtype=np.float32), 0.0, None) * np.asarray(patch.valid_mask, dtype=np.float32)


def _score_ph_attention(patch) -> np.ndarray:
    return np.asarray(patch.soft_attention, dtype=np.float32) * np.asarray(patch.valid_mask, dtype=np.float32)


def _score_brightness_ph_attention(patch) -> np.ndarray:
    return _score_brightness(patch) * np.asarray(patch.soft_attention, dtype=np.float32)


def _score_lifetime_ph_attention(patch) -> np.ndarray:
    return _score_ph_attention(patch) * np.float32(max(float(patch.lifetime), 0.0))


BASELINES: dict[str, Callable[[Any], np.ndarray]] = {
    "valid_uniform": _score_uniform,
    "brightness": _score_brightness,
    "ph_attention": _score_ph_attention,
    "brightness_ph_attention": _score_brightness_ph_attention,
    "lifetime_ph_attention": _score_lifetime_ph_attention,
}


def _safe_ratio(num: float, den: float, eps: float = 1.0e-8) -> float:
    return float(num / max(den, eps))


def _normalized(arr: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    total = float(np.asarray(arr, dtype=np.float32).sum())
    if total <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32) / total


def calibrate_baselines(
    records: list[SceneSplitRecord],
    *,
    config: dict[str, Any],
    resolver: GroundTruthResolver,
) -> dict[str, dict[str, float]]:
    totals = {name: {"score_sum": 0.0, "target_sum": 0.0} for name in BASELINES}
    for record in records:
        print(f"[calibrate] {record.split} {record.scene_key}")
        _scene, _valid_mask, _gt, patches, _metrics = build_all_scene_patches(record, config=config, resolver=resolver)
        for patch in patches:
            target_sum = float(np.asarray(patch.target_density, dtype=np.float32).sum())
            for name, score_fn in BASELINES.items():
                totals[name]["score_sum"] += float(score_fn(patch).sum())
                totals[name]["target_sum"] += target_sum
    calibrated: dict[str, dict[str, float]] = {}
    for name, item in totals.items():
        score_sum = float(item["score_sum"])
        target_sum = float(item["target_sum"])
        calibrated[name] = {
            "scale": _safe_ratio(target_sum, score_sum),
            "score_sum": score_sum,
            "target_sum": target_sum,
        }
    return calibrated


def evaluate_scene_baselines(
    record: SceneSplitRecord,
    *,
    config: dict[str, Any],
    resolver: GroundTruthResolver,
    calibration: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    scene, _valid_mask, _gt, patches, build_metrics = build_all_scene_patches(record, config=config, resolver=resolver)
    rows: list[dict[str, Any]] = []
    for name, score_fn in BASELINES.items():
        pred_canvas = np.zeros(scene.shape, dtype=np.float32)
        target_canvas = np.zeros(scene.shape, dtype=np.float32)
        raw_canvas = np.zeros(scene.shape, dtype=np.float32)
        scale = float(calibration[name]["scale"])
        for patch in patches:
            h, w = patch.shape
            r0, r1, c0, c1 = [int(v) for v in patch.crop_rc]
            valid = np.asarray(patch.valid_mask, dtype=np.float32)[:h, :w] > 0
            score = np.asarray(score_fn(patch), dtype=np.float32)[:h, :w]
            pred_local = score * np.float32(scale)
            target_local = np.asarray(patch.target_density, dtype=np.float32)[:h, :w]
            raw_local = np.asarray(patch.raw_count, dtype=np.float32)[:h, :w]
            pred_view = pred_canvas[r0 : r1 + 1, c0 : c1 + 1]
            target_view = target_canvas[r0 : r1 + 1, c0 : c1 + 1]
            raw_view = raw_canvas[r0 : r1 + 1, c0 : c1 + 1]
            pred_view[valid] += pred_local[valid]
            target_view[valid] += target_local[valid]
            raw_view[valid] += raw_local[valid]

        pred_sum = float(pred_canvas.sum())
        target_sum = float(target_canvas.sum())
        overlap_sum = float(np.minimum(pred_canvas, target_canvas).sum())
        spatial_overlap = float(np.minimum(_normalized(pred_canvas), _normalized(target_canvas)).sum())
        rows.append(
            {
                "split": record.split,
                "scene_key": record.scene_key,
                "baseline": name,
                "scale": scale,
                "pred_sum": pred_sum,
                "target_sum": target_sum,
                "raw_sum": float(raw_canvas.sum()),
                "abs_count_error": float(abs(pred_sum - target_sum)),
                "pred_target_ratio": _safe_ratio(pred_sum, target_sum),
                "target_explained": _safe_ratio(overlap_sum, target_sum),
                "pred_matched": _safe_ratio(overlap_sum, pred_sum),
                "spatial_overlap": spatial_overlap,
                "patch_count": int(build_metrics.get("patch_count_total", len(patches))),
                "ph_anchor_count": int(build_metrics.get("partition_summary", {}).get("ph_anchor_count", 0)),
                "fallback_grid_count": int(build_metrics.get("partition_summary", {}).get("fallback_grid_count", 0)),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in BASELINES:
        items = [row for row in rows if row["baseline"] == name]
        if not items:
            continue
        pred_sum = float(sum(float(row["pred_sum"]) for row in items))
        target_sum = float(sum(float(row["target_sum"]) for row in items))
        out.append(
            {
                "baseline": name,
                "scene_count": int(len(items)),
                "pred_sum": pred_sum,
                "target_sum": target_sum,
                "pred_target_ratio": _safe_ratio(pred_sum, target_sum),
                "scene_count_mae": float(np.mean([float(row["abs_count_error"]) for row in items])),
                "target_explained_mean": float(np.mean([float(row["target_explained"]) for row in items])),
                "pred_matched_mean": float(np.mean([float(row["pred_matched"]) for row in items])),
                "spatial_overlap_mean": float(np.mean([float(row["spatial_overlap"]) for row in items])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = read_json(args.config)
    records = read_scene_split(args.scene_split_csv)
    calibration_records = _records_for_split(records, str(args.calibration_split), int(args.limit_calibration_scenes))
    eval_records = _records_for_split(records, str(args.eval_split), int(args.limit_eval_scenes))
    if not calibration_records:
        raise RuntimeError(f"No calibration records for split={args.calibration_split}")
    if not eval_records:
        raise RuntimeError(f"No eval records for split={args.eval_split}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = GroundTruthResolver(STEP3 / "bboxes_JPSS-2")
    calibration = calibrate_baselines(calibration_records, config=config, resolver=resolver)
    scene_rows: list[dict[str, Any]] = []
    for record in eval_records:
        print(f"[evaluate] {record.split} {record.scene_key}")
        scene_rows.extend(evaluate_scene_baselines(record, config=config, resolver=resolver, calibration=calibration))
    summary_rows = summarize_rows(scene_rows)

    scene_csv = output_dir / f"baseline_scene_metrics_{args.eval_split}.csv"
    summary_csv = output_dir / f"baseline_summary_{args.eval_split}.csv"
    calibration_json = output_dir / "baseline_calibration.json"
    write_csv(scene_csv, scene_rows)
    write_csv(summary_csv, summary_rows)
    calibration_json.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scene_csv": str(scene_csv),
                "summary_csv": str(summary_csv),
                "calibration_json": str(calibration_json),
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
