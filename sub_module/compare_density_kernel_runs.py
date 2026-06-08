from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "occupancy_f1",
    "occupancy_precision",
    "occupancy_recall",
    "occupancy_brier",
    "spatial_overlap_mean_positive",
    "target_explained",
    "pred_matched",
    "pred_target_ratio",
    "patch_count",
    "positive_patch_count",
    "zero_target_patch_count",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PH-assisted density/OX run summaries.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in label=/path/to/run_dir form. The first run is used as the delta baseline.",
    )
    parser.add_argument("--output-md", type=Path, default=None, help="Optional markdown table output path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be label=path: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label is empty: {spec}")
    run_dir = Path(raw_path).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")
    return label, run_dir


def load_config(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    snapshot = summary.get("outputs", {}).get("config_snapshot")
    candidates = []
    if snapshot:
        candidates.append(Path(str(snapshot)))
    candidates.append(run_dir / "config_snapshot.json")
    config_path = summary.get("config_path")
    if config_path:
        candidates.append(Path(str(config_path)))
    for path in candidates:
        try:
            if path.exists():
                return read_json(path)
        except OSError:
            continue
    return {}


def nested_get(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json")
    config = load_config(summary, run_dir)
    target = config.get("target", {}) if isinstance(config.get("target"), dict) else {}
    loss = nested_get(config, ["training", "loss"], {})
    if not isinstance(loss, dict):
        loss = {}
    test = summary.get("test", {})
    if not isinstance(test, dict):
        raise ValueError(f"Run summary has no test metrics: {run_dir}")
    final_val = None
    train_history = summary.get("train_history", [])
    if isinstance(train_history, list) and train_history:
        last = train_history[-1]
        if isinstance(last, dict) and isinstance(last.get("val"), dict):
            final_val = last["val"]
    row = {
        "label": label,
        "run_dir": str(run_dir),
        "config_path": summary.get("config_path"),
        "epochs": summary.get("epochs"),
        "sigma_pixels": target.get("sigma_pixels"),
        "radius_pixels": target.get("radius_pixels"),
        "occupancy_weight": loss.get("occupancy_weight"),
        "spatial_weight": loss.get("spatial_weight"),
        "final_val_occupancy_f1": None if final_val is None else final_val.get("occupancy_f1"),
    }
    for key in METRIC_KEYS:
        row[f"test_{key}"] = test.get(key)
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_delta(value: Any, baseline: Any) -> Any:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return float(value) - float(baseline)
    return None


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "label",
        "sigma_pixels",
        "radius_pixels",
        "occupancy_weight",
        "spatial_weight",
        "epochs",
        "test_occupancy_f1",
        "delta_f1",
        "test_occupancy_precision",
        "test_occupancy_recall",
        "test_occupancy_brier",
        "delta_brier",
        "test_spatial_overlap_mean_positive",
        "delta_spatial_overlap",
        "test_pred_target_ratio",
        "test_patch_count",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = [summarize_run(*parse_run_spec(spec)) for spec in args.run]
    if rows:
        baseline = rows[0]
        for row in rows:
            row["delta_f1"] = metric_delta(row.get("test_occupancy_f1"), baseline.get("test_occupancy_f1"))
            row["delta_brier"] = metric_delta(row.get("test_occupancy_brier"), baseline.get("test_occupancy_brier"))
            row["delta_spatial_overlap"] = metric_delta(
                row.get("test_spatial_overlap_mean_positive"),
                baseline.get("test_spatial_overlap_mean_positive"),
            )
    output = {"schema_version": 1, "kind": "density_kernel_run_comparison", "runs": rows}
    table = markdown_table(rows)
    print(table)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table, encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
