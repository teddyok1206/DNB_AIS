from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


STAGE_ORDER = [
    "preprocess",
    "scene_load",
    "ph",
    "graph",
    "train",
    "infer",
    "assemble",
]

PREPROCESS_MANIFEST_COLUMNS = [
    "run_tag",
    "scene_mode",
    "scene_key",
    "raw_scene_tif",
    "effective_scene_tif",
    "mask_applied",
    "segment_policy",
    "segment_count",
    "segment_top5_areas",
    "downsample_applied",
    "downsample_factor",
    "downsample_src_shape",
    "downsample_dst_shape",
    "timestamp",
]

GRAPH_MANIFEST_COLUMNS = [
    "run_tag",
    "scene_mode",
    "scene_key",
    "graph_cache_key",
    "graph_cache_path",
    "graph_count",
    "total_nodes",
    "total_edges",
    "node_count_min",
    "node_count_median",
    "node_count_max",
    "used_cache",
    "timestamp",
]


def _to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_serializable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_cache_key(
    scene_key: str,
    preprocess_config: Any,
    ph_config: Any,
    graph_config: Any,
    *,
    version: str = "v1",
) -> str:
    payload = {
        "version": str(version),
        "scene_key": str(scene_key),
        "preprocess": _to_serializable(preprocess_config),
        "ph": _to_serializable(ph_config),
        "graph": _to_serializable(graph_config),
    }
    digest = hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest_row(path: str | Path, columns: list[str], row: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {column: _to_serializable(row.get(column)) for column in columns}
    frame = pd.DataFrame([payload], columns=columns)
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, frame], ignore_index=True)
    else:
        merged = frame
    merged.to_csv(path, index=False)
    return path


@dataclass
class RunContext:
    root: Path
    step3: Path
    output_root: Path
    seed: int
    run_tag: str
    scene_mode: str
    run_mode: str
    device: str
    resume_stage: str = "none"
    profile: str = "default"
    stage_order: list[str] = field(default_factory=lambda: list(STAGE_ORDER))
    run_dir: Path | None = None
    stage_dir: Path | None = None

    def bind_run_dir(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = self.run_dir / "_stage_state"
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def _stage_index(self, stage_name: str) -> int:
        if stage_name not in self.stage_order:
            raise ValueError(f"Unknown stage name: {stage_name}")
        return self.stage_order.index(stage_name)

    def stage_path(self, stage_name: str) -> Path:
        if self.stage_dir is None:
            raise RuntimeError("RunContext.bind_run_dir() must be called before stage operations.")
        return self.stage_dir / f"{stage_name}.json"

    def mark_stage(self, stage_name: str, payload: dict[str, Any] | None = None) -> Path:
        payload = dict(payload or {})
        payload["stage"] = stage_name
        payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        return save_json(self.stage_path(stage_name), payload)

    def load_stage(self, stage_name: str) -> dict[str, Any] | None:
        path = self.stage_path(stage_name)
        if not path.exists():
            return None
        return load_json(path)

    def should_skip_stage(self, stage_name: str) -> bool:
        if self.resume_stage in {"", "none", None}:
            return False
        current_idx = self._stage_index(stage_name)
        resume_idx = self._stage_index(str(self.resume_stage))
        if current_idx >= resume_idx:
            return False
        return self.stage_path(stage_name).exists()


def scene_quality_report(scene_image: np.ndarray, gt_count_map: np.ndarray) -> dict[str, Any]:
    if scene_image.shape != gt_count_map.shape:
        raise ValueError(f"Shape mismatch: scene={scene_image.shape}, gt={gt_count_map.shape}")

    nan_count = int(np.isnan(scene_image).sum())
    inf_count = int(np.isinf(scene_image).sum())
    nonzero = int((scene_image != 0).sum())
    total = int(scene_image.size)
    gt_nonzero = int((gt_count_map > 0).sum())
    gt_total = float(gt_count_map.sum())
    warnings: list[str] = []
    if nan_count > 0:
        warnings.append(f"scene contains NaN values: {nan_count}")
    if inf_count > 0:
        warnings.append(f"scene contains Inf values: {inf_count}")
    if nonzero == 0:
        warnings.append("scene contains only zeros")
    if gt_nonzero == 0:
        warnings.append("gt_count_map has no positive pixels")

    return {
        "scene_shape": [int(scene_image.shape[0]), int(scene_image.shape[1])],
        "scene_nonzero_ratio": float(nonzero / max(total, 1)),
        "scene_nan_count": nan_count,
        "scene_inf_count": inf_count,
        "gt_nonzero_pixels": gt_nonzero,
        "gt_total_count": gt_total,
        "warnings": warnings,
    }


def graph_quality_report(graphs: Iterable[Any]) -> dict[str, Any]:
    graph_list = list(graphs)
    node_counts = np.array([int(graph.num_nodes) for graph in graph_list], dtype=np.int64) if graph_list else np.array([], dtype=np.int64)
    edge_counts = np.array([int(graph.edge_index.shape[1]) for graph in graph_list], dtype=np.int64) if graph_list else np.array([], dtype=np.int64)

    warnings: list[str] = []
    if len(graph_list) == 0:
        warnings.append("graph list is empty")
    if node_counts.size and int((node_counts < 8).sum()) > 0:
        warnings.append(f"graphs with node_count<8: {int((node_counts < 8).sum())}")

    return {
        "graph_count": int(len(graph_list)),
        "total_nodes": int(node_counts.sum()) if node_counts.size else 0,
        "total_edges": int(edge_counts.sum()) if edge_counts.size else 0,
        "node_count_min": int(node_counts.min()) if node_counts.size else 0,
        "node_count_median": int(np.median(node_counts)) if node_counts.size else 0,
        "node_count_max": int(node_counts.max()) if node_counts.size else 0,
        "warnings": warnings,
    }


def evaluate_predictions_summary(
    graphs: list[Any],
    predictions: dict[int, np.ndarray],
    gt_count_map: np.ndarray,
    heatmap: np.ndarray,
) -> dict[str, Any]:
    all_y = np.concatenate([graph.y.detach().cpu().numpy() for graph in graphs]) if graphs else np.array([], dtype=np.float32)
    all_pred = np.concatenate(
        [predictions[int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])] for graph in graphs]
    ) if graphs else np.array([], dtype=np.float32)
    positive_mask = all_y > 0
    positive_count = int(positive_mask.sum())

    graph_topk_hits = 0
    if positive_count > 0 and all_pred.size >= positive_count:
        top_idx = np.argpartition(all_pred, -positive_count)[-positive_count:]
        graph_topk_hits = int((all_y[top_idx] > 0).sum())

    scene_positive_mask = gt_count_map > 0
    scene_positive_count = int(scene_positive_mask.sum())
    scene_topk_hits = 0
    if scene_positive_count > 0 and np.any(heatmap > 0):
        flat_heatmap = heatmap.reshape(-1)
        top_scene_idx = np.argpartition(flat_heatmap, -scene_positive_count)[-scene_positive_count:]
        scene_topk_hits = int(scene_positive_mask.reshape(-1)[top_scene_idx].sum())

    return {
        "graph_positive_count": positive_count,
        "graph_topk_hits": graph_topk_hits,
        "graph_topk_hit_rate": float(graph_topk_hits / max(positive_count, 1)),
        "pred_sum_graph_nodes": float(all_pred.sum()) if all_pred.size else 0.0,
        "gt_sum_graph_nodes": float(all_y.sum()) if all_y.size else 0.0,
        "pred_sum_ratio": float(all_pred.sum() / max(float(all_y.sum()), 1.0e-6)) if all_pred.size else 0.0,
        "scene_positive_count": scene_positive_count,
        "scene_topk_hits": scene_topk_hits,
        "scene_topk_hit_rate": float(scene_topk_hits / max(scene_positive_count, 1)),
        "heatmap_nonzero_pixels": int((heatmap > 0).sum()),
        "heatmap_min": float(np.min(heatmap)),
        "heatmap_max": float(np.max(heatmap)),
        "heatmap_mean": float(np.mean(heatmap)),
        "heatmap_sum": float(np.sum(heatmap)),
    }


def build_scene_feature_frame(metadata_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "tif_name" not in df.columns:
        raise ValueError("metadata CSV must contain 'tif_name'")

    scenes = df.loc[df["tif_name"].notna(), ["tif_name"]].copy()
    scenes["scene_key"] = scenes["tif_name"].astype(str).str.replace(".tif", "", regex=False)
    scenes = scenes.drop_duplicates("scene_key").reset_index(drop=True)

    if "KR_Sea_overlap" in df.columns:
        overlap = (
            df.loc[df["tif_name"].notna(), ["tif_name", "KR_Sea_overlap"]]
            .copy()
            .dropna(subset=["KR_Sea_overlap"])
        )
        if not overlap.empty:
            overlap["scene_key"] = overlap["tif_name"].astype(str).str.replace(".tif", "", regex=False)
            overlap_agg = overlap.groupby("scene_key", as_index=False)["KR_Sea_overlap"].mean()
            scenes = scenes.merge(overlap_agg, on="scene_key", how="left")
        else:
            scenes["KR_Sea_overlap"] = np.nan
    else:
        scenes["KR_Sea_overlap"] = np.nan

    scenes["doy"] = scenes["scene_key"].str.extract(r"A(\d{4})(\d{3})_", expand=True)[1].astype(float)
    scenes["hhmm"] = scenes["scene_key"].str.extract(r"A\d{7}_(\d{4})_", expand=True)[0].astype(float)

    overlap_series = scenes["KR_Sea_overlap"].fillna(scenes["KR_Sea_overlap"].median() if scenes["KR_Sea_overlap"].notna().any() else 0.0)
    if len(scenes) >= 6 and overlap_series.nunique() > 1:
        scenes["coastal_bin"] = pd.qcut(overlap_series, q=3, labels=["low", "mid", "high"], duplicates="drop")
    else:
        scenes["coastal_bin"] = "single"

    if len(scenes) >= 6 and scenes["hhmm"].nunique() > 1:
        scenes["time_bin"] = pd.qcut(scenes["hhmm"], q=3, labels=["early", "mid", "late"], duplicates="drop")
    else:
        scenes["time_bin"] = "single"

    scenes["strata"] = scenes["coastal_bin"].astype(str) + "__" + scenes["time_bin"].astype(str)
    return scenes[["scene_key", "tif_name", "KR_Sea_overlap", "doy", "hhmm", "coastal_bin", "time_bin", "strata"]]


def stratified_split_by_scene(
    scene_df: pd.DataFrame,
    *,
    seed: int,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    scene_col: str = "scene_key",
    strata_col: str = "strata",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not math.isclose(sum(split_ratios), 1.0, rel_tol=1.0e-6):
        raise ValueError("split_ratios must sum to 1.0")

    frame = scene_df.copy()
    if strata_col not in frame.columns:
        frame[strata_col] = "default"
    frame = frame.drop_duplicates(scene_col).reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    assignments: list[tuple[str, str]] = []
    for _, group in frame.groupby(strata_col, dropna=False):
        keys = group[scene_col].astype(str).tolist()
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * split_ratios[0]))
        n_val = int(round(n * split_ratios[1]))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        # Keep each split non-empty when stratum has enough samples.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_train and n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            while n_train + n_val + n_test < n:
                n_train += 1

        for key in keys[:n_train]:
            assignments.append((key, "train"))
        for key in keys[n_train : n_train + n_val]:
            assignments.append((key, "val"))
        for key in keys[n_train + n_val :]:
            assignments.append((key, "test"))

    split_df = pd.DataFrame(assignments, columns=[scene_col, "split"])
    split_df = frame.merge(split_df, on=scene_col, how="left")
    split_df["split"] = split_df["split"].fillna("train")
    split_df = split_df.sort_values(scene_col).reset_index(drop=True)

    meta = {
        "seed": int(seed),
        "split_ratios": list(split_ratios),
        "scene_count": int(len(split_df)),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "strata_counts": split_df[strata_col].value_counts(dropna=False).to_dict(),
    }
    return split_df, meta
