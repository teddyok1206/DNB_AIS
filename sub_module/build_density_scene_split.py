from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STEP3 = ROOT / "[3]_DNB_AIS - (STEP 3)"


@dataclass(frozen=True)
class SceneRecord:
    scene_key: str
    day_key: str
    tif_name: str
    tif_path: Path
    geojson_path: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a day-grouped DNB density smoke split snapshot.")
    parser.add_argument("--completed-txt", type=Path, default=STEP3 / "bboxes_JPSS-2" / "Bboxes_completed.txt")
    parser.add_argument("--geojson-dir", type=Path, default=STEP3 / "bboxes_JPSS-2")
    parser.add_argument("--tif-root", type=Path, default=Path("/Volumes/SAMSUNG/JPSS-2_VIIRS"))
    parser.add_argument("--output-dir", type=Path, default=STEP3 / "outputs" / "density_smoke_split_10_3_2")
    parser.add_argument("--train-days", type=int, default=10)
    parser.add_argument("--val-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--require-existing-tif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-geojson", action=argparse.BooleanOptionalAction, default=True)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_completed_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if not value.endswith(".tif"):
            value = f"{value}.tif"
        names.append(value)
    return names


def index_tifs(tif_root: Path) -> dict[str, Path]:
    tif_paths: dict[str, Path] = {}
    for suffix in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for path in tif_root.rglob(suffix):
            tif_paths.setdefault(path.name, path.resolve())
    return tif_paths


def build_records(
    *,
    completed_names: list[str],
    tif_paths: dict[str, Path],
    geojson_dir: Path,
    require_existing_tif: bool,
    require_geojson: bool,
) -> tuple[list[SceneRecord], dict[str, Any]]:
    records: list[SceneRecord] = []
    missing_tif: list[str] = []
    missing_geojson: list[str] = []
    duplicate_completed = [name for name, count in Counter(completed_names).items() if count > 1]

    for tif_name in completed_names:
        scene_key = Path(tif_name).stem
        day_key = scene_key.split("_")[0]
        tif_path = tif_paths.get(tif_name)
        geojson_path = geojson_dir / f"{scene_key}.geojson"
        if tif_path is None:
            missing_tif.append(tif_name)
            if require_existing_tif:
                continue
            tif_path = Path("")
        if not geojson_path.exists():
            missing_geojson.append(f"{scene_key}.geojson")
            if require_geojson:
                continue
        records.append(
            SceneRecord(
                scene_key=scene_key,
                day_key=day_key,
                tif_name=tif_name,
                tif_path=tif_path,
                geojson_path=geojson_path.resolve(),
            )
        )

    diagnostics = {
        "completed_count": int(len(completed_names)),
        "unique_completed_count": int(len(set(completed_names))),
        "duplicate_completed": duplicate_completed,
        "tif_index_count": int(len(tif_paths)),
        "usable_scene_count": int(len(records)),
        "usable_day_count": int(len({record.day_key for record in records})),
        "missing_tif_count": int(len(missing_tif)),
        "missing_tif_examples": missing_tif[:20],
        "missing_geojson_count": int(len(missing_geojson)),
        "missing_geojson_examples": missing_geojson[:20],
    }
    return records, diagnostics


def assign_day_splits(
    days: list[str],
    *,
    train_days: int,
    val_days: int,
    test_days: int,
    seed: int,
) -> dict[str, str]:
    requested = int(train_days) + int(val_days) + int(test_days)
    if len(days) < requested:
        raise ValueError(f"Not enough usable days: requested {requested}, available {len(days)}")

    shuffled = sorted(days)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)

    train_set = set(shuffled[: int(train_days)])
    val_set = set(shuffled[int(train_days) : int(train_days) + int(val_days)])
    test_set = set(shuffled[int(train_days) + int(val_days) : requested])
    return {
        day: ("train" if day in train_set else "val" if day in val_set else "test" if day in test_set else "unused")
        for day in days
    }


def write_scene_split(path: Path, records: list[SceneRecord], day_split: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_order = {"train": 0, "val": 1, "test": 2, "unused": 3}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "day_key", "scene_key", "tif_name", "tif_path", "geojson_path"],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: (split_order[day_split[item.day_key]], item.day_key, item.scene_key)):
            split = day_split[record.day_key]
            if split == "unused":
                continue
            writer.writerow(
                {
                    "split": split,
                    "day_key": record.day_key,
                    "scene_key": record.scene_key,
                    "tif_name": record.tif_name,
                    "tif_path": str(record.tif_path),
                    "geojson_path": str(record.geojson_path),
                }
            )


def write_day_split(path: Path, records: list[SceneRecord], day_split: dict[str, str]) -> None:
    scenes_by_day: dict[str, list[str]] = defaultdict(list)
    for record in records:
        scenes_by_day[record.day_key].append(record.scene_key)
    split_order = {"train": 0, "val": 1, "test": 2, "unused": 3}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "day_key", "scene_count", "scene_keys"])
        writer.writeheader()
        for day in sorted(scenes_by_day, key=lambda item: (split_order[day_split[item]], item)):
            split = day_split[day]
            if split == "unused":
                continue
            scene_keys = sorted(scenes_by_day[day])
            writer.writerow(
                {
                    "split": split,
                    "day_key": day,
                    "scene_count": int(len(scene_keys)),
                    "scene_keys": ";".join(scene_keys),
                }
            )


def split_counts(records: list[SceneRecord], day_split: dict[str, str]) -> dict[str, Any]:
    day_counts = Counter(day_split.values())
    scene_counts = Counter(day_split[record.day_key] for record in records)
    return {
        "days": {key: int(day_counts.get(key, 0)) for key in ("train", "val", "test", "unused")},
        "scenes": {key: int(scene_counts.get(key, 0)) for key in ("train", "val", "test", "unused")},
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    completed_txt = args.completed_txt.expanduser().resolve()
    geojson_dir = args.geojson_dir.expanduser().resolve()
    tif_root = args.tif_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_names = read_completed_names(completed_txt)
    tif_paths = index_tifs(tif_root)
    records, diagnostics = build_records(
        completed_names=completed_names,
        tif_paths=tif_paths,
        geojson_dir=geojson_dir,
        require_existing_tif=bool(args.require_existing_tif),
        require_geojson=bool(args.require_geojson),
    )
    days = sorted({record.day_key for record in records})
    day_split = assign_day_splits(
        days,
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
    )

    scene_split_path = output_dir / "scene_split.csv"
    day_split_path = output_dir / "day_split.csv"
    snapshot_path = output_dir / "Bboxes_completed.snapshot.txt"
    summary_path = output_dir / "split_summary.json"

    write_scene_split(scene_split_path, records, day_split)
    write_day_split(day_split_path, records, day_split)
    snapshot_path.write_text("\n".join(completed_names) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "kind": "smoke_split_snapshot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "requested_days": {
            "train": int(args.train_days),
            "val": int(args.val_days),
            "test": int(args.test_days),
        },
        "source": {
            "completed_txt": str(completed_txt),
            "completed_txt_sha256": file_sha256(completed_txt),
            "geojson_dir": str(geojson_dir),
            "tif_root": str(tif_root),
            "require_existing_tif": bool(args.require_existing_tif),
            "require_geojson": bool(args.require_geojson),
        },
        "diagnostics": diagnostics,
        "split_counts": split_counts(records, day_split),
        "outputs": {
            "scene_split_csv": str(scene_split_path),
            "day_split_csv": str(day_split_path),
            "completed_snapshot_txt": str(snapshot_path),
        },
        "note": "This is a smoke split snapshot, not the final 250/60/55 split.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary["split_counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
