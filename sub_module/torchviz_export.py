from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.data import Data
from torchviz import make_dot

from sub_module.dnb_gat_pipeline import load_model_checkpoint


def build_dummy_graph(in_channels: int, num_nodes: int = 12) -> Data:
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32, requires_grad=True)
    src: list[int] = []
    dst: list[int] = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src.extend([i, j])
        dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data._x_for_dot = x
    return data


def discover_checkpoints(root: Path) -> list[Path]:
    return sorted(root.glob("outputs/DNB_GAT_v1/**/run_*/**/*_gat_checkpoint.pt"))


def export_torchviz_for_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_nodes: int,
    map_location: str = "cpu",
) -> tuple[Path, Path]:
    model, bundle = load_model_checkpoint(checkpoint_path, map_location=map_location)
    model.train()
    in_channels = int(bundle.get("architecture", {}).get("in_channels", 3))
    graph = build_dummy_graph(in_channels=in_channels, num_nodes=int(num_nodes))

    output = model(graph)
    scalar = output.sum()
    dot = make_dot(scalar, params={**dict(model.named_parameters()), "x": graph._x_for_dot})

    stem = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_torchviz"
    dot_path = output_dir / f"{stem}.dot"
    png_path = output_dir / f"{stem}.png"
    dot.save(str(dot_path))
    dot.render(filename=stem, directory=str(output_dir), format="png", cleanup=True)
    return png_path, dot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export torchviz autograd graph PNG/DOT for one or more GAT .pt checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path. Can be passed multiple times. If omitted, auto-discovers *_gat_checkpoint.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PNG/DOT. Default: <repo>/outputs/DNB_GAT_v1/model_viz",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=12,
        help="Dummy graph node count used to build autograd graph. Default: 12",
    )
    parser.add_argument(
        "--map-location",
        default="cpu",
        help="torch.load map_location. Default: cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.output_dir is None:
        output_dir = root / "outputs" / "DNB_GAT_v1" / "model_viz"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        checkpoints = [Path(path).resolve() for path in args.checkpoint]
    else:
        checkpoints = discover_checkpoints(root)

    if not checkpoints:
        raise RuntimeError("No checkpoint files found. Provide --checkpoint or create *_gat_checkpoint.pt first.")

    for checkpoint in checkpoints:
        if not checkpoint.exists():
            print(f"SKIP (missing): {checkpoint}")
            continue
        png_path, dot_path = export_torchviz_for_checkpoint(
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            num_nodes=int(args.num_nodes),
            map_location=str(args.map_location),
        )
        print(str(png_path))
        print(str(dot_path))


if __name__ == "__main__":
    main()
