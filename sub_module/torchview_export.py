from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torchview import draw_graph

from sub_module.dnb_gat_pipeline import load_model_checkpoint


def build_dummy_graph(in_channels: int, num_nodes: int = 12) -> Data:
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    src: list[int] = []
    dst: list[int] = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src.extend([i, j])
        dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class _GATWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        data = Data(x=x, edge_index=edge_index)
        return self.model(data)


def discover_checkpoints(root: Path) -> list[Path]:
    return sorted(root.glob("outputs/DNB_GAT_v1/**/run_*/**/*_gat_checkpoint.pt"))


def export_torchview_for_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_nodes: int,
    depth: int,
    expand_nested: bool,
    hide_inner_tensors: bool,
    map_location: str = "cpu",
) -> tuple[Path, Path]:
    model, bundle = load_model_checkpoint(checkpoint_path, map_location=map_location)
    model.eval()
    in_channels = int(bundle.get("architecture", {}).get("in_channels", 3))
    data = build_dummy_graph(in_channels=in_channels, num_nodes=int(num_nodes))
    wrapper = _GATWrapper(model)
    wrapper.eval()

    graph = draw_graph(
        wrapper,
        input_data=(data.x, data.edge_index),
        graph_name=f"{checkpoint_path.stem}_torchview",
        depth=int(depth),
        expand_nested=bool(expand_nested),
        show_shapes=True,
        hide_inner_tensors=bool(hide_inner_tensors),
        mode="eval",
        strict=True,
    )

    stem = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_torchview"
    dot_path = output_dir / f"{stem}.dot"
    png_path = output_dir / f"{stem}.png"
    graph.visual_graph.save(str(dot_path))
    graph.visual_graph.render(filename=stem, directory=str(output_dir), format="png", cleanup=True)
    return png_path, dot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export module-level graph PNG/DOT for one or more GAT checkpoints via torchview."
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
    parser.add_argument("--num-nodes", type=int, default=12, help="Dummy graph node count. Default: 12")
    parser.add_argument("--depth", type=int, default=6, help="torchview depth. Default: 6")
    parser.add_argument("--expand-nested", action="store_true", help="Expand nested modules in graph output")
    parser.add_argument(
        "--show-inner-tensors",
        action="store_true",
        help="Show inner tensors (by default hidden for readability)",
    )
    parser.add_argument("--map-location", default="cpu", help="torch.load map_location. Default: cpu")
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
        png_path, dot_path = export_torchview_for_checkpoint(
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            num_nodes=int(args.num_nodes),
            depth=int(args.depth),
            expand_nested=bool(args.expand_nested),
            hide_inner_tensors=not bool(args.show_inner_tensors),
            map_location=str(args.map_location),
        )
        print(str(png_path))
        print(str(dot_path))


if __name__ == "__main__":
    main()
