from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DensityModelConfig:
    name: str = "MaskedDensityUNet"
    in_channels: int = 6
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    activation: str = "softplus"
    dropout: float = 0.0


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _apply_output_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    activation = str(activation).lower()
    if activation == "softplus":
        return F.softplus(x)
    if activation == "relu":
        return F.relu(x)
    if activation in {"exp", "log_count_exp"}:
        return torch.exp(torch.clamp(x, min=-20.0, max=20.0))
    if activation in {"expm1", "log_count_expm1"}:
        return torch.expm1(torch.clamp(x, min=0.0, max=20.0))
    if activation in {"identity", "linear", "none", ""}:
        return x
    raise ValueError(f"Unsupported output activation: {activation}")


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0, dilation: int = 1) -> None:
        super().__init__()
        padding = int(dilation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.skip = nn.Identity() if int(in_channels) == int(out_channels) else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MaskedDensityUNet(nn.Module):
    """Small U-Net/ResUNet for PH-hierarchical DNB density regression."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        activation: str = "softplus",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depth = max(int(depth), 2)
        base_channels = int(base_channels)
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.config = DensityModelConfig(
            name="MaskedDensityUNet",
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_channels=base_channels,
            depth=depth,
            activation=str(activation),
            dropout=float(dropout),
        )
        self.activation = str(activation)
        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualConvBlock(int(in_channels), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[idx], channels[idx - 1], channels[idx - 1], dropout=float(dropout)))
        self.head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1)

    def architecture_dict(self) -> dict[str, Any]:
        return self.config.__dict__.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoder):
            if idx > 0:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        x = skips[-1]
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)
        return _apply_output_activation(self.head(x), self.activation)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ResidualConvBlock(channels, channels, dropout=dropout, dilation=int(dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskedDilatedDensityNet(nn.Module):
    """Archival fast density baseline; current pipeline uses MaskedDensityUNet."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2),
        activation: str = "softplus",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {
            "name": "MaskedDilatedDensityNet",
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_channels": int(base_channels),
            "dilations": [int(v) for v in dilations],
            "activation": str(activation),
            "dropout": float(dropout),
        }
        self.activation = str(activation)
        self.stem = ResidualConvBlock(int(in_channels), int(base_channels), dropout=float(dropout))
        self.blocks = nn.Sequential(
            *[DilatedResidualBlock(int(base_channels), dilation=int(dilation), dropout=float(dropout)) for dilation in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(int(base_channels), int(base_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(base_channels)), int(base_channels)),
            nn.SiLU(),
            nn.Conv2d(int(base_channels), int(out_channels), kernel_size=1),
        )

    def architecture_dict(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return _apply_output_activation(self.head(x), self.activation)


class CountSpatialDensityUNet(nn.Module):
    """U-Net that separates total-count prediction from spatial density allocation."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        count_hidden_channels: int | None = None,
        count_activation: str = "softplus",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depth = max(int(depth), 2)
        base_channels = int(base_channels)
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.config = {
            "name": "CountSpatialDensityUNet",
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_channels": base_channels,
            "depth": depth,
            "count_hidden_channels": count_hidden_channels,
            "count_activation": str(count_activation),
            "dropout": float(dropout),
        }
        self.count_activation = str(count_activation)
        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualConvBlock(int(in_channels), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[idx], channels[idx - 1], channels[idx - 1], dropout=float(dropout)))
        self.spatial_head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1)

        hidden = int(count_hidden_channels or max(channels[-1] // 2, base_channels))
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def architecture_dict(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoder):
            if idx > 0:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        bottleneck = skips[-1]
        decoded = bottleneck
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            decoded = block(decoded, skip)
        count_raw = self.count_head(bottleneck)
        count = _apply_output_activation(count_raw, self.count_activation)
        return {
            "spatial_logits": self.spatial_head(decoded),
            "count": count,
            "count_raw": count_raw,
        }


class OccupancySpatialUNet(nn.Module):
    """U-Net that predicts ship presence and allocates positive mass spatially.

    This is the active count-free path: the scalar head estimates whether a
    patch contains at least one ship, and the spatial head estimates where the
    positive evidence lies within the valid crop.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        occupancy_hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depth = max(int(depth), 2)
        base_channels = int(base_channels)
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.config = {
            "name": "OccupancySpatialUNet",
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_channels": base_channels,
            "depth": depth,
            "occupancy_hidden_channels": occupancy_hidden_channels,
            "dropout": float(dropout),
        }
        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualConvBlock(int(in_channels), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[idx], channels[idx - 1], channels[idx - 1], dropout=float(dropout)))
        self.spatial_head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1)

        hidden = int(occupancy_hidden_channels or max(channels[-1] // 2, base_channels))
        self.occupancy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def architecture_dict(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoder):
            if idx > 0:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        bottleneck = skips[-1]
        decoded = bottleneck
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            decoded = block(decoded, skip)
        return {
            "spatial_logits": self.spatial_head(decoded),
            "occupancy_logit": self.occupancy_head(bottleneck),
        }


class DualRadianceCountSpatialUNet(nn.Module):
    """Count-spatial U-Net with separate spatial and raw-radiance count inputs."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        count_hidden_channels: int | None = None,
        count_activation: str = "softplus",
        dropout: float = 0.0,
        spatial_channel_indices: tuple[int, ...] | list[int] = (0, 2, 3, 4, 5, 6),
        count_channel_indices: tuple[int, ...] | list[int] = (1,),
    ) -> None:
        super().__init__()
        depth = max(int(depth), 2)
        base_channels = int(base_channels)
        spatial_indices = tuple(int(idx) for idx in spatial_channel_indices)
        count_indices = tuple(int(idx) for idx in count_channel_indices)
        if not spatial_indices:
            raise ValueError("spatial_channel_indices must not be empty")
        if not count_indices:
            raise ValueError("count_channel_indices must not be empty")
        if max(spatial_indices + count_indices) >= int(in_channels) or min(spatial_indices + count_indices) < 0:
            raise ValueError("channel index out of range for in_channels")

        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.config = {
            "name": "DualRadianceCountSpatialUNet",
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_channels": base_channels,
            "depth": depth,
            "count_hidden_channels": count_hidden_channels,
            "count_activation": str(count_activation),
            "dropout": float(dropout),
            "spatial_channel_indices": list(spatial_indices),
            "count_channel_indices": list(count_indices),
        }
        self.count_activation = str(count_activation)
        self.register_buffer("_spatial_channel_index", torch.tensor(spatial_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_count_channel_index", torch.tensor(count_indices, dtype=torch.long), persistent=False)

        self.spatial_encoder = nn.ModuleList()
        self.spatial_encoder.append(ResidualConvBlock(len(spatial_indices), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.spatial_encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[idx], channels[idx - 1], channels[idx - 1], dropout=float(dropout)))
        self.spatial_head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1)

        self.count_encoder = nn.ModuleList()
        self.count_encoder.append(ResidualConvBlock(len(count_indices), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.count_encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))

        hidden = int(count_hidden_channels or max(channels[-1] // 2, base_channels))
        stats_channels = len(count_indices) * 4
        self.count_head = nn.Sequential(
            nn.Linear(channels[-1] + stats_channels, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def architecture_dict(self) -> dict[str, Any]:
        return dict(self.config)

    def _encode(self, x: torch.Tensor, blocks: nn.ModuleList) -> list[torch.Tensor]:
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(blocks):
            if idx > 0:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        return skips

    @staticmethod
    def _raw_radiance_stats(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        return torch.cat(
            [
                flat.sum(dim=2),
                flat.mean(dim=2),
                flat.amax(dim=2),
                flat.std(dim=2, unbiased=False),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_x = x.index_select(1, self._spatial_channel_index.to(device=x.device))
        count_x = x.index_select(1, self._count_channel_index.to(device=x.device))

        spatial_skips = self._encode(spatial_x, self.spatial_encoder)
        decoded = spatial_skips[-1]
        for block, skip in zip(self.decoder, reversed(spatial_skips[:-1])):
            decoded = block(decoded, skip)

        count_skips = self._encode(count_x, self.count_encoder)
        pooled = F.adaptive_avg_pool2d(count_skips[-1], 1).flatten(1)
        stats = self._raw_radiance_stats(count_x)
        count_raw = self.count_head(torch.cat([pooled, stats], dim=1))
        count = _apply_output_activation(count_raw, self.count_activation)
        return {
            "spatial_logits": self.spatial_head(decoded),
            "count": count,
            "count_raw": count_raw,
        }


def build_density_model(name: str, **kwargs: Any) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized in {"main", "unet", "resunet", "masked_density_unet", "maskeddensityunet"}:
        return MaskedDensityUNet(**kwargs)
    if normalized in {
        "occupancy_spatial",
        "occupancy_spatial_unet",
        "occupancyspatialunet",
        "ship_ox_spatial",
        "ship_presence_spatial",
    }:
        return OccupancySpatialUNet(**kwargs)
    if normalized in {"count_spatial", "count_spatial_unet", "countspatialdensityunet", "count_spatial_density_unet"}:
        return CountSpatialDensityUNet(**kwargs)
    if normalized in {
        "dual_radiance_count_spatial",
        "dual_radiance_count_spatial_unet",
        "dualradiancecountspatialunet",
    }:
        return DualRadianceCountSpatialUNet(**kwargs)
    if normalized in {"fast", "dilated", "csrnet", "masked_dilated_density_net", "maskeddilateddensitynet"}:
        kwargs = dict(kwargs)
        kwargs.pop("depth", None)
        return MaskedDilatedDensityNet(**kwargs)
    raise ValueError(f"Unsupported density model: {name}")
