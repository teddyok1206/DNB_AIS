from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DensityModelConfig:
    name: str = "PixelBinaryOccupancyUNet"
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.0
    occupancy_hidden_channels: int | None = None


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


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


class PixelBinaryOccupancyUNet(nn.Module):
    """U-Net for hard pixel-level ship presence plus auxiliary patch O/X.

    The pixel head emits independent logits. It is intentionally not a spatial
    softmax and does not force each patch to allocate unit probability mass.
    """

    def __init__(
        self,
        in_channels: int = 3,
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
        self.config = DensityModelConfig(
            name="PixelBinaryOccupancyUNet",
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_channels=base_channels,
            depth=depth,
            dropout=float(dropout),
            occupancy_hidden_channels=occupancy_hidden_channels,
        )
        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualConvBlock(int(in_channels), channels[0], dropout=float(dropout)))
        for idx in range(1, depth):
            self.encoder.append(ResidualConvBlock(channels[idx - 1], channels[idx], dropout=float(dropout)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[idx], channels[idx - 1], channels[idx - 1], dropout=float(dropout)))
        self.pixel_head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1)

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
        return self.config.__dict__.copy()

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
            "pixel_logits": self.pixel_head(decoded),
            "occupancy_logit": self.occupancy_head(bottleneck),
        }


def build_density_model(name: str, **kwargs: Any) -> nn.Module:
    normalized = str(name).strip().lower()
    active_names = {
        "main",
        "unet",
        "pixel_binary_occupancy",
        "pixel_binary_occupancy_unet",
        "pixelbinaryoccupancyunet",
        "pixel_occupancy",
        "pixel_ox",
        "ship_pixel_presence",
    }
    if normalized in active_names:
        return PixelBinaryOccupancyUNet(**kwargs)
    retired = {
        "masked_density_unet",
        "maskeddensityunet",
        "masked_dilated_density_net",
        "maskeddilateddensitynet",
        "fast",
        "dilated",
        "csrnet",
        "count_spatial",
        "count_spatial_unet",
        "countspatialdensityunet",
        "count_spatial_density_unet",
        "occupancy_spatial",
        "occupancy_spatial_unet",
        "occupancyspatialunet",
        "ship_ox_spatial",
        "ship_presence_spatial",
        "occupancy_only",
        "occupancy_only_unet",
        "occupancyonlyunet",
        "spatial_only",
        "spatial_only_unet",
        "spatialonlyunet",
        "dual_radiance_count_spatial",
        "dual_radiance_count_spatial_unet",
        "dualradiancecountspatialunet",
    }
    if normalized in retired:
        raise ValueError(
            f"Retired density model {name!r} was archived under "
            "_archive/retired_density_complexity_20260609/sub_module/dnb_density_models_legacy_full.py"
        )
    raise ValueError(f"Unsupported density model: {name}")
