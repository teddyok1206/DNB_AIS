from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DensityModelConfig:
    name: str = "MaskedDensityUNet"
    in_channels: int = 2
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
    """Small U-Net/ResUNet for PH-masked DNB density regression."""

    def __init__(
        self,
        in_channels: int = 2,
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
    """CSRNet-inspired fast density baseline with dilated convolutions only."""

    def __init__(
        self,
        in_channels: int = 2,
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


def build_density_model(name: str, **kwargs: Any) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized in {"main", "unet", "resunet", "masked_density_unet", "maskeddensityunet"}:
        return MaskedDensityUNet(**kwargs)
    if normalized in {"fast", "dilated", "csrnet", "masked_dilated_density_net", "maskeddilateddensitynet"}:
        kwargs = dict(kwargs)
        kwargs.pop("depth", None)
        return MaskedDilatedDensityNet(**kwargs)
    raise ValueError(f"Unsupported density model: {name}")
