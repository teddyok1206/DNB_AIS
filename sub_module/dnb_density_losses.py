from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DensityLossConfig:
    name: str = "structured_density_loss"
    pixel_weight: float = 0.45
    count_weight: float = 0.22
    batch_count_weight: float = 0.08
    local_count_weight: float = 0.20
    background_weight: float = 0.05
    pixel_loss: str = "huber"
    huber_delta: float = 0.05
    count_loss: str = "relative_huber"
    count_huber_delta: float = 0.25
    count_normalizer: float = 1.0
    local_count_windows: tuple[int, ...] = (16, 32, 64)
    local_count_stride_factor: int = 2
    background_target_threshold: float = 1.0e-6
    eps: float = 1.0e-8
    report_components: bool = True


def _as_tuple_of_ints(values: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    return tuple(int(value) for value in values)


def density_loss_config_from_dict(config: dict[str, Any] | None) -> DensityLossConfig:
    if not config:
        return DensityLossConfig()
    allowed = {item.name for item in fields(DensityLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    if "local_count_windows" in values:
        values["local_count_windows"] = _as_tuple_of_ints(
            values["local_count_windows"],
            default=DensityLossConfig.local_count_windows,
        )
    return DensityLossConfig(**values)


def _masked_mean(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.clamp(weight.sum(), min=float(eps))
    return (values * weight).sum() / denom


def _huber(values: torch.Tensor, delta: float) -> torch.Tensor:
    delta = float(delta)
    abs_values = values.abs()
    quadratic = torch.minimum(abs_values, torch.tensor(delta, dtype=values.dtype, device=values.device))
    linear = abs_values - quadratic
    return 0.5 * quadratic.square() + delta * linear


def weighted_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    eps: float,
) -> torch.Tensor:
    diff = pred - target
    normalized = str(loss_name).lower()
    if normalized in {"mse", "l2"}:
        element = diff.square()
    elif normalized in {"mae", "l1"}:
        element = diff.abs()
    elif normalized in {"huber", "smooth_l1"}:
        element = _huber(diff, float(huber_delta))
    else:
        raise ValueError(f"Unsupported pixel density loss: {loss_name}")
    return _masked_mean(element, weight, float(eps))


def _relative_count_error_loss(
    pred_count: torch.Tensor,
    target_count: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    denom = target_count.detach() + float(normalizer)
    error = (pred_count - target_count) / torch.clamp(denom, min=float(eps))

    normalized = str(loss_name).lower()
    if normalized in {"mse", "relative_mse", "l2"}:
        loss = error.square()
    elif normalized in {"mae", "relative_mae", "l1"}:
        loss = error.abs()
    elif normalized in {"huber", "relative_huber", "smooth_l1"}:
        loss = _huber(error, float(huber_delta))
    else:
        raise ValueError(f"Unsupported count loss: {loss_name}")
    return loss.mean()


def count_conservation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    pred_count = (pred * valid_mask).flatten(1).sum(dim=1)
    target_count = (target * valid_mask).flatten(1).sum(dim=1)
    return _relative_count_error_loss(
        pred_count,
        target_count,
        loss_name=loss_name,
        huber_delta=float(huber_delta),
        normalizer=float(normalizer),
        eps=float(eps),
    )


def batch_count_calibration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    pred_count = (pred * valid_mask).sum().reshape(1)
    target_count = (target * valid_mask).sum().reshape(1)
    return _relative_count_error_loss(
        pred_count,
        target_count,
        loss_name=loss_name,
        huber_delta=float(huber_delta),
        normalizer=float(normalizer),
        eps=float(eps),
    )


def local_count_conservation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    windows: tuple[int, ...],
    stride_factor: int,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    stride_factor = max(int(stride_factor), 1)
    height, width = [int(v) for v in pred.shape[-2:]]
    for window in windows:
        kernel_h = min(int(window), height)
        kernel_w = min(int(window), width)
        if kernel_h <= 1 or kernel_w <= 1:
            continue
        stride_h = max(kernel_h // stride_factor, 1)
        stride_w = max(kernel_w // stride_factor, 1)
        kernel = pred.new_ones((1, 1, kernel_h, kernel_w))
        pred_sum = F.conv2d(
            pred * valid_mask,
            kernel,
            stride=(stride_h, stride_w),
        )
        target_sum = F.conv2d(
            target * valid_mask,
            kernel,
            stride=(stride_h, stride_w),
        )
        valid_sum = F.conv2d(
            valid_mask,
            kernel,
            stride=(stride_h, stride_w),
        )
        local_valid = (valid_sum > 0).to(dtype=pred.dtype)
        denom = target_sum.detach() + float(normalizer)
        error = (pred_sum - target_sum) / torch.clamp(denom, min=float(eps))

        normalized = str(loss_name).lower()
        if normalized in {"mse", "relative_mse", "l2"}:
            element = error.square()
        elif normalized in {"mae", "relative_mae", "l1"}:
            element = error.abs()
        elif normalized in {"huber", "relative_huber", "smooth_l1"}:
            element = _huber(error, float(huber_delta))
        else:
            raise ValueError(f"Unsupported local count loss: {loss_name}")
        losses.append(_masked_mean(element, local_valid, float(eps)))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def background_suppression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_threshold: float,
    eps: float,
) -> torch.Tensor:
    background = valid_mask * (target <= float(target_threshold)).to(dtype=pred.dtype)
    return _masked_mean(pred.square(), background, float(eps))


class StructuredDensityLoss(nn.Module):
    """Density-map loss with pixel, integral-count, local-count, and background terms."""

    def __init__(self, config: DensityLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or DensityLossConfig()
        self.last_components: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        target = batch["target"]
        valid_mask = batch["valid_mask"]
        loss_weight = batch.get("loss_weight", valid_mask)
        cfg = self.config

        pixel = weighted_density_loss(
            pred,
            target,
            loss_weight,
            loss_name=cfg.pixel_loss,
            huber_delta=float(cfg.huber_delta),
            eps=float(cfg.eps),
        )
        count = count_conservation_loss(
            pred,
            target,
            valid_mask,
            loss_name=cfg.count_loss,
            huber_delta=float(cfg.count_huber_delta),
            normalizer=float(cfg.count_normalizer),
            eps=float(cfg.eps),
        )
        batch_count = batch_count_calibration_loss(
            pred,
            target,
            valid_mask,
            loss_name=cfg.count_loss,
            huber_delta=float(cfg.count_huber_delta),
            normalizer=float(cfg.count_normalizer),
            eps=float(cfg.eps),
        )
        local = local_count_conservation_loss(
            pred,
            target,
            valid_mask,
            windows=cfg.local_count_windows,
            stride_factor=int(cfg.local_count_stride_factor),
            loss_name=cfg.count_loss,
            huber_delta=float(cfg.count_huber_delta),
            normalizer=float(cfg.count_normalizer),
            eps=float(cfg.eps),
        )
        background = background_suppression_loss(
            pred,
            target,
            valid_mask,
            target_threshold=float(cfg.background_target_threshold),
            eps=float(cfg.eps),
        )
        total = (
            float(cfg.pixel_weight) * pixel
            + float(cfg.count_weight) * count
            + float(cfg.batch_count_weight) * batch_count
            + float(cfg.local_count_weight) * local
            + float(cfg.background_weight) * background
        )
        if bool(cfg.report_components):
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_pixel": float(pixel.detach().cpu()),
                "loss_count": float(count.detach().cpu()),
                "loss_batch_count": float(batch_count.detach().cpu()),
                "loss_local_count": float(local.detach().cpu()),
                "loss_background": float(background.detach().cpu()),
                "loss_weight_pixel": float(cfg.pixel_weight),
                "loss_weight_count": float(cfg.count_weight),
                "loss_weight_batch_count": float(cfg.batch_count_weight),
                "loss_weight_local_count": float(cfg.local_count_weight),
                "loss_weight_background": float(cfg.background_weight),
            }
        return total


def build_density_loss(config: dict[str, Any] | DensityLossConfig | None = None) -> StructuredDensityLoss:
    if isinstance(config, DensityLossConfig):
        return StructuredDensityLoss(config)
    return StructuredDensityLoss(density_loss_config_from_dict(config))
