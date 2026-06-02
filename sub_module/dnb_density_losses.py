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
    foreground_weight: float = 0.0
    count_weight: float = 0.22
    batch_count_weight: float = 0.08
    local_count_weight: float = 0.20
    empty_count_weight: float = 0.0
    background_weight: float = 0.05
    pixel_loss: str = "huber"
    foreground_loss: str = "huber"
    huber_delta: float = 0.05
    count_loss: str = "relative_huber"
    count_huber_delta: float = 0.25
    count_normalizer: float = 1.0
    local_count_windows: tuple[int, ...] = (16, 32, 64)
    local_count_stride_factor: int = 2
    positive_local_count_only: bool = False
    local_count_target_threshold: float = 1.0e-6
    foreground_target_threshold: float = 1.0e-6
    foreground_weight_power: float = 0.0
    empty_count_target_threshold: float = 1.0e-6
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


def _masked_values(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    active = weight > 0
    return torch.where(active, values, torch.zeros_like(values))


def _apply_mask(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    active = weight > 0
    return torch.where(active, values * weight, torch.zeros_like(values))


def _masked_mean(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.clamp(weight.sum(), min=float(eps))
    return (_masked_values(values, weight) * weight).sum() / denom


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
    diff = _masked_values(pred - target, weight)
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


def _count_error_element(
    pred_count: torch.Tensor,
    target_count: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    normalized = str(loss_name).lower()
    if normalized in {"log_mse", "log_l2", "log_count_mse"}:
        error = torch.log1p(torch.clamp(pred_count, min=0.0)) - torch.log1p(torch.clamp(target_count, min=0.0))
        return error.square()
    if normalized in {"log_mae", "log_l1", "log_count_mae"}:
        error = torch.log1p(torch.clamp(pred_count, min=0.0)) - torch.log1p(torch.clamp(target_count, min=0.0))
        return error.abs()
    if normalized in {"log_huber", "log_count_huber"}:
        error = torch.log1p(torch.clamp(pred_count, min=0.0)) - torch.log1p(torch.clamp(target_count, min=0.0))
        return _huber(error, float(huber_delta))

    denom = target_count.detach() + float(normalizer)
    error = (pred_count - target_count) / torch.clamp(denom, min=float(eps))
    if normalized in {"mse", "relative_mse", "l2"}:
        return error.square()
    if normalized in {"mae", "relative_mae", "l1"}:
        return error.abs()
    if normalized in {"huber", "relative_huber", "smooth_l1"}:
        return _huber(error, float(huber_delta))
    raise ValueError(f"Unsupported count loss: {loss_name}")


def _count_error_loss(
    pred_count: torch.Tensor,
    target_count: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    normalizer: float,
    eps: float,
) -> torch.Tensor:
    return _count_error_element(
        pred_count,
        target_count,
        loss_name=loss_name,
        huber_delta=float(huber_delta),
        normalizer=float(normalizer),
        eps=float(eps),
    ).mean()


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
    pred_count = _apply_mask(pred, valid_mask).flatten(1).sum(dim=1)
    target_count = _apply_mask(target, valid_mask).flatten(1).sum(dim=1)
    return _count_error_loss(
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
    pred_count = _apply_mask(pred, valid_mask).sum().reshape(1)
    target_count = _apply_mask(target, valid_mask).sum().reshape(1)
    return _count_error_loss(
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
    positive_only: bool,
    target_threshold: float,
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
            _apply_mask(pred, valid_mask),
            kernel,
            stride=(stride_h, stride_w),
        )
        target_sum = F.conv2d(
            _apply_mask(target, valid_mask),
            kernel,
            stride=(stride_h, stride_w),
        )
        valid_sum = F.conv2d(
            valid_mask,
            kernel,
            stride=(stride_h, stride_w),
        )
        local_valid_bool = valid_sum > 0
        if bool(positive_only):
            local_valid_bool = local_valid_bool & (target_sum > float(target_threshold))
        local_valid = local_valid_bool.to(dtype=pred.dtype)
        element = _count_error_element(
            pred_sum,
            target_sum,
            loss_name=loss_name,
            huber_delta=float(huber_delta),
            normalizer=float(normalizer),
            eps=float(eps),
        )
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


def foreground_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
    target_threshold: float,
    weight_power: float,
    eps: float,
) -> torch.Tensor:
    foreground = valid_mask * (target > float(target_threshold)).to(dtype=pred.dtype)
    if float(weight_power) > 0.0:
        foreground = foreground * torch.clamp(target.detach(), min=float(target_threshold)).pow(float(weight_power))
    return weighted_density_loss(
        pred,
        target,
        foreground,
        loss_name=loss_name,
        huber_delta=float(huber_delta),
        eps=float(eps),
    )


def empty_window_false_positive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    windows: tuple[int, ...],
    stride_factor: int,
    target_threshold: float,
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
        pred_sum = F.conv2d(_apply_mask(pred, valid_mask), kernel, stride=(stride_h, stride_w))
        target_sum = F.conv2d(_apply_mask(target, valid_mask), kernel, stride=(stride_h, stride_w))
        valid_sum = F.conv2d(valid_mask, kernel, stride=(stride_h, stride_w))
        empty_valid = ((valid_sum > 0) & (target_sum <= float(target_threshold))).to(dtype=pred.dtype)
        mean_pred_density = pred_sum / torch.clamp(valid_sum, min=float(eps))
        losses.append(_masked_mean(mean_pred_density.square(), empty_valid, float(eps)))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


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
        foreground = foreground_density_loss(
            pred,
            target,
            valid_mask,
            loss_name=cfg.foreground_loss,
            huber_delta=float(cfg.huber_delta),
            target_threshold=float(cfg.foreground_target_threshold),
            weight_power=float(cfg.foreground_weight_power),
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
            positive_only=bool(cfg.positive_local_count_only),
            target_threshold=float(cfg.local_count_target_threshold),
            eps=float(cfg.eps),
        )
        empty_count = empty_window_false_positive_loss(
            pred,
            target,
            valid_mask,
            windows=cfg.local_count_windows,
            stride_factor=int(cfg.local_count_stride_factor),
            target_threshold=float(cfg.empty_count_target_threshold),
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
            + float(cfg.foreground_weight) * foreground
            + float(cfg.count_weight) * count
            + float(cfg.batch_count_weight) * batch_count
            + float(cfg.local_count_weight) * local
            + float(cfg.empty_count_weight) * empty_count
            + float(cfg.background_weight) * background
        )
        if bool(cfg.report_components):
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_pixel": float(pixel.detach().cpu()),
                "loss_foreground": float(foreground.detach().cpu()),
                "loss_count": float(count.detach().cpu()),
                "loss_batch_count": float(batch_count.detach().cpu()),
                "loss_local_count": float(local.detach().cpu()),
                "loss_empty_count": float(empty_count.detach().cpu()),
                "loss_background": float(background.detach().cpu()),
                "loss_weight_pixel": float(cfg.pixel_weight),
                "loss_weight_foreground": float(cfg.foreground_weight),
                "loss_weight_count": float(cfg.count_weight),
                "loss_weight_batch_count": float(cfg.batch_count_weight),
                "loss_weight_local_count": float(cfg.local_count_weight),
                "loss_weight_empty_count": float(cfg.empty_count_weight),
                "loss_weight_background": float(cfg.background_weight),
            }
        return total


@dataclass(frozen=True)
class CountSpatialLossConfig:
    name: str = "count_spatial_density_loss"
    count_weight: float = 0.50
    spatial_weight: float = 0.40
    density_weight: float = 0.10
    background_weight: float = 0.0
    count_loss: str = "log_mse"
    count_huber_delta: float = 1.0
    spatial_loss: str = "kl"
    spatial_temperature: float = 1.0
    density_loss: str = "huber"
    huber_delta: float = 0.05
    background_target_threshold: float = 1.0e-6
    min_target_count_for_spatial: float = 1.0e-6
    eps: float = 1.0e-8
    report_components: bool = True


def count_spatial_loss_config_from_dict(config: dict[str, Any] | None) -> CountSpatialLossConfig:
    if not config:
        return CountSpatialLossConfig()
    allowed = {item.name for item in fields(CountSpatialLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return CountSpatialLossConfig(**values)


def _spatial_prob_from_logits(
    spatial_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    temperature = max(float(temperature), float(eps))
    valid = valid_mask > 0
    logits = spatial_logits / temperature
    logits = torch.where(valid, logits, torch.full_like(logits, -1.0e9))
    flat_logits = logits.flatten(1)
    flat_valid = valid.flatten(1).to(dtype=spatial_logits.dtype)
    max_logits = flat_logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(flat_logits - max_logits) * flat_valid
    denom = torch.clamp(exp_logits.sum(dim=1, keepdim=True), min=float(eps))
    return (exp_logits / denom).reshape_as(spatial_logits)


def density_from_model_output(
    output: torch.Tensor | dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    spatial_temperature: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if "density" in output:
        return output["density"]
    if "spatial_logits" not in output or "count" not in output:
        raise ValueError("Count-spatial model output must contain spatial_logits and count")
    prob = _spatial_prob_from_logits(
        output["spatial_logits"],
        batch["valid_mask"],
        temperature=float(spatial_temperature),
        eps=float(eps),
    )
    count = output["count"].reshape(-1, 1, 1, 1)
    return prob * count


class CountSpatialDensityLoss(nn.Module):
    """Loss for density = predicted_count * masked spatial probability."""

    def __init__(self, config: CountSpatialLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or CountSpatialLossConfig()
        self.last_components: dict[str, float] = {}

    def density_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return density_from_model_output(
            output,
            batch,
            spatial_temperature=float(self.config.spatial_temperature),
            eps=float(self.config.eps),
        )

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            raise ValueError("count_spatial_density_loss requires a dict output with spatial_logits and count")
        target = batch["target"]
        valid_mask = batch["valid_mask"]
        cfg = self.config

        pred_density = self.density_from_output(output, batch)
        pred_prob = _spatial_prob_from_logits(
            output["spatial_logits"],
            valid_mask,
            temperature=float(cfg.spatial_temperature),
            eps=float(cfg.eps),
        )
        target_mass = _apply_mask(target, valid_mask)
        target_count = target_mass.flatten(1).sum(dim=1)
        pred_count = output["count"].reshape(-1)

        count = _count_error_loss(
            pred_count,
            target_count,
            loss_name=cfg.count_loss,
            huber_delta=float(cfg.count_huber_delta),
            normalizer=1.0,
            eps=float(cfg.eps),
        )

        target_prob = target_mass / torch.clamp(target_count.reshape(-1, 1, 1, 1), min=float(cfg.eps))
        positive = (target_count > float(cfg.min_target_count_for_spatial)).to(dtype=target.dtype).reshape(-1, 1, 1, 1)
        normalized = str(cfg.spatial_loss).lower()
        if normalized in {"kl", "kld", "kl_div"}:
            spatial_element = target_prob * (torch.log(torch.clamp(target_prob, min=float(cfg.eps))) - torch.log(torch.clamp(pred_prob, min=float(cfg.eps))))
        elif normalized in {"ce", "cross_entropy", "nll"}:
            spatial_element = -target_prob * torch.log(torch.clamp(pred_prob, min=float(cfg.eps)))
        else:
            raise ValueError(f"Unsupported spatial loss: {cfg.spatial_loss}")
        spatial_per_sample = spatial_element.flatten(1).sum(dim=1)
        positive_flat = positive.reshape(-1)
        spatial = (spatial_per_sample * positive_flat).sum() / torch.clamp(positive_flat.sum(), min=1.0)

        density = weighted_density_loss(
            pred_density,
            target,
            valid_mask,
            loss_name=cfg.density_loss,
            huber_delta=float(cfg.huber_delta),
            eps=float(cfg.eps),
        )
        background = background_suppression_loss(
            pred_density,
            target,
            valid_mask,
            target_threshold=float(cfg.background_target_threshold),
            eps=float(cfg.eps),
        )
        total = (
            float(cfg.count_weight) * count
            + float(cfg.spatial_weight) * spatial
            + float(cfg.density_weight) * density
            + float(cfg.background_weight) * background
        )
        if bool(cfg.report_components):
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_count": float(count.detach().cpu()),
                "loss_spatial": float(spatial.detach().cpu()),
                "loss_density": float(density.detach().cpu()),
                "loss_background": float(background.detach().cpu()),
                "loss_weight_count": float(cfg.count_weight),
                "loss_weight_spatial": float(cfg.spatial_weight),
                "loss_weight_density": float(cfg.density_weight),
                "loss_weight_background": float(cfg.background_weight),
                "pred_count_mean": float(pred_count.detach().mean().cpu()),
                "target_count_mean": float(target_count.detach().mean().cpu()),
            }
        return total


def build_density_loss(
    config: dict[str, Any] | DensityLossConfig | CountSpatialLossConfig | None = None,
) -> nn.Module:
    if isinstance(config, CountSpatialLossConfig):
        return CountSpatialDensityLoss(config)
    if isinstance(config, DensityLossConfig):
        return StructuredDensityLoss(config)
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "count_spatial_density_loss",
        "count_spatial",
        "count_spatial_loss",
    }:
        return CountSpatialDensityLoss(count_spatial_loss_config_from_dict(config))
    return StructuredDensityLoss(density_loss_config_from_dict(config))
