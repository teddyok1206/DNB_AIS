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
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
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


def _weighted_sample_mean(values: torch.Tensor, sample_weight: torch.Tensor, eps: float) -> torch.Tensor:
    weight = sample_weight.reshape(values.shape[0], *([1] * (values.ndim - 1))).to(dtype=values.dtype, device=values.device)
    return (values * weight).sum() / torch.clamp(weight.expand_as(values).sum(), min=float(eps))


def _lifetime_sample_weight(
    batch: dict[str, Any],
    *,
    mode: str,
    strength: float,
    min_weight: float,
    max_weight: float,
    normalize: bool,
    eps: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(reference.shape[0])
    mode = str(mode).lower()
    if mode in {"none", "off", "false", "0"} or float(strength) <= 0.0 or "lifetime" not in batch:
        return reference.new_ones((batch_size,), dtype=reference.dtype)
    lifetime = torch.clamp(batch["lifetime"].to(device=reference.device, dtype=reference.dtype).reshape(-1), min=0.0)
    if mode in {"linear", "lifetime"}:
        signal = lifetime
    elif mode in {"log", "log1p", "log_lifetime", "log1p_batch"}:
        signal = torch.log1p(lifetime)
    elif mode in {"sqrt", "sqrt_lifetime"}:
        signal = torch.sqrt(lifetime)
    else:
        raise ValueError(f"Unsupported lifetime_weight_mode: {mode}")
    max_signal = torch.clamp(signal.max().detach(), min=float(eps))
    scaled = signal / max_signal
    weight = 1.0 + float(strength) * scaled
    weight = torch.clamp(weight, min=float(min_weight), max=float(max_weight))
    if bool(normalize):
        weight = weight / torch.clamp(weight.mean().detach(), min=float(eps))
    return weight


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


def _occupancy_logit_from_output(output: dict[str, torch.Tensor]) -> torch.Tensor:
    if "occupancy_logit" in output:
        return output["occupancy_logit"].reshape(-1)
    if "occupancy_logits" in output:
        return output["occupancy_logits"].reshape(-1)
    raise ValueError("Occupancy-spatial model output must contain occupancy_logit")


def occupancy_spatial_density_from_model_output(
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
    if "spatial_logits" not in output:
        raise ValueError("Occupancy-spatial model output must contain spatial_logits")
    pred_prob = _spatial_prob_from_logits(
        output["spatial_logits"],
        batch["valid_mask"],
        temperature=float(spatial_temperature),
        eps=float(eps),
    )
    occupancy_prob = torch.sigmoid(_occupancy_logit_from_output(output)).reshape(-1, 1, 1, 1)
    return pred_prob * occupancy_prob


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
        sample_weight = _lifetime_sample_weight(
            batch,
            mode=cfg.lifetime_weight_mode,
            strength=float(cfg.lifetime_weight_strength),
            min_weight=float(cfg.lifetime_weight_min),
            max_weight=float(cfg.lifetime_weight_max),
            normalize=bool(cfg.lifetime_weight_normalize),
            eps=float(cfg.eps),
            reference=target,
        )
        sample_weight_map = sample_weight.reshape(-1, 1, 1, 1)

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

        count_element = _count_error_element(
            pred_count,
            target_count,
            loss_name=cfg.count_loss,
            huber_delta=float(cfg.count_huber_delta),
            normalizer=1.0,
            eps=float(cfg.eps),
        )
        count = _weighted_sample_mean(count_element, sample_weight, float(cfg.eps))

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
        spatial_weight = positive_flat * sample_weight
        spatial = (spatial_per_sample * spatial_weight).sum() / torch.clamp(spatial_weight.sum(), min=1.0)

        density = weighted_density_loss(
            pred_density,
            target,
            valid_mask * sample_weight_map,
            loss_name=cfg.density_loss,
            huber_delta=float(cfg.huber_delta),
            eps=float(cfg.eps),
        )
        background = background_suppression_loss(
            pred_density,
            target,
            valid_mask * sample_weight_map,
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
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
                "pred_count_mean": float(pred_count.detach().mean().cpu()),
                "target_count_mean": float(target_count.detach().mean().cpu()),
            }
        return total


@dataclass(frozen=True)
class OccupancySpatialLossConfig:
    name: str = "occupancy_spatial_loss"
    occupancy_weight: float = 0.60
    spatial_weight: float = 0.40
    occupancy_pos_weight: float = 1.0
    spatial_loss: str = "kl"
    spatial_temperature: float = 1.0
    min_target_count_for_positive: float = 1.0e-6
    min_target_count_for_spatial: float = 1.0e-6
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
    eps: float = 1.0e-8
    report_components: bool = True


@dataclass(frozen=True)
class OccupancyOnlyLossConfig:
    name: str = "occupancy_only_loss"
    occupancy_pos_weight: float = 1.0
    min_target_count_for_positive: float = 1.0e-6
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
    eps: float = 1.0e-8
    report_components: bool = True


@dataclass(frozen=True)
class SpatialOnlyLossConfig:
    name: str = "spatial_only_loss"
    spatial_loss: str = "kl"
    spatial_temperature: float = 1.0
    min_target_count_for_spatial: float = 1.0e-6
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
    eps: float = 1.0e-8
    report_components: bool = True


@dataclass(frozen=True)
class PixelBinaryOccupancyLossConfig:
    name: str = "pixel_binary_occupancy_loss"
    pixel_weight: float = 0.85
    occupancy_weight: float = 0.15
    pixel_pos_weight: float = 50.0
    occupancy_pos_weight: float = 1.0
    dice_weight: float = 0.0
    min_raw_count_for_positive: float = 0.0
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
    eps: float = 1.0e-8
    report_components: bool = True


def occupancy_spatial_loss_config_from_dict(config: dict[str, Any] | None) -> OccupancySpatialLossConfig:
    if not config:
        return OccupancySpatialLossConfig()
    allowed = {item.name for item in fields(OccupancySpatialLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return OccupancySpatialLossConfig(**values)


def occupancy_only_loss_config_from_dict(config: dict[str, Any] | None) -> OccupancyOnlyLossConfig:
    if not config:
        return OccupancyOnlyLossConfig()
    allowed = {item.name for item in fields(OccupancyOnlyLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return OccupancyOnlyLossConfig(**values)


def spatial_only_loss_config_from_dict(config: dict[str, Any] | None) -> SpatialOnlyLossConfig:
    if not config:
        return SpatialOnlyLossConfig()
    allowed = {item.name for item in fields(SpatialOnlyLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return SpatialOnlyLossConfig(**values)


def pixel_binary_occupancy_loss_config_from_dict(config: dict[str, Any] | None) -> PixelBinaryOccupancyLossConfig:
    if not config:
        return PixelBinaryOccupancyLossConfig()
    allowed = {item.name for item in fields(PixelBinaryOccupancyLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return PixelBinaryOccupancyLossConfig(**values)


class OccupancySpatialLoss(nn.Module):
    """Loss for active O/X + spatial distribution training.

    The model does not predict ship count. Its patch-level mass is the
    probability that at least one ship exists in the patch.
    """

    def __init__(self, config: OccupancySpatialLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or OccupancySpatialLossConfig()
        self.last_components: dict[str, float] = {}

    def target_count(self, batch: dict[str, Any]) -> torch.Tensor:
        target_mass = _apply_mask(batch["target"], batch["valid_mask"])
        return target_mass.flatten(1).sum(dim=1)

    def occupancy_target(self, batch: dict[str, Any]) -> torch.Tensor:
        return (self.target_count(batch) > float(self.config.min_target_count_for_positive)).to(dtype=batch["target"].dtype)

    def target_density_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        target_mass = _apply_mask(batch["target"], batch["valid_mask"])
        target_count = target_mass.flatten(1).sum(dim=1).reshape(-1, 1, 1, 1)
        positive = (target_count > float(self.config.min_target_count_for_positive)).to(dtype=target_mass.dtype)
        target_prob = target_mass / torch.clamp(target_count, min=float(self.config.eps))
        return target_prob * positive

    def density_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return occupancy_spatial_density_from_model_output(
            output,
            batch,
            spatial_temperature=float(self.config.spatial_temperature),
            eps=float(self.config.eps),
        )

    def occupancy_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(output):
            pred_density = output * batch["valid_mask"]
            pred_prob = torch.clamp(pred_density.flatten(1).sum(dim=1), min=0.0, max=1.0)
        else:
            pred_prob = torch.sigmoid(_occupancy_logit_from_output(output))
        return pred_prob, self.occupancy_target(batch)

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            raise ValueError("occupancy_spatial_loss requires a dict output with spatial_logits and occupancy_logit")
        target = batch["target"]
        valid_mask = batch["valid_mask"]
        cfg = self.config
        sample_weight = _lifetime_sample_weight(
            batch,
            mode=cfg.lifetime_weight_mode,
            strength=float(cfg.lifetime_weight_strength),
            min_weight=float(cfg.lifetime_weight_min),
            max_weight=float(cfg.lifetime_weight_max),
            normalize=bool(cfg.lifetime_weight_normalize),
            eps=float(cfg.eps),
            reference=target,
        )

        target_count = self.target_count(batch)
        occupancy_target = (target_count > float(cfg.min_target_count_for_positive)).to(dtype=target.dtype)
        occupancy_logit = _occupancy_logit_from_output(output)
        bce = F.binary_cross_entropy_with_logits(occupancy_logit, occupancy_target, reduction="none")
        if float(cfg.occupancy_pos_weight) != 1.0:
            class_weight = torch.where(
                occupancy_target > 0,
                occupancy_target.new_tensor(float(cfg.occupancy_pos_weight)),
                occupancy_target.new_tensor(1.0),
            )
            bce = bce * class_weight
        occupancy = (bce * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=float(cfg.eps))

        pred_prob = _spatial_prob_from_logits(
            output["spatial_logits"],
            valid_mask,
            temperature=float(cfg.spatial_temperature),
            eps=float(cfg.eps),
        )
        target_mass = _apply_mask(target, valid_mask)
        target_prob = target_mass / torch.clamp(target_count.reshape(-1, 1, 1, 1), min=float(cfg.eps))
        positive = (target_count > float(cfg.min_target_count_for_spatial)).to(dtype=target.dtype)
        normalized = str(cfg.spatial_loss).lower()
        if normalized in {"kl", "kld", "kl_div"}:
            spatial_element = target_prob * (
                torch.log(torch.clamp(target_prob, min=float(cfg.eps)))
                - torch.log(torch.clamp(pred_prob, min=float(cfg.eps)))
            )
        elif normalized in {"ce", "cross_entropy", "nll"}:
            spatial_element = -target_prob * torch.log(torch.clamp(pred_prob, min=float(cfg.eps)))
        else:
            raise ValueError(f"Unsupported spatial loss: {cfg.spatial_loss}")
        spatial_per_sample = spatial_element.flatten(1).sum(dim=1)
        spatial_weight = positive * sample_weight
        spatial = (spatial_per_sample * spatial_weight).sum() / torch.clamp(spatial_weight.sum(), min=1.0)

        total = float(cfg.occupancy_weight) * occupancy + float(cfg.spatial_weight) * spatial
        if bool(cfg.report_components):
            occupancy_prob = torch.sigmoid(occupancy_logit)
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_occupancy": float(occupancy.detach().cpu()),
                "loss_spatial": float(spatial.detach().cpu()),
                "loss_weight_occupancy": float(cfg.occupancy_weight),
                "loss_weight_spatial": float(cfg.spatial_weight),
                "occupancy_target_mean": float(occupancy_target.detach().mean().cpu()),
                "occupancy_pred_mean": float(occupancy_prob.detach().mean().cpu()),
                "target_count_mean_raw": float(target_count.detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
        return total


class OccupancyOnlyLoss(nn.Module):
    """Patch-level binary ship-presence loss.

    Density helpers return a uniform valid-mask allocation only so the shared
    training runner can keep writing diagnostics. Use occupancy metrics as the
    authoritative result for this loss.
    """

    def __init__(self, config: OccupancyOnlyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or OccupancyOnlyLossConfig()
        self.last_components: dict[str, float] = {}

    def target_count(self, batch: dict[str, Any]) -> torch.Tensor:
        target_mass = _apply_mask(batch["target"], batch["valid_mask"])
        return target_mass.flatten(1).sum(dim=1)

    def occupancy_target(self, batch: dict[str, Any]) -> torch.Tensor:
        return (self.target_count(batch) > float(self.config.min_target_count_for_positive)).to(dtype=batch["target"].dtype)

    def _uniform_density(self, mass: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid = (valid_mask > 0).to(dtype=valid_mask.dtype)
        denom = torch.clamp(valid.flatten(1).sum(dim=1).reshape(-1, 1, 1, 1), min=float(self.config.eps))
        return valid * mass.reshape(-1, 1, 1, 1) / denom

    def target_density_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._uniform_density(self.occupancy_target(batch), batch["valid_mask"])

    def density_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            pred_prob = torch.clamp(output.flatten(1).mean(dim=1), min=0.0, max=1.0)
        else:
            pred_prob = torch.sigmoid(_occupancy_logit_from_output(output))
        return self._uniform_density(pred_prob, batch["valid_mask"])

    def occupancy_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(output):
            pred_density = output * batch["valid_mask"]
            pred_prob = torch.clamp(pred_density.flatten(1).sum(dim=1), min=0.0, max=1.0)
        else:
            pred_prob = torch.sigmoid(_occupancy_logit_from_output(output))
        return pred_prob, self.occupancy_target(batch)

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            raise ValueError("occupancy_only_loss requires a dict output with occupancy_logit")
        cfg = self.config
        target = batch["target"]
        target_count = self.target_count(batch)
        occupancy_target = (target_count > float(cfg.min_target_count_for_positive)).to(dtype=target.dtype)
        occupancy_logit = _occupancy_logit_from_output(output)
        sample_weight = _lifetime_sample_weight(
            batch,
            mode=cfg.lifetime_weight_mode,
            strength=float(cfg.lifetime_weight_strength),
            min_weight=float(cfg.lifetime_weight_min),
            max_weight=float(cfg.lifetime_weight_max),
            normalize=bool(cfg.lifetime_weight_normalize),
            eps=float(cfg.eps),
            reference=target,
        )
        bce = F.binary_cross_entropy_with_logits(occupancy_logit, occupancy_target, reduction="none")
        if float(cfg.occupancy_pos_weight) != 1.0:
            class_weight = torch.where(
                occupancy_target > 0,
                occupancy_target.new_tensor(float(cfg.occupancy_pos_weight)),
                occupancy_target.new_tensor(1.0),
            )
            bce = bce * class_weight
        occupancy = (bce * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=float(cfg.eps))
        if bool(cfg.report_components):
            occupancy_prob = torch.sigmoid(occupancy_logit)
            self.last_components = {
                "loss_total": float(occupancy.detach().cpu()),
                "loss_occupancy": float(occupancy.detach().cpu()),
                "occupancy_target_mean": float(occupancy_target.detach().mean().cpu()),
                "occupancy_pred_mean": float(occupancy_prob.detach().mean().cpu()),
                "target_count_mean_raw": float(target_count.detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
        return occupancy


class SpatialOnlyLoss(nn.Module):
    """Positive-patch spatial distribution loss with no patch O/X objective."""

    def __init__(self, config: SpatialOnlyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialOnlyLossConfig()
        self.last_components: dict[str, float] = {}

    def target_count(self, batch: dict[str, Any]) -> torch.Tensor:
        target_mass = _apply_mask(batch["target"], batch["valid_mask"])
        return target_mass.flatten(1).sum(dim=1)

    def target_density_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        target_mass = _apply_mask(batch["target"], batch["valid_mask"])
        target_count = target_mass.flatten(1).sum(dim=1).reshape(-1, 1, 1, 1)
        positive = (target_count > float(self.config.min_target_count_for_spatial)).to(dtype=target_mass.dtype)
        target_prob = target_mass / torch.clamp(target_count, min=float(self.config.eps))
        return target_prob * positive

    def density_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            return output * batch["valid_mask"]
        if "spatial_logits" not in output:
            raise ValueError("spatial_only_loss requires spatial_logits")
        pred_prob = _spatial_prob_from_logits(
            output["spatial_logits"],
            batch["valid_mask"],
            temperature=float(self.config.spatial_temperature),
            eps=float(self.config.eps),
        )
        # Metrics are defined only where a positive spatial target exists.
        positive = (self.target_count(batch).reshape(-1, 1, 1, 1) > float(self.config.min_target_count_for_spatial)).to(dtype=pred_prob.dtype)
        return pred_prob * positive

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            raise ValueError("spatial_only_loss requires a dict output with spatial_logits")
        if "spatial_logits" not in output:
            raise ValueError("spatial_only_loss requires spatial_logits")
        cfg = self.config
        target = batch["target"]
        valid_mask = batch["valid_mask"]
        sample_weight = _lifetime_sample_weight(
            batch,
            mode=cfg.lifetime_weight_mode,
            strength=float(cfg.lifetime_weight_strength),
            min_weight=float(cfg.lifetime_weight_min),
            max_weight=float(cfg.lifetime_weight_max),
            normalize=bool(cfg.lifetime_weight_normalize),
            eps=float(cfg.eps),
            reference=target,
        )
        target_count = self.target_count(batch)
        target_mass = _apply_mask(target, valid_mask)
        target_prob = target_mass / torch.clamp(target_count.reshape(-1, 1, 1, 1), min=float(cfg.eps))
        pred_prob = _spatial_prob_from_logits(
            output["spatial_logits"],
            valid_mask,
            temperature=float(cfg.spatial_temperature),
            eps=float(cfg.eps),
        )
        normalized = str(cfg.spatial_loss).lower()
        if normalized in {"kl", "kld", "kl_div"}:
            spatial_element = target_prob * (
                torch.log(torch.clamp(target_prob, min=float(cfg.eps)))
                - torch.log(torch.clamp(pred_prob, min=float(cfg.eps)))
            )
        elif normalized in {"ce", "cross_entropy", "nll"}:
            spatial_element = -target_prob * torch.log(torch.clamp(pred_prob, min=float(cfg.eps)))
        else:
            raise ValueError(f"Unsupported spatial loss: {cfg.spatial_loss}")
        spatial_per_sample = spatial_element.flatten(1).sum(dim=1)
        positive = (target_count > float(cfg.min_target_count_for_spatial)).to(dtype=target.dtype)
        spatial_weight = positive * sample_weight
        spatial = (spatial_per_sample * spatial_weight).sum() / torch.clamp(spatial_weight.sum(), min=1.0)
        if bool(cfg.report_components):
            self.last_components = {
                "loss_total": float(spatial.detach().cpu()),
                "loss_spatial": float(spatial.detach().cpu()),
                "positive_patch_weight_sum": float(spatial_weight.detach().sum().cpu()),
                "positive_patch_mean": float(positive.detach().mean().cpu()),
                "target_count_mean_raw": float(target_count.detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
        return spatial


class PixelBinaryOccupancyLoss(nn.Module):
    """Hard pixel-level O/X loss from raw AIS count pixels.

    The smoothed density target remains available for older losses and
    visualization, but this loss builds its main target from raw_count > 0.
    """

    def __init__(self, config: PixelBinaryOccupancyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or PixelBinaryOccupancyLossConfig()
        self.last_components: dict[str, float] = {}

    def pixel_target(self, batch: dict[str, Any]) -> torch.Tensor:
        raw_count = batch.get("raw_count")
        if raw_count is None:
            raw_count = batch["target"]
        return (raw_count > float(self.config.min_raw_count_for_positive)).to(dtype=batch["valid_mask"].dtype) * batch["valid_mask"]

    def target_count(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.pixel_target(batch).flatten(1).sum(dim=1)

    def occupancy_target(self, batch: dict[str, Any]) -> torch.Tensor:
        return (self.target_count(batch) > 0).to(dtype=batch["valid_mask"].dtype)

    def target_density_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.pixel_target(batch)

    def density_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            return torch.sigmoid(output) * batch["valid_mask"]
        if "pixel_logits" not in output:
            raise ValueError("pixel_binary_occupancy_loss requires pixel_logits")
        return torch.sigmoid(output["pixel_logits"]) * batch["valid_mask"]

    def occupancy_from_output(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(output):
            pixel_prob = torch.sigmoid(output) * batch["valid_mask"]
            # Noisy fallback: patch probability is max pixel probability.
            pred_prob = pixel_prob.flatten(1).max(dim=1).values
        else:
            if "occupancy_logit" in output:
                pred_prob = torch.sigmoid(_occupancy_logit_from_output(output))
            elif "pixel_logits" in output:
                pred_prob = (torch.sigmoid(output["pixel_logits"]) * batch["valid_mask"]).flatten(1).max(dim=1).values
            else:
                raise ValueError("pixel_binary_occupancy_loss requires occupancy_logit or pixel_logits")
        return pred_prob, self.occupancy_target(batch)

    def pixel_occupancy_from_output(
        self,
        output: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.density_from_output(output, batch)
        target = self.pixel_target(batch)
        valid = batch["valid_mask"]
        return pred, target, valid

    def _soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float) -> torch.Tensor:
        prob = torch.sigmoid(logits) * valid_mask
        target = target * valid_mask
        intersection = (prob * target).flatten(1).sum(dim=1)
        denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
        dice = (2.0 * intersection + float(eps)) / torch.clamp(denom + float(eps), min=float(eps))
        return 1.0 - dice.mean()

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(output):
            pixel_logits = output
            occupancy_logit = None
        else:
            if "pixel_logits" not in output:
                raise ValueError("pixel_binary_occupancy_loss requires pixel_logits")
            pixel_logits = output["pixel_logits"]
            occupancy_logit = _occupancy_logit_from_output(output) if "occupancy_logit" in output or "occupancy_logits" in output else None

        cfg = self.config
        valid_mask = batch["valid_mask"]
        target = self.pixel_target(batch)
        sample_weight = _lifetime_sample_weight(
            batch,
            mode=cfg.lifetime_weight_mode,
            strength=float(cfg.lifetime_weight_strength),
            min_weight=float(cfg.lifetime_weight_min),
            max_weight=float(cfg.lifetime_weight_max),
            normalize=bool(cfg.lifetime_weight_normalize),
            eps=float(cfg.eps),
            reference=target,
        ).reshape(-1, 1, 1, 1)
        weight = valid_mask * sample_weight
        pos_weight = pixel_logits.new_tensor(float(cfg.pixel_pos_weight))
        pixel_element = F.binary_cross_entropy_with_logits(
            pixel_logits,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )
        pixel = (pixel_element * weight).sum() / torch.clamp(weight.sum(), min=float(cfg.eps))
        if float(cfg.dice_weight) > 0.0:
            dice = self._soft_dice_loss(pixel_logits, target, weight, float(cfg.eps))
        else:
            dice = pixel.new_tensor(0.0)

        occupancy = pixel.new_tensor(0.0)
        occupancy_target = self.occupancy_target(batch)
        if occupancy_logit is not None and float(cfg.occupancy_weight) > 0.0:
            patch_weight = sample_weight.reshape(-1)
            occ_element = F.binary_cross_entropy_with_logits(occupancy_logit, occupancy_target, reduction="none")
            if float(cfg.occupancy_pos_weight) != 1.0:
                class_weight = torch.where(
                    occupancy_target > 0,
                    occupancy_target.new_tensor(float(cfg.occupancy_pos_weight)),
                    occupancy_target.new_tensor(1.0),
                )
                occ_element = occ_element * class_weight
            occupancy = (occ_element * patch_weight).sum() / torch.clamp(patch_weight.sum(), min=float(cfg.eps))

        total = float(cfg.pixel_weight) * pixel + float(cfg.occupancy_weight) * occupancy + float(cfg.dice_weight) * dice
        if bool(cfg.report_components):
            pred_prob = torch.sigmoid(pixel_logits)
            positive_pixels = target > 0.5
            valid_pixels = valid_mask > 0
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_pixel_bce": float(pixel.detach().cpu()),
                "loss_patch_occupancy": float(occupancy.detach().cpu()),
                "loss_pixel_dice": float(dice.detach().cpu()),
                "loss_weight_pixel": float(cfg.pixel_weight),
                "loss_weight_patch_occupancy": float(cfg.occupancy_weight),
                "loss_weight_dice": float(cfg.dice_weight),
                "pixel_pos_weight": float(cfg.pixel_pos_weight),
                "pixel_target_positive_fraction": float((positive_pixels & valid_pixels).to(dtype=target.dtype).sum().detach().cpu() / torch.clamp(valid_mask.sum().detach().cpu(), min=1.0)),
                "pixel_pred_mean": float((pred_prob * valid_mask).sum().detach().cpu() / torch.clamp(valid_mask.sum().detach().cpu(), min=1.0)),
                "patch_occupancy_target_mean": float(occupancy_target.detach().mean().cpu()),
                "target_pixel_count_mean": float(self.target_count(batch).detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
        return total


def build_density_loss(
    config: dict[str, Any] | DensityLossConfig | CountSpatialLossConfig | OccupancySpatialLossConfig | OccupancyOnlyLossConfig | SpatialOnlyLossConfig | PixelBinaryOccupancyLossConfig | None = None,
) -> nn.Module:
    if isinstance(config, PixelBinaryOccupancyLossConfig):
        return PixelBinaryOccupancyLoss(config)
    if isinstance(config, SpatialOnlyLossConfig):
        return SpatialOnlyLoss(config)
    if isinstance(config, OccupancyOnlyLossConfig):
        return OccupancyOnlyLoss(config)
    if isinstance(config, OccupancySpatialLossConfig):
        return OccupancySpatialLoss(config)
    if isinstance(config, CountSpatialLossConfig):
        return CountSpatialDensityLoss(config)
    if isinstance(config, DensityLossConfig):
        return StructuredDensityLoss(config)
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "pixel_binary_occupancy_loss",
        "pixel_binary_occupancy",
        "pixel_occupancy_loss",
        "pixel_occupancy",
        "pixel_ox",
        "ship_pixel_presence",
    }:
        return PixelBinaryOccupancyLoss(pixel_binary_occupancy_loss_config_from_dict(config))
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "spatial_only_loss",
        "spatial_only",
        "distribution_only",
        "ship_spatial",
    }:
        return SpatialOnlyLoss(spatial_only_loss_config_from_dict(config))
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "occupancy_only_loss",
        "occupancy_only",
        "ship_ox",
        "ship_presence",
    }:
        return OccupancyOnlyLoss(occupancy_only_loss_config_from_dict(config))
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "occupancy_spatial_loss",
        "occupancy_spatial",
        "ship_ox_spatial",
        "ship_presence_spatial",
    }:
        return OccupancySpatialLoss(occupancy_spatial_loss_config_from_dict(config))
    if isinstance(config, dict) and str(config.get("name", "structured_density_loss")).lower() in {
        "count_spatial_density_loss",
        "count_spatial",
        "count_spatial_loss",
    }:
        return CountSpatialDensityLoss(count_spatial_loss_config_from_dict(config))
    return StructuredDensityLoss(density_loss_config_from_dict(config))
