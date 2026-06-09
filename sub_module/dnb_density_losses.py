from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PixelBinaryOccupancyLossConfig:
    name: str = "pixel_binary_occupancy_loss"
    target_mode: str = "hard"
    pixel_loss_type: str = "bce"
    pixel_weight: float = 0.85
    occupancy_weight: float = 0.15
    pixel_pos_weight: float = 50.0
    field_weight_strength: float = 0.0
    smooth_l1_beta: float = 0.1
    occupancy_pos_weight: float = 1.0
    dice_weight: float = 0.0
    min_raw_count_for_positive: float = 0.0
    radius_probability_sigma_pixels: float = 4.0
    radius_probability_radius_pixels: int = 0
    radius_probability_truncate: float = 3.0
    probability_target_threshold: float = 0.25
    lifetime_weight_mode: str = "none"
    lifetime_weight_strength: float = 0.0
    lifetime_weight_min: float = 0.25
    lifetime_weight_max: float = 2.0
    lifetime_weight_normalize: bool = True
    eps: float = 1.0e-8
    report_components: bool = True


def pixel_binary_occupancy_loss_config_from_dict(config: dict[str, Any] | None) -> PixelBinaryOccupancyLossConfig:
    if not config:
        return PixelBinaryOccupancyLossConfig()
    allowed = {item.name for item in fields(PixelBinaryOccupancyLossConfig)}
    values = {key: value for key, value in dict(config).items() if key in allowed}
    return PixelBinaryOccupancyLossConfig(**values)


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


def _occupancy_logit_from_output(output: dict[str, torch.Tensor]) -> torch.Tensor:
    if "occupancy_logit" in output:
        return output["occupancy_logit"].reshape(-1)
    if "occupancy_logits" in output:
        return output["occupancy_logits"].reshape(-1)
    raise ValueError("Pixel-binary model output must contain occupancy_logit")


class PixelBinaryOccupancyLoss(nn.Module):
    """Pixel probability-field loss from raw AIS seed pixels.

    In hard mode the target is exact raw_count > threshold. In probability
    mode the exact AIS pixels are only seeds for a Gaussian 0..1 proximity
    field; ship count mass is not spread or conserved as a density label.
    """

    def __init__(self, config: PixelBinaryOccupancyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or PixelBinaryOccupancyLossConfig()
        self.last_components: dict[str, float] = {}
        self.pixel_metric_target_threshold = float(self.config.probability_target_threshold)
        mode = str(self.config.target_mode).strip().lower()
        self.target_display_name = (
            "target radius probability field"
            if mode in {"radius_probability", "probability_field", "gaussian_radius", "proximity"}
            else "target hard pixel occupancy"
        )

    def source_pixel_target_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        raw_count = batch.get("raw_count")
        if raw_count is None:
            raw_count = batch["target"]
        return (raw_count > float(self.config.min_raw_count_for_positive)).to(dtype=batch["valid_mask"].dtype) * batch["valid_mask"]

    def _radius_probability_target(self, source: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        sigma = float(cfg.radius_probability_sigma_pixels)
        radius = int(cfg.radius_probability_radius_pixels)
        if radius <= 0:
            radius = int(math.ceil(max(sigma, 0.0) * max(float(cfg.radius_probability_truncate), 0.0)))
        source = (source > 0.5).to(dtype=valid_mask.dtype) * valid_mask
        if radius <= 0 or sigma <= 0.0:
            return source
        target = torch.zeros_like(source)
        height = int(source.shape[-2])
        width = int(source.shape[-1])
        sigma = max(sigma, 1.0e-6)
        for dy in range(-radius, radius + 1):
            src_r0 = max(0, -dy)
            src_r1 = height - max(0, dy)
            dst_r0 = max(0, dy)
            dst_r1 = height - max(0, -dy)
            if src_r0 >= src_r1:
                continue
            for dx in range(-radius, radius + 1):
                src_c0 = max(0, -dx)
                src_c1 = width - max(0, dx)
                dst_c0 = max(0, dx)
                dst_c1 = width - max(0, -dx)
                if src_c0 >= src_c1:
                    continue
                weight = math.exp(-0.5 * ((dy / sigma) ** 2 + (dx / sigma) ** 2))
                if weight <= float(cfg.eps):
                    continue
                shifted = source[..., src_r0:src_r1, src_c0:src_c1] * source.new_tensor(float(weight))
                current = target[..., dst_r0:dst_r1, dst_c0:dst_c1]
                target[..., dst_r0:dst_r1, dst_c0:dst_c1] = torch.maximum(current, shifted)
        return torch.clamp(target * valid_mask, min=0.0, max=1.0)

    def pixel_target(self, batch: dict[str, Any]) -> torch.Tensor:
        source = self.source_pixel_target_from_batch(batch)
        mode = str(self.config.target_mode).strip().lower()
        if mode in {"hard", "exact", "binary", "pixel_binary"}:
            return source
        if mode in {"radius_probability", "probability_field", "gaussian_radius", "proximity"}:
            return self._radius_probability_target(source, batch["valid_mask"])
        raise ValueError(f"Unsupported pixel target mode: {self.config.target_mode}")

    def target_count(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.source_pixel_target_from_batch(batch).flatten(1).sum(dim=1)

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
            pred_prob = pixel_prob.flatten(1).max(dim=1).values
        else:
            if "occupancy_logit" in output or "occupancy_logits" in output:
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
        return self.density_from_output(output, batch), self.pixel_target(batch), batch["valid_mask"]

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
        pixel_loss_type = str(cfg.pixel_loss_type).strip().lower()
        pixel_weight_map = weight
        if pixel_loss_type in {"bce", "binary_cross_entropy", "binary_cross_entropy_with_logits"}:
            pixel_element = F.binary_cross_entropy_with_logits(
                pixel_logits,
                target,
                reduction="none",
                pos_weight=pixel_logits.new_tensor(float(cfg.pixel_pos_weight)),
            )
        elif pixel_loss_type in {"smooth_l1", "weighted_smooth_l1", "huber", "field_smooth_l1"}:
            pred_prob_for_loss = torch.sigmoid(pixel_logits)
            pixel_element = F.smooth_l1_loss(
                pred_prob_for_loss,
                target,
                reduction="none",
                beta=max(float(cfg.smooth_l1_beta), float(cfg.eps)),
            )
            if float(cfg.field_weight_strength) > 0.0:
                pixel_weight_map = weight * (1.0 + float(cfg.field_weight_strength) * torch.clamp(target, min=0.0, max=1.0))
        else:
            raise ValueError(f"Unsupported pixel_loss_type: {cfg.pixel_loss_type}")
        pixel = (pixel_element * pixel_weight_map).sum() / torch.clamp(pixel_weight_map.sum(), min=float(cfg.eps))
        dice = self._soft_dice_loss(pixel_logits, target, weight, float(cfg.eps)) if float(cfg.dice_weight) > 0.0 else pixel.new_tensor(0.0)

        occupancy = pixel.new_tensor(0.0)
        occupancy_target = None
        if occupancy_logit is not None and float(cfg.occupancy_weight) > 0.0:
            occupancy_target = self.occupancy_target(batch)
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
            valid_pixels = valid_mask > 0
            valid_count = torch.clamp(valid_mask.sum().detach().cpu(), min=1.0)
            source_target = self.source_pixel_target_from_batch(batch)
            metric_positive = target >= float(cfg.probability_target_threshold)
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_pixel_field": float(pixel.detach().cpu()),
                "loss_pixel_dice": float(dice.detach().cpu()),
                "pixel_loss_type": 1.0 if pixel_loss_type in {"smooth_l1", "weighted_smooth_l1", "huber", "field_smooth_l1"} else 0.0,
                "field_weight_strength": float(cfg.field_weight_strength),
                "field_weight_mean": float((pixel_weight_map * valid_mask).sum().detach().cpu() / valid_count),
                "smooth_l1_beta": float(cfg.smooth_l1_beta),
                "loss_weight_pixel": float(cfg.pixel_weight),
                "loss_weight_dice": float(cfg.dice_weight),
                "target_mode": 1.0 if str(cfg.target_mode).strip().lower() in {"radius_probability", "probability_field", "gaussian_radius", "proximity"} else 0.0,
                "radius_probability_sigma_pixels": float(cfg.radius_probability_sigma_pixels),
                "radius_probability_radius_pixels": float(cfg.radius_probability_radius_pixels),
                "probability_target_threshold": float(cfg.probability_target_threshold),
                "pixel_pos_weight": float(cfg.pixel_pos_weight),
                "pixel_target_positive_fraction": float((metric_positive & valid_pixels).to(dtype=target.dtype).sum().detach().cpu() / valid_count),
                "source_pixel_positive_fraction": float((source_target > 0.5).to(dtype=target.dtype).sum().detach().cpu() / valid_count),
                "probability_target_mean": float((target * valid_mask).sum().detach().cpu() / valid_count),
                "pixel_pred_mean": float((pred_prob * valid_mask).sum().detach().cpu() / valid_count),
                "target_pixel_count_mean": float(self.target_count(batch).detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
            if float(cfg.occupancy_weight) > 0.0:
                if occupancy_target is None:
                    occupancy_target = self.occupancy_target(batch)
                self.last_components.update(
                    {
                        "loss_patch_occupancy": float(occupancy.detach().cpu()),
                        "loss_weight_patch_occupancy": float(cfg.occupancy_weight),
                        "patch_occupancy_target_mean": float(occupancy_target.detach().mean().cpu()),
                    }
                )
        return total


def build_density_loss(config: dict[str, Any] | PixelBinaryOccupancyLossConfig | None = None) -> nn.Module:
    if isinstance(config, PixelBinaryOccupancyLossConfig):
        return PixelBinaryOccupancyLoss(config)
    if config is None:
        return PixelBinaryOccupancyLoss(PixelBinaryOccupancyLossConfig())
    if isinstance(config, dict):
        normalized = str(config.get("name", "pixel_binary_occupancy_loss")).lower()
        if normalized in {
            "pixel_binary_occupancy_loss",
            "radius_probability_loss",
            "weighted_smooth_l1_probability_loss",
            "smooth_l1_probability_loss",
            "probability_field_smooth_l1_loss",
            "probability_field_loss",
            "radius_probability_occupancy_loss",
            "probability_field_occupancy_loss",
            "probability_field_occupancy",
            "radius_probability_occupancy",
            "pixel_binary_occupancy",
            "pixel_occupancy_loss",
            "pixel_occupancy",
            "pixel_ox",
            "ship_pixel_presence",
        }:
            return PixelBinaryOccupancyLoss(pixel_binary_occupancy_loss_config_from_dict(config))
        raise ValueError(
            f"Retired density loss {normalized!r} was archived under "
            "_archive/retired_density_complexity_20260609/sub_module/dnb_density_losses_legacy_full.py"
        )
    raise TypeError(f"Unsupported density loss config type: {type(config).__name__}")
