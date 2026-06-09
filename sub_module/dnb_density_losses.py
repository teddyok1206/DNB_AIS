from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


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
    """Hard pixel-level O/X loss from raw AIS count pixels.

    The smoothed density target can still be used by preview/diagnostic code,
    but the supervised label here is raw_count > min_raw_count_for_positive.
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
        pixel_element = F.binary_cross_entropy_with_logits(
            pixel_logits,
            target,
            reduction="none",
            pos_weight=pixel_logits.new_tensor(float(cfg.pixel_pos_weight)),
        )
        pixel = (pixel_element * weight).sum() / torch.clamp(weight.sum(), min=float(cfg.eps))
        dice = self._soft_dice_loss(pixel_logits, target, weight, float(cfg.eps)) if float(cfg.dice_weight) > 0.0 else pixel.new_tensor(0.0)

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
            valid_count = torch.clamp(valid_mask.sum().detach().cpu(), min=1.0)
            self.last_components = {
                "loss_total": float(total.detach().cpu()),
                "loss_pixel_bce": float(pixel.detach().cpu()),
                "loss_patch_occupancy": float(occupancy.detach().cpu()),
                "loss_pixel_dice": float(dice.detach().cpu()),
                "loss_weight_pixel": float(cfg.pixel_weight),
                "loss_weight_patch_occupancy": float(cfg.occupancy_weight),
                "loss_weight_dice": float(cfg.dice_weight),
                "pixel_pos_weight": float(cfg.pixel_pos_weight),
                "pixel_target_positive_fraction": float((positive_pixels & valid_pixels).to(dtype=target.dtype).sum().detach().cpu() / valid_count),
                "pixel_pred_mean": float((pred_prob * valid_mask).sum().detach().cpu() / valid_count),
                "patch_occupancy_target_mean": float(occupancy_target.detach().mean().cpu()),
                "target_pixel_count_mean": float(self.target_count(batch).detach().mean().cpu()),
                "lifetime_weight_mode": 0.0 if str(cfg.lifetime_weight_mode).lower() in {"none", "off", "false", "0"} else 1.0,
                "lifetime_weight_strength": float(cfg.lifetime_weight_strength),
                "lifetime_weight_mean": float(sample_weight.detach().mean().cpu()),
                "lifetime_weight_max": float(sample_weight.detach().max().cpu()),
            }
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
