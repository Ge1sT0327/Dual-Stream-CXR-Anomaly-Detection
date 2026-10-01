"""Texture stream: ResNet50 mid-level feature fusion.

Implements the top (blue) branch of Figure 1 and Eq. 1 of the paper:

    F_tex(x) = Concat( phi2(x), Upsample(phi3(x)) )                (Eq. 1)

where ``phi_j(x)`` is the feature map produced by the j-th residual stage.
Only the middle stages (Layer 2, Layer 3) are used, because Layer 1 is too
low-level and Layer 4 has lost too much spatial resolution.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TextureStreamConfig


class TextureStream(nn.Module):
    """Frozen ResNet50 encoder returning multi-scale local texture features."""

    def __init__(self, config: Optional[TextureStreamConfig] = None) -> None:
        super().__init__()
        self.config = config or TextureStreamConfig()
        self.feature_layers: Tuple[int, ...] = tuple(self.config.feature_layers)
        if self.feature_layers != (2, 3):
            # The paper fixes Layer 2 + Layer 3; anything else is a variant.
            if len(self.feature_layers) != 2:
                raise ValueError("Texture stream expects exactly two stages")

        self.backbone = self._build_backbone()
        if self.config.frozen:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

        #: Channel counts of the stock ResNet50 stages, needed by downstream
        #: consumers that allocate buffers for F_tex.
        self.stage_channels: Dict[int, int] = {1: 256, 2: 512, 3: 1024, 4: 2048}
        self.out_channels: int = sum(self.stage_channels[j] for j in self.feature_layers)

    # ------------------------------------------------------------------ build

    def _build_backbone(self) -> nn.Module:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if self.config.pretraining == "imagenet"
            else None
        )
        model = resnet50(
            weights=weights,
            replace_stride_with_dilation=(
                [False, False, False]
                if not self.config.replace_stride_with_dilation
                else [False, False, True]
            ),
        )
        # A plain Sequential over the stem and the four residual stages makes
        # stage indexing unambiguous and avoids relying on attribute names.
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    # --------------------------------------------------------------- forward

    def stage_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Return the output of every residual stage as a dict ``{j: tensor}``."""
        out: Dict[int, torch.Tensor] = {}
        h = x
        for idx, module in enumerate(self.backbone):
            h = module(h)
            # Modules 0..3 are stem (conv/bn/relu/maxpool); stage j is at 3 + j.
            if idx >= 4:
                out[idx - 3] = h
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``F_tex(x)`` with shape ``(B, 1536, 28, 28)`` for 224x224 input."""
        stages = self.stage_features(x)
        chosen = [stages[j] for j in self.feature_layers]

        # Eq. 1: the deepest selected stage is bilinearly upsampled to the
        # spatial size of the shallowest one, then concatenated on channels.
        target_hw = chosen[0].shape[-2:]
        upsampled = [
            feat if feat.shape[-2:] == target_hw else _upsample(feat, target_hw)
            for feat in chosen
        ]
        return torch.cat(upsampled, dim=1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-mode wrapper used by the pipeline."""
        self.eval()
        return self.forward(x)


def _upsample(feat: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinear upsampling that matches the paper's wording for Eq. 1."""
    return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
