"""Frozen backbones and the preprocessing helper shared by both streams."""

from __future__ import annotations

import torch

from ..config import PreprocessConfig
from .context_stream import ContextStream, assert_patch_grid
from .texture_stream import TextureStream

__all__ = [
    "ContextStream",
    "TextureStream",
    "assert_patch_grid",
    "normalize_batch",
]


def normalize_batch(
    images: torch.Tensor, config: PreprocessConfig
) -> torch.Tensor:
    """Apply the paper's preprocessing to a batch of raw [0, 255] images.

    Section 2.1: force RGB, resize to 224x224 with bilinear interpolation, then
    normalize with ImageNet statistics. Resizing is a no-op when the batch is
    already 224x224.
    """
    import torch.nn.functional as F

    x = images.float()
    if x.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
    if config.to_rgb and x.shape[1] != 3:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"cannot convert {x.shape[1]} channels to RGB")
    if x.max() > 2.0:  # values still on the [0, 255] scale
        x = x / 255.0
    size = config.image_size
    if x.shape[-2:] != (size, size):
        x = F.interpolate(
            x, size=(size, size), mode=config.interpolation, align_corners=False
        )
    mean = torch.tensor(config.mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
