"""Anatomy-guided background suppression (paper Section 2.3, Eq. 3).

The physical justification is X-ray attenuation: air outside the body absorbs
little radiation, so pixels outside the patient are dark. Splitting the image
into patches ``P_i`` and masking patches whose mean optical density
``mu(P_i)`` does not exceed ``tau`` removes non-anatomical sensor noise, so the
anomaly score depends only on lung-tissue pathology.

    M_i = 1  if mu(P_i) > tau,   else 0                            (Eq. 3)

The paper fixes the empirical threshold at ``tau = 30`` on the [0, 255]
intensity scale.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import AnatomicalPriorConfig


def optical_density(
    images: torch.Tensor, grid: Tuple[int, int] = (14, 14)
) -> torch.Tensor:
    """Mean intensity of every patch in ``grid``.

    Parameters
    ----------
    images:
        ``(B, C, H, W)`` tensor. The paper's threshold ``tau = 30`` is defined on
        the [0, 255] intensity scale, so pass raw intensities (before ImageNet
        normalization) for Eq. 3 to carry its physical meaning.
    grid:
        Patch grid, ``(14, 14)`` giving 196 patches for a 224x224 input.

    Returns
    -------
    ``(B, N)`` tensor of per-patch mean optical densities.
    """
    if images.dim() != 4:
        raise ValueError(f"expected a 4-D image tensor, got {tuple(images.shape)}")
    pooled = F.adaptive_avg_pool2d(images, output_size=grid)
    if pooled.shape[1] > 1:  # average the colour channels
        pooled = pooled.mean(dim=1, keepdim=True)
    return pooled.flatten(1)


class AnatomicalPriorFilter(torch.nn.Module):
    """Builds the binary ROI mask ``M`` and applies it to a patch embedding."""

    def __init__(self, config: Optional[AnatomicalPriorConfig] = None) -> None:
        super().__init__()
        self.config = config or AnatomicalPriorConfig()

    def mask(
        self, images: torch.Tensor, grid: Tuple[int, int] = (14, 14)
    ) -> torch.Tensor:
        """Return the boolean ROI mask ``M`` with shape ``(B, N)``."""
        mu = optical_density(images, grid=grid)
        mask = mu > self.config.tau
        if self.config.mask_dilation > 0:
            mask = self._dilate(mask, grid, self.config.mask_dilation)
        return mask

    def _dilate(
        self, mask: torch.Tensor, grid: Tuple[int, int], iterations: int
    ) -> torch.Tensor:
        """Binary dilation on the patch grid, keeping the mask non-empty."""
        b, n = mask.shape
        m = mask.view(b, 1, grid[0], grid[1]).float()
        for _ in range(iterations):
            m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        dilated = m.view(b, n) > 0.5
        # Never let dilation destroy a sample: fall back to the original mask.
        empty = dilated.sum(dim=1) == 0
        if empty.any():
            dilated[empty] = mask[empty]
        return dilated

    def forward(
        self,
        embeddings: torch.Tensor,
        images: torch.Tensor,
        grid: Tuple[int, int] = (14, 14),
        fill_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Suppress background patches.

        Parameters
        ----------
        embeddings:
            ``(B, N, D)`` patch descriptors from the context stream.
        images:
            ``(B, C, H, W)`` images used only to compute the optical densities.

        Returns
        -------
        ``(filtered, mask)`` where ``filtered`` has background patch rows zeroed
        and ``mask`` is the ``(B, N)`` boolean ROI mask.
        """
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be (B, N, D)")
        mask = self.mask(images, grid=grid)
        if mask.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"mask has {mask.shape[1]} patches but embeddings have "
                f"{embeddings.shape[1]}"
            )
        filtered = embeddings.masked_fill(~mask.unsqueeze(-1), fill_value)
        return filtered, mask
