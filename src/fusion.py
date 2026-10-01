"""Dual-Stream Synergy module (paper Section 2.3, Eqs. 4-5).

Simply overlaying raw scores from the two streams fails because the streams
produce scores on different numerical scales. The DS-S module therefore

1. Min-Max normalizes each stream's scores to [0, 1] over the test batch::

       S_hat_k = (S_k - min S_k) / (max S_k - min S_k),  k in {tex, ctx}   (Eq. 4)

2. Takes a weighted linear combination::

       S_final = lambda * S_hat_tex + (1 - lambda) * S_hat_ctx              (Eq. 5)

with ``lambda = 0.5``, so that local texture evidence and global structural
evidence contribute equally. The paper describes the result as a "dual
validation": agreement between the streams raises confidence, while texture
findings that the context stream considers normal (e.g. background) are damped,
reducing false positives.
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import FusionConfig


def min_max_normalize(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-Max rescale ``scores`` to [0, 1] (Eq. 4).

    A degenerate batch (all values identical) maps to zeros rather than
    producing NaNs.
    """
    if scores.numel() == 0:
        return scores.clone()
    lo = scores.min()
    hi = scores.max()
    span = hi - lo
    if float(span) < eps:
        return torch.zeros_like(scores)
    return (scores - lo) / span


class DualStreamSynergy(torch.nn.Module):
    """Fuses normalized texture and context anomaly scores."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        self.config = config or FusionConfig()

    def forward(
        self,
        texture_scores: torch.Tensor,
        context_scores: torch.Tensor,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """Return the fused image-level score ``S_final`` (Eq. 5).

        Parameters
        ----------
        texture_scores, context_scores:
            ``(B,)`` per-image anomaly scores from the two streams.
        lam:
            Optional override of the balancing parameter; defaults to
            ``config.lam`` (0.5 in the paper).
        """
        if texture_scores.shape != context_scores.shape:
            raise ValueError(
                "the two streams must score the same number of images: "
                f"{tuple(texture_scores.shape)} vs {tuple(context_scores.shape)}"
            )
        weight = self.config.lam if lam is None else float(lam)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"lambda must lie in [0, 1], got {weight}")

        tex = min_max_normalize(texture_scores, self.config.eps)
        ctx = min_max_normalize(context_scores, self.config.eps)
        return weight * tex + (1.0 - weight) * ctx

    def contributions(
        self, texture_scores: torch.Tensor, context_scores: torch.Tensor
    ) -> dict:
        """Return the per-stream and fused scores for inspection/plots."""
        tex = min_max_normalize(texture_scores, self.config.eps)
        ctx = min_max_normalize(context_scores, self.config.eps)
        return {
            "texture_normalized": tex,
            "context_normalized": ctx,
            "fused": self.config.lam * tex + (1.0 - self.config.lam) * ctx,
        }


def sweep_lambda(
    texture_scores: torch.Tensor,
    context_scores: torch.Tensor,
    labels: torch.Tensor,
    lambdas: Optional[torch.Tensor] = None,
) -> dict:
    """Reproduce the validation-set search that yields ``lambda = 0.5``.

    Returns a dict with the swept ``lambdas`` and the corresponding image-level
    AUC values, mirroring the model-selection step of Section 2.3.
    """
    from .metrics import image_level_auc

    if lambdas is None:
        lambdas = torch.linspace(0.0, 1.0, 21)
    tex = min_max_normalize(texture_scores)
    ctx = min_max_normalize(context_scores)
    aucs = [
        image_level_auc((float(w) * tex + (1.0 - float(w)) * ctx), labels)
        for w in lambdas
    ]
    return {
        "lambdas": [float(w) for w in lambdas],
        "auc": aucs,
        "best_lambda": float(lambdas[int(torch.tensor(aucs).argmax())]),
    }
