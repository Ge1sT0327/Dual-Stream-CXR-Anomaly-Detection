"""k-NN anomaly scoring (paper Section 3.1).

The paper uses scikit-learn's k-NN with ``K = 50`` and Euclidean distance on
frozen embeddings. The reference implementation below is PatchCore-style: the
image/patch anomaly score is the distance to the k-th nearest neighbour in the
normal training memory bank, which is the variant that matches the reported
numbers more closely than a plain mean-of-k-distances.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .config import KNNConfig


class KNNAnomalyDetector:
    """Memory-bank nearest-neighbour anomaly detector.

    Parameters
    ----------
    config:
        ``KNNConfig`` holding ``n_neighbors=50`` and ``metric="euclidean"``.
    """

    def __init__(self, config: Optional[KNNConfig] = None) -> None:
        self.config = config or KNNConfig()
        self._bank: Optional[torch.Tensor] = None
        self._bank_norm: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ fit

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "KNNAnomalyDetector":
        """Store the normal-training memory bank.

        Parameters
        ----------
        features:
            ``(M, D)`` tensor of embeddings from the 1,341 normal training
            radiographs. Multi-dimensional inputs are flattened to ``(M, D)``.
        """
        if features.dim() > 2:
            features = features.reshape(-1, features.shape[-1])
        if features.dim() != 2:
            raise ValueError("features must be (M, D) or (..., D)")
        if features.shape[0] < self.config.n_neighbors:
            raise ValueError(
                f"memory bank holds {features.shape[0]} vectors, fewer than "
                f"K = {self.config.n_neighbors}"
            )
        self._bank = features.detach().float().contiguous()
        self._bank_norm = (self._bank**2).sum(dim=1)
        return self

    # -------------------------------------------------------------- scoring

    @property
    def is_fitted(self) -> bool:
        return self._bank is not None

    def _require_fitted(self) -> torch.Tensor:
        if self._bank is None:
            raise RuntimeError("call fit() before scoring")
        return self._bank

    @torch.no_grad()
    def score(self, queries: torch.Tensor) -> torch.Tensor:
        """Distance to the k-th nearest neighbour for every query vector.

        Parameters
        ----------
        queries:
            ``(Q, D)`` query embeddings.

        Returns
        -------
        ``(Q,)`` tensor of anomaly scores. Higher means more anomalous.
        """
        bank = self._require_fitted()
        if queries.dim() != 2:
            raise ValueError("queries must be (Q, D)")
        if queries.shape[1] != bank.shape[1]:
            raise ValueError(
                f"dimension mismatch: bank D={bank.shape[1]}, queries D={queries.shape[1]}"
            )
        if self.config.metric != "euclidean":
            raise NotImplementedError(
                f"only euclidean is reported in the paper, got {self.config.metric}"
            )

        query = queries.detach().float()
        # Squared Euclidean distances via the expansion ||q||^2 - 2 q.b + ||b||^2.
        dist = (
            (query**2).sum(dim=1, keepdim=True)
            - 2.0 * query @ bank.t()
            + self._bank_norm.unsqueeze(0)
        ).clamp_min_(0.0)
        k = min(self.config.n_neighbors, bank.shape[0])
        knn = torch.topk(dist, k=k, dim=1, largest=False).values
        if self.config.use_kth_distance:
            return knn[:, -1].sqrt()
        return knn.mean(dim=1).sqrt()

    @torch.no_grad()
    def score_map(
        self, patch_features: torch.Tensor, grid: Tuple[int, int] = (14, 14)
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Patch-level scoring used for the localization maps of Figure 2.

        Parameters
        ----------
        patch_features:
            ``(B, N, D)`` patch embeddings of one batch.

        Returns
        -------
        ``(image_scores, patch_scores)`` where ``image_scores`` is ``(B,)`` and
        ``patch_scores`` is a NumPy array of shape ``(B, grid[0], grid[1])``.
        """
        if patch_features.dim() != 3:
            raise ValueError("patch_features must be (B, N, D)")
        b, n, d = patch_features.shape
        scores = self.score(patch_features.reshape(b * n, d))
        maps = scores.reshape(b, grid[0], grid[1]).cpu().numpy()
        image_scores = scores.reshape(b, n).max(dim=1).values
        return image_scores, maps
