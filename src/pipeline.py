"""End-to-end inference pipeline for the dual-stream framework.

Orchestrates the four stages of Figure 1:

1. **Texture stream** -- ResNet50 Layer 2 + Layer 3, Eq. 1, scored with k-NN.
2. **Context stream** -- DINOv3 ViT-L Layer 6 + Layer 12 + Layer 18 (MLFA),
   Eq. 2, background-suppressed by the anatomical prior filter, Eq. 3, scored
   with k-NN.
3. **Dual-Stream Synergy** -- Min-Max normalization (Eq. 4) and weighted linear
   fusion (Eq. 5).
4. **Outputs** -- an image-level diagnostic score and a localization heatmap.

Only forward inference is performed: both backbones stay frozen, there are no
gradient updates and no fine-tuning (paper Section 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .anatomical_prior import AnatomicalPriorFilter
from .config import DualStreamConfig
from .fusion import DualStreamSynergy, min_max_normalize
from .knn import KNNAnomalyDetector
from .metrics import anomaly_maps_from_embeddings, image_level_scores
from .streams import ContextStream, TextureStream

#: The patch grid used for every map and score (196 patches for 224x224 input).
PATCH_GRID: Tuple[int, int] = (14, 14)
NUM_PATCHES: int = PATCH_GRID[0] * PATCH_GRID[1]


@dataclass
class StreamOutput:
    """Everything produced by one stream for one batch."""

    #: ``(B,)`` per-image anomaly score.
    image_scores: torch.Tensor
    #: ``(B, N)`` per-patch anomaly scores on the 14x14 grid.
    patch_scores: torch.Tensor
    #: ``(B, 1, H, W)`` upsampled localization heatmap.
    heatmap: torch.Tensor


class DualStreamAnomalyDetector:
    """Zero-shot dual-stream anomaly detector.

    Parameters
    ----------
    config:
        Full configuration; defaults to the paper's settings.
    texture_stream, context_stream:
        Optional pre-built streams. Supplying them lets callers share the
        expensive frozen backbones across detector instances (useful for the
        ablation study, where the streams are toggled independently).
    """

    def __init__(
        self,
        config: Optional[DualStreamConfig] = None,
        texture_stream: Optional[TextureStream] = None,
        context_stream: Optional[ContextStream] = None,
    ) -> None:
        self.config = (config or DualStreamConfig()).validate()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self._texture_stream = texture_stream
        self._context_stream = context_stream

        self.prior_filter = AnatomicalPriorFilter(self.config.prior).to(self.device)
        self.synergy = DualStreamSynergy(self.config.fusion).to(self.device)

        self.texture_knn = KNNAnomalyDetector(self.config.knn)
        self.context_knn = KNNAnomalyDetector(self.config.knn)

    # ----------------------------------------------------------- lazy loaders

    @property
    def texture_stream(self) -> TextureStream:
        if self._texture_stream is None:
            self._texture_stream = TextureStream(self.config.texture).to(self.device)
            self._texture_stream.eval()
        return self._texture_stream

    @property
    def context_stream(self) -> ContextStream:
        if self._context_stream is None:
            self._context_stream = ContextStream(self.config.context).to(self.device)
            self._context_stream.eval()
        return self._context_stream

    @property
    def is_fitted(self) -> bool:
        return self.texture_knn.is_fitted and self.context_knn.is_fitted

    # ------------------------------------------------------------------- fit

    @torch.no_grad()
    def fit(
        self,
        normal_images: torch.Tensor,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> "DualStreamAnomalyDetector":
        """Build both memory banks from the 1,341 normal training radiographs.

        Parameters
        ----------
        normal_images:
            ``(M, C, H, W)`` normalized normal chest X-rays. The background mask
            is computed from ``raw_images`` when available; pass images that are
            still on the [0, 255] scale via :meth:`fit_raw` if you need exact
            Eq. 3 thresholds during fitting.
        """
        tex_bank: List[torch.Tensor] = []
        ctx_bank: List[torch.Tensor] = []

        for start in range(0, normal_images.shape[0], batch_size):
            batch = normal_images[start : start + batch_size].to(self.device)
            tex_bank.append(self._texture_patches(batch).cpu())
            ctx_bank.append(self._context_patches(batch, raw_batch=batch).cpu())
            if verbose:
                done = min(start + batch_size, normal_images.shape[0])
                print(f"  fitted {done}/{normal_images.shape[0]} normal images")

        self.texture_knn.fit(torch.cat(tex_bank, dim=0))
        self.context_knn.fit(torch.cat(ctx_bank, dim=0))
        return self

    @torch.no_grad()
    def fit_raw(
        self,
        raw_images: torch.Tensor,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> "DualStreamAnomalyDetector":
        """Like :meth:`fit` but takes raw [0, 255] images.

        The anatomical prior mask must be evaluated on the raw intensity scale
        for ``tau = 30`` to carry the physical meaning given in the paper, so
        this entry point normalizes internally after masking.
        """
        from .streams import normalize_batch

        tex_bank: List[torch.Tensor] = []
        ctx_bank: List[torch.Tensor] = []
        for start in range(0, raw_images.shape[0], batch_size):
            raw = raw_images[start : start + batch_size].to(self.device)
            norm = normalize_batch(raw, self.config.preprocess)
            tex_bank.append(self._texture_patches(norm).cpu())
            ctx_bank.append(self._context_patches(norm, raw_batch=raw).cpu())
            if verbose:
                done = min(start + batch_size, raw_images.shape[0])
                print(f"  fitted {done}/{raw_images.shape[0]} normal images")

        self.texture_knn.fit(torch.cat(tex_bank, dim=0))
        self.context_knn.fit(torch.cat(ctx_bank, dim=0))
        return self

    # --------------------------------------------------------------- scoring

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        raw_images: Optional[torch.Tensor] = None,
        batch_size: int = 8,
        return_per_stream: bool = True,
    ) -> Dict[str, object]:
        """Score a test set.

        Parameters
        ----------
        images:
            Normalized ``(T, C, H, W)`` test images.
        raw_images:
            Optional ``(T, C, H, W)`` images on the [0, 255] scale, used for the
            anatomical prior mask of Eq. 3. Because ``tau = 30`` is defined on the
            physical intensity scale, always pass raw images here; passing
            normalized images would make the threshold meaningless.

        Returns
        -------
        A dict with ``fused`` (``(T,)``), ``texture`` and ``context`` image
        scores, the per-patch score maps, and the upsampled heatmaps.
        """
        if not self.is_fitted:
            raise RuntimeError("call fit()/fit_raw() before score()")
        if raw_images is None:
            raw_images = images

        tex_scores: List[torch.Tensor] = []
        ctx_scores: List[torch.Tensor] = []
        tex_patch: List[torch.Tensor] = []
        ctx_patch: List[torch.Tensor] = []
        heatmaps: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        total = images.shape[0]
        for start in range(0, total, batch_size):
            stop = start + batch_size
            batch = images[start:stop].to(self.device)
            raw_batch = raw_images[start:stop].to(self.device)

            tex_feat = self._texture_patches(batch)
            ctx_feat, mask = self._context_patches(batch, raw_batch=raw_batch, return_mask=True)

            tex_s, tex_map = self.texture_knn.score_map(tex_feat, grid=PATCH_GRID)
            ctx_s, ctx_map = self.context_knn.score_map(ctx_feat, grid=PATCH_GRID)
            ctx_map_t = torch.as_tensor(ctx_map, device=self.device).flatten(1)

            tex_scores.append(tex_s.cpu())
            ctx_scores.append(ctx_s.cpu())
            tex_patch.append(torch.as_tensor(tex_map, device=self.device).flatten(1).cpu())
            ctx_patch.append(ctx_map_t.cpu())
            masks.append(mask.cpu())

            # Localization: the fused patch score is the same weighted
            # combination as Eq. 5, applied spatially instead of per image.
            lam = self.config.fusion.lam
            fused_map = lam * _patch_normalize(
                torch.as_tensor(tex_map, device=self.device).flatten(1)
            ) + (1.0 - lam) * _patch_normalize(ctx_map_t)
            heatmaps.append(
                anomaly_maps_from_embeddings(
                    fused_map.detach().cpu(),
                    grid=PATCH_GRID,
                    output_size=self.config.preprocess.image_size,
                    smooth_sigma=0.6,
                )
            )

        texture_scores = torch.cat(tex_scores)
        context_scores = torch.cat(ctx_scores)
        fused = self.synergy(texture_scores, context_scores)

        result: Dict[str, object] = {
            "fused": fused,
            "heatmaps": torch.cat(heatmaps, dim=0),
            "roi_mask": torch.cat(masks, dim=0),
        }
        if return_per_stream:
            result.update(
                {
                    "texture": texture_scores,
                    "context": context_scores,
                    "texture_patch_scores": torch.cat(tex_patch, dim=0),
                    "context_patch_scores": torch.cat(ctx_patch, dim=0),
                    "texture_normalized": min_max_normalize(texture_scores),
                    "context_normalized": min_max_normalize(context_scores),
                }
            )
        return result

    # ------------------------------------------------------------- ablations

    @torch.no_grad()
    def score_single_stream(
        self,
        images: torch.Tensor,
        stream: str,
        raw_images: Optional[torch.Tensor] = None,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Score using exactly one stream, for the Table 2 ablation study.

        ``stream="context"`` still applies the anatomical prior filter and MLFA,
        matching Experiment 3; use :meth:`score_vanilla_vit` for Experiment 1/2.
        """
        if stream not in {"texture", "context"}:
            raise ValueError("stream must be 'texture' or 'context'")
        raw_images = images if raw_images is None else raw_images
        scores: List[torch.Tensor] = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(self.device)
            raw_batch = raw_images[start : start + batch_size].to(self.device)
            if stream == "texture":
                feat = self._texture_patches(batch)
                knn = self.texture_knn
            else:
                feat = self._context_patches(batch, raw_batch=raw_batch)
                knn = self.context_knn
            patch = knn.score(feat.reshape(-1, feat.shape[-1]))
            scores.append(
                image_level_scores(patch.view(feat.shape[0], feat.shape[1])).cpu()
            )
        return torch.cat(scores)

    @torch.no_grad()
    def score_vanilla_vit(
        self,
        images: torch.Tensor,
        batch_size: int = 8,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Reproduce the Section 2.2 vanilla-ViT baseline (Layer 24, AUC 0.6552).

        Because the paper reports this baseline on the deep Layer-24 features
        only, this method bypasses MLFA and the anatomical prior filter.
        """
        layer = self.config.context.vanilla_layer if layer is None else layer
        tokens: List[torch.Tensor] = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(self.device)
            tokens.append(self.context_stream.patch_tokens(batch, layers=(layer,))[layer])
        all_tokens = torch.cat(tokens, dim=0)

        bank = getattr(self, "_vanilla_bank", None)
        if bank is None:
            raise RuntimeError(
                "call fit_vanilla_vit() on the normal training set before scoring"
            )
        detector = KNNAnomalyDetector(self.config.knn)
        detector._bank = bank  # noqa: SLF001 - intentional reuse of the bank
        detector._bank_norm = (bank**2).sum(dim=1)  # noqa: SLF001
        patch = detector.score(all_tokens.reshape(-1, all_tokens.shape[-1]))
        return image_level_scores(
            patch.view(all_tokens.shape[0], all_tokens.shape[1])
        ).cpu()

    @torch.no_grad()
    def fit_vanilla_vit(
        self, images: torch.Tensor, batch_size: int = 8, layer: Optional[int] = None
    ) -> "DualStreamAnomalyDetector":
        """Build the Layer-24 memory bank for :meth:`score_vanilla_vit`."""
        layer = self.config.context.vanilla_layer if layer is None else layer
        tokens: List[torch.Tensor] = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(self.device)
            tokens.append(
                self.context_stream.patch_tokens(batch, layers=(layer,))[layer].cpu()
            )
        all_tokens = torch.cat(tokens, dim=0)
        self._vanilla_bank = all_tokens.reshape(-1, all_tokens.shape[-1])
        return self

    # ------------------------------------------------------------- internals

    def _texture_patches(self, batch: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` texture descriptors aligned to the 14x14 patch grid."""
        feat = self.texture_stream.embed(batch)  # (B, 1536, 28, 28)
        pooled = F.adaptive_avg_pool2d(feat, output_size=PATCH_GRID)
        return pooled.flatten(2).transpose(1, 2).contiguous()

    def _context_patches(
        self,
        batch: torch.Tensor,
        raw_batch: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ):
        """``(B, N, 3D)`` context descriptors after MLFA and prior filtering."""
        stream = self.context_stream
        layers = stream.feature_layers
        tokens = stream.patch_tokens(batch, layers=layers)

        # MLFA operates at full token resolution, then we average-pool down to
        # the shared 14x14 grid so both streams live in the same spatial space.
        merged = torch.cat([tokens[l] for l in layers], dim=-1)  # Eq. 2
        merged = merged.transpose(1, 2)  # (B, 3D, N)
        merged = F.adaptive_avg_pool1d(merged, NUM_PATCHES).transpose(1, 2)

        mask_source = raw_batch if raw_batch is not None else batch
        filtered, mask = self.prior_filter(
            merged, mask_source, grid=PATCH_GRID
        )  # Eq. 3
        if return_mask:
            return filtered, mask
        return filtered


def _patch_normalize(patch_scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image Min-Max normalization of patch scores (spatial Eq. 4)."""
    lo = patch_scores.min(dim=1, keepdim=True).values
    hi = patch_scores.max(dim=1, keepdim=True).values
    span = (hi - lo).clamp_min(eps)
    return (patch_scores - lo) / span


def summarize(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    """Convenience wrapper computing Table 1 metrics for a score vector."""
    from .metrics import binary_classification_metrics

    return binary_classification_metrics(scores, labels)
