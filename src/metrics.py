"""Evaluation metrics (paper Section 3.1, "Evaluation metrics").

The paper uses a two-part evaluation strategy:

* **Global discrimination** -- image-level AUC plus confusion-matrix derived
  Accuracy, Precision (the paper's PPV), Recall, and F1. The paper also reports
  the clinically valuable Negative Predictive Value (NPV).
* **Localization** -- qualitative assessment of the anomaly heatmaps
  (paper Section 3.4 and Figure 2).

Thresholds for the confusion matrix are selected on a validation split and then
applied unchanged to the test set; :func:`binary_classification_metrics` takes
the threshold explicitly for that reason.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------- image level


def image_level_auc(scores, labels) -> float:
    """Area under the ROC curve for image-level anomaly scores."""
    from sklearn.metrics import roc_auc_score

    score_np = _to_numpy(scores).ravel()
    label_np = _to_numpy(labels).ravel()
    if np.unique(label_np).size < 2:
        raise ValueError("AUC needs both normal and abnormal samples")
    return float(roc_auc_score(label_np, score_np))


def image_level_scores(patch_scores: torch.Tensor) -> torch.Tensor:
    """Reduce patch scores to one score per image.

    Max pooling is the reduction used throughout the paper's reported numbers.
    """
    if patch_scores.dim() == 3:  # (B, H, W)
        return patch_scores.flatten(1).max(dim=1).values
    if patch_scores.dim() == 2:  # (B, N)
        return patch_scores.max(dim=1).values
    raise ValueError(f"expected (B, N) or (B, H, W), got {tuple(patch_scores.shape)}")


def binary_classification_metrics(
    scores,
    labels,
    threshold: Optional[float] = None,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Confusion-matrix metrics matching the paper's Table 1 columns.

    Returns ``AUC``, ``ACC``, ``Precision``, ``Recall``, ``F1-score``, ``NPV``
    and the ``threshold`` actually used. When ``threshold`` is ``None`` the
    value maximizing Youden's J statistic on the supplied data is used; for a
    faithful reproduction pass a threshold chosen on the validation split.
    """
    from sklearn.metrics import roc_curve

    score_np = _to_numpy(scores).ravel().astype(np.float64)
    label_np = _to_numpy(labels).ravel().astype(np.int64)
    abnormal = label_np == positive_label

    if threshold is None:
        fpr, tpr, thr = roc_curve(abnormal.astype(np.int64), score_np)
        threshold = float(thr[int(np.argmax(tpr - fpr))])

    predicted = score_np > threshold
    tp = int(np.sum(predicted & abnormal))
    fp = int(np.sum(predicted & ~abnormal))
    fn = int(np.sum(~predicted & abnormal))
    tn = int(np.sum(~predicted & ~abnormal))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / max(len(label_np), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        # AUC is undefined without both classes present; report None instead of
        # raising so that single-class diagnostic slices still yield a report.
        "AUC": (
            image_level_auc(score_np, label_np)
            if np.unique(label_np).size > 1
            else None
        ),
        "ACC": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-score": float(f1),
        "NPV": float(npv),
        "threshold": float(threshold),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


# ------------------------------------------------------------- localization


def anomaly_maps_from_embeddings(
    patch_scores: torch.Tensor,
    grid: Tuple[int, int] = (14, 14),
    output_size: Optional[int] = 224,
    smooth_sigma: float = 0.0,
) -> torch.Tensor:
    """Turn ``(B, N)`` patch scores into ``(B, 1, H, W)`` heatmaps.

    Parameters
    ----------
    patch_scores:
        ``(B, N)`` per-patch anomaly scores.
    grid:
        Patch grid, ``(14, 14)`` for 224x224 inputs.
    output_size:
        Spatial size of the returned map. ``None`` keeps the patch grid.
    smooth_sigma:
        Optional Gaussian blur applied on the patch grid before upsampling, to
        suppress the "fragmented high-frequency noise" visible in Figure 2(b).
    """
    if patch_scores.dim() != 2:
        raise ValueError("patch_scores must be (B, N)")
    b, n = patch_scores.shape
    if n != grid[0] * grid[1]:
        raise ValueError(f"expected {grid[0] * grid[1]} patches, got {n}")

    maps = patch_scores.view(b, 1, grid[0], grid[1])
    if smooth_sigma > 0:
        maps = _gaussian_blur(maps, smooth_sigma)
    if output_size is not None:
        maps = F.interpolate(
            maps, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
    return maps


def normalization_stats(maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Min/max of every map, for consistent heatmap colour scaling."""
    flat = maps.flatten(1)
    return flat.min(dim=1).values, flat.max(dim=1).values


def localization_score(
    heatmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    top_fraction: float = 0.1,
) -> float:
    """Mean anomaly mass inside the top-scoring region, a soft-localization proxy.

    The paper does not report a bounding-box metric (no box supervision is
    available); Section 3.4 argues for qualitative "soft localization". This
    helper quantifies how concentrated the heatmap is and can be combined with
    ground-truth lesion boxes if a reader supplies them.
    """
    flat = heatmap.flatten()
    if mask is not None:
        flat = flat[mask.flatten().bool()]
    if flat.numel() == 0:
        return 0.0
    k = max(1, int(round(flat.numel() * top_fraction)))
    top = torch.topk(flat, k=k).values
    total = flat.sum()
    if float(total) <= 0:
        return 0.0
    return float(top.sum() / total)


# ------------------------------------------------------------------ helpers


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Small separable Gaussian blur without external dependencies."""
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = maps.shape[1]
    kx = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    padded = F.pad(maps, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(F.conv2d(padded, kx, groups=channels), ky, groups=channels)


def summarize_runs(
    per_method_scores: Dict[str, Tuple[Sequence[float], Sequence[int]]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute Table 1 style metrics for several methods at once.

    ``per_method_scores`` maps a method name to ``(scores, labels)``. Thresholds
    may be supplied per method to enforce the validation-selected operating
    point.
    """
    thresholds = thresholds or {}
    out: Dict[str, Dict[str, float]] = {}
    for name, (scores, labels) in per_method_scores.items():
        out[name] = binary_classification_metrics(
            scores, labels, threshold=thresholds.get(name)
        )
    return out
