"""Dual-Stream Feature Synergy for chest X-ray anomaly detection.

Reference
---------
Chenxi Gao, "Dual-Stream Feature Synergy and Prior Fusion Method for Chest
X-ray Anomaly Detection", ICIAAI 2026.
DOI: 10.4108/eai.22-5-2026.2365105

This package contains a clean re-implementation of the paper's method. It is
intended for reference and for reproduction of the published results; the
published numbers themselves are frozen in ``results/reported_metrics.json``.
"""

from .anatomical_prior import AnatomicalPriorFilter, optical_density
from .config import DualStreamConfig, PreprocessConfig
from .fusion import DualStreamSynergy, min_max_normalize
from .knn import KNNAnomalyDetector
from .metrics import (
    anomaly_maps_from_embeddings,
    binary_classification_metrics,
    image_level_scores,
    image_level_auc,
)
from .pipeline import DualStreamAnomalyDetector
from .streams import ContextStream, TextureStream, assert_patch_grid

__all__ = [
    "AnatomicalPriorFilter",
    "ContextStream",
    "DualStreamAnomalyDetector",
    "DualStreamConfig",
    "DualStreamSynergy",
    "KNNAnomalyDetector",
    "PreprocessConfig",
    "TextureStream",
    "anomaly_maps_from_embeddings",
    "assert_patch_grid",
    "binary_classification_metrics",
    "image_level_auc",
    "image_level_scores",
    "min_max_normalize",
    "optical_density",
]

__version__ = "1.0.0"
