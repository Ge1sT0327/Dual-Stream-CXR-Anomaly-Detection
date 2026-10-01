"""Configuration objects for the dual-stream framework.

Every hyper-parameter below is the exact value reported in the paper
(Section 3.1 "Implementation details" and Section 2.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

#: DINOv3 ViT-L/16 patch grid for a 224x224 input: N = 196 patch tokens.
DINOV3_PATCH_GRID: Tuple[int, int] = (14, 14)
DINOV3_PATCH_TOKENS: int = 196
DINOV3_EMBED_DIM: int = 1024


@dataclass
class PreprocessConfig:
    """Image preprocessing (paper Section 2.1)."""

    image_size: int = 224
    interpolation: str = "bilinear"
    #: ImageNet statistics [12], as named in the paper.
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    #: Images are forced to RGB before normalization.
    to_rgb: bool = True


@dataclass
class TextureStreamConfig:
    """ResNet50 texture stream (paper Section 2.3, Eq. 1).

    Layer indices follow the paper's convention where ``Layer j`` is the output
    of the ``j``-th residual stage. Only the middle stages ``Layer 2`` and
    ``Layer 3`` are used, giving 1/8 and 1/16 relative spatial resolution.
    """

    backbone: str = "resnet50"
    pretraining: str = "imagenet"
    #: ``Layer 2`` and ``Layer 3`` residual stages.
    feature_layers: Tuple[int, ...] = (2, 3)
    frozen: bool = True
    #: Stage-1 stride trick is NOT used; the paper keeps stock ResNet50.
    replace_stride_with_dilation: bool = False


@dataclass
class ContextStreamConfig:
    """DINOv3 ViT-Large context stream with MLFA (paper Section 2.3, Eq. 2).

    ``feature_layers`` is the layer set S = {6, 12, 18}: shallow texture/edges,
    mid-scale anatomical sub-parts, and global context. Layer 24 is deliberately
    excluded to avoid the over-smoothed "macroscopic" semantics that cause the
    semantic gap analysed in Section 2.2.
    """

    backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    #: Local directory holding the DINOv3 weights, or ``None`` to use the
    #: Hugging Face hub id in ``backbone``. See README for download notes.
    local_files_only: bool = True
    weights_dir: str | None = None
    feature_layers: Tuple[int, ...] = (6, 12, 18)
    #: Layer 24 is the vanilla ViT baseline and is excluded from MLFA.
    vanilla_layer: int = 24
    patch_grid: Tuple[int, int] = DINOV3_PATCH_GRID
    embed_dim: int = DINOV3_EMBED_DIM
    #: Number of register tokens emitted by DINOv3 before the patch tokens.
    num_register_tokens: int = 4
    frozen: bool = True


@dataclass
class AnatomicalPriorConfig:
    """Anatomy-guided background suppression (paper Section 2.3, Eq. 3).

    Pixels outside the body have low X-ray attenuation and therefore low
    intensity. Patches whose mean optical density mu(P_i) is not strictly
    greater than ``tau`` are masked out.
    """

    tau: float = 30.0
    #: Dilation (in patches) applied after thresholding to keep ROI borders.
    mask_dilation: int = 0


@dataclass
class FusionConfig:
    """Dual-Stream Synergy module (paper Section 2.3, Eqs. 4-5)."""

    #: Balancing parameter lambda in Eq. 5, tuned on the validation set.
    lam: float = 0.5
    #: Min-Max normalization is computed over the test batch (Eq. 4).
    normalize: str = "minmax"
    eps: float = 1e-8


@dataclass
class KNNConfig:
    """k-NN anomaly scoring (paper Section 3.1)."""

    n_neighbors: int = 50
    metric: str = "euclidean"
    #: Use the k-th neighbour distance (PatchCore style) rather than the mean.
    use_kth_distance: bool = True


@dataclass
class DualStreamConfig:
    """Top-level configuration aggregating every sub-config."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    texture: TextureStreamConfig = field(default_factory=TextureStreamConfig)
    context: ContextStreamConfig = field(default_factory=ContextStreamConfig)
    prior: AnatomicalPriorConfig = field(default_factory=AnatomicalPriorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    device: str = "cuda"
    seed: int = 0

    def validate(self) -> "DualStreamConfig":
        """Fail fast on configurations that contradict the paper."""
        if not 0.0 <= self.fusion.lam <= 1.0:
            raise ValueError(f"lambda must lie in [0, 1], got {self.fusion.lam}")
        if self.context.vanilla_layer in self.context.feature_layers:
            raise ValueError(
                "Layer 24 must not be part of the MLFA layer set S = {6, 12, 18}"
            )
        if self.knn.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.prior.tau < 0:
            raise ValueError("tau must be non-negative")
        return self
