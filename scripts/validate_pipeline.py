"""Self-contained sanity check for the dual-stream framework.

Runs the complete pipeline -- preprocessing, Eq. 1, Eq. 2, Eq. 3, Eq. 4, Eq. 5,
k-NN scoring, metrics and heatmap generation -- on **synthetic** tensors and
**stub backbones**. Nothing is downloaded, no dataset is required and no GPU is
needed, so this doubles as a smoke test in CI.

It validates the *implementation*, not the paper's accuracy. The published
numbers live in ``results/reported_metrics.json`` and were produced with the real
Kermany dataset and the real frozen weights.

Usage
-----
    python scripts/validate_pipeline.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.anatomical_prior import AnatomicalPriorFilter, optical_density  # noqa: E402
from src.config import (  # noqa: E402
    AnatomicalPriorConfig,
    ContextStreamConfig,
    DualStreamConfig,
    FusionConfig,
    KNNConfig,
    PreprocessConfig,
    TextureStreamConfig,
)
from src.fusion import DualStreamSynergy, min_max_normalize, sweep_lambda  # noqa: E402
from src.knn import KNNAnomalyDetector  # noqa: E402
from src.metrics import (  # noqa: E402
    anomaly_maps_from_embeddings,
    binary_classification_metrics,
    image_level_auc,
    image_level_scores,
)
from src.pipeline import PATCH_GRID, DualStreamAnomalyDetector  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [ok]   {name}")
    else:
        FAILED.append(f"{name} -- {detail}")
        print(f"  [FAIL] {name} {detail}")


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


# --------------------------------------------------------------------- stubs


class StubTextureStream(nn.Module):
    """Stand-in for ResNet50 producing Layer 2 + Layer 3 shaped features."""

    feature_layers = (2, 3)
    out_channels = 512 + 1024

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        b = batch.shape[0]
        generator = torch.Generator().manual_seed(self.seed)
        # Mid-level stages of a 224x224 input: 1/8 and 1/16 resolution.
        l2 = torch.rand(b, 512, 28, 28, generator=generator)
        l3 = torch.rand(b, 1024, 14, 14, generator=generator)
        up = torch.nn.functional.interpolate(
            l3, size=(28, 28), mode="bilinear", align_corners=False
        )
        return torch.cat([l2, up], dim=1)


class StubContextStream(nn.Module):
    """Stand-in for DINOv3 ViT-L exposing Layer 6/12/18 patch tokens."""

    feature_layers = (6, 12, 18)
    vanilla_layer = 24
    embed_dim = 1024

    def __init__(self, seed: int = 1) -> None:
        super().__init__()
        self.seed = seed

    def patch_tokens(self, batch: torch.Tensor, layers=None):
        layers = tuple(layers or self.feature_layers)
        b = batch.shape[0]
        return {
            layer: torch.rand(
                b, 196, self.embed_dim, generator=torch.Generator().manual_seed(self.seed + layer)
            )
            for layer in layers
        }


# -------------------------------------------------------------------- checks


def check_preprocessing() -> None:
    banner("Preprocessing (Section 2.1)")
    from src.streams import normalize_batch

    config = PreprocessConfig()
    raw = torch.randint(0, 256, (2, 1, 260, 200)).float()
    out = normalize_batch(raw, config)
    check("output is (B, 3, 224, 224)", tuple(out.shape) == (2, 3, 224, 224), str(out.shape))
    check("grayscale is expanded to RGB", out.shape[1] == 3)
    check(
        "normalized values are roughly zero-mean",
        abs(float(out.mean())) < 1.0,
        f"mean={float(out.mean()):.4f}",
    )


def check_texture_stream() -> None:
    banner("Texture stream: Eq. 1 (ResNet50 Layer 2 + Layer 3)")
    stream = StubTextureStream()
    feat = stream.embed(torch.zeros(2, 3, 224, 224))
    check("F_tex has 1536 channels", feat.shape[1] == 1536, str(feat.shape))
    check(
        "F_tex spatial size matches the shallowest stage (28x28)",
        feat.shape[-2:] == (28, 28),
        str(feat.shape),
    )

    from src.streams import TextureStream

    real = TextureStream(TextureStreamConfig(pretraining="none"))
    check(
        "real ResNet50 wrapper declares 1536 output channels",
        real.out_channels == 1536,
        str(real.out_channels),
    )
    check(
        "real ResNet50 wrapper is frozen",
        all(not p.requires_grad for p in real.parameters()),
    )


def check_context_stream() -> None:
    banner("Context stream: Eq. 2 (MLFA over layers {6, 12, 18})")
    config = ContextStreamConfig()
    check("layer set S = {6, 12, 18}", tuple(config.feature_layers) == (6, 12, 18))
    check("Layer 24 is excluded from MLFA", config.vanilla_layer not in config.feature_layers)

    stream = StubContextStream()
    tokens = stream.patch_tokens(torch.zeros(2, 3, 224, 224))
    merged = torch.cat([tokens[l] for l in stream.feature_layers], dim=-1)
    check("F_ctx is (B, 196, 3072)", tuple(merged.shape) == (2, 196, 3072), str(merged.shape))

    from src.streams import assert_patch_grid

    try:
        assert_patch_grid(196, (14, 14))
        ok = True
    except ValueError:
        ok = False
    check("expected patch grid N = 196 is accepted", ok)
    try:
        assert_patch_grid(197, (14, 14))
        bad_rejected = False
    except ValueError:
        bad_rejected = True
    check("wrong patch count is rejected", bad_rejected)


def check_anatomical_prior() -> None:
    banner("Anatomical prior filter: Eq. 3 (tau = 30)")
    prior = AnatomicalPriorFilter(AnatomicalPriorConfig(tau=30.0))

    # A dark border (air) with a bright central block (tissue).
    image = torch.full((1, 1, 224, 224), 10.0)
    image[:, :, 56:168, 56:168] = 200.0
    mu = optical_density(image, grid=PATCH_GRID)
    check("optical density returns N = 196 patches", mu.shape == (1, 196), str(mu.shape))

    mask = prior.mask(image, grid=PATCH_GRID)
    check("background patches are masked out", int(mask.sum()) < 196, f"kept={int(mask.sum())}")
    check("ROI patches are kept", int(mask.sum()) > 0)

    embeddings = torch.ones(1, 196, 8)
    filtered, returned_mask = prior(embeddings, image, grid=PATCH_GRID)
    check("masked embeddings are zeroed", bool((filtered[~returned_mask] == 0).all()))
    check("ROI embeddings are untouched", bool((filtered[returned_mask] == 1).all()))

    bright = torch.full((1, 1, 224, 224), 255.0)
    check(
        "a fully bright image keeps every patch",
        int(prior.mask(bright, grid=PATCH_GRID).sum()) == 196,
    )


def check_knn() -> None:
    banner("k-NN scoring (K = 50, Euclidean)")
    config = KNNConfig(n_neighbors=50, metric="euclidean")
    detector = KNNAnomalyDetector(config)

    generator = torch.Generator().manual_seed(3)
    bank = torch.randn(300, 16, generator=generator)
    detector.fit(bank)

    near = bank[0] + 0.001 * torch.randn(5, 16, generator=generator)
    far = bank[0] + 5.0 * torch.randn(5, 16, generator=generator)
    near_scores = detector.score(near)
    far_scores = detector.score(far)
    check(
        "in-distribution queries score lower than out-of-distribution ones",
        float(near_scores.mean()) < float(far_scores.mean()),
        f"{float(near_scores.mean()):.4f} vs {float(far_scores.mean()):.4f}",
    )
    # A bank member is its own nearest neighbour at distance 0, so its k-th
    # neighbour distance can never exceed that of a perturbed query. This is the
    # scale-free correctness check on the memory bank.
    bank_self = float(detector.score(bank[:1])[0])
    check(
        "a bank member never scores above a perturbed query",
        bank_self <= float(near_scores.max()),
        f"self={bank_self:.6f} max_perturbed={float(near_scores.max()):.6f}",
    )

    patch = torch.randn(3, 196, 16, generator=generator)
    image_scores, maps = detector.score_map(patch, grid=PATCH_GRID)
    check("score_map returns (B,) image scores", image_scores.shape == (3,), str(image_scores.shape))
    check("score_map returns (B, 14, 14) maps", maps.shape == (3, 14, 14), str(maps.shape))

    try:
        KNNAnomalyDetector(KNNConfig(n_neighbors=50)).score(torch.randn(2, 4))
        raised = False
    except RuntimeError:
        raised = True
    check("scoring before fit fails loudly", raised)

    try:
        KNNAnomalyDetector(KNNConfig(n_neighbors=500)).fit(torch.randn(100, 4))
        raised = False
    except ValueError:
        raised = True
    check("undersized memory bank is rejected", raised)


def check_fusion() -> None:
    banner("Dual-Stream Synergy: Eq. 4 and Eq. 5 (lambda = 0.5)")
    synergy = DualStreamSynergy(FusionConfig(lam=0.5))

    raw = torch.tensor([0.2, 5.0, 9.9])
    normalized = min_max_normalize(raw)
    check("Min-Max maps the minimum to 0", abs(float(normalized.min())) < 1e-6)
    check("Min-Max maps the maximum to 1", abs(float(normalized.max()) - 1.0) < 1e-6)
    check(
        "Min-Max preserves sample order",
        bool(torch.all(torch.argsort(raw) == torch.argsort(normalized))),
    )
    check(
        "a degenerate batch normalizes to zeros",
        bool((min_max_normalize(torch.ones(4)) == 0).all()),
    )

    tex = torch.tensor([0.1, 0.9, 0.5, 0.3])
    ctx = torch.tensor([0.2, 0.8, 0.4, 0.6])
    check(
        "lambda = 0 reproduces the context stream",
        torch.allclose(synergy(tex, ctx, lam=0.0), min_max_normalize(ctx), atol=1e-6),
    )
    check(
        "lambda = 1 reproduces the texture stream",
        torch.allclose(synergy(tex, ctx, lam=1.0), min_max_normalize(tex), atol=1e-6),
    )
    mid = synergy(tex, ctx, lam=0.5)
    check(
        "lambda = 0.5 is the midpoint of the two streams",
        torch.allclose(
            mid, 0.5 * min_max_normalize(tex) + 0.5 * min_max_normalize(ctx), atol=1e-6
        ),
    )
    check("fused scores stay in [0, 1]", bool((mid >= -1e-6).all() and (mid <= 1 + 1e-6).all()))

    labels = torch.tensor([0, 1, 0, 1])
    sweep = sweep_lambda(tex, ctx, labels)
    check("lambda sweep returns 21 operating points", len(sweep["lambdas"]) == 21, str(len(sweep["lambdas"])))


def check_metrics() -> None:
    banner("Evaluation metrics (Section 3.1)")
    scores = torch.tensor([0.1, 0.2, 0.85, 0.95])
    labels = torch.tensor([0, 0, 1, 1])
    check("AUC is 1.0 for a perfectly separated set", abs(image_level_auc(scores, labels) - 1.0) < 1e-9)
    check(
        "AUC is 0.0 for a fully inverted set",
        abs(image_level_auc(-scores, labels) - 0.0) < 1e-9,
    )
    check(
        "AUC is 0.5 for an uninformative constant predictor",
        abs(image_level_auc(torch.ones(4), labels) - 0.5) < 1e-9,
    )

    metrics = binary_classification_metrics(scores, labels, threshold=0.5)
    check("TP/FP/FN/TN are computed", metrics["confusion_matrix"] == {"TP": 2, "FP": 0, "FN": 0, "TN": 2})
    check("Accuracy is 1.0", abs(metrics["ACC"] - 1.0) < 1e-9)
    check("Precision is 1.0", abs(metrics["Precision"] - 1.0) < 1e-9)
    check("Recall is 1.0", abs(metrics["Recall"] - 1.0) < 1e-9)
    check("F1 is 1.0", abs(metrics["F1-score"] - 1.0) < 1e-9)
    check("NPV is 1.0", abs(metrics["NPV"] - 1.0) < 1e-9)

    # Two normal images, both called abnormal at threshold 0.5: TN = 0, FN = 0,
    # so NPV collapses to 0.
    mixed = binary_classification_metrics(torch.tensor([0.9, 0.8]), torch.tensor([0, 0]), threshold=0.5)
    check("NPV collapses when normals are called abnormal", abs(mixed["NPV"] - 0.0) < 1e-9, str(mixed["NPV"]))
    all_clear = binary_classification_metrics(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]), threshold=0.5)
    check("NPV is 1.0 when no normal is flagged", abs(all_clear["NPV"] - 1.0) < 1e-9, str(all_clear["NPV"]))
    check("AUC is None for a single-class slice", mixed["AUC"] is None)

    patch = torch.rand(2, 196)
    check(
        "max reduction picks the hottest patch",
        torch.allclose(image_level_scores(patch), patch.max(dim=1).values),
    )
    maps = anomaly_maps_from_embeddings(patch, grid=PATCH_GRID, output_size=224)
    check("heatmaps are (B, 1, 224, 224)", tuple(maps.shape) == (2, 1, 224, 224), str(maps.shape))
    maps_smooth = anomaly_maps_from_embeddings(patch, grid=PATCH_GRID, output_size=224, smooth_sigma=0.6)
    check("Gaussian smoothing keeps the shape", maps_smooth.shape == maps.shape)


def check_pipeline() -> None:
    banner("End-to-end pipeline")
    config = DualStreamConfig(
        texture=TextureStreamConfig(pretraining="none"),
        context=ContextStreamConfig(),
        device="cpu",
    ).validate()

    detector = DualStreamAnomalyDetector(
        config,
        texture_stream=StubTextureStream(),
        context_stream=StubContextStream(),
    )
    check("detector starts unfitted", not detector.is_fitted)

    normal = torch.randint(40, 220, (12, 3, 224, 224)).float()
    # Give the normal set a consistent, tissue-like bright centre.
    normal[:, :, 56:168, 56:168] += 20.0
    detector.fit(normal, batch_size=4)
    check("both memory banks are built", detector.is_fitted)

    test = torch.randint(40, 220, (5, 3, 224, 224)).float()
    out = detector.score(test, batch_size=2)
    for key in ("fused", "texture", "context", "heatmaps", "roi_mask"):
        check(f"score() returns '{key}'", key in out)
    check("fused scores are (T,)", tuple(out["fused"].shape) == (5,), str(out["fused"].shape))
    check("fused scores lie in [0, 1]", bool((out["fused"] >= 0).all() and (out["fused"] <= 1).all()))
    check("heatmaps are (T, 1, 224, 224)", tuple(out["heatmaps"].shape) == (5, 1, 224, 224), str(out["heatmaps"].shape))
    check("ROI masks are (T, 196)", tuple(out["roi_mask"].shape) == (5, 196))
    check("ROI mask keeps at least one patch per image", bool((out["roi_mask"].sum(dim=1) > 0).all()))

    tex_only = detector.score_single_stream(test, "texture")
    ctx_only = detector.score_single_stream(test, "context")
    check("single-stream ablation matches the batch size", tuple(tex_only.shape) == (5,))
    check("context-only ablation matches the batch size", tuple(ctx_only.shape) == (5,))

    detector.fit_vanilla_vit(normal, batch_size=4)
    vanilla = detector.score_vanilla_vit(test, batch_size=2)
    check("vanilla Layer-24 baseline returns (T,)", tuple(vanilla.shape) == (5,))

    try:
        DualStreamAnomalyDetector(config).score(test)
        raised = False
    except RuntimeError:
        raised = True
    check("scoring an unfitted detector fails loudly", raised)


def check_config_guards() -> None:
    banner("Configuration guards")
    try:
        DualStreamConfig(fusion=FusionConfig(lam=1.5)).validate()
        raised = False
    except ValueError:
        raised = True
    check("lambda outside [0, 1] is rejected", raised)

    try:
        DualStreamConfig(context=ContextStreamConfig(feature_layers=(6, 24))).validate()
        raised = False
    except ValueError:
        raised = True
    check("including Layer 24 in MLFA is rejected", raised)

    check("the default configuration satisfies the paper", DualStreamConfig().validate() is not None)
    default = DualStreamConfig()
    check("default lambda is 0.5", default.fusion.lam == 0.5)
    check("default tau is 30", default.prior.tau == 30.0)
    check("default K is 50", default.knn.n_neighbors == 50)
    check("default input size is 224", default.preprocess.image_size == 224)


def main() -> int:
    print("Dual-Stream CXR Anomaly Detection -- implementation sanity check")
    print("Synthetic tensors and stub backbones only; nothing is downloaded.")
    torch.manual_seed(0)

    check_preprocessing()
    check_texture_stream()
    check_context_stream()
    check_anatomical_prior()
    check_knn()
    check_fusion()
    check_metrics()
    check_pipeline()
    check_config_guards()

    banner("Summary")
    print(f"  passed: {len(PASSED)}")
    print(f"  failed: {len(FAILED)}")
    for item in FAILED:
        print(f"    - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
