"""Qualitative heatmap rendering for the Figure 2 layout.

Reproduces the four-column comparison of paper Figure 2:

    (a) original chest X-ray
    (b) texture stream (ResNet) -- fragmented high-frequency noise
    (c) context stream (DINOv3) -- over-smooth, coarse localization
    (d) dual-stream -- best background suppression and lesion coverage

This module only *renders*; it never invents activations. Pass it the maps your
own inference run produced.

Usage
-----
    python scripts/render_heatmaps.py --npz run.npz --out results/figures/fig7_localization.png

where ``run.npz`` holds ``images`` (T, H, W), ``texture`` (T, h, w),
``context`` (T, h, w) and ``fused`` (T, h, w).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COLUMN_TITLES = [
    "(a) Original",
    "(b) Texture stream (ResNet)",
    "(c) Context stream (DINOv3)",
    "(d) Dual-Stream (Ours)",
]


def _resize(map_2d: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to ``size`` x ``size`` using NumPy only."""
    from numpy import interp

    h, w = map_2d.shape
    ys = np.linspace(0, h - 1, size)
    xs = np.linspace(0, w - 1, size)
    rows = interp(ys, np.arange(h), map_2d, axis=0)
    return interp(xs, np.arange(w), rows, axis=1)


def render(
    images: np.ndarray,
    texture: np.ndarray,
    context: np.ndarray,
    fused: np.ndarray,
    out_path: str,
    max_rows: int = 4,
    image_size: int = 224,
) -> str:
    """Render the four-column qualitative comparison."""
    n = min(len(images), max_rows)
    fig, axes = plt.subplots(n, 4, figsize=(3.1 * 4, 3.1 * n), squeeze=False)

    for row in range(n):
        panels = [
            (images[row], None, "gray"),
            (texture[row], "jet", None),
            (context[row], "jet", None),
            (fused[row], "jet", None),
        ]
        for col, (data, cmap, base_cmap) in enumerate(panels):
            ax = axes[row][col]
            if base_cmap == "gray":
                ax.imshow(data, cmap="gray")
            else:
                base = _resize(np.asarray(images[row], dtype=np.float32), image_size)
                ax.imshow(base, cmap="gray")
                overlay = _resize(np.asarray(data, dtype=np.float32), image_size)
                # Per-map Min-Max so every heatmap uses the full colour range.
                lo, hi = float(np.min(overlay)), float(np.max(overlay))
                span = max(hi - lo, 1e-8)
                ax.imshow((overlay - lo) / span, cmap=cmap, alpha=0.45, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#BDC3C7")
            if row == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=10, fontweight="bold")

    fig.suptitle(
        "Soft localization of pneumonia: texture, context, and dual-stream heatmaps",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_from_npz(npz_path: str, out_path: str, max_rows: int = 4) -> str:
    """Load a prepared ``.npz`` bundle and render it."""
    bundle = np.load(npz_path)
    required = {"images", "texture", "context", "fused"}
    missing = required - set(bundle.files)
    if missing:
        raise KeyError(f"{npz_path} is missing arrays: {sorted(missing)}")
    return render(
        bundle["images"],
        bundle["texture"],
        bundle["context"],
        bundle["fused"],
        out_path,
        max_rows=max_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="prepared heatmap bundle")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "figures", "fig7_localization.png"),
        help="output PNG path",
    )
    parser.add_argument("--max-rows", type=int, default=4)
    args = parser.parse_args()

    path = render_from_npz(args.npz, args.out, max_rows=args.max_rows)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
