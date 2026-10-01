"""Generate the reported-result figures used in the README.

This script plots the *published* numbers frozen in
``results/reported_metrics.json``. It never runs the model and never downloads
anything, so the figures are a faithful rendering of the paper's Tables 1 and 2.

Usage
-----
    python scripts/make_reported_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results", "reported_metrics.json")
OUT_DIR = os.path.join(ROOT, "results", "figures")

# Method colours: our variants in a warm accent, baselines in neutral grey/blue.
PALETTE = {
    "PatchCore (Re-impl)": "#8C8C8C",
    "Vanilla ViT": "#B0B7C3",
    "Only Texture Stream": "#3498DB",
    "Context Stream": "#E67E22",
    "Dual-Stream": "#27AE60",
}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def load_metrics() -> dict:
    with open(METRICS, encoding="utf-8") as handle:
        return json.load(handle)


def figure_main_auc(data: dict, out_dir: str) -> str:
    """Figure: image-level AUC of every method in Table 1."""
    rows = data["main_results"]
    names = [r["method"] for r in rows]
    aucs = [r["AUC"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(
        range(len(names)),
        aucs,
        color=[PALETTE.get(n, "#7F8C8D") for n in names],
        width=0.62,
        edgecolor="black",
        linewidth=0.6,
    )
    # The dual-stream result is the headline of the paper: mark it.
    best = float(np.argmax(aucs))
    bars[int(best)].set_edgecolor("#1E8449")
    bars[int(best)].set_linewidth(2.0)

    for idx, value in enumerate(aucs):
        ax.text(
            idx,
            value + 0.004,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if idx == best else "normal",
        )

    # Dashed reference at the strongest single-stream backbone.
    tex = next(r["AUC"] for r in rows if r["method"] == "Only Texture Stream")
    ax.axhline(tex, color="#3498DB", linestyle="--", linewidth=1.1, alpha=0.8)
    tex_idx = names.index("Only Texture Stream")
    ax.text(
        tex_idx,
        tex - 0.0045,
        f"best single stream ({tex:.4f})",
        ha="center",
        va="top",
        fontsize=8,
        color="#2C7BB6",
        fontweight="bold",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [n.replace(" (Re-impl)", "\n(re-impl)") for n in names], fontsize=9
    )
    ax.set_ylabel("Image-level AUC")
    ax.set_ylim(0.60, 0.93)
    ax.set_title(
        "Kermany chest X-ray: image-level AUC by method (Table 1)", fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_main_auc.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_ablation(data: dict, out_dir: str) -> str:
    """Figure: staged ablation of Table 2."""
    rows = data["ablation"]
    # Spell out which components are active, mirroring the Table 2 columns.
    labels = [
        "Exp.{} {}\nCNN:{} ViT:{}\nMLFA:{} Prior:{}".format(
            r["id"],
            "(Ours)" if r["id"] == 5 else "",
            "Y" if r["stream_a_cnn"] else "-",
            "Y" if r["stream_b_vit"] else "-",
            "Y" if r["mlfa_layers"] else "-",
            "Y" if r["anatomy_filter"] else "-",
        )
        for r in rows
    ]
    aucs = [r["AUC"] for r in rows]
    colors = ["#B0B7C3", "#E67E22", "#F0A65A", "#3498DB", "#27AE60"]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bars = ax.bar(
        range(len(labels)),
        aucs,
        color=colors,
        width=0.6,
        edgecolor="black",
        linewidth=0.6,
    )
    bars[-1].set_linewidth(2.0)
    bars[-1].set_edgecolor("#1E8449")

    for idx, row in enumerate(rows):
        ax.text(
            idx,
            row["AUC"] + 0.004,
            f"{row['AUC']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if idx == len(rows) - 1 else "normal",
        )
        if row["improvement_label"] != "Baseline":
            ax.text(
                idx,
                row["AUC"] - 0.02,
                row["improvement_label"],
                ha="center",
                va="top",
                fontsize=8.5,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="#34495E", ec="none"),
            )
        else:
            ax.text(
                idx,
                row["AUC"] - 0.02,
                "Baseline",
                ha="center",
                va="top",
                fontsize=8.5,
                color="#34495E",
                fontweight="bold",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Image-level AUC")
    ax.set_ylim(0.60, 0.93)
    ax.set_title(
        "Staged component ablation: MLFA, anatomical prior, and dual-stream fusion "
        "(Table 2)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_ablation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_metrics(data: dict, out_dir: str) -> str:
    """Figure: all Table 1 metrics side by side."""
    rows = data["main_results"]
    metrics = ["AUC", "ACC", "Precision", "Recall", "F1-score"]

    x = np.arange(len(metrics))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2) * width
        ax.bar(
            x + offset,
            [row[m] for m in metrics],
            width=width,
            label=row["method"],
            color=PALETTE.get(row["method"], "#7F8C8D"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.60, 1.06)
    # Place the legend outside the axes so it never overlaps the bars.
    ax.legend(
        fontsize=8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )
    ax.set_title(
        "Discrimination metrics across methods (Table 1)", fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_metrics_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_improvement_chain(data: dict, out_dir: str) -> str:
    """Figure: AUC change reported at each stage of Table 2.

    Each value is the ``Improvement`` column exactly as printed in the paper. Note
    that stages 2 -> 3 -> 4 are not all additive component insertions: Exp.4
    replaces the context stream with the texture stream, so its +7.12% is a
    cross-stream comparison rather than a marginal gain on top of Exp.3.
    """
    rows = [r for r in data["ablation"] if r["improvement_percent"] is not None]
    labels = [
        "MLFA over deep\nLayer-24 features\n(Exp.1 $\\rightarrow$ Exp.2)",
        "Anatomical\nprior filter\n(Exp.2 $\\rightarrow$ Exp.3)",
        "Texture stream only\nreplaces context stream\n(Exp.3 $\\rightarrow$ Exp.4)",
        "Dual-stream fusion\n(Exp.4 $\\rightarrow$ Exp.5, Ours)",
    ]
    values = [r["improvement_percent"] for r in rows]
    colors = ["#E67E22", "#F0A65A", "#3498DB", "#27AE60"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars = ax.barh(
        range(len(labels)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        height=0.62,
    )
    bars[-1].set_linewidth(2.0)
    bars[-1].set_edgecolor("#1E8449")
    for idx, value in enumerate(values):
        ax.text(
            value + 0.18,
            idx,
            f"+{value:.2f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Image-level AUC change reported in Table 2")
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_title(
        "Ablation: reported AUC change at each stage (Table 2)", fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_improvement_chain.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_metrics()
    written = [
        figure_main_auc(data, OUT_DIR),
        figure_ablation(data, OUT_DIR),
        figure_metrics(data, OUT_DIR),
        figure_improvement_chain(data, OUT_DIR),
    ]
    for path in written:
        print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
