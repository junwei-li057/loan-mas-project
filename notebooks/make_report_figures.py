"""Generate Junwei's report figures (Fig 3-4) for the NeurIPS write-up.

Figures owned by other team members (Shiqi's Table 1, Zixuan's Fig 5-6) are
produced separately and intentionally not generated here.

Outputs (PNG @200dpi + PDF) under docs/figures/:
  fig3_confidence_reliability.{png,pdf}
  fig4_selective_fusion_curve.{png,pdf}

Data source:
  docs/eval_report_sample1.md  (Section 6 tables, embedded below)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_HL = "#1a73e8"     # blue   (fused / highlight)
C_WARN = "#d93025"   # red    (trend / highlight)
C_OK = "#188038"     # green


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- Fig 3
def fig3_confidence_reliability():
    # eval_report_sample1.md  Section 6 -- confidence reliability bins
    bins = pd.DataFrame({
        "label": ["0.40-0.46", "0.46-0.51", "0.51-0.57",
                  "0.62-0.68", "0.74-0.79", "0.79-0.85"],
        "n": [39, 29, 108, 2529, 965, 529],
        "mean_conf": [0.4021, 0.5000, 0.5500, 0.6500, 0.7500, 0.8500],
        "accuracy": [0.1026, 0.1724, 0.2222, 0.2985, 0.6062, 0.6843],
    })
    x, y, n = bins["mean_conf"].values, bins["accuracy"].values, bins["n"].values

    # Authoritative sample-level slope reported in eval_report_sample1.md (Sec. 6).
    # We draw the trend line through the size-weighted centroid so the picture is
    # consistent with that reported number rather than re-deriving a bin-level one.
    REPORTED_SLOPE = 1.40
    w = n / n.sum()
    xbar, ybar = np.average(x, weights=w), np.average(y, weights=w)
    intercept = ybar - REPORTED_SLOPE * xbar

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    sizes = 60 + 380 * (n / n.max())
    ax.scatter(x, y, s=sizes, color=C_HL, alpha=0.75, edgecolor="white",
               linewidth=1.2, zorder=3, label="confidence bin (area $\\propto$ n)")
    xs = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
    ax.plot(xs, intercept + REPORTED_SLOPE * xs, "--", color=C_WARN, lw=1.8,
            zorder=2, label=f"trend, slope = {REPORTED_SLOPE:.2f}")
    for xi, yi, ni in zip(x, y, n):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                    xytext=(9, -13), fontsize=8, color="#444")
    ax.set_xlabel("Mean LLM-claimed confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Confidence is a real trust signal", pad=10)
    ax.legend(loc="upper left")
    ax.set_xlim(0.36, 0.90)
    ax.set_ylim(0, 0.80)
    _save(fig, "fig3_confidence_reliability")


# ---------------------------------------------------------------- Fig 4
def fig4_selective_fusion_curve():
    # eval_report_sample1.md  Section 6 -- selective-fusion curve
    cov = np.arange(0.05, 1.0001, 0.05)
    d_auc = [0.0126, 0.0048, 0.0037, 0.0026, 0.0016, 0.0015, 0.0009, 0.0010,
             0.0012, 0.0011, 0.0013, 0.0013, 0.0012, 0.0011, 0.0012, 0.0011,
             0.0011, 0.0010, 0.0010, 0.0009]
    d_auprc = [0.0129, 0.0038, 0.0054, 0.0026, 0.0021, 0.0016, 0.0020, 0.0018,
               0.0014, 0.0014, 0.0015, 0.0012, 0.0015, 0.0014, 0.0014, 0.0012,
               0.0015, 0.0015, 0.0015, 0.0014]

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.axhline(0, color="#999", lw=0.8)
    ax.plot(cov, d_auprc, "-o", color=C_HL, ms=4, lw=1.8,
            label="$\\Delta$AUPRC (fused $-$ baseline)")
    ax.plot(cov, d_auc, "-s", color=C_OK, ms=3.5, lw=1.4, alpha=0.85,
            label="$\\Delta$AUC (fused $-$ baseline)")
    # highlight most-trusted 5%
    ax.scatter([cov[0]], [d_auprc[0]], s=160, facecolor="none",
               edgecolor=C_WARN, lw=2, zorder=5)
    ax.annotate(f"most-trusted 5%: $\\Delta$AUPRC = +{d_auprc[0]:.4f}",
                (cov[0], d_auprc[0]), textcoords="offset points",
                xytext=(40, -22), fontsize=9, color=C_WARN,
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.2))
    ax.annotate(f"full coverage: +{d_auprc[-1]:.4f}",
                (cov[-1], d_auprc[-1]), textcoords="offset points",
                xytext=(-150, 22), fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.set_xlabel("Coverage (fraction kept, most-trusted first)")
    ax.set_ylabel("Metric gain over baseline")
    ax.set_title("Gain concentrates where text is trusted", pad=10)
    ax.set_ylim(-0.0005, 0.0145)
    ax.legend(loc="upper right")
    _save(fig, "fig4_selective_fusion_curve")


if __name__ == "__main__":
    fig3_confidence_reliability()
    fig4_selective_fusion_curve()
    print("\nFig 3-4 written to", OUT)
