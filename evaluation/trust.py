"""
trust.py — does the MAS know when NOT to trust the text?

Two diagnostics:

  - confidence_reliability  : is the LLM's self-reported confidence a real
                              trust signal? Bin rows by claimed confidence and
                              measure the text-score's discrimination within
                              each bin. If confidence is well-calibrated, high-
                              confidence bins should give higher text AUC.

  - selective_fusion_curve  : as we restrict fusion to only the highest-trust
                              rows (top X% by some trust_score), does the
                              fused-vs-baseline advantage grow? This is the
                              MAS's "selective participation" certificate.

  - trust_calibration       : convenience wrapper returning both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def confidence_reliability(
    text_score: np.ndarray,
    text_conf: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 8,
    min_bin_n: int = 5,
) -> pd.DataFrame:
    """Bin rows by LLM-claimed confidence; compute per-bin accuracy of the
    text-only verdict (text_score > 0.5 vs y).

    Flat accuracy across bins → the LLM's self-reported confidence is not a
    real trust signal (the conf_threshold gate in evaluate_fusion is then
    effectively a placebo). Upward slope → higher claimed confidence is
    genuinely more reliable.

    Returns
    -------
    DataFrame with columns:
        conf_bin, conf_bin_lo, conf_bin_hi, n,
        mean_conf, mean_text_risk, default_rate, accuracy
    Sorted by conf_bin_lo ascending.

    A `slope` attribute on the result holds the OLS slope of accuracy on
    mean_conf — the headline number that quantifies reliability.
    """
    text_score = np.asarray(text_score, dtype=float)
    text_conf = np.asarray(text_conf, dtype=float)
    y = np.asarray(y_true).astype(int)
    pred = (text_score > 0.5).astype(int)
    correct = (pred == y).astype(int)

    lo_all, hi_all = float(text_conf.min()), float(text_conf.max())
    edges = np.linspace(lo_all, hi_all, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            mask = (text_conf >= lo) & (text_conf < hi)
        else:
            mask = (text_conf >= lo) & (text_conf <= hi)
        if int(mask.sum()) < min_bin_n:
            continue
        rows.append(
            {
                "conf_bin": f"[{lo:.2f}, {hi:.2f}]",
                "conf_bin_lo": lo,
                "conf_bin_hi": hi,
                "n": int(mask.sum()),
                "mean_conf": float(text_conf[mask].mean()),
                "mean_text_risk": float(text_score[mask].mean()),
                "default_rate": float(y[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) >= 2:
        slope = float(np.polyfit(df["mean_conf"].values, df["accuracy"].values, 1)[0])
    else:
        slope = float("nan")
    df.attrs["slope"] = slope
    df.attrs["n_total"] = int(len(y))
    return df


def selective_fusion_curve(
    y_true: np.ndarray,
    p_baseline: np.ndarray,
    p_fused: np.ndarray,
    trust_score: np.ndarray,
    *,
    n_steps: int = 20,
    coverage_min: float = 0.05,
    min_n: int = 20,
) -> pd.DataFrame:
    """Sweep coverage k% (top-k by `trust_score`); compute baseline vs fused
    metrics on those rows.

    `trust_score` should be the per-row signal one would use to decide whether
    the text is worth trusting. Typical choices:
      - text_confidence (LLM self-report)
      - |text_risk_score - p_baseline| (model disagreement)

    Curve shape answers the headline "when to trust the text" question:
      - Monotonically decreasing (top first, then declines) → trust only at
        the very top (selective wins).
      - Inverted-U → an optimal coverage τ* exists.
      - Flat → the trust score carries no useful signal.

    Returns
    -------
    DataFrame with columns:
        coverage, n_kept,
        baseline_auc, fused_auc, delta_auc,
        baseline_auprc, fused_auprc, delta_auprc
    """
    y = np.asarray(y_true).astype(int)
    p_b = np.asarray(p_baseline, dtype=float)
    p_f = np.asarray(p_fused, dtype=float)
    trust = np.asarray(trust_score, dtype=float)

    n = len(y)
    order = np.argsort(-trust)  # high → low
    rows = []
    for k in np.linspace(coverage_min, 1.0, n_steps):
        cutoff = int(round(k * n))
        if cutoff < min_n:
            continue
        idx = order[:cutoff]
        yk = y[idx]
        if yk.sum() == 0 or yk.sum() == cutoff:
            continue
        b_auc = float(roc_auc_score(yk, p_b[idx]))
        f_auc = float(roc_auc_score(yk, p_f[idx]))
        b_auprc = float(average_precision_score(yk, p_b[idx]))
        f_auprc = float(average_precision_score(yk, p_f[idx]))
        rows.append(
            {
                "coverage": float(k),
                "n_kept": cutoff,
                "baseline_auc": b_auc,
                "fused_auc": f_auc,
                "delta_auc": f_auc - b_auc,
                "baseline_auprc": b_auprc,
                "fused_auprc": f_auprc,
                "delta_auprc": f_auprc - b_auprc,
            }
        )
    return pd.DataFrame(rows)


def trust_calibration(
    text_score: np.ndarray,
    text_conf: np.ndarray,
    y_true: np.ndarray,
    p_baseline: np.ndarray,
    p_fused: np.ndarray,
    *,
    n_bins: int = 8,
    n_steps: int = 20,
) -> dict:
    """Combined trust diagnostics.

    Runs `confidence_reliability` and `selective_fusion_curve` (using
    `text_conf` as the trust score for the latter) and returns both, plus
    a one-line verdict in `headline`.
    """
    reliability = confidence_reliability(text_score, text_conf, y_true, n_bins=n_bins)
    selective = selective_fusion_curve(
        y_true, p_baseline, p_fused, text_conf, n_steps=n_steps
    )

    slope = reliability.attrs.get("slope", float("nan"))
    top_delta = (
        float(selective.iloc[0]["delta_auprc"])
        if len(selective) > 0
        else float("nan")
    )
    if np.isnan(slope) or np.isnan(top_delta):
        headline = "insufficient data to verdict"
    elif slope > 0.05 and top_delta > 0:
        headline = "confidence is a real trust signal; selective fusion helps"
    elif slope > 0.05:
        headline = "confidence tracks accuracy but fusion gains are flat at the top"
    elif top_delta > 0:
        headline = "confidence does not track accuracy, but selective fusion still helps"
    else:
        headline = "confidence is not informative; selective fusion does not help"

    return {
        "reliability": reliability,
        "selective": selective,
        "slope": slope,
        "top_delta_auprc": top_delta,
        "headline": headline,
    }
