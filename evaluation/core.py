"""
core.py — head-to-head comparison of the two pipelines on a held-out set.

Pipelines compared:
  - baseline-only           XGBoost on numeric features, no text fusion
  - Strategist (5 scalars)  MAS-tuned grade_weights, logit-space fusion

Key MAS question answered here: does the multi-agent system produce a
statistically meaningful improvement over the numeric-only baseline?
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

# Below this evaluation-set size, bootstrap CIs widen to the point where
# pipeline differences cannot be meaningfully distinguished from noise. We
# still compute the metrics, but warn so the caller and the report reader
# do not over-interpret the numbers.
_MIN_INFORMATIVE_N = 100
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss as _log_loss,
    brier_score_loss,
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def pipeline_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Compute AUPRC, AUC, log-loss, Brier with bootstrap CIs for one pipeline.

    Returns
    -------
    dict with keys:
        name, n, default_rate,
        auc, auc_ci_lo, auc_ci_hi,
        auprc, auprc_ci_lo, auprc_ci_hi,
        log_loss, brier
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_pred_proba, dtype=float), 1e-7, 1 - 1e-7)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        raise ValueError(
            f"pipeline_metrics({name}): need ≥1 sample and both classes present"
        )

    auc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    ll = float(_log_loss(y, p, labels=[0, 1]))
    brier = float(brier_score_loss(y, p))

    rng = np.random.default_rng(seed)
    auc_samples = np.empty(n_bootstrap, dtype=float)
    auprc_samples = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_samples[valid] = roc_auc_score(y_b, p_b)
        auprc_samples[valid] = average_precision_score(y_b, p_b)
        valid += 1
    auc_samples = auc_samples[:valid]
    auprc_samples = auprc_samples[:valid]
    auc_lo, auc_hi = (float(np.quantile(auc_samples, q)) for q in (0.025, 0.975))
    auprc_lo, auprc_hi = (float(np.quantile(auprc_samples, q)) for q in (0.025, 0.975))

    return {
        "name": name,
        "n": int(n),
        "default_rate": float(y.mean()),
        "auc": auc,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "auprc": auprc,
        "auprc_ci_lo": auprc_lo,
        "auprc_ci_hi": auprc_hi,
        "log_loss": ll,
        "brier": brier,
    }


def _delong_auc_pvalue(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> tuple[float, float, float]:
    """DeLong's test for two correlated ROC curves on the same y_true.

    Implements Sun & Xu (2014) fast algorithm for the structural variance/covariance
    of the U-statistic estimator of AUC. Returns (auc_a, auc_b, two-sided p-value).

    Reference: Xu Sun & Weichao Xu, "Fast implementation of DeLong's algorithm for
    comparing the areas under correlated receiver operating characteristic curves"
    IEEE Signal Processing Letters, 2014.
    """
    from scipy.stats import norm

    y = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        ranks = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and x_sorted[j] == x_sorted[i]:
                j += 1
            ranks[i:j] = 0.5 * (i + j + 1)
            i = j
        out = np.empty(len(x), dtype=float)
        out[order] = ranks
        return out

    def _structural(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x = scores[pos]
        y_ = scores[neg]
        tx = _midrank(x)
        ty = _midrank(y_)
        tz = _midrank(scores)
        auc = (tz[pos].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[pos] - tx) / n
        v10 = 1.0 - (tz[neg] - ty) / m
        return float(auc), v01, v10

    auc_a, v01_a, v10_a = _structural(p_a)
    auc_b, v01_b, v10_b = _structural(p_b)
    s01 = np.cov(np.stack([v01_a, v01_b]), ddof=1)
    s10 = np.cov(np.stack([v10_a, v10_b]), ddof=1)
    cov = s01 / m + s10 / n
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        return auc_a, auc_b, float("nan")
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    return auc_a, auc_b, p_value


def pipeline_comparison(
    y_true: np.ndarray,
    preds_by_pipeline: dict[str, np.ndarray],
    *,
    reference: str = "baseline",
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Side-by-side metrics for all pipelines + DeLong p-values vs `reference`.

    Returns
    -------
    DataFrame indexed by pipeline name with columns:
        n, default_rate, auc, auc_ci, auprc, auprc_ci,
        log_loss, brier, delong_p_vs_ref
    Where *_ci is a (lo, hi) tuple. delong_p_vs_ref is NaN for the reference row.
    """
    if reference not in preds_by_pipeline:
        raise ValueError(
            f"reference='{reference}' not found in preds_by_pipeline keys: "
            f"{list(preds_by_pipeline)}"
        )
    y = np.asarray(y_true).astype(int)
    if len(y) < _MIN_INFORMATIVE_N:
        warnings.warn(
            f"pipeline_comparison: n={len(y)} is below the recommended minimum of "
            f"{_MIN_INFORMATIVE_N}; bootstrap CIs and DeLong p-values will be wide "
            f"and conclusions may not be statistically meaningful.",
            stacklevel=2,
        )
    p_ref = np.asarray(preds_by_pipeline[reference], dtype=float)

    rows = []
    for name, preds in preds_by_pipeline.items():
        m = pipeline_metrics(name, y, preds, n_bootstrap=n_bootstrap, seed=seed)
        if name == reference:
            p_value = float("nan")
        else:
            _, _, p_value = _delong_auc_pvalue(y, p_ref, np.asarray(preds, dtype=float))
        rows.append(
            {
                "pipeline": name,
                "n": m["n"],
                "default_rate": m["default_rate"],
                "auc": m["auc"],
                "auc_ci": (m["auc_ci_lo"], m["auc_ci_hi"]),
                "auprc": m["auprc"],
                "auprc_ci": (m["auprc_ci_lo"], m["auprc_ci_hi"]),
                "log_loss": m["log_loss"],
                "brier": m["brier"],
                "delong_p_vs_ref": p_value,
            }
        )
    return pd.DataFrame(rows).set_index("pipeline")


def run_pipelines(
    green: Any,
    subset_df: pd.DataFrame,
    strategy: dict,
) -> dict[str, np.ndarray]:
    """Compute predicted probabilities under each pipeline on the same set.

    Parameters
    ----------
    green     : an object exposing `.model` (sklearn-style with `predict_proba`)
                and `.num_features` (list of column names)
    subset_df : DataFrame with columns the strategy needs
                (text_risk_score, text_confidence, grade, num features)
    strategy  : dict with 'grade_weights' and 'conf_threshold'

    Returns
    -------
    dict {'baseline': preds, 'strategist': preds}

    Notes
    -----
    Fusion is reproduced inline (logit-space, confidence-thresholded) to keep
    this module independent of GreenAgent.evaluate_fusion. The formula matches
    the Strategist branch of cell 15 of the notebook:

        fused_logit = logit(baseline) + text_weight × conf × [logit(text) − logit(0.5)]

    with `text_weight = grade_weights[grade]` when `text_confidence ≥ conf_threshold`,
    else `text_weight = 0`.
    """
    df = subset_df.reset_index(drop=True)
    X = df[green.num_features].fillna(0)
    baseline = green.model.predict_proba(X)[:, 1].astype(float)

    grade_weights = dict(strategy.get("grade_weights", {}))
    conf_threshold = float(strategy.get("conf_threshold", 0.65))

    text_weight = df["grade"].map(grade_weights).fillna(0.0).astype(float).values
    conf = df["text_confidence"].astype(float).values
    text_weight = np.where(conf < conf_threshold, 0.0, text_weight)

    text_evidence = _logit(df["text_risk_score"].values) - _logit(np.full_like(baseline, 0.5))
    fused_logit = _logit(baseline) + text_weight * conf * text_evidence
    strategist = np.where(text_weight == 0, baseline, _sigmoid(fused_logit))

    return {"baseline": baseline, "strategist": strategist}
