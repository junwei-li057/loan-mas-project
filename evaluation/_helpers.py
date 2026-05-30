"""
Internal helpers shared by the eval_verify driver scripts.

Underscore-prefixed because nothing here is part of the public `evaluation`
API surface — these are utilities the verification / report-generation
scripts use to reconstruct the evaluation subset and to provide a stand-in
decision_log when no real one has been dumped from training.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def build_subset(
    df_split: pd.DataFrame,
    text_pkl: Path,
    grade_norm_stats: dict,
) -> pd.DataFrame:
    """Legacy idx-based join, kept for backward compatibility.

    Joins cached text-analyst outputs onto a train/test split by pandas row
    `idx` and per-grade z-normalizes the text risk score. Use this only when
    you have the old `data/text_analyst_results_{train,test}_matched_sample1.pkl`
    files; for the consolidated pkl built by text_analyst.py, prefer
    `build_subset_from_combined` (id-based join, no row-order dependency).
    """
    with open(text_pkl, "rb") as f:
        results = pickle.load(f)
    rdf = pd.DataFrame(results).set_index("idx")
    if "text_risk_score" not in rdf.columns:
        rdf["text_risk_score"] = rdf["risk_label"].astype(float)
    sub = df_split.join(rdf[["text_risk_score", "confidence"]], how="left")
    sub = sub.rename(columns={"confidence": "text_confidence"})
    sub = sub.dropna(subset=["text_risk_score", "text_confidence"]).copy()
    return _apply_grade_zscore(sub, grade_norm_stats)


def build_subset_from_combined(
    df_split: pd.DataFrame,
    combined_pkl: Path,
    grade_norm_stats: dict,
) -> pd.DataFrame:
    """Id-based counterpart to `build_subset`, for the consolidated pkl.

    Reads `text_ana_results/text_analysis_combined.pkl` (or whatever path is
    passed) and inner-joins it onto `df_split` by LendingClub loan `id`. Every
    row in the consolidated pkl carries a non-null `id`, so the join does not
    depend on sample csv row order or on a particular train/test split having
    been used during text analysis.

    Per-grade z-normalization of text_risk_score uses `grade_norm_stats` from
    the saved model artifact, matching cell 12 of the training notebook so the
    online (`app/ui/Home.py`) and offline (here) paths normalize identically.
    """
    with open(combined_pkl, "rb") as f:
        rows = pickle.load(f)
    rdf = pd.DataFrame(rows)
    rdf = rdf[["id", "text_risk_score", "confidence"]].rename(
        columns={"confidence": "text_confidence"}
    )
    rdf = rdf.dropna(subset=["id", "text_risk_score", "text_confidence"])
    sub = df_split.merge(rdf, on="id", how="inner")
    return _apply_grade_zscore(sub, grade_norm_stats)


def _apply_grade_zscore(sub: pd.DataFrame, grade_norm_stats: dict) -> pd.DataFrame:
    """Per-grade z-score normalization → clipped to [0.05, 0.95].

    Shared by both build_subset variants so the normalization rule stays in
    one place; tweak it here, both paths update.
    """
    def _normalize(score, grade):
        if grade not in grade_norm_stats:
            return 0.5
        m, s = grade_norm_stats[grade]
        z = (score - m) / s
        return round(min(max(0.5 + z * 0.15, 0.05), 0.95), 4)

    sub["text_risk_score"] = sub.apply(
        lambda r: _normalize(r["text_risk_score"], r["grade"]), axis=1
    )
    return sub


def synthesize_decision_log() -> tuple[list[dict], dict]:
    """Build a 4-iter decision_log with mixed events, plus a final_result.

    Used as a stand-in when no real `decision_log_sample1.pkl` is available
    (see README "Producing a real decision_log"). The report header records
    which source was used so readers know whether §3 / §4 reflect a real run.
    """
    log = [
        {
            "iteration": 0,
            "event": "strategy_update",
            "veto": False,
            "advocate_mode": "rule",
            "expectation": {"expected_delta_auc": 0.01},
            "reflection": {"surprise": 0.03},
            "overall_baseline": 0.72,
            "overall_fused": 0.73,
            "overall_baseline_auprc": 0.45,
            "overall_fused_auprc": 0.47,
            "advocate_opportunities": ["grade_C_underweighted"],
        },
        {
            "iteration": 1,
            "event": "green_arbitrate",
            "veto": True,
            "advocate_mode": "LLM",
            "expectation": {"expected_delta_auc": 0.005},
            "reflection": {"surprise": 0.08},
            "overall_baseline": 0.72,
            "overall_fused": 0.725,
            "overall_baseline_auprc": 0.45,
            "overall_fused_auprc": 0.46,
            "advocate_opportunities": [],
        },
        {
            "iteration": 2,
            "event": "strategy_update",
            "veto": False,
            "advocate_mode": "rule",
            "expectation": {"expected_delta_auc": 0.008},
            "reflection": {"surprise": 0.02},
            "overall_baseline": 0.72,
            "overall_fused": 0.735,
            "overall_baseline_auprc": 0.45,
            "overall_fused_auprc": 0.48,
            "advocate_opportunities": ["grade_D_boost"],
        },
        {
            "iteration": 3,
            "event": "converged",
            "veto": False,
            "advocate_mode": "rule",
            "expectation": {"expected_delta_auc": 0.0005},
            "reflection": {"surprise": 0.01},
            "overall_baseline": 0.72,
            "overall_fused": 0.735,
            "overall_baseline_auprc": 0.45,
            "overall_fused_auprc": 0.48,
            "advocate_opportunities": [],
        },
    ]
    final_result = {"overall_baseline_auprc": 0.45, "overall_fused_auprc": 0.48}
    return log, final_result
