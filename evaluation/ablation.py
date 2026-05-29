"""
ablation.py — is the multi-agent approach actually worth its complexity?

Two skeptical comparisons:

  - meta_model_comparison  : a vanilla Logistic-Regression that takes the
                              numeric features PLUS the text score, confidence,
                              and a few regex flags as additional features.
                              Can a single linear model match the MAS?

  - grid_search_comparison : brute-force grid over `grade_weights` (up to a
                              few hundred combinations) to find the best test
                              AUC achievable with the same fusion mechanism the
                              Strategist uses. Tests whether the LLM Strategist
                              meaningfully beats dumb search.

  - ablation_summary        : consolidates the above + the pipeline comparison
                              into one decision-grade table.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import run_pipelines


def _add_text_flags(df: pd.DataFrame, desc_col: str = "desc") -> pd.DataFrame:
    """Regex-based binary text flags used as extra LR features."""
    out = df.copy()
    desc = out[desc_col].fillna("").astype(str).str.lower() if desc_col in out.columns else None
    if desc is None:
        for f in ("flag_consolidation", "flag_repayment_plan", "flag_stress", "flag_stable_income"):
            out[f] = 0.0
        return out
    out["flag_consolidation"] = desc.str.contains("consolidat").astype(float)
    out["flag_repayment_plan"] = desc.str.contains(
        r"will pay|plan to|repay|pay off", regex=True
    ).astype(float)
    out["flag_stress"] = desc.str.contains(
        r"behind|urgent|emergency|struggling|desperate", regex=True
    ).astype(float)
    out["flag_stable_income"] = desc.str.contains(
        r"stable|steady|permanent|full.time|full time", regex=True
    ).astype(float)
    return out


def meta_model_comparison(
    subset_train: pd.DataFrame,
    subset_test: pd.DataFrame,
    num_features: list[str],
    *,
    add_text_flags: bool = True,
    label_col: str = "label",
    grade_col: str = "grade",
    C: float = 0.1,
    max_iter: int = 1000,
) -> dict:
    """Train an LR meta-model (numeric + text features) and evaluate it.

    The LR baseline is the simplest "use text as features" approach. If it
    matches or beats the MAS Strategist, the multi-agent overhead is hard
    to justify.

    Features used (when present in `subset_train`):
        num_features
        + text_risk_score, text_confidence
        + flag_consolidation, flag_repayment_plan, flag_stress, flag_stable_income
          (regex flags, added if `add_text_flags=True` and `desc` column exists)

    Returns
    -------
    dict with keys:
        meta_test_auc, meta_test_auprc,
        per_grade_meta_auc        : {grade: {n, auc, auprc}}
        features_used             : list[str]
        n_train, n_test
    """
    tr = subset_train.copy()
    te = subset_test.copy()
    if add_text_flags:
        tr = _add_text_flags(tr)
        te = _add_text_flags(te)

    feats = list(num_features)
    for col in ("text_risk_score", "text_confidence"):
        if col in tr.columns and col in te.columns:
            feats.append(col)
    if add_text_flags:
        for col in ("flag_consolidation", "flag_repayment_plan",
                    "flag_stress", "flag_stable_income"):
            if col in tr.columns:
                feats.append(col)

    X_tr = tr[feats].fillna(0)
    y_tr = tr[label_col].astype(int).values
    X_te = te[feats].fillna(0)
    y_te = te[label_col].astype(int).values

    lr = LogisticRegression(class_weight="balanced", max_iter=max_iter, C=C)
    lr.fit(X_tr, y_tr)
    p_te = lr.predict_proba(X_te)[:, 1]

    per_grade: dict[str, dict] = {}
    if grade_col in te.columns:
        for g in sorted(te[grade_col].unique().tolist()):
            mask = (te[grade_col] == g).values
            y_g = y_te[mask]
            if mask.sum() < 10 or len(np.unique(y_g)) < 2:
                continue
            per_grade[str(g)] = {
                "n": int(mask.sum()),
                "auc": float(roc_auc_score(y_g, p_te[mask])),
                "auprc": float(average_precision_score(y_g, p_te[mask])),
            }

    return {
        "meta_test_auc": float(roc_auc_score(y_te, p_te)),
        "meta_test_auprc": float(average_precision_score(y_te, p_te)),
        "per_grade_meta_auc": per_grade,
        "features_used": feats,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


_DEFAULT_GRID = {
    "C": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "D": [0.0, 0.05, 0.1, 0.15, 0.2],
    "E": [0.0, 0.05, 0.1],
    "F": [0.0],
    "G": [0.0],
    "thr": [0.55, 0.60, 0.65, 0.70, 0.75],
}


def grid_search_comparison(
    green: Any,
    subset_train: pd.DataFrame,
    subset_test: pd.DataFrame,
    strategist_strategy: dict,
    *,
    cands_c: list[float] | None = None,
    cands_d: list[float] | None = None,
    cands_e: list[float] | None = None,
    cands_f: list[float] | None = None,
    cands_g: list[float] | None = None,
    cands_thr: list[float] | None = None,
    label_col: str = "label",
) -> dict:
    """Brute-force grid over `grade_weights` and `conf_threshold`.

    Picks the candidate that maximizes TRAIN AUC, then reports its TEST AUC.
    The Strategist must beat this grid (or at least match it) to justify the
    multi-agent overhead.

    Returns
    -------
    dict with keys:
        n_candidates,
        best_grid_strategy,
        grid_train_auc, grid_test_auc, grid_test_auprc,
        strategist_test_auc, strategist_test_auprc,
        delta_strategist_vs_grid_auc, delta_strategist_vs_grid_auprc
    """
    cc = cands_c if cands_c is not None else _DEFAULT_GRID["C"]
    cd = cands_d if cands_d is not None else _DEFAULT_GRID["D"]
    ce = cands_e if cands_e is not None else _DEFAULT_GRID["E"]
    cf = cands_f if cands_f is not None else _DEFAULT_GRID["F"]
    cg = cands_g if cands_g is not None else _DEFAULT_GRID["G"]
    ct = cands_thr if cands_thr is not None else _DEFAULT_GRID["thr"]

    y_tr = subset_train[label_col].astype(int).values
    y_te = subset_test[label_col].astype(int).values

    best_auc = -1.0
    best_strat: dict | None = None
    n_combos = 0
    for w_c, w_d, w_e, w_f, w_g, thr in itertools.product(cc, cd, ce, cf, cg, ct):
        n_combos += 1
        cand = {
            "grade_weights": {"C": w_c, "D": w_d, "E": w_e, "F": w_f, "G": w_g},
            "conf_threshold": thr,
        }
        preds = run_pipelines(green, subset_train, cand)
        try:
            auc_tr = roc_auc_score(y_tr, preds["strategist"])
        except ValueError:
            continue
        if auc_tr > best_auc:
            best_auc = float(auc_tr)
            best_strat = cand

    if best_strat is None:
        raise RuntimeError("grid_search_comparison: no valid candidate produced a usable AUC")

    # Evaluate winner on test
    preds_te = run_pipelines(green, subset_test, best_strat)
    grid_test_auc = float(roc_auc_score(y_te, preds_te["strategist"]))
    grid_test_auprc = float(average_precision_score(y_te, preds_te["strategist"]))

    # Strategist's performance on the same test set
    preds_strat = run_pipelines(green, subset_test, strategist_strategy)
    strat_test_auc = float(roc_auc_score(y_te, preds_strat["strategist"]))
    strat_test_auprc = float(average_precision_score(y_te, preds_strat["strategist"]))

    return {
        "n_candidates": n_combos,
        "best_grid_strategy": best_strat,
        "grid_train_auc": best_auc,
        "grid_test_auc": grid_test_auc,
        "grid_test_auprc": grid_test_auprc,
        "strategist_test_auc": strat_test_auc,
        "strategist_test_auprc": strat_test_auprc,
        "delta_strategist_vs_grid_auc": strat_test_auc - grid_test_auc,
        "delta_strategist_vs_grid_auprc": strat_test_auprc - grid_test_auprc,
    }


def ablation_summary(
    pipeline_comparison_df: pd.DataFrame,
    meta_result: dict,
    grid_result: dict,
) -> pd.DataFrame:
    """Consolidate the 'is MAS worth it' answers into one decision-grade table.

    Inputs are the outputs of:
      - pipeline_comparison(...)    → indexed by pipeline name
      - meta_model_comparison(...)  → dict
      - grid_search_comparison(...) → dict

    Returns
    -------
    DataFrame with columns:
        method, test_auc, test_auprc, vs_baseline_auc, vs_baseline_auprc, note
    Rows:
        baseline-only          (XGBoost on numeric)
        LR stacking            (LR over numeric + text features + regex flags)
        Grid search (best)     (upper bound for grade_weights at the same fusion mechanism)
        MAS Strategist         (the multi-agent system, 5 LLM-tuned scalars)
    """
    pcd = pipeline_comparison_df
    if "baseline" not in pcd.index or "strategist" not in pcd.index:
        raise ValueError(
            "ablation_summary expects pipeline_comparison_df indexed by 'baseline' and 'strategist'"
        )
    base_auc = float(pcd.loc["baseline", "auc"])
    base_auprc = float(pcd.loc["baseline", "auprc"])
    strat_auc = float(pcd.loc["strategist", "auc"])
    strat_auprc = float(pcd.loc["strategist", "auprc"])

    rows = [
        {
            "method": "baseline-only",
            "test_auc": base_auc,
            "test_auprc": base_auprc,
            "vs_baseline_auc": 0.0,
            "vs_baseline_auprc": 0.0,
            "note": "XGBoost on numeric features",
        },
        {
            "method": "LR stacking",
            "test_auc": float(meta_result["meta_test_auc"]),
            "test_auprc": float(meta_result["meta_test_auprc"]),
            "vs_baseline_auc": float(meta_result["meta_test_auc"]) - base_auc,
            "vs_baseline_auprc": float(meta_result["meta_test_auprc"]) - base_auprc,
            "note": f"LogReg, {len(meta_result['features_used'])} features",
        },
        {
            "method": "Grid search (best)",
            "test_auc": float(grid_result["grid_test_auc"]),
            "test_auprc": float(grid_result["grid_test_auprc"]),
            "vs_baseline_auc": float(grid_result["grid_test_auc"]) - base_auc,
            "vs_baseline_auprc": float(grid_result["grid_test_auprc"]) - base_auprc,
            "note": f"{grid_result['n_candidates']}-combo brute force",
        },
        {
            "method": "MAS Strategist",
            "test_auc": strat_auc,
            "test_auprc": strat_auprc,
            "vs_baseline_auc": strat_auc - base_auc,
            "vs_baseline_auprc": strat_auprc - base_auprc,
            "note": "Multi-agent LLM-tuned 5-scalar weights",
        },
    ]
    return pd.DataFrame(rows).set_index("method")
