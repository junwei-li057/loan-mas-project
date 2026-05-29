"""
P2 verification: run the implemented core + agent functions on real sample1
data using the saved model artifact.

Required files:
    models/loan_default_model.pkl
    data/text_analyst_results_train_matched_sample1.pkl
    data/text_analyst_results_test_matched_sample1.pkl
    loan_default.csv

Reproduces enough of cells 2, 4, 12 (data split + load_text_subset + grade
z-score normalization) to produce subset_test, then exercises:
    - evaluation.core.run_pipelines
    - evaluation.core.pipeline_metrics
    - evaluation.core.pipeline_comparison

For agent functions, builds a synthetic decision_log + final_result with the
exact field shape the milestone checks expect, then verifies:
    - evaluation.agent.per_agent_kpi
    - evaluation.agent.loop_dynamics

Run from project root:
    python notebooks/eval_verify_p2.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluation as ev  # noqa: E402
from evaluation._helpers import build_subset, synthesize_decision_log  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main verification
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 78)
    print("P2 verification — evaluation.core + evaluation.agent on sample1")
    print("=" * 78)

    # Load model artifact
    with open(REPO_ROOT / "models/loan_default_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    NUM_FEATURES = artifact["features"]
    best_strategy = artifact["best_strategy"]
    grade_norm_stats = artifact["grade_norm_stats"]
    print(f"\nLoaded model artifact")
    print(f"  features         : {len(NUM_FEATURES)}")
    print(f"  best_strategy    : {best_strategy}")

    # Reproduce train/test split from cell 4
    df_all = pd.read_csv(REPO_ROOT / "loan_default.csv", low_memory=False)
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
    )
    target_test = test_df[test_df["grade"].isin(["C", "D", "E", "F", "G"])].copy()
    subset_test = build_subset(
        target_test,
        REPO_ROOT / "data/text_analyst_results_test_matched_sample1.pkl",
        grade_norm_stats,
    )
    print(f"\nReconstructed subset_test : {len(subset_test)} rows  "
          f"default_rate={subset_test['label'].mean():.3f}")

    # Synthetic green
    green = SimpleNamespace(model=artifact["model"], num_features=NUM_FEATURES)

    # --- core.run_pipelines ---
    print("\n--- core.run_pipelines ---")
    preds = ev.run_pipelines(green, subset_test, best_strategy)
    print(f"  pipelines produced : {list(preds.keys())}")
    print(f"  baseline    range  : [{preds['baseline'].min():.4f}, {preds['baseline'].max():.4f}]  mean={preds['baseline'].mean():.4f}")
    print(f"  strategist  range  : [{preds['strategist'].min():.4f}, {preds['strategist'].max():.4f}]  mean={preds['strategist'].mean():.4f}")
    diff = preds["strategist"] - preds["baseline"]
    print(f"  text-shifted rows  : {int((np.abs(diff) > 1e-9).sum())} / {len(diff)}")

    # --- core.pipeline_metrics on baseline alone ---
    print("\n--- core.pipeline_metrics (baseline) ---")
    m = ev.pipeline_metrics("baseline", subset_test["label"].values, preds["baseline"], n_bootstrap=200)
    for k in ("name", "n", "default_rate", "auc", "auc_ci_lo", "auc_ci_hi",
              "auprc", "auprc_ci_lo", "auprc_ci_hi", "log_loss", "brier"):
        print(f"  {k:>14s} : {m[k]}")

    # --- core.pipeline_comparison ---
    print("\n--- core.pipeline_comparison ---")
    cmp = ev.pipeline_comparison(
        subset_test["label"].values, preds, reference="baseline", n_bootstrap=200
    )
    with pd.option_context("display.max_colwidth", 30, "display.width", 140):
        print(cmp.to_string())

    # --- agent.per_agent_kpi ---
    print("\n--- agent.per_agent_kpi (synthetic decision_log) ---")
    syn_log, syn_final = synthesize_decision_log()
    kpi_df = ev.per_agent_kpi(syn_log, syn_final)
    print(kpi_df.to_string())
    print(f"\n  attrs:")
    print(f"    n_iters          : {kpi_df.attrs['n_iters']}")
    print(f"    milestones_fired : {kpi_df.attrs['milestones_fired']}")
    print(f"    total milestones : {len(kpi_df.attrs['milestones'])}")

    # --- agent.loop_dynamics ---
    print("\n--- agent.loop_dynamics ---")
    dyn = ev.loop_dynamics(syn_log)
    print(f"  n_iters              : {dyn['n_iters']}")
    print(f"  converged            : {dyn['converged']}")
    print(f"  veto_rate            : {dyn['veto_rate']:.3f}")
    print(f"  event_counts         : {dyn['event_counts']}")
    print(f"  advocate_modes       : {dyn['advocate_modes']}")
    print(f"  arbitration_outcomes : {dyn['arbitration_outcomes']}")
    print(f"  per_iter_trace[0]    : {dyn['per_iter_trace'][0]}")

    print("\n" + "=" * 78)
    print("OK: all P2 functions executed successfully on real / synthetic data.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
