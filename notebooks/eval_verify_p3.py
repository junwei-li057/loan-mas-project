"""
P3 end-to-end verification.

Exercises every public function in `evaluation` and writes a real MD report
to `docs/eval_report_sample1.md`.

Uses sample1's train/test text pkls and the saved model artifact. The
decision_log is synthetic (we don't re-run the iterative loop here).

Run from project root:
    python notebooks/eval_verify_p3.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluation as ev  # noqa: E402
from evaluation._helpers import build_subset, synthesize_decision_log  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("P3 end-to-end verification — full evaluation suite on sample1")
    print("=" * 78)

    # 1. Load artifacts
    with open(REPO_ROOT / "models/loan_default_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    NUM_FEATURES = artifact["features"]
    best_strategy = artifact["best_strategy"]
    grade_norm_stats = artifact["grade_norm_stats"]

    df_all = pd.read_csv(REPO_ROOT / "loan_default.csv", low_memory=False)
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
    )
    target_train = train_df[train_df["grade"].isin(["C", "D", "E", "F", "G"])].copy()
    target_test = test_df[test_df["grade"].isin(["C", "D", "E", "F", "G"])].copy()

    subset_train = build_subset(
        target_train,
        REPO_ROOT / "data/text_analyst_results_train_matched_sample1.pkl",
        grade_norm_stats,
    )
    subset_test = build_subset(
        target_test,
        REPO_ROOT / "data/text_analyst_results_test_matched_sample1.pkl",
        grade_norm_stats,
    )
    print(f"\n  subset_train : {len(subset_train)} rows  default={subset_train['label'].mean():.3f}")
    print(f"  subset_test  : {len(subset_test)} rows  default={subset_test['label'].mean():.3f}")

    green = SimpleNamespace(model=artifact["model"], num_features=NUM_FEATURES)

    # 2. core: predictions + comparison
    print("\n[core] run_pipelines + pipeline_comparison …")
    preds = ev.run_pipelines(green, subset_test, best_strategy)
    pipeline_cmp = ev.pipeline_comparison(
        subset_test["label"].values, preds, reference="baseline", n_bootstrap=200
    )
    print(pipeline_cmp.to_string())

    # 3. agent: KPI + loop_dynamics
    # Prefer a real decision_log pickled by the training notebook (see README
    # for the dump snippet). Fall back to a 4-iter synthetic log only when no
    # real one is available; the report metadata records which one was used so
    # readers know whether §3 / §4 reflect a real run or a stand-in.
    decision_log_path = REPO_ROOT / "data" / "decision_log_sample1.pkl"
    if decision_log_path.exists():
        with open(decision_log_path, "rb") as f:
            log_blob = pickle.load(f)
        syn_log = log_blob["decision_log"]
        syn_final = log_blob["final_result"]
        decision_log_source = f"real ({len(syn_log)}-iter, from {decision_log_path.name})"
        print(f"\n[agent] per_agent_kpi + loop_dynamics on REAL {len(syn_log)}-iter log "
              f"({decision_log_path.name}) …")
    else:
        syn_log, syn_final = synthesize_decision_log()
        decision_log_source = "synthetic (4-iter; no real log found)"
        print(f"\n[agent] ⚠️  no real decision_log at {decision_log_path}; "
              f"using synthetic 4-iter log …")
    agent_kpi = ev.per_agent_kpi(syn_log, syn_final)
    loop_dyn = ev.loop_dynamics(syn_log)
    print(agent_kpi.to_string())
    print(f"  converged={loop_dyn['converged']}  veto_rate={loop_dyn['veto_rate']:.3f}  "
          f"events={loop_dyn['event_counts']}")

    # 4. ablation: meta-LR + grid search + summary
    print("\n[ablation] meta_model_comparison …")
    meta_res = ev.meta_model_comparison(subset_train, subset_test, NUM_FEATURES)
    print(f"  meta_test_auc={meta_res['meta_test_auc']:.4f}  "
          f"meta_test_auprc={meta_res['meta_test_auprc']:.4f}  "
          f"n_features={len(meta_res['features_used'])}")

    print("\n[ablation] grid_search_comparison (this may take a few seconds) …")
    grid_res = ev.grid_search_comparison(
        green, subset_train, subset_test, best_strategy,
        cands_c=[0.0, 0.05, 0.1, 0.2, 0.3],
        cands_d=[0.0, 0.05, 0.1, 0.15],
        cands_e=[0.0, 0.05, 0.1],
        # Symmetric F / G coverage so grid search is not handicapped vs MAS,
        # which can pick non-zero weights for any grade.
        cands_f=[0.0, 0.05, 0.10],
        cands_g=[0.0, 0.05, 0.10],
        cands_thr=[0.60, 0.65, 0.70],
    )
    print(f"  n_candidates={grid_res['n_candidates']}  "
          f"grid_test_auc={grid_res['grid_test_auc']:.4f}  "
          f"strategist_test_auc={grid_res['strategist_test_auc']:.4f}  "
          f"Δ={grid_res['delta_strategist_vs_grid_auc']:+.4f}")

    print("\n[ablation] ablation_summary …")
    ablation = ev.ablation_summary(pipeline_cmp, meta_res, grid_res)
    print(ablation.to_string())

    # 5. trust
    print("\n[trust] trust_calibration …")
    trust = ev.trust_calibration(
        subset_test["text_risk_score"].values,
        subset_test["text_confidence"].values,
        subset_test["label"].values,
        preds["baseline"],
        preds["strategist"],
    )
    print(f"  headline    : {trust['headline']}")
    print(f"  slope       : {trust['slope']:.4f}")
    print(f"  top ΔAUPRC  : {trust['top_delta_auprc']:+.4f}")
    print(f"  reliability bins : {len(trust['reliability'])}")
    print(f"  selective points : {len(trust['selective'])}")

    # 6. report: generate the consolidated MD
    print("\n[report] generate_report …")
    out_path = REPO_ROOT / "docs" / "eval_report_sample1.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = ev.generate_report(
        pipeline_comparison_df=pipeline_cmp,
        agent_kpi_df=agent_kpi,
        loop_dyn=loop_dyn,
        ablation_df=ablation,
        trust_diag=trust,
        out_path=out_path,
        meta={
            "data": "sample1 (train/test)",
            "model_artifact": "models/loan_default_model.pkl",
            "decision_log": decision_log_source,
        },
    )
    print(f"  wrote {written}  ({written.stat().st_size} bytes)")

    print("\n" + "=" * 78)
    print(f"OK: P3 end-to-end complete. Report → {written}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
