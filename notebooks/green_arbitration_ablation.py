"""
Green-as-Arbitrator ablation.

This is a lightweight arbitration-layer comparison, not a full XGBoost replay.
It asks: if a high-confidence text signal wants to influence the prediction,
what changes when Green owns arbitration and applies subgroup reliability checks?

Inputs:
  - text_ana_results/text_analysis_combined.pkl
  - a LendingClub csv with at least id, grade, label, and optionally desc

Example:
    python notebooks/green_arbitration_ablation.py --loan-csv loan_default.csv

Outputs:
  - outputs/green_arbitration_ablation/*.csv
  - docs/green_arbitration_ablation.md
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMBINED = REPO_ROOT / "text_ana_results" / "text_analysis_combined.pkl"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "green_arbitration_ablation"
DEFAULT_REPORT = REPO_ROOT / "docs" / "green_arbitration_ablation.md"


def safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = y_true.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score.astype(float).to_numpy()))


def short_text(value: object, width: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[: width - 3] + "..." if len(text) > width else text


def load_joined(loan_csv: Path, combined_pkl: Path) -> pd.DataFrame:
    if not loan_csv.exists():
        raise FileNotFoundError(
            f"Loan CSV not found: {loan_csv}. Pass --loan-csv /path/to/loan_default.csv "
            "or another sample csv containing id, grade, label."
        )
    if not combined_pkl.exists():
        raise FileNotFoundError(f"Combined text-analysis pkl not found: {combined_pkl}")

    loans = pd.read_csv(loan_csv, low_memory=False)
    required = {"id", "grade", "label"}
    missing = required - set(loans.columns)
    if missing:
        raise ValueError(f"{loan_csv} is missing required columns: {sorted(missing)}")

    with open(combined_pkl, "rb") as f:
        text_rows = pickle.load(f)
    text_df = pd.DataFrame(text_rows)
    score_col = (
        "custom_v2_text_risk_score"
        if "custom_v2_text_risk_score" in text_df.columns
        else "text_risk_score"
    )
    keep = [
        "id",
        score_col,
        "confidence",
        "reasoning",
        "desc_preview",
        "fairness_action",
    ]
    keep = [c for c in keep if c in text_df.columns]
    text_df = text_df[keep].rename(columns={score_col: "text_score"})
    text_df = text_df.dropna(subset=["id", "text_score", "confidence"])

    loan_cols = ["id", "grade", "label"] + [c for c in ["desc", "loan_status"] if c in loans.columns]
    joined = loans[loan_cols].merge(text_df, on="id", how="inner")
    joined = joined.dropna(subset=["grade", "label", "text_score", "confidence"]).copy()
    joined["label"] = joined["label"].astype(int)
    joined["grade"] = joined["grade"].astype(str)
    joined["text_score"] = joined["text_score"].astype(float)
    joined["confidence"] = joined["confidence"].astype(float)
    joined["text_gap"] = (joined["text_score"] - 0.5).abs()
    joined["text_direction_default"] = joined["text_score"] >= 0.5
    joined["text_correct"] = joined["text_direction_default"].astype(int) == joined["label"]
    return joined


def run_ablation(
    df: pd.DataFrame,
    *,
    confidence_threshold: float,
    text_gap_threshold: float,
    grade_auc_support_threshold: float,
    extreme_confidence_threshold: float,
    extreme_gap_threshold: float,
) -> dict[str, pd.DataFrame]:
    grade_rows = []
    grade_auc: dict[str, float] = {}
    for grade, sub in df.groupby("grade"):
        auc = safe_auc(sub["label"], sub["text_score"])
        grade_auc[grade] = auc
        high_conf = sub["confidence"] >= confidence_threshold
        grade_rows.append(
            {
                "grade": grade,
                "n": len(sub),
                "default_rate": sub["label"].mean(),
                "text_auc": auc,
                "high_conf_n": int(high_conf.sum()),
                "high_conf_text_direction_accuracy": sub.loc[high_conf, "text_correct"].mean(),
            }
        )
    grade_df = pd.DataFrame(grade_rows).sort_values("grade")

    # Old-design proxy: a separate White Arbitrator accepts high-confidence,
    # nontrivial text evidence without owning the subgroup safety policy.
    old_acts = (df["confidence"] >= confidence_threshold) & (df["text_gap"] >= text_gap_threshold)

    # New-design proxy: Green lets text act only when subgroup text history is
    # supportive or when borrower-level text is extreme enough for an exception.
    grade_support = df["grade"].map(
        lambda g: grade_auc.get(str(g), float("nan")) >= grade_auc_support_threshold
    )
    extreme_exception = (
        (df["confidence"] >= extreme_confidence_threshold)
        & (df["text_gap"] >= extreme_gap_threshold)
    )
    new_acts = old_acts & (grade_support | extreme_exception)

    work = df.copy()
    work["old_design_action"] = old_acts
    work["new_green_action"] = new_acts
    work["green_suppressed_old_action"] = old_acts & ~new_acts
    work["old_wrong_new_suppressed"] = work["green_suppressed_old_action"] & ~work["text_correct"]
    work["old_right_new_suppressed"] = work["green_suppressed_old_action"] & work["text_correct"]

    summary = pd.DataFrame(
        [
            summarize_design(work, old_acts, "old_proxy_separate_arbitrator"),
            summarize_design(work, new_acts, "new_green_as_arbitrator"),
        ]
    )
    events = pd.DataFrame(
        [
            {"event": "old_design_acted", "count": int(old_acts.sum())},
            {"event": "new_green_acted", "count": int(new_acts.sum())},
            {"event": "green_suppressed_old_action", "count": int((old_acts & ~new_acts).sum())},
            {"event": "old_wrong_new_suppressed", "count": int(work["old_wrong_new_suppressed"].sum())},
            {"event": "old_right_new_suppressed", "count": int(work["old_right_new_suppressed"].sum())},
            {"event": "extreme_exception_allowed", "count": int((old_acts & extreme_exception & ~grade_support).sum())},
        ]
    )

    case_cols = [
        c
        for c in [
            "id",
            "grade",
            "label",
            "loan_status",
            "text_score",
            "confidence",
            "text_gap",
            "desc_preview",
            "desc",
            "reasoning",
        ]
        if c in work.columns
    ]
    better = work.loc[work["old_wrong_new_suppressed"], case_cols].copy()
    better["why_new_better"] = (
        "Old proxy would trust high-confidence text in a weak subgroup; "
        "Green suppresses it because grade-level text history is below the support threshold."
    )
    cost = work.loc[work["old_right_new_suppressed"], case_cols].copy()
    cost["why_old_better"] = (
        "Old proxy would trust text and be right; Green suppresses it because subgroup history is weak."
    )
    for frame in (better, cost):
        if "desc_preview" in frame.columns:
            frame["desc_preview"] = frame["desc_preview"].map(short_text)
        if "desc" in frame.columns:
            frame["desc"] = frame["desc"].map(short_text)

    better = better.sort_values(["confidence", "text_gap"], ascending=False).head(25)
    cost = cost.sort_values(["confidence", "text_gap"], ascending=False).head(25)

    return {
        "grade_text_reliability": grade_df,
        "design_summary": summary,
        "event_counts": events,
        "cases_new_green_better": better,
        "cases_old_proxy_better": cost,
    }


def summarize_design(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, float | int | str]:
    says_default = mask & df["text_direction_default"]
    says_safe = mask & ~df["text_direction_default"]
    return {
        "design": name,
        "acted_n": int(mask.sum()),
        "acted_share": float(mask.mean()),
        "action_accuracy": float(df.loc[mask, "text_correct"].mean()) if mask.any() else float("nan"),
        "default_precision_when_text_says_default": float(df.loc[says_default, "label"].mean())
        if says_default.any()
        else float("nan"),
        "nondefault_precision_when_text_says_safe": float(1 - df.loc[says_safe, "label"].mean())
        if says_safe.any()
        else float("nan"),
    }


def write_outputs(results: dict[str, pd.DataFrame], outdir: Path, report_path: Path, meta: dict[str, object]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)

    summary = results["design_summary"]
    events = results["event_counts"]
    grades = results["grade_text_reliability"]

    old = summary.iloc[0]
    new = summary.iloc[1]
    precision_delta = (
        new["default_precision_when_text_says_default"]
        - old["default_precision_when_text_says_default"]
    )
    accuracy_delta = new["action_accuracy"] - old["action_accuracy"]

    markdown = f"""# Green Arbitration Ablation

This is a lightweight arbitration-layer comparison, not a full XGBoost replay.
It compares two policy proxies on a joined LendingClub/text-analysis sample:

- **Old proxy, separate Arbitrator:** high-confidence, nontrivial text signals are allowed to act broadly.
- **New Green-as-Arbitrator:** the same signals act only when grade-level text history supports them, or when borrower-level evidence is extreme enough for an exception.

## Data

- Loan CSV: `{display_path(Path(meta["loan_csv"]))}`
- Text analysis: `{display_path(Path(meta["combined_pkl"]))}`
- Joined rows: `{meta["n_joined"]}`
- Confidence threshold: `{meta["confidence_threshold"]:.2f}`
- Text-gap threshold: `{meta["text_gap_threshold"]:.2f}`
- Grade support threshold: text AUC >= `{meta["grade_auc_support_threshold"]:.2f}`

## Result Summary

{df_to_markdown(summary)}

## Event Counts

{df_to_markdown(events)}

## Grade Text Reliability

{df_to_markdown(grades)}

## Interpretation

Green-as-Arbitrator is a selective-trust mechanism, not a universal metric booster.
In this run it changed action coverage from `{old["acted_n"]}` rows to `{new["acted_n"]}` rows.
Action accuracy changed by `{accuracy_delta:+.4f}`, while precision when text says default changed by `{precision_delta:+.4f}`.

The key tradeoff is visible in the case files:

- `cases_new_green_better.csv`: old proxy would trust text and be wrong; Green suppresses it.
- `cases_old_proxy_better.csv`: old proxy would trust text and be right; Green suppresses it.

Presentation-safe takeaway:

> Green arbitration trades coverage for selective trust. Its job is not to let text influence every confident case; its job is to decide when text is reliable enough to deserve influence.
"""
    report_path.write_text(markdown, encoding="utf-8")


def display_path(path: Path) -> str:
    """Prefer repo-relative display paths; avoid leaking local home paths."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return path.name


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional deps."""
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            return f"{float(value):.4f}"
        return str(value)

    headers = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loan-csv",
        type=Path,
        default=REPO_ROOT / "loan_default.csv",
        help="LendingClub csv with id, grade, label columns.",
    )
    parser.add_argument("--combined-pkl", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--text-gap-threshold", type=float, default=0.10)
    parser.add_argument("--grade-auc-support-threshold", type=float, default=0.51)
    parser.add_argument("--extreme-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--extreme-gap-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_joined(args.loan_csv, args.combined_pkl)
    results = run_ablation(
        df,
        confidence_threshold=args.confidence_threshold,
        text_gap_threshold=args.text_gap_threshold,
        grade_auc_support_threshold=args.grade_auc_support_threshold,
        extreme_confidence_threshold=args.extreme_confidence_threshold,
        extreme_gap_threshold=args.extreme_gap_threshold,
    )
    write_outputs(
        results,
        args.outdir,
        args.report_path,
        {
            "loan_csv": args.loan_csv,
            "combined_pkl": args.combined_pkl,
            "n_joined": len(df),
            "confidence_threshold": args.confidence_threshold,
            "text_gap_threshold": args.text_gap_threshold,
            "grade_auc_support_threshold": args.grade_auc_support_threshold,
        },
    )

    print("\n=== Design summary ===")
    print(results["design_summary"].to_string(index=False))
    print("\n=== Event counts ===")
    print(results["event_counts"].to_string(index=False))
    print(f"\nWrote report: {args.report_path}")
    print(f"Wrote CSV outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
