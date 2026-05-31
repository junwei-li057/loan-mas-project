"""
API-backed Green arbitration evaluation.

This script runs a small formal multi-agent comparison on real borrower cases.
It reuses cached Text Analyst scores from text_analysis_combined.pkl, then calls
MiniMax for:

  1. Feature Strategist
  2. Subgroup Advocate
  3. old proxy: standalone White Arbitrator
  4. new design: Green-as-Arbitrator

The run is resumable. Partial JSONL rows are written after every case.

Example:
    MINIMAX_API_KEY=... python notebooks/green_arbitration_api_eval.py \
      --loan-csv /Users/linzhong/stats461/loan_default_sample2.csv --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
COMBINED_PKL = REPO_ROOT / "text_ana_results" / "text_analysis_combined.pkl"
OUTDIR = REPO_ROOT / "outputs" / "green_arbitration_api_eval"
MINIMAX_URL = os.getenv("MINIMAX_URL", "https://api.minimax.io/v1/chat/completions")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
DEFAULT_MAX_TOKENS = int(os.getenv("MINIMAX_MAX_TOKENS", "6000"))
DEFAULT_TIMEOUT = int(os.getenv("MINIMAX_TIMEOUT", "180"))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def call_llm(
    prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    retries: int = 1,
) -> tuple[dict, str]:
    api_key = os.getenv("MINIMAX_API_KEY", "")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set in this process.")
    payload = {
        "model": MINIMAX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                MINIMAX_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            data = resp.json()
            if resp.status_code != 200:
                raise RuntimeError(f"MiniMax HTTP {resp.status_code}: {data}")
            raw = data["choices"][0]["message"]["content"]
            return extract_json(raw), raw
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise last_error


def extract_json(raw: str) -> dict:
    cleaned = str(raw or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.I).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError(f"No JSON object found: {cleaned[:500]}")


def historical_text_impact(delta: float) -> str:
    if delta <= -0.04:
        return "text has historically reduced reliability for this grade"
    if delta < -0.01:
        return "text has historically been weak or mildly harmful for this grade"
    if delta < 0.01:
        return "text history is close to neutral for this grade"
    return "text has historically helped this grade"


def load_joined(loan_csv: Path) -> pd.DataFrame:
    loans = pd.read_csv(loan_csv, low_memory=False)
    with open(COMBINED_PKL, "rb") as f:
        text_rows = pickle.load(f)
    text_df = pd.DataFrame(text_rows)
    score_col = "custom_v2_text_risk_score" if "custom_v2_text_risk_score" in text_df.columns else "text_risk_score"
    text_df = text_df[
        ["id", score_col, "confidence", "reasoning", "desc_preview"]
    ].rename(columns={score_col: "text_score"})
    joined = loans.merge(text_df, on="id", how="inner")
    joined = joined.dropna(subset=["grade", "label", "text_score", "confidence"]).copy()
    joined["text_score"] = joined["text_score"].astype(float)
    joined["confidence"] = joined["confidence"].astype(float)
    joined["text_gap"] = (joined["text_score"] - 0.5).abs()
    return joined


def select_cases(joined: pd.DataFrame, limit: int) -> pd.DataFrame:
    joined = joined[(joined["confidence"] >= 0.65) & (joined["text_gap"] >= 0.10)].copy()
    joined = joined.sort_values(["confidence", "text_gap"], ascending=False)
    cols = [
        "id", "grade", "label", "loan_status", "desc", "desc_preview",
        "text_score", "confidence", "reasoning", "text_gap",
    ]
    return joined[[c for c in cols if c in joined.columns]].head(limit).reset_index(drop=True)


def grade_stats_from_cases(df: pd.DataFrame) -> dict[str, dict]:
    stats = {}
    for grade, sub in df.groupby("grade"):
        # Direction accuracy is the robust signal available without XGBoost.
        pred_default = sub["text_score"] >= 0.5
        acc = float((pred_default.astype(int) == sub["label"].astype(int)).mean())
        # Convert to a small signed history delta so prompts can reason.
        delta = acc - 0.5
        stats[str(grade)] = {
            "direction_accuracy": acc,
            "history_delta": delta,
            "n": int(len(sub)),
        }
    return stats


def strategist_prompt(row: pd.Series, learned_w: float, hist_delta: float) -> str:
    return f"""
You are the Feature Strategist in a loan-default multi-agent system.
Return ONLY compact JSON. No markdown. No <think>. Keep all strings under 120 chars.

Facts: grade={row.grade}; text_score={row.text_score:.3f} (>0.5 means default risk);
confidence={row.confidence:.3f}; learned_weight={learned_w:.3f};
grade_history="{historical_text_impact(hist_delta)}" delta={hist_delta:+.3f};
borrower_text="{str(row.get('desc', row.get('desc_preview', '')))[:360]}";
text_reason="{str(row.reasoning)[:260]}".

Schema:
{{
  "strategy_type": "keep_learned|case_specific_exception|explanation_only|human_review",
  "proposed_weight": <float 0.00-0.20>,
  "rationale": "<one short sentence>",
  "key_evidence": ["<short evidence>", "<short evidence>"]
}}
"""


def advocate_prompt(row: pd.Series, learned_w: float, hist_delta: float, strat: dict) -> str:
    return f"""
You are the Subgroup Advocate. Decide whether the Strategist's proposed text weight is safe
for this grade, given subgroup reliability history.
Return ONLY compact JSON. No markdown. No <think>. Keep all strings under 120 chars.

Facts: grade={row.grade}; text_score={row.text_score:.3f}; confidence={row.confidence:.3f};
learned_weight={learned_w:.3f}; strategist_weight={float(strat.get('proposed_weight', learned_w)):.3f};
strategy={strat.get('strategy_type')}; strategist_reason="{str(strat.get('rationale'))[:220]}";
grade_history="{historical_text_impact(hist_delta)}" delta={hist_delta:+.3f};
borrower_text="{str(row.get('desc', row.get('desc_preview', '')))[:320]}".

Schema:
{{
  "decision": "pass|soft_veto|hard_veto",
  "max_allowed_weight": <float 0.00-0.20>,
  "rationale": "<one short sentence>",
  "constraints": ["<short constraint>"]
}}
"""


def old_arbitrator_prompt(row: pd.Series, learned_w: float, strat: dict, adv: dict) -> str:
    proposed = float(strat.get("proposed_weight", learned_w))
    return f"""
You are the OLD DESIGN standalone White Arbitrator.
You mediate between Strategist and Advocate, but you do NOT own the final numerical safety policy.
Choose a compromise text weight for this case.
Return ONLY compact JSON. No markdown. No <think>. Keep resolution under 140 chars.

Facts: grade={row.grade}; text_score={row.text_score:.3f}; confidence={row.confidence:.3f};
learned_weight={learned_w:.3f}; strategist_weight={proposed:.3f};
strategist_reason="{str(strat.get('rationale'))[:180]}";
advocate_decision={adv.get('decision')}; advocate_max={adv.get('max_allowed_weight')};
advocate_reason="{str(adv.get('rationale'))[:180]}".

Schema:
{{
  "final_weight": <float 0.00-0.20>,
  "ruling": "strategist|advocate|balanced",
  "resolution": "<one short sentence>"
}}
"""


def green_arbitrator_prompt(row: pd.Series, learned_w: float, hist_delta: float, strat: dict, adv: dict) -> str:
    proposed = float(strat.get("proposed_weight", learned_w))
    return f"""
You are the NEW DESIGN Green Agent acting as arbitrator and final trust authority.
Unlike the old standalone Arbitrator, you own subgroup reliability and numerical safety.
Choose the final text weight. Be conservative when grade history is weak.
Return ONLY compact JSON. No markdown. No <think>. Keep strings under 140 chars.

Facts: grade={row.grade}; text_score={row.text_score:.3f}; confidence={row.confidence:.3f};
learned_weight={learned_w:.3f}; strategist_weight={proposed:.3f};
advocate_decision={adv.get('decision')}; advocate_max={adv.get('max_allowed_weight')};
grade_history="{historical_text_impact(hist_delta)}" delta={hist_delta:+.3f};
advocate_reason="{str(adv.get('rationale'))[:180]}";
borrower_text="{str(row.get('desc', row.get('desc_preview', '')))[:280]}".

Schema:
{{
  "final_weight": <float 0.00-0.20>,
  "ruling": "strategist|advocate|balanced|suppress",
  "resolution": "<one short sentence>",
  "safety_reason": "<one short sentence>"
}}
"""


def clamp_weight(value: object, default: float) -> float:
    try:
        return round(float(np.clip(float(value), 0.0, 0.20)), 3)
    except Exception:
        return round(default, 3)


def run_case(row: pd.Series, learned_weights: dict[str, float], grade_stats: dict[str, dict]) -> dict:
    grade = str(row.grade)
    learned_w = float(learned_weights.get(grade, 0.03))
    hist_delta = float(grade_stats.get(grade, {}).get("history_delta", 0.0))

    strat, strat_raw = call_llm(strategist_prompt(row, learned_w, hist_delta))
    strat["proposed_weight"] = clamp_weight(strat.get("proposed_weight"), learned_w)
    adv, adv_raw = call_llm(advocate_prompt(row, learned_w, hist_delta, strat))
    adv["max_allowed_weight"] = clamp_weight(adv.get("max_allowed_weight"), strat["proposed_weight"])
    old, old_raw = call_llm(old_arbitrator_prompt(row, learned_w, strat, adv))
    green, green_raw = call_llm(green_arbitrator_prompt(row, learned_w, hist_delta, strat, adv))

    old_w = clamp_weight(old.get("final_weight"), learned_w)
    green_w = clamp_weight(green.get("final_weight"), learned_w)
    text_says_default = bool(row.text_score >= 0.5)
    label = int(row.label)
    text_correct = int(text_says_default) == label
    old_acts = old_w > learned_w + 0.005
    green_acts = green_w > learned_w + 0.005

    if old_acts and not green_acts and not text_correct:
        comparison = "green_better_suppressed_wrong_text"
    elif old_acts and not green_acts and text_correct:
        comparison = "old_better_trusted_correct_text"
    elif green_acts and not old_acts and text_correct:
        comparison = "green_better_trusted_correct_text"
    elif green_acts and not old_acts and not text_correct:
        comparison = "old_better_suppressed_wrong_text"
    else:
        comparison = "same_or_ambiguous"

    return {
        "id": row.id,
        "grade": grade,
        "label": label,
        "loan_status": row.get("loan_status", ""),
        "text_score": float(row.text_score),
        "confidence": float(row.confidence),
        "text_correct_direction": bool(text_correct),
        "learned_weight": learned_w,
        "strategist_type": strat.get("strategy_type"),
        "strategist_weight": strat["proposed_weight"],
        "strategist_rationale": strat.get("rationale"),
        "advocate_decision": adv.get("decision"),
        "advocate_max_weight": adv["max_allowed_weight"],
        "advocate_rationale": adv.get("rationale"),
        "old_weight": old_w,
        "old_ruling": old.get("ruling"),
        "old_resolution": old.get("resolution"),
        "green_weight": green_w,
        "green_ruling": green.get("ruling"),
        "green_resolution": green.get("resolution"),
        "green_safety_reason": green.get("safety_reason"),
        "comparison": comparison,
        "desc_preview": str(row.get("desc_preview", row.get("desc", "")))[:260],
        "raw": {
            "strategist": strat,
            "advocate": adv,
            "old_arbitrator": old,
            "green_arbitrator": green,
        },
    }


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("ok") is True:
                done.add(str(obj["id"]))
        except Exception:
            pass
    return done


def summarize(jsonl_path: Path, outdir: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return
    all_df = df.copy()
    all_df.to_csv(outdir / "api_eval_results_all.csv", index=False)
    if "ok" in df.columns:
        df = df[df["ok"] == True].copy()
    if df.empty:
        (outdir / "README.md").write_text(
            "# Green Arbitration API Evaluation\n\nNo successful API rows yet. "
            "See `api_eval_results_all.csv` for recorded errors.\n",
            encoding="utf-8",
        )
        return
    df.to_csv(outdir / "api_eval_results.csv", index=False)
    counts = df["comparison"].value_counts().rename_axis("comparison").reset_index(name="count")
    counts.to_csv(outdir / "api_eval_comparison_counts.csv", index=False)
    summary = pd.DataFrame([
        {
            "n": len(df),
            "old_avg_weight": df["old_weight"].mean(),
            "green_avg_weight": df["green_weight"].mean(),
            "old_acted_n": int((df["old_weight"] > df["learned_weight"] + 0.005).sum()),
            "green_acted_n": int((df["green_weight"] > df["learned_weight"] + 0.005).sum()),
            "text_direction_accuracy": df["text_correct_direction"].mean(),
        }
    ])
    summary.to_csv(outdir / "api_eval_summary.csv", index=False)
    report = "# Green Arbitration API Evaluation\n\n"
    report += "This run calls MiniMax for Strategist, Advocate, old standalone Arbitrator proxy, and Green-as-Arbitrator.\n\n"
    report += "## Summary\n\n" + summary.to_string(index=False) + "\n\n"
    report += "## Comparison Counts\n\n" + counts.to_string(index=False) + "\n"
    (outdir / "README.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loan-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "api_eval_results.jsonl"
    joined = load_joined(args.loan_csv)
    grade_stats = grade_stats_from_cases(joined)
    cases = select_cases(joined, args.limit)
    learned_weights = {"C": 0.03, "D": 0.10, "E": 0.05, "F": 0.03, "G": 0.03}
    done = load_done(jsonl_path)
    print(f"Selected {len(cases)} cases; already completed {len(done)}.", flush=True)

    with jsonl_path.open("a", encoding="utf-8") as f:
        for i, row in cases.iterrows():
            if str(row.id) in done:
                continue
            print(f"[{i + 1}/{len(cases)}] id={row.id} grade={row.grade} score={row.text_score:.2f} conf={row.confidence:.2f}", flush=True)
            try:
                result = run_case(row, learned_weights, grade_stats)
                result["ok"] = True
            except Exception as e:
                result = {"id": row.id, "grade": row.grade, "ok": False, "error": str(e)}
                print(f"  ERROR: {e}", flush=True)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(args.sleep)
    summarize(jsonl_path, args.outdir)
    print(f"Wrote results to {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
