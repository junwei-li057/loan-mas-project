# Loan Default — Multi-Agent Demo

A multi-agent loan-default scoring system. An XGBoost baseline produces a numerical PD on Grade C–G loans; a panel of LLM agents reads the borrower's free-text description, audits the proposed weight changes, and fuses text-derived risk with the model probability in logit space. A Streamlit app provides an interactive single-page demo.

## Agent architecture

| Role | Type | Job |
|---|---|---|
| **Text Analyst** | LLM | Scores the borrower description, returns `text_risk_score` + `confidence` plus matched protective/risk signals. Falls back to a regex-rule scorer when the LLM call cannot be parsed. |
| **Feature Strategist** | LLM | Picks one of five `strategy_type`s — `keep_learned`, `case_specific_exception`, `fuzzy_only`, `explanation_only`, `human_review` — and proposes a per-grade text weight. Decision heuristics push it toward `case_specific_exception` (with `proposed_weight` strictly greater than learned) whenever the text is high-confidence AND strongly disagrees with the numeric baseline. |
| **Subgroup Advocate** | LLM | Reasons about grade-level historical text impact; emits `pass / soft_veto / hard_veto` on the Strategist's proposal. Soft vetoes go to arbitration; hard vetoes are enforced. |
| **Green Agent** | LLM + ML | Hub agent: runs XGBoost baseline, evaluates fusion, generates expectations / reflects (Reflexion), arbitrates on veto, and validates with adaptive thresholds. |
| **Reporter** | LLM (+ deterministic fallback) | Generates a two-paragraph natural-language explanation: P1 covers the prediction result (numeric tension, fused vs. threshold, verdict), P2 covers whether the text deserved to be trusted (based on `effective_weight`, `confidence`, subgroup history). |

The Arbitrator role is owned by **Green** (`green.arbitrate()` in the notebook; the *Green Agent — Arbitration (on-veto)* panel in the demo) — it is **not** a separate White Agent.

## Demo flow & preset cases

The Streamlit page ships with four preset borrowers in [app/ui/Home.py](app/ui/Home.py), designed so each one drives a different branch of the multi-agent flow:

| Case | Grade | What it demonstrates |
|---|---|---|
| **Case 1 — Arbitrator sides with Advocate** | C | Strongly protective text + grade where text is historically harmful → Strategist proposes increase → Advocate **hard-vetoes** → Green arbitration enforces the veto. Fused score equals baseline. |
| **Case 2 — Arbitrator sides with Strategist** | E | Same borrower profile, grade where text is only marginally harmful → Advocate **soft-vetoes** → Green arbitration negotiates a compromise weight. |
| **Case 3 — Strategist passes cleanly** | D | Protective text in a grade where text history is favorable → no veto → fusion lowers the prediction. |
| **Case 4 — Low confidence, text ignored** | D | Empty / vague description → `text_confidence` falls below `conf_threshold` → text gate closes, fused = baseline. |

## Repository layout

```
.
├── app/
│   ├── loan_demo.py                 # Streamlit entry-point shim
│   └── ui/
│       └── Home.py                  # Single-page Streamlit demo (loads model + calls LLM)
├── notebooks/
│   ├── loan_default.ipynb           # Training + multi-agent fusion notebook
│   ├── eval_verify_p2.py            # Smoke test for evaluation.{core,agent}
│   └── eval_verify_p3.py            # End-to-end eval suite → docs/eval_report_sample1.md
├── evaluation/                      # Offline evaluation toolkit (5 modules)
│   ├── core.py                      #   - 2-pipeline head-to-head + DeLong
│   ├── agent.py                     #   - Per-Agent KPI + loop dynamics
│   ├── ablation.py                  #   - LR stacking + grid-search vs MAS
│   ├── trust.py                     #   - Confidence reliability + selective fusion
│   └── report.py                    #   - Consolidated Markdown report
├── models/
│   └── loan_default_model.pkl       # Trained XGBoost model + metadata
├── data/
│   ├── README.md                    # How to obtain the raw dataset
│   ├── text_analyst_results_train_matched_sample1.pkl
│   └── text_analyst_results_test_matched_sample1.pkl
├── docs/
│   ├── eval_report_sample1.md       # Latest offline evaluation report (read by the UI)
│   ├── system_design.tex            # System-design write-up (LaTeX source)
│   └── system_design.pdf            # System-design write-up (rendered)
├── requirements.txt
├── .gitignore
└── README.md
```

The raw `loan_default.csv` (~88 MB) is **not** committed. See [data/README.md](data/README.md) for how to obtain it.

The text-analyst driver script (`text_analyst.py`) and its bulk outputs under `text_ana_results/` are intentionally ignored by `.gitignore` — the driver contains user-specific API credentials and the outputs are large and regenerable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Streamlit demo

The demo calls the [Minimax](https://api.minimaxi.com) chat-completions API for the LLM agents. Provide your own key via the `MINIMAX_API_KEY` environment variable (or paste it into the in-app field on first load):

```bash
export MINIMAX_API_KEY=your_key_here
streamlit run app/loan_demo.py
```

Environment variables:

- `MINIMAX_API_KEY` — required for the LLM-agent panels (falls back to an in-app prompt if unset)
- `MINIMAX_MODEL` — optional, defaults to `MiniMax-M2.7`
- `LOAN_MODEL_PATH` — optional, overrides the path to `models/loan_default_model.pkl`

The single-page UI loads the model directly from the pickle and calls Minimax directly — there is no separate backend service to start.

For hosting on Streamlit Community Cloud, see [docs/deploy.md](docs/deploy.md).

## Reproducing the training notebook

1. Download the raw dataset and place it as `loan_default.csv` at the repo root (see [data/README.md](data/README.md)).
2. Set `MINIMAX_API_KEY` in the notebook's first cell.
3. Run [notebooks/loan_default.ipynb](notebooks/loan_default.ipynb) top-to-bottom. It will retrain the XGBoost baseline, run the LLM agents on a matched sample, and rewrite the pickle artifacts in `models/` and `data/`.

## Regenerating the offline evaluation report

The "Offline evaluation" panel in the Streamlit UI is rendered from [docs/eval_report_sample1.md](docs/eval_report_sample1.md). To refresh it after retraining:

```bash
python notebooks/eval_verify_p3.py    # rewrites docs/eval_report_sample1.md
```

This drives the full `evaluation/` suite (pipeline comparison, per-agent KPI, ablation, trust calibration) on the saved sample-1 artifacts and writes a consolidated Markdown report.

### Producing a real `decision_log` (optional but recommended)

`per_agent_kpi` and `loop_dynamics` (sections §3 and §4 of the report) consume a `decision_log` from the iterative training loop. If `data/decision_log_sample1.pkl` exists, `eval_verify_p3.py` loads it; otherwise it falls back to a 4-iter synthetic log and labels the report header accordingly. To dump a real one, add this cell at the end of the training notebook right after `run_iterative_loop` returns:

```python
import pickle
from pathlib import Path

with open(Path("data/decision_log_sample1.pkl"), "wb") as f:
    pickle.dump({"decision_log": decision_log, "final_result": final_result}, f)
```

The pickle is small (one dict of per-iter records); commit it if you want the published report to reflect a real run.

## Implementation notes

Three conventions are load-bearing for stable behavior with the MiniMax-M2.7 backend; preserve them when extending the system.

**1. All LLM calls use JSON output mode.** MiniMax-M2.7 is a thinking model. In free-text mode it leaks `<think>…</think>` blocks or stream-of-consciousness reasoning that exhausts the token budget *before* it ever produces the actual answer — the user-visible output ends up empty or truncated. JSON mode (`response_format_json=True` in `call_llm`) reliably suppresses this. Every agent (Text Analyst, Strategist, Advocate, Green LLM, Arbitrator, Reporter) emits a JSON object that the UI parses locally with `_extract_json_object`.

**2. The Reporter has a deterministic Python fallback.** `_reporter_deterministic_fallback` in [app/ui/Home.py](app/ui/Home.py) synthesises the two-paragraph narrative from the numeric inputs alone — same two-section structure (prediction result / trust assessment), same `effective_weight == 0` vs `> 0` branching as the LLM prompt. It runs whenever the LLM returns invalid or truncated JSON, so the Reporter panel can never show "hidden reasoning only" or a blank box even when the LLM misbehaves.

**3. The `reasoning` field carries borrower-level analysis, never system status.** `fallback_text_analyst` writes `reasoning` as a sentence describing what was detected in the borrower's text (e.g. *"Identified 3 protective signals … and no risk signals; adjusted from 0.45 to 0.150."*). System-failure detail (e.g. *"LLM JSON parsing failed: …"*) goes into a separate `_fallback_reason` field that stays in the debug raw panel and never flows into another agent's prompt. This is what prevents downstream agents — especially the Reporter — from confusing system state for borrower narrative quality.
