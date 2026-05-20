# Loan Default — Multi-Agent Demo

A multi-agent loan-default scoring system. An XGBoost baseline produces a numerical PD on Grade C–G loans; a panel of LLM agents reads the borrower's free-text description, audits the proposed weight changes, and fuses text-derived risk with the model probability in logit space. A Streamlit app provides an interactive demo.

## Agent architecture

| Role | Type | Job |
|---|---|---|
| **Text Analyst** | LLM | Scores the borrower description, returns `text_risk_score` + `confidence` |
| **Feature Strategist** | LLM | Proposes per-grade text weight, classifies signals (PERSISTENT_WIN / NOISY / etc.) using trajectory |
| **Subgroup Advocate** | LLM | Vetoes weight changes that harm specific grades; flags positive opportunities |
| **Green Agent** | LLM + ML | Hub agent: runs XGBoost baseline, evaluates fusion, generates expectations / reflects (Reflexion), arbitrates on veto, validates with adaptive thresholds, and trains the Tier-A learned gate |
| **Reporter** | LLM | Generates the natural-language explanation per loan |

The Arbitrator role is owned by **Green** (`green.arbitrate()` in the notebook; the *Green Agent — Arbitration (on-veto)* panel in the demo) — it is **not** a separate White Agent.

## Repository layout

```
.
├── app/
│   └── loan_demo.py                 # Streamlit demo (entry point)
├── notebooks/
│   └── loan_default.ipynb           # Training + multi-agent fusion notebook
├── models/
│   └── loan_default_model.pkl       # Trained XGBoost model + metadata
├── data/
│   ├── README.md                    # How to obtain the raw dataset
│   ├── text_analyst_results_train_matched_sample1.pkl
│   └── text_analyst_results_test_matched_sample1.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

The raw `loan_default.csv` (~88 MB) is **not** committed to the repo. See [data/README.md](data/README.md) for how to obtain it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Streamlit demo

The demo calls the [Minimax](https://api.minimaxi.com) chat-completions API for the LLM agents. Provide your own key via the `MINIMAX_API_KEY` environment variable:

```bash
export MINIMAX_API_KEY=your_key_here
streamlit run app/loan_demo.py
```

Optional environment variables:

- `MINIMAX_API_KEY` — required for the LLM-agent panels to function
- `LOAN_MODEL_PATH` — override the path to `loan_default_model.pkl`

## Reproducing the training notebook

1. Download the raw dataset and place it as `loan_default.csv` at the repo root (see [data/README.md](data/README.md)).
2. Set `MINIMAX_API_KEY` in the notebook's first cell.
3. Run [notebooks/loan_default.ipynb](notebooks/loan_default.ipynb) top-to-bottom. It will retrain the XGBoost baseline, run the LLM agents on a matched sample, and rewrite the pickle artifacts in `models/` and `data/`.

## Security note

Earlier revisions of `loan_demo.py` contained a hardcoded Minimax API key. It has been removed; the key is now read from the `MINIMAX_API_KEY` environment variable. If you fork an older snapshot, rotate any exposed key.
