# Deploying the demo to Streamlit Community Cloud

This page covers the three-click deployment of the single-page Streamlit demo
to [share.streamlit.io](https://share.streamlit.io). The repository is
already cloud-ready; no code changes are required.

## Prerequisites

- A Streamlit Community Cloud account (free) signed in with the GitHub
  account that has access to `junwei-li057/loan-mas-project`.
- A valid `MINIMAX_API_KEY` with billing balance. Without one, the demo
  still starts and the XGBoost baseline, preset selector, and "Offline
  evaluation" panel all work, but the five LLM-agent panels error out with
  HTTP 429 / `insufficient_balance_error`.

## One-time deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Fill in:
   - **Repository**: `junwei-li057/loan-mas-project`
   - **Branch**: `main`
   - **Main file path**: `app/ui/Home.py`
   - **Python version** (Advanced settings): `3.11`
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   MINIMAX_API_KEY = "your_minimax_key_here"
   ```
   Optional overrides:
   ```toml
   MINIMAX_MODEL = "MiniMax-M2.7"        # default; change to test other models
   ```
4. Click **Deploy**.

First boot takes ~2–4 minutes (pip-install of xgboost + scikit-learn +
streamlit). Subsequent reboots are seconds.

## How the app finds the API key (fallback order)

`get_minimax_api_key()` in `app/ui/Home.py` looks in this order:

1. `MINIMAX_API_KEY` environment variable (set by Streamlit Cloud if you
   put it in Secrets).
2. `st.session_state["minimax_api_key"]` (in-app password field — useful
   for visitors who bring their own key).
3. `st.secrets["MINIMAX_API_KEY"]` (the Secrets TOML; equivalent to #1 on
   cloud).

So:

- If you ship the key via Secrets → every visitor uses your account.
- If you leave Secrets empty → visitors get a password field at the top
  of the page and can paste their own key for the session.

## What the deployed app needs from the repo

These files must stay in `main`:

| Path | Purpose |
|---|---|
| `app/ui/Home.py` | The single-page app — Streamlit Cloud entry point |
| `models/loan_default_model.pkl` | Loaded at startup; required |
| `docs/eval_report_sample1.md` | Rendered by the "Offline evaluation" panel |
| `.streamlit/config.toml` | Pins theme to light (matches Home.py CSS) |
| `requirements.txt` | Pip dependencies installed by the cloud runner |

(`app/loan_demo.py` is a `runpy` wrapper that re-runs `Home.py` so the legacy
local command `streamlit run app/loan_demo.py` keeps working; it is not used
by the cloud deployment.)

Files that are NOT needed at runtime (kept out of the cloud image by
`.gitignore`): `loan_default.csv`, `text_analyst.py`, `text_ana_results/`
sample csvs, the executed-notebook output.

## After deploy: smoke test

Once the app is up, click each of the four preset borrowers and confirm:

1. The header shows the new learned weights
   (`C=0.03, D=0.10, E=0.05, F=0.03, G=0.03`, `conf_threshold=0.65`).
2. The "Offline evaluation" panel renders the report whose header says
   `combined.pkl (16801 train, 4199 test; id-joined to loan_default.csv)`.
3. The Reporter panel produces two paragraphs (LLM if the key works,
   deterministic fallback if not) — it should never be blank.

If panels look blank or the page renders with broken chip colors,
hard-refresh (`Cmd+Shift+R` / `Ctrl+Shift+R`) — the cached old CSS may
need clearing.

## Updating the deployed app

Streamlit Cloud auto-redeploys on every push to the configured branch.
Push → wait ~30 seconds → refresh the URL. To force a rebuild from
scratch (rare; mostly when changing Python version or pinning new deps),
use the **Reboot app** button in the app's Streamlit Cloud settings page.

## Cost / quota notes

- Streamlit Cloud Community tier: free, public app, 1 GB RAM, sleeps
  after 7 days of inactivity (any visit wakes it).
- MiniMax: each preset case fires ~5 LLM calls
  (TextAnalyst + Strategist + Advocate + Green + Reporter, occasionally
  +Arbitrator). Budget the API key accordingly.
