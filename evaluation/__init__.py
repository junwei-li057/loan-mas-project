"""
Multi-agent evaluation suite for the loan-default MAS.

SCOPE — MODEL-LEVEL ONLY.
All functions here operate on held-out test sets and produce offline reports
or artifacts. They are NOT for runtime per-case use.

Per-case (single-row) prediction, comparison, and explanation lives in
`app/api/inference.py` and is served via the FastAPI backend to the
Streamlit frontend.

Five metric families, all MAS-specific:

  1. core       — 2-pipeline head-to-head (baseline / Strategist) with bootstrap
                  CIs and DeLong significance.
  2. agent      — Per-Agent KPI (MultiAgentBench §3.3) and loop dynamics
                  (convergence, LLM call counts, veto / arbitration rates).
  3. ablation   — Is the MAS worth it? vs LR-stacking, vs brute-force grid search.
  4. trust      — Does the MAS know when NOT to trust the text? Confidence
                  reliability + selective-fusion curve.
  5. report     — One-page consolidated report (Markdown / HTML).
"""

from .core import (
    pipeline_metrics,
    pipeline_comparison,
    run_pipelines,
)
from .agent import (
    per_agent_kpi,
    loop_dynamics,
)
from .ablation import (
    meta_model_comparison,
    grid_search_comparison,
    ablation_summary,
)
from .trust import (
    confidence_reliability,
    selective_fusion_curve,
    trust_calibration,
)
from .report import generate_report

__all__ = [
    # core
    "pipeline_metrics",
    "pipeline_comparison",
    "run_pipelines",
    # agent
    "per_agent_kpi",
    "loop_dynamics",
    # ablation
    "meta_model_comparison",
    "grid_search_comparison",
    "ablation_summary",
    # trust
    "confidence_reliability",
    "selective_fusion_curve",
    "trust_calibration",
    # report
    "generate_report",
]
