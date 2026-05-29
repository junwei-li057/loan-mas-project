"""
agent.py — agent-level diagnostics.

Two views into the multi-agent collaboration:

  - per_agent_kpi   : MultiAgentBench §3.3 milestone-attribution KPI,
                      decomposing the system's success into per-agent load.
                      Identifies hub agents and bottleneck agents.

  - loop_dynamics   : how the orchestrator behaved over the iterations:
                      convergence speed, total/per-agent LLM calls, fallback
                      rate, veto rate, arbitration-outcome distribution.

Both consume the decision_log produced by `run_iterative_loop` (cell 21 in
the notebook).
"""
from __future__ import annotations

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Milestone definitions  (MultiAgentBench §3.3)
# ──────────────────────────────────────────────────────────────────────────────
# A milestone fires when the system reaches a specific desired state during a
# loop iteration (per-iter) or across the whole run (once). Each milestone is
# attributed to one or more agents — that agent is credited when the milestone
# fires. KPI_j = (milestones agent j is credited for) / (total milestone slots).

PER_ITER_MILESTONES: list[dict] = [
    {
        "name": "M_expect_generated",
        "description": "Green generated an expectation before evaluate_fusion",
        "agents": ["Green"],
        "check": lambda e: "expectation" in e,
    },
    {
        "name": "M_low_surprise",
        "description": "Green's prediction error matched reality (surprise < 0.05)",
        "agents": ["Green"],
        "check": lambda e: e.get("reflection", {}).get("surprise", float("inf")) < 0.05,
    },
    {
        "name": "M_strategy_proposed",
        "description": "Strategist produced a non-trivial proposal (won or vetoed)",
        "agents": ["Strategist"],
        # `green_llm:*` is Green's standalone fusion decision — not a Strategist
        # action — so it must not credit the Strategist. `strategy_update` =
        # proposal accepted; `fallback` = Strategist LLM failed and rule path
        # ran; `green_arbitrate*` = proposal vetoed and routed to Green's
        # arbitration. All three are Strategist work.
        "check": lambda e: (
            e.get("event") in ("strategy_update", "fallback")
            or str(e.get("event", "")).startswith("green_arbitrate")
        ),
    },
    {
        "name": "M_no_veto",
        "description": "Advocate let the strategy pass (subgroup-safe)",
        "agents": ["Advocate"],
        "check": lambda e: not e.get("veto", False),
    },
    {
        "name": "M_opportunity_flagged",
        "description": "Advocate flagged ≥1 positive opportunity",
        "agents": ["Advocate"],
        "check": lambda e: bool(e.get("advocate_opportunities", [])),
    },
    {
        # Renamed from M_arbitration_succeeded — the original docstring claimed
        # this counted arbitrations that resolved without the LLM safety net,
        # but the check has no way to detect "rule path vs. LLM fallback" from
        # the decision_log shape. Treating any invoked arbitration as a hit
        # keeps the docstring and the implementation honest.
        "name": "M_arbitration_invoked",
        "description": "On veto, Green's arbitrate was invoked (rule path or LLM safety net)",
        "agents": ["Green", "Strategist", "Advocate"],
        "check": lambda e: (
            e.get("veto", False)
            and str(e.get("event", "")).startswith("green_arbitrate")
        ),
    },
    {
        "name": "M_auprc_improved_iter",
        "description": "Fused AUPRC exceeded baseline AUPRC this iteration",
        "agents": ["Strategist", "TextAnalyst"],
        "check": lambda e: (
            e.get("overall_fused_auprc", 0) > e.get("overall_baseline_auprc", 0)
        ),
    },
]

ONCE_MILESTONES: list[dict] = [
    {
        "name": "M_loop_converged",
        "description": "Loop reached convergence before max_iter",
        "agents": ["Green", "Strategist", "Advocate"],
        "check": lambda log, fr: any(e.get("event") == "converged" for e in log),
    },
    {
        "name": "M_test_above_baseline",
        "description": "Final fused AUPRC on test > baseline AUPRC",
        "agents": ["Strategist", "TextAnalyst"],
        "check": lambda log, fr: (
            fr.get("overall_fused_auprc", 0) > fr.get("overall_baseline_auprc", 0)
        ),
    },
]

DEFAULT_AGENTS = ("TextAnalyst", "Strategist", "Advocate", "Green")


def per_agent_kpi(
    decision_log: list[dict],
    final_result: dict,
    *,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
) -> pd.DataFrame:
    """MultiAgentBench §3.3 KPI: contributions per agent.

    KPI_j = n_j / M  where n_j is the number of milestones agent j is credited
    for and M is the total milestone slot count. The summary row at the end
    holds the mean KPI across agents.

    Parameters
    ----------
    decision_log : list of per-iter dicts as produced by run_iterative_loop
                   (cell 21 in the notebook).
    final_result : output of green.evaluate_fusion on the final strategy.
    agents       : tuple of agent names to score (others are ignored even if
                   listed in milestone.agents). Default covers all 4 White-Agent
                   roles in the current design.

    Returns
    -------
    DataFrame indexed by agent name with columns:
        n_hit  — milestones the agent was credited for
        n_total — total milestone slots (same for every agent row)
        kpi    — n_hit / n_total
    Plus a final summary row 'overall_kpi' with mean(kpi) across the agents.

    The full instance log (which milestone fired in which iter) is exposed on
    the result's `.attrs['milestones']` for downstream reporting.
    """
    instances: list[dict] = []

    for entry in decision_log:
        i = entry.get("iteration")
        for m in PER_ITER_MILESTONES:
            try:
                fired = bool(m["check"](entry))
            except Exception:
                fired = False
            instances.append(
                {
                    "iter": i,
                    "name": m["name"],
                    "agents": list(m["agents"]),
                    "fired": fired,
                }
            )

    for m in ONCE_MILESTONES:
        try:
            fired = bool(m["check"](decision_log, final_result))
        except Exception:
            fired = False
        instances.append(
            {
                "iter": None,
                "name": m["name"],
                "agents": list(m["agents"]),
                "fired": fired,
            }
        )

    total_M = len(instances)
    by_agent = {a: {"n_hit": 0, "contributed_to": []} for a in agents}
    for inst in instances:
        if not inst["fired"]:
            continue
        tag = (
            f"iter{inst['iter']}:{inst['name']}"
            if inst["iter"] is not None
            else inst["name"]
        )
        for a in inst["agents"]:
            if a in by_agent:
                by_agent[a]["n_hit"] += 1
                by_agent[a]["contributed_to"].append(tag)

    rows = []
    for a in agents:
        n_hit = by_agent[a]["n_hit"]
        rows.append(
            {
                "agent": a,
                "n_hit": n_hit,
                "n_total": total_M,
                "kpi": n_hit / total_M if total_M else 0.0,
            }
        )
    overall = sum(r["kpi"] for r in rows) / len(rows) if rows else 0.0
    rows.append({"agent": "overall_kpi", "n_hit": None, "n_total": total_M, "kpi": overall})

    df = pd.DataFrame(rows).set_index("agent")
    df.attrs["milestones"] = instances
    df.attrs["milestones_fired"] = sum(1 for x in instances if x["fired"])
    df.attrs["n_iters"] = len(decision_log)
    return df


def loop_dynamics(decision_log: list[dict]) -> dict:
    """Loop-level diagnostics extracted from decision_log.

    Parameters
    ----------
    decision_log : the per-iter log produced by run_iterative_loop.

    Returns
    -------
    dict with keys:
        n_iters                 : iterations recorded
        converged               : True iff any entry has event == 'converged'
        veto_rate               : fraction of iters where Advocate vetoed
        advocate_modes          : counts of {'LLM', 'rule'} for the Advocate
        event_counts            : counts by event type (strategy_update,
                                  green_arbitrate, fallback, converged, ...)
        arbitration_outcomes    : on-veto event distribution
        per_iter_trace          : list of compact per-iter rows for tabular display
    """
    n_iters = len(decision_log)
    converged = any(e.get("event") == "converged" for e in decision_log)

    veto_count = sum(1 for e in decision_log if e.get("veto", False))
    veto_rate = veto_count / n_iters if n_iters else 0.0

    advocate_modes: dict[str, int] = {}
    for e in decision_log:
        m = e.get("advocate_mode")
        if m is not None:
            advocate_modes[m] = advocate_modes.get(m, 0) + 1

    event_counts: dict[str, int] = {}
    for e in decision_log:
        ev = e.get("event") or "_no_event"
        event_counts[ev] = event_counts.get(ev, 0) + 1

    arbitration_outcomes: dict[str, int] = {}
    for e in decision_log:
        if not e.get("veto", False):
            continue
        ev = e.get("event") or "_no_event"
        arbitration_outcomes[ev] = arbitration_outcomes.get(ev, 0) + 1

    per_iter_trace = []
    for e in decision_log:
        per_iter_trace.append(
            {
                "iter": e.get("iteration"),
                "event": e.get("event"),
                "veto": bool(e.get("veto", False)),
                "advocate_mode": e.get("advocate_mode"),
                "overall_baseline_auc": e.get("overall_baseline"),
                "overall_fused_auc": e.get("overall_fused"),
                "overall_baseline_auprc": e.get("overall_baseline_auprc"),
                "overall_fused_auprc": e.get("overall_fused_auprc"),
                "advocate_opportunities": e.get("advocate_opportunities", []),
                "surprise": e.get("reflection", {}).get("surprise"),
            }
        )

    return {
        "n_iters": n_iters,
        "converged": converged,
        "veto_rate": veto_rate,
        "advocate_modes": advocate_modes,
        "event_counts": event_counts,
        "arbitration_outcomes": arbitration_outcomes,
        "per_iter_trace": per_iter_trace,
    }
