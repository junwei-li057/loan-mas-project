"""
report.py — one-page consolidated evaluation report.

Stitches together the five metric families into a single artifact (Markdown
by default, optional HTML render).

Sections produced:
  1. Executive summary  : one-paragraph verdict + headline numbers
  2. Pipeline head-to-head : table from pipeline_comparison() with CIs and DeLong p
  3. Per-Agent KPI       : table from per_agent_kpi()
  4. Loop dynamics       : per-iter trace + summary stats from loop_dynamics()
  5. Ablation            : "is MAS worth it" table from ablation_summary()
  6. Trust calibration   : confidence_reliability + selective_fusion_curve tables
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def _fmt(x, prec: int = 4) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "—"
        return f"{x:.{prec}f}"
    except Exception:
        return str(x)


def _cell(v) -> str:
    if isinstance(v, tuple) and len(v) == 2:
        return f"({_fmt(v[0])}, {_fmt(v[1])})"
    if isinstance(v, (int, float)):
        return _fmt(v)
    if v is None:
        return ""
    return str(v)


def _md_table(df: pd.DataFrame, index_label: str | None = None) -> str:
    """Render a pandas DataFrame as a GitHub-flavored Markdown table without
    any external dependency (avoids needing `tabulate`)."""
    out = df.copy()
    if index_label is not None:
        out.index.name = index_label
    out = out.reset_index()
    cols = [str(c) for c in out.columns]
    rows = [[_cell(out.iloc[i, j]) for j in range(len(cols))] for i in range(len(out))]

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _exec_summary(
    pipeline_comparison_df: pd.DataFrame,
    agent_kpi_df: pd.DataFrame,
    loop_dyn: dict,
    ablation_df: pd.DataFrame,
    trust_diag: dict,
) -> str:
    base_auc = pipeline_comparison_df.loc["baseline", "auc"]
    strat_auc = pipeline_comparison_df.loc["strategist", "auc"]
    delta_auc = strat_auc - base_auc
    delong_p = pipeline_comparison_df.loc["strategist", "delong_p_vs_ref"]
    sig = "significant" if (not np.isnan(delong_p) and delong_p < 0.05) else "not significant"
    overall_kpi = float(agent_kpi_df.loc["overall_kpi", "kpi"])
    converged = loop_dyn.get("converged", False)
    n_iters = loop_dyn.get("n_iters", 0)
    veto_rate = loop_dyn.get("veto_rate", 0.0)
    mas_row = ablation_df.loc["MAS Strategist"]
    grid_row = ablation_df.loc["Grid search (best)"]
    mas_vs_grid_auc = float(mas_row["test_auc"]) - float(grid_row["test_auc"])
    trust_head = trust_diag.get("headline", "—")

    bullets = [
        f"- **Discrimination**: Strategist AUC = {_fmt(strat_auc)} vs baseline {_fmt(base_auc)} "
        f"(Δ = {_fmt(delta_auc, 4)}, DeLong p = {_fmt(delong_p)}; {sig}).",
        f"- **Agent contribution**: overall KPI = {_fmt(overall_kpi, 3)} over "
        f"{n_iters} loop iter(s), advocate veto rate = {_fmt(veto_rate, 3)} "
        f"({'converged' if converged else 'did not converge'}).",
        f"- **MAS vs grid search**: Δ test AUC = {_fmt(mas_vs_grid_auc, 4)} "
        f"({'MAS beats brute-force' if mas_vs_grid_auc > 0 else 'grid search matches or beats MAS'}).",
        f"- **Trust calibration**: {trust_head}.",
    ]
    return "\n".join(bullets)


def generate_report(
    *,
    pipeline_comparison_df: pd.DataFrame,
    agent_kpi_df: pd.DataFrame,
    loop_dyn: dict,
    ablation_df: pd.DataFrame,
    trust_diag: dict,
    out_path: str | Path = "eval_report.md",
    fmt: Literal["md", "html"] = "md",
    title: str = "Multi-Agent Loan-Default System — Evaluation Report",
    meta: dict | None = None,
) -> Path:
    """Produce one consolidated evaluation report.

    Parameters
    ----------
    pipeline_comparison_df : from `pipeline_comparison(...)`
    agent_kpi_df           : from `per_agent_kpi(...)`
    loop_dyn               : from `loop_dynamics(...)`
    ablation_df            : from `ablation_summary(...)`
    trust_diag             : from `trust_calibration(...)`
    out_path               : output path (overwritten)
    fmt                    : 'md' or 'html'
    meta                   : optional dict for the report header
                             (e.g. {'model_version', 'data_window',
                              'evaluated_on', 'git_sha'})

    Returns
    -------
    Path to the written report.
    """
    if fmt not in ("md", "html"):
        raise ValueError(f"fmt must be 'md' or 'html'; got {fmt!r}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = meta or {}
    header_lines = [f"# {title}", ""]
    header_lines.append(
        f"_generated_: `{_dt.datetime.now().isoformat(timespec='seconds')}`"
    )
    for k, v in meta.items():
        header_lines.append(f"_{k}_: `{v}`")
    header = "\n".join(header_lines)

    sec_exec = "## 1. Executive Summary\n\n" + _exec_summary(
        pipeline_comparison_df, agent_kpi_df, loop_dyn, ablation_df, trust_diag
    )

    sec_pipe = (
        "## 2. Pipeline Head-to-Head\n\n"
        "Bootstrap 95% CIs on AUC and AUPRC. `delong_p_vs_ref` is the two-sided "
        "p-value vs the reference pipeline (NaN for the reference itself).\n\n"
        + _md_table(pipeline_comparison_df, index_label="pipeline")
    )

    sec_agent = (
        "## 3. Per-Agent KPI (MultiAgentBench §3.3)\n\n"
        f"Iterations recorded: **{loop_dyn.get('n_iters', 0)}**. "
        f"Milestones fired: **{agent_kpi_df.attrs.get('milestones_fired', '—')}** / "
        f"{int(agent_kpi_df['n_total'].iloc[0]) if len(agent_kpi_df) else 0}.\n\n"
        + _md_table(agent_kpi_df, index_label="agent")
    )

    # Loop dynamics
    trace_df = pd.DataFrame(loop_dyn.get("per_iter_trace", []))
    if len(trace_df) > 0:
        # Stringify list-valued columns so the MD renderer stays simple.
        for col in trace_df.columns:
            if trace_df[col].apply(lambda v: isinstance(v, (list, dict))).any():
                trace_df[col] = trace_df[col].apply(
                    lambda v: "" if not v else ("[" + ", ".join(map(str, v)) + "]") if isinstance(v, list) else str(v)
                )
        trace_md = _md_table(trace_df.set_index(trace_df.columns[0]), index_label=trace_df.columns[0])
    else:
        trace_md = "_no iterations recorded_"
    sec_loop = (
        "## 4. Loop Dynamics\n\n"
        f"- Iterations: **{loop_dyn.get('n_iters', 0)}**  \n"
        f"- Converged: **{loop_dyn.get('converged', False)}**  \n"
        f"- Advocate veto rate: **{_fmt(loop_dyn.get('veto_rate', 0.0), 3)}**  \n"
        f"- Advocate modes: `{loop_dyn.get('advocate_modes', {})}`  \n"
        f"- Event counts: `{loop_dyn.get('event_counts', {})}`  \n"
        f"- Arbitration outcomes: `{loop_dyn.get('arbitration_outcomes', {})}`  \n"
    )
    rq = loop_dyn.get("reflexion_quality")
    if rq is not None:
        sec_loop += (
            f"- Reflexion accuracy: mean |ΔAUC err| = "
            f"**{_fmt(rq.get('mean_abs_delta_auc_err'), 4)}**, "
            f"mean |mean_shift err| = "
            f"**{_fmt(rq.get('mean_abs_mean_shift_err'), 4)}** "
            f"(n={rq.get('n_records', '—')})\n"
        )
    sec_loop += "\n### Per-iter trace\n\n" + trace_md

    sec_abl = (
        "## 5. Ablation — Is MAS Worth Its Complexity?\n\n"
        "Each row is an alternative method. `vs_baseline_*` columns are "
        "absolute differences (positive = better than the XGBoost-only baseline).\n\n"
        + _md_table(ablation_df, index_label="method")
    )

    reliability = trust_diag.get("reliability")
    selective = trust_diag.get("selective")
    rel_md = (
        _md_table(reliability, index_label=None)
        if isinstance(reliability, pd.DataFrame) and len(reliability)
        else "_no reliability bins computed_"
    )
    sel_md = (
        _md_table(selective, index_label=None)
        if isinstance(selective, pd.DataFrame) and len(selective)
        else "_no selective-fusion points computed_"
    )
    sec_trust = (
        "## 6. Trust Calibration\n\n"
        f"**Verdict**: {trust_diag.get('headline', '—')}  \n"
        f"- LLM-confidence-vs-accuracy slope: **{_fmt(trust_diag.get('slope'), 4)}** "
        f"(positive ⇒ higher claimed confidence really is more reliable)  \n"
        f"- Top-coverage ΔAUPRC (fused − baseline at the most-trusted "
        f"{_fmt(selective.iloc[0]['coverage'], 2) if isinstance(selective, pd.DataFrame) and len(selective) else '—'}): "
        f"**{_fmt(trust_diag.get('top_delta_auprc'), 4)}**\n\n"
        "### Confidence reliability bins\n\n"
        + rel_md
        + "\n\n### Selective-fusion curve\n\n"
        + sel_md
    )

    parts = [header, "", sec_exec, "", sec_pipe, "", sec_agent, "", sec_loop, "", sec_abl, "", sec_trust, ""]
    md = "\n".join(parts)

    if fmt == "md":
        out.write_text(md, encoding="utf-8")
    else:
        try:
            import markdown as _md_lib
        except ImportError as e:
            raise RuntimeError(
                "fmt='html' requires `pip install markdown`; or set fmt='md'."
            ) from e
        html = _md_lib.markdown(md, extensions=["tables"])
        out.write_text(
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:980px;"
            "margin:2em auto;padding:0 1em;line-height:1.5}"
            "table{border-collapse:collapse}"
            "th,td{border:1px solid #ccc;padding:.35em .75em}"
            "th{background:#f0f0f0;text-align:left}"
            "code{background:#f6f6f6;padding:.1em .3em;border-radius:3px}"
            "</style></head><body>" + html + "</body></html>",
            encoding="utf-8",
        )
    return out
