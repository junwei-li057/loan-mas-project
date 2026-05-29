import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import requests
import json
import re
import time
import os
import html
from pathlib import Path
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_URL     = "https://api.minimaxi.com/v1/chat/completions"
MINIMAX_MODEL   = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
MODEL_PATH      = Path(
    os.getenv("LOAN_MODEL_PATH", PROJECT_ROOT / "models" / "loan_default_model.pkl")
).expanduser()
REPORT_PATH = PROJECT_ROOT / "docs" / "eval_report_sample1.md"

st.set_page_config(page_title="Loan Default — Multi-Agent Demo", layout="wide", page_icon="🏦")

def get_minimax_api_key() -> str:
    if MINIMAX_API_KEY:
        return MINIMAX_API_KEY
    if st.session_state.get("minimax_api_key"):
        return st.session_state["minimax_api_key"]
    try:
        return st.secrets.get("MINIMAX_API_KEY", "")
    except Exception:
        return ""

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

saved = load_model()
model             = saved["model"]
NUM_FEATURES      = saved["features"]
BEST_STRATEGY     = saved["best_strategy"]
GRADE_TRAIN_STATS = saved.get("grade_train_stats", {})

# ── Math helpers ──────────────────────────────────────────────────────────────
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def _stable_pick(options, key):
    return options[sum(ord(c) for c in str(key)) % len(options)]

def historical_text_impact(delta: float, key: str = "") -> str:
    if delta <= -0.04:
        return _stable_pick([
            "prior cases suggest text has reduced reliability for this segment",
            "the training record shows text has been a poor fit for this grade",
            "similar borrowers have seen weaker decisions when text was given more influence",
        ], key)
    if delta < -0.01:
        return _stable_pick([
            "past results are unfavorable enough to treat extra text influence cautiously",
            "text has behaved inconsistently for comparable cases, with downside showing up in training",
            "the grade-level record gives a caution signal rather than clear support for text",
        ], key)
    if delta < 0:
        return _stable_pick([
            "text has offered only weak and uneven value for similar borrowers",
            "the history is mixed, so any extra text weight should stay limited",
            "prior evidence is not strong enough to fully trust text without constraints",
        ], key)
    if delta < 0.01:
        return _stable_pick([
            "text has been broadly acceptable for this grade, though the lift is modest",
            "past cases do not show subgroup harm, but the benefit is incremental",
            "the historical signal is mildly supportive rather than decisive",
        ], key)
    return _stable_pick([
        "prior cases indicate text can add useful signal for this grade",
        "the grade-level history supports giving text some influence",
        "similar borrowers have generally benefited from incorporating text evidence",
    ], key)

# ── LLM helpers ───────────────────────────────────────────────────────────────
def call_llm(prompt, max_tokens=300, temperature=0.3, response_format_json=False):
    api_key = get_minimax_api_key()
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set. Set it before running the Streamlit demo.")

    payload = {
        "model": MINIMAX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        MINIMAX_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(
            f"MiniMax returned non-JSON response: status={resp.status_code}, body={resp.text[:800]}"
        )

    if resp.status_code != 200:
        raise RuntimeError(f"MiniMax HTTP {resp.status_code}: {data}")
    if "choices" not in data or not data.get("choices"):
        raise RuntimeError(f"MiniMax response missing choices: {data}")

    choice = data["choices"][0]
    try:
        return choice["message"]["content"]
    except Exception:
        raise RuntimeError(f"MiniMax response has unexpected choice format: {choice}")

def _extract_json_object(raw: str) -> dict:
    """Parse the first JSON object from an LLM response, with useful errors."""
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

    raise ValueError(f"No JSON object found in LLM response: {cleaned[:1000]}")

def confidence_cap_for_text(desc_text: str) -> tuple[float, str]:
    """Deterministic cap: confidence measures evidence quality, not sentiment strength."""
    text = re.sub(r"<[^>]+>", " ", str(desc_text)).strip()
    words = re.findall(r"[A-Za-z0-9$%]+", text)
    word_count = len(words)
    has_money_or_rate = bool(re.search(r"\$[\d,]+|\d+%|\d+\s*(?:months?|years?)", text, re.I))
    has_employment = bool(re.search(
        r"\b(stable|steady|employed|employment|job|nurse|teacher|engineer|manager|salary|income|years?)\b",
        text, re.I
    ))
    has_repayment = bool(re.search(r"\b(will pay|plan to|repay|pay off|budget|monthly|no missed payments)\b", text, re.I))
    has_specific_purpose = bool(re.search(r"\b(consolidat|medical|home improvement|car|business|tuition|moving)\b", text, re.I))
    detail_count = sum([has_money_or_rate, has_employment, has_repayment, has_specific_purpose])

    if word_count <= 3:
        return 0.45, "extremely short description"
    if word_count <= 8 and detail_count == 0:
        return 0.55, "short vague description with no concrete details"
    if word_count <= 15 and detail_count == 0:
        return 0.62, "limited description with no concrete details"
    if detail_count == 0:
        return 0.70, "no concrete income, employment, repayment, or purpose details"
    if detail_count == 1 and word_count < 25:
        return 0.80, "only one concrete detail in a short description"
    return 0.95, "sufficient textual detail"

def fallback_text_analysis(desc_text: str, reason: str, raw: str = "") -> dict:
    """Deterministic backup when the LLM response is not valid JSON."""
    text = re.sub(r"<[^>]+>", " ", str(desc_text)).strip()
    rules = [
        ("protective", -0.15, [
            r"\b(?:registered\s+)?nurse\b",
            r"\b\d+\s+years?\s+of\s+continuous\s+employment\b",
            r"\bstable\s+(?:annual\s+)?salary\b",
            r"\bemployed\s+at\s+the\s+same\s+\w+\b",
        ]),
        ("protective", -0.10, [
            r"\bnever\s+missed\s+(?:a\s+single\s+)?payment(?:\s+on\s+any\s+account)?\b",
            r"\bno\s+missed\s+payments\b",
            r"\baccounts?\s+in\s+good\s+standing\b",
            r"\b(?:plan|intend|commit)\s+to\s+repay\b",
        ]),
        ("protective", -0.08, [
            r"\bannual\s+salary\s+is\s+\$[\d,]+\b",
            r"\bstable\s+annual\s+salary\s+of\s+\$[\d,]+\b",
            r"\$[\d,]+",
            r"\d+%",
        ]),
        ("protective", -0.05, [
            r"\bconsolidating\s+two\s+low-balance\s+credit\s+cards\b",
            r"\bconsolidat\w+\s+(?:credit\s+card\s+)?debt\b",
            r"\breduce\s+(?:my\s+)?interest\s+rate\b",
            r"\bsimplify\s+(?:my\s+)?finances\b",
        ]),
        ("risk", 0.15, [r"\b(?:urgent|desperate|emergency|need immediately)\b"]),
        ("risk", 0.12, [r"\b(?:behind|collections|late payments|overdue)\b"]),
        ("risk", 0.10, [r"\b(?:unemployed|no income|lost my job)\b"]),
        ("risk", 0.08, [r"\b(?:maxed|multiple debts|struggling)\b"]),
    ]
    score = 0.45
    risk_signals = []
    protective_signals = []
    for kind, delta, patterns in rules:
        matched_phrase = None
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                matched_phrase = match.group(0)
                break
        if not matched_phrase:
            continue
        score += delta
        if kind == "risk":
            risk_signals.append(matched_phrase)
        else:
            protective_signals.append(matched_phrase)

    conf_cap, conf_reason = confidence_cap_for_text(text)
    score = round(min(max(score, 0.05), 0.95), 3)
    evidence_count = len(risk_signals) + len(protective_signals)
    fallback_conf = min(0.50 + 0.09 * evidence_count, 0.86)
    confidence = round(min(fallback_conf if evidence_count else 0.5, conf_cap), 3)
    if protective_signals and not risk_signals:
        brief = "Stable employment, clean payment history, income detail, and consolidation purpose point to lower text risk."
    elif risk_signals and not protective_signals:
        brief = "The description contains credit-relevant stress signals that increase text-based risk."
    elif risk_signals and protective_signals:
        brief = "The description mixes protective borrower details with risk signals, so text evidence remains balanced."
    else:
        brief = "The description is too vague for strong text-based adjustment."

    # `reasoning` is consumed by downstream agents (Strategist / Reporter) as a
    # borrower-level analysis sentence. It must describe the borrower's text, not
    # the system's parsing state. System failure detail goes in `_fallback_reason`,
    # which is debug-only and never reaches another LLM prompt.
    if protective_signals and not risk_signals:
        reasoning = (
            f"Identified {len(protective_signals)} protective signal(s) "
            f"({', '.join(repr(s) for s in protective_signals[:3])}) and no risk signals; "
            f"adjusted from 0.45 to {score:.3f}."
        )
    elif risk_signals and not protective_signals:
        reasoning = (
            f"Identified {len(risk_signals)} risk signal(s) "
            f"({', '.join(repr(s) for s in risk_signals[:3])}) and no protective signals; "
            f"adjusted from 0.45 to {score:.3f}."
        )
    elif risk_signals and protective_signals:
        reasoning = (
            f"Identified {len(protective_signals)} protective and {len(risk_signals)} risk "
            f"signal(s) in the description; net adjustment from 0.45 to {score:.3f}."
        )
    else:
        reasoning = (
            f"No protective or risk phrases matched; kept score near the neutral 0.45 baseline "
            f"({score:.3f})."
        )

    return {
        "text_risk_score": score,
        "confidence": confidence,
        "_raw_confidence": confidence,
        "_confidence_cap": round(conf_cap, 3),
        "_confidence_cap_reason": conf_reason,
        "risk_signals": risk_signals[:3],
        "protective_signals": protective_signals[:4],
        "reasoning": reasoning,
        "display_brief": brief,
        "_fallback_reason": f"Fallback parser used because LLM JSON parsing failed: {reason}",
        "_raw": raw or reason,
    }

def build_text_display_brief(result: dict) -> str:
    """Create a short UI sentence from parsed text-analysis fields."""
    risk_signals = [str(s) for s in result.get("risk_signals", []) if str(s).strip()]
    protective_signals = [str(s) for s in result.get("protective_signals", []) if str(s).strip()]
    score = float(result.get("text_risk_score", 0.45))

    if protective_signals and not risk_signals:
        if score < 0.25:
            return "Specific employment, payment, income, and loan-purpose details indicate low text-based default risk."
        return "Protective borrower details outweigh risk signals in the description."
    if risk_signals and not protective_signals:
        return "The description contains borrower stress signals that raise text-based default risk."
    if risk_signals and protective_signals:
        return "The description contains both protective context and risk signals, so text evidence is mixed."
    return "The description provides limited credit-relevant evidence, so text influence should remain modest."

def concise_text(text: str, max_chars: int = 260) -> str:
    """Compact long agent prose for first-pass display."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    out = ""
    for sentence in sentences:
        candidate = f"{out} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        out = candidate
    if not out:
        out = cleaned[:max_chars].rsplit(" ", 1)[0]
    return f"{out} ..."

def strip_llm_thinking(text: str) -> str:
    """Remove reasoning traces if a model leaks <think> content."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.I).strip()
    if cleaned:
        return cleaned
    # If the model starts a think block and truncates before closing it, hide it.
    cleaned = re.sub(r"<think>.*", "", str(text or ""), flags=re.DOTALL | re.I).strip()
    return cleaned

def render_highlighted_description(container, desc_text: str, signals: list[str]) -> None:
    """Render borrower text with matched signal phrases highlighted."""
    text = str(desc_text or "")
    intervals: list[tuple[int, int]] = []
    lowered = text.lower()
    for signal in signals:
        signal = str(signal or "").strip().strip('"')
        if not signal:
            continue
        start = lowered.find(signal.lower())
        if start >= 0:
            intervals.append((start, start + len(signal)))

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start >= merged[-1][1]:
            merged.append((start, end))

    if not merged:
        container.caption("Text Analyst evidence will be highlighted here after a successful run.")
        return

    parts = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(text[cursor:start]))
        parts.append(f'<strong class="evidence-hit">{html.escape(text[start:end])}</strong>')
        cursor = end
    parts.append(html.escape(text[cursor:]))
    container.markdown(
        f"""
<div class="evidence-map">
  <div class="evidence-map-title">Mapped Text Analyst evidence</div>
  <div class="evidence-map-body">{''.join(parts)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

def text_analyst(desc_text: str) -> dict:
    desc_text = re.sub(r"<[^>]+>", " ", str(desc_text)).strip()[:500]
    word_count  = len(desc_text.split())
    has_numbers = bool(re.search(r"\$[\d,]+|\d+%|\d+ months?|\d+ years?", desc_text))
    has_plan    = bool(re.search(r"will pay|plan to|intend|commit|currently|stable", desc_text, re.I))
    has_stress  = bool(re.search(r"emergency|urgent|desperate|behind|struggling|need immediately", desc_text, re.I))

    prompt = (
        "You are evaluating loan applications from Grade C-G borrowers on LendingClub.\n"
        "Score the DESCRIPTION TEXT ONLY — not the borrower's credit grade.\n\n"
        f"Description: {desc_text}\n\n"
        "Scoring rules — read carefully:\n\n"
        "START at 0.45 (baseline for a vague C-G description), then ADJUST:\n\n"
        "LOWER the score (subtract from 0.45) for each protective signal present:\n"
        "  -0.15  mentions stable employment or specific job title\n"
        "  -0.10  states a concrete repayment plan or budget\n"
        "  -0.10  explicitly says 'no missed payments' or 'accounts in good standing'\n"
        "  -0.08  provides specific dollar amount of income or savings\n"
        "  -0.05  mentions a single clear purpose (e.g. one consolidation, one purchase)\n\n"
        "RAISE the score (add to 0.45) for each risk signal present:\n"
        "  +0.15  urgent or desperate language ('need immediately', 'emergency')\n"
        "  +0.12  explicitly behind on payments or facing collections\n"
        "  +0.10  unemployed or no income mentioned at all\n"
        "  +0.08  multiple problem debts with no repayment plan\n"
        "  +0.05  vague about income or employment\n\n"
        "Clamp final score to [0.05, 0.95].\n\n"
        "Examples:\n"
        "  'I need to consolidate my debt.' → 0.45 (nothing to adjust)\n"
        "  'Stable nurse 8 yrs, consolidating 2 cards, never missed a payment.' "
        "→ 0.45 - 0.15 - 0.10 - 0.10 = 0.10 → clamp to 0.15\n"
        "  'Behind on rent, multiple maxed cards, no job.' "
        "→ 0.45 + 0.12 + 0.08 + 0.10 = 0.75\n\n"
        "CRITICAL OUTPUT FORMAT: respond with ONLY the JSON object. No reasoning, no markdown code fences, no commentary. Your response must start with { and end with }. Any text outside the JSON braces will break the parser.\n\n" + "Output a single JSON object and nothing else:\n"
        '{"text_risk_score": <computed float>,\n'
        ' "confidence": <float 0.4-0.95>,\n'
        ' "risk_signals": [<up to 3 exact quoted phrases from the description>],\n'
        ' "protective_signals": [<up to 3 exact quoted phrases from the description>],\n'
        ' "reasoning": "<one sentence showing your adjustment math>"}\n\n'
        "Confidence must reflect how much evidence the text contains, not how strong the "
        "risk/protective direction feels. Extremely short or vague descriptions must have "
        "low confidence: 'I need.' or 'Need loan.' should be around 0.40-0.45; one vague "
        "sentence with no income, employment, repayment plan, or purpose detail should be "
        "at most 0.55-0.62."
    )
    llm_raw = call_llm(prompt, max_tokens=800, temperature=0.1)
    try:
        result = _extract_json_object(llm_raw)
    except Exception as e:
        return fallback_text_analysis(desc_text, str(e), llm_raw)

    raw_confidence = min(max(float(result.get("confidence", 0.6)), 0.4), 0.95)
    conf_cap, conf_cap_reason = confidence_cap_for_text(desc_text)
    result["confidence"] = round(min(raw_confidence, conf_cap), 3)
    result["_raw_confidence"] = round(raw_confidence, 3)
    result["_confidence_cap"] = round(conf_cap, 3)
    result["_confidence_cap_reason"] = conf_cap_reason
    result["text_risk_score"] = round(min(max(float(result.get("text_risk_score", 0.45)), 0.05), 0.95), 3)
    result["display_brief"] = build_text_display_brief(result)
    result["_raw"] = llm_raw   # keep raw for debug display
    return result

def reporter(row_dict, baseline, fused, grade, text_score, text_reasoning,
             effective_weight, confidence=0.0, veto_active=False, hist_delta=0.0,
             threshold=0.5):
    delta = fused - baseline
    abs_pp = abs(delta) * 100
    margin_pp = abs(fused - threshold) * 100
    direction = "raised" if delta > 0 else "lowered" if delta < 0 else "did not change"
    baseline_verdict = "DEFAULT" if baseline >= threshold else "NON-DEFAULT"
    fused_verdict = "DEFAULT" if fused >= threshold else "NON-DEFAULT"
    verdict_changed = baseline_verdict != fused_verdict
    if verdict_changed:
        verdict_effect = f"changed the verdict from {baseline_verdict} to {fused_verdict}"
    else:
        verdict_effect = f"did not change the verdict ({fused_verdict})"

    if effective_weight == 0:
        weight_rule = (
            f"\n\nHARD RULE — MUST FOLLOW:\n"
            f"The effective text weight for Grade {grade} is 0.00. "
            f"The text score ({text_score:.2f}) did NOT change the fused score mathematically "
            f"(fused = {fused:.3f} = baseline {baseline:.3f}, unchanged). "
            f"You MUST clearly state that the text did not affect the prediction. "
            f"Do NOT write phrases like 'text further increases', 'text highlights', "
            f"or 'text-driven'. The text is contextual evidence only, suppressed by subgroup policy."
        )
        if veto_active:
            weight_rule += (
                f"\nThe reason text is suppressed: for Grade {grade}, "
                f"{historical_text_impact(hist_delta, grade)} — Subgroup Advocate blocked additional text influence."
            )
    else:
        weight_rule = (
            f"\nFusion method: logit space. "
            f"text evidence = logit({text_score:.2f}) − logit(0.5); "
            f"fused_logit = logit(baseline) + {effective_weight:.2f} × confidence × text_evidence; "
            f"fused_pred = sigmoid(fused_logit) = {fused:.3f}. "
            f"Describe the text's directional influence (raised/lowered the log-odds) proportionally. "
            f"If the probability shift is small, explain why it still matters or why it remains limited."
        )

    if effective_weight == 0:
        p2_guidance = (
            "Paragraph 2 must explain WHY the system DISTRUSTED the text for this Grade "
            f"{grade}: subgroup history showed text was historically unreliable or harmful "
            "for this grade, so the Advocate suppressed it. Frame it as 'the text was heard "
            "but not trusted, because…'. Do NOT say the text moved the score (it did not). "
            "Discuss whether the trust decision feels right given the borrower's narrative quality."
        )
    else:
        p2_guidance = (
            "Paragraph 2 must judge whether the text earned its influence. The system "
            f"PARTIALLY trusted it (effective weight {effective_weight:.2f}, confidence {confidence:.2f}, "
            f"causing a {abs_pp:.1f}-pp shift from {baseline:.3f} to {fused:.3f}). Decide whether "
            "the borrower's narrative is concrete and specific enough that this nudge feels earned, "
            "or whether the system is granting more trust than the evidence justifies. End with an "
            "explicit verdict on whether the text earned its influence in this case."
        )

    prompt = (
        f"You are a senior loan default risk analyst writing a concise model-audit insight "
        f"on a Grade {grade} borrower whose fused score sits in the borderline zone.\n\n"
        f"=== Numeric facts ===\n"
        f"Baseline (numeric-only): {baseline:.3f}  |  Fused: {fused:.3f}  |  "
        f"Threshold: {threshold:.2f}  |  Verdict: {fused_verdict}\n"
        f"Distance from threshold: {margin_pp:.1f} pp ({'above' if fused >= threshold else 'below'})\n"
        f"Key features: loan=${row_dict.get('loan_amnt',0):,.0f}, "
        f"int_rate={row_dict.get('int_rate',0):.1f}%, "
        f"annual_inc=${row_dict.get('annual_inc',0):,.0f}, "
        f"dti={row_dict.get('dti',0):.1f}, fico={row_dict.get('fico_range_low',0):.0f}\n\n"
        f"=== Text facts ===\n"
        f"Text risk score: {text_score:.3f}  |  Text confidence: {confidence:.3f}  |  "
        f"Effective text weight: {effective_weight:.2f}\n"
        f"Computed text impact: text {direction} the default probability by {abs_pp:.1f} pp "
        f"({baseline:.3f} → {fused:.3f}); {verdict_effect}.\n"
        f"Text Analyst reasoning: {text_reasoning}\n"
        f"{weight_rule}\n\n"
        f"=== Writing task ===\n"
        f"Produce two paragraphs of finished prose. Each paragraph is 3-4 sentences, roughly "
        f"70-100 words. No bullets, headings, labels, or meta commentary.\n\n"
        f"Paragraph 1 must focus ONLY on THE PREDICTION RESULT. Pick the 2-3 numeric features "
        f"that matter most for this borrower and explain the tension between risk drivers and "
        f"mitigants. State explicitly where fused={fused:.3f} sits vs. threshold={threshold:.2f} "
        f"and the resulting verdict ({fused_verdict}, {margin_pp:.1f} pp away). Do NOT discuss "
        f"the text in paragraph 1.\n\n"
        f"{p2_guidance}\n\n"
        f"Avoid filler phrases like 'presents a borderline case', 'provides reassurance', or "
        f"'moderate level of risk' unless tied to specific evidence.\n\n"
        f"CRITICAL OUTPUT FORMAT: respond with ONLY a JSON object. No reasoning, no markdown "
        f"code fences, no commentary. Your response must start with {{ and end with }}. Any text "
        f"outside the JSON braces will break the parser.\n\n"
        f'Output exactly this JSON object: {{"paragraph_1": "<finished prose for paragraph 1>", '
        f'"paragraph_2": "<finished prose for paragraph 2>"}}'
    )
    raw = call_llm(prompt, max_tokens=3000, temperature=0.25, response_format_json=True)
    try:
        obj = _extract_json_object(raw)
        p1 = str(obj.get("paragraph_1", "")).strip()
        p2 = str(obj.get("paragraph_2", "")).strip()
        if p1 and p2:
            return f"{p1}\n\n{p2}"
        if p1 or p2:
            return p1 or p2
    except Exception:
        pass
    return _reporter_deterministic_fallback(
        row_dict, baseline, fused, grade, text_score, text_reasoning,
        effective_weight, confidence, veto_active, hist_delta, threshold,
    )


def _reporter_deterministic_fallback(
    row_dict, baseline, fused, grade, text_score, text_reasoning,
    effective_weight, confidence, veto_active, hist_delta, threshold,
) -> str:
    """Python-only two-paragraph narrative used when the LLM returns nothing usable.

    Guarantees the user sees a real, faithful, two-paragraph audit even when the
    thinking-model leaks reasoning and exhausts the token budget before producing
    valid JSON. Numbers and verdict logic are derived directly from the inputs, so
    the fallback can never contradict the actual prediction.
    """
    delta = fused - baseline
    abs_pp = abs(delta) * 100
    margin_pp = abs(fused - threshold) * 100
    fused_verdict = "DEFAULT" if fused >= threshold else "NON-DEFAULT"
    above_or_below = "above" if fused >= threshold else "below"

    loan = float(row_dict.get("loan_amnt", 0) or 0)
    rate = float(row_dict.get("int_rate", 0) or 0)
    inc = float(row_dict.get("annual_inc", 0) or 0)
    dti = float(row_dict.get("dti", 0) or 0)
    fico = float(row_dict.get("fico_range_low", 0) or 0)

    risk_bits, mit_bits = [], []
    if rate >= 17.0:
        risk_bits.append(f"a high interest rate of {rate:.1f}%")
    elif rate >= 12.0:
        risk_bits.append(f"an elevated interest rate of {rate:.1f}%")
    if dti >= 25.0:
        risk_bits.append(f"a stretched DTI of {dti:.1f}")
    elif dti <= 15.0 and dti > 0:
        mit_bits.append(f"a comfortable DTI of {dti:.1f}")
    if fico >= 720:
        mit_bits.append(f"a strong FICO of {fico:.0f}")
    elif fico >= 680 and fico < 720:
        mit_bits.append(f"a serviceable FICO of {fico:.0f}")
    elif fico > 0 and fico < 680:
        risk_bits.append(f"a thin FICO of {fico:.0f}")
    if inc >= 80000:
        mit_bits.append(f"an annual income of ${inc:,.0f}")
    elif inc > 0 and inc < 50000:
        risk_bits.append(f"a modest income of ${inc:,.0f}")
    if loan > 0 and inc > 0 and loan / inc >= 0.30:
        risk_bits.append(f"a loan-to-income ratio near {loan / inc:.0%}")

    risk_clause = ("Risk drivers include " + ", ".join(risk_bits) + ".") if risk_bits else ""
    mit_clause = ("Mitigants include " + ", ".join(mit_bits) + ".") if mit_bits else ""
    if not risk_bits and not mit_bits:
        risk_clause = (
            f"The numeric profile (rate {rate:.1f}%, DTI {dti:.1f}, FICO {fico:.0f}, "
            f"income ${inc:,.0f}) is broadly mid-range."
        )

    p1 = (
        f"For this Grade {grade} borrower the numeric model produced a baseline default "
        f"probability of {baseline:.3f}, and the fused score settled at {fused:.3f}. "
        f"{risk_clause} {mit_clause} "
        f"That places the fused score {margin_pp:.1f} percentage points {above_or_below} "
        f"the {threshold:.2f} decision threshold, yielding a {fused_verdict} verdict — "
        f"{'comfortably clear of' if margin_pp >= 5 else 'right on'} the boundary."
    ).strip()

    if effective_weight == 0:
        reason_phrase = historical_text_impact(hist_delta, grade) if veto_active else (
            f"the borrower's narrative did not clear the trust bar for Grade {grade}"
        )
        p2 = (
            f"The text was heard but not trusted here. Despite a text confidence of "
            f"{confidence:.2f}, the effective weight was driven to 0.00 — "
            f"{reason_phrase}, so the Subgroup Advocate suppressed any text influence. "
            f"As a result the fused score equals the numeric baseline and the text did not "
            f"move the prediction. Given the subgroup's history this trust decision is the "
            f"conservative call: the narrative is treated as context rather than evidence."
        )
    else:
        direction_word = "downward" if delta < 0 else ("upward" if delta > 0 else "neutral")
        earned_clause = (
            "the nudge is proportionate — small enough to respect the subgroup history while "
            "letting a credible narrative register"
            if abs_pp <= 4.0
            else "the nudge is on the larger side, so the narrative needs to be concrete to justify it"
        )
        p2 = (
            f"The text was partially trusted: confidence {confidence:.2f} combined with an "
            f"effective weight of {effective_weight:.2f} produced a {direction_word} shift of "
            f"{abs_pp:.1f} percentage points ({baseline:.3f} → {fused:.3f}). "
            f"{earned_clause}. "
            f"On balance, the text earned its limited influence in this case, even though it "
            f"{'did not flip' if (baseline >= threshold) == (fused >= threshold) else 'flipped'} "
            f"the verdict."
        )

    return f"{p1}\n\n{p2}"

def run_strategist_llm(grade, baseline_pred, text_score, text_confidence,
                       learned_w, train_stats, desc_text, threshold=0.5):
    """
    LLM Strategist: reads the full conflict context, outputs a rich strategy object.
    Called only when genuine conflict or complexity is detected.
    """
    hist_delta   = train_stats.get("delta_at_high_weight", 0.0)
    hist_reason  = train_stats.get("reason", "unknown")
    history_summary = historical_text_impact(hist_delta, grade)
    in_fuzzy     = 0.30 <= baseline_pred <= 0.65
    score_gap    = round(text_score - baseline_pred, 3)
    near_threshold = abs(baseline_pred - threshold) < 0.05

    baseline_w_desc = (
        f"Static learned text weight (grade-level): {learned_w:.2f}\n"
        f"Historical text impact: {history_summary}\n"
        f"Reason: {hist_reason}"
    )
    weight_rules = (
        "- The static weight is conservative grade policy. case_specific_exception or fuzzy_only "
        "means propose an INCREASE above this baseline when the case warrants extra text influence.\n"
        "- proposed_weight must be STRICTLY GREATER than learned when you choose case_specific_exception "
        "or fuzzy_only (typically learned + 0.03 to learned + 0.10). If you would propose a value equal "
        "to learned, you must instead choose keep_learned or explanation_only — do NOT label a no-op as "
        "case_specific_exception.\n"
        "- proposed_weight must equal learned for keep_learned / explanation_only / human_review.\n"
        "- Use a modest increase when historical text impact is unstable or harmful, because the "
        "Advocate may veto and Green will then arbitrate. Do not refuse to propose simply because of "
        "veto risk — that is what the Advocate and Arbitrator are for.\n"
        "- proposed_weight must be in [0.00, 0.20].\n"
    )

    decision_heuristics = (
        "=== Decision heuristics (read carefully) ===\n"
        "Pick strategy_type by what the case actually warrants, not by what feels safe:\n\n"
        "- case_specific_exception: choose this when text_confidence >= 0.80 AND |score_gap| >= 0.25 "
        "AND the description contains concrete verifiable details (employment tenure, exact income, "
        "explicit payment-history claim, named profession, specific debt purpose). In this regime the "
        "text is carrying real signal that disagrees with the numeric model, and your job is to let "
        "it nudge the weight up. A typical exception is learned + 0.05, capped by max_prediction_shift "
        "around 0.05; the Advocate decides whether to veto.\n"
        "- fuzzy_only: choose when baseline is in fuzzy zone but the text justification rests primarily "
        "on the fuzzy-zone status rather than on description-specific evidence.\n"
        "- explanation_only: this is NOT a polite default. Use it only when the text is vague, short, "
        "lacks specifics, or merely echoes the numeric verdict. A confident, detailed, strongly "
        "disagreeing text is the OPPOSITE of explanation_only — it is the prototypical "
        "case_specific_exception case. Do not retreat to explanation_only just because the historical "
        "subgroup impact is mixed.\n"
        "- keep_learned: text adds no new information (low confidence, or aligns with baseline).\n"
        "- human_review: reserve for genuine ambiguity where automated reasoning is unsafe.\n\n"
    )

    prompt = (
        "You are the Feature Strategist in a loan default prediction multi-agent system.\n"
        "Your job is to propose a strategy for how the text risk score should influence the final prediction.\n"
        "You must reason carefully — you are not a rule engine.\n\n"
        f"=== Mode ===\n"
        f"Baseline-weight source: Static grade-level policy\n\n"
        f"=== Current case context ===\n"
        f"Grade: {grade}\n"
        f"Numeric baseline prediction: {baseline_pred:.3f} (threshold = {threshold})\n"
        f"Text risk score: {text_score:.3f}  (text confidence: {text_confidence:.3f})\n"
        f"Score gap (text - baseline): {score_gap:+.3f}\n"
        f"Baseline in fuzzy zone [0.30–0.65]: {'YES' if in_fuzzy else 'NO'}\n"
        f"Near decision threshold (±0.05): {'YES' if near_threshold else 'NO'}\n\n"
        f"=== Baseline text weight ===\n"
        f"{baseline_w_desc}\n\n"
        f"=== Borrower description snippet ===\n{desc_text[:300]}\n\n"
        "Choose ONE strategy_type:\n"
        "  keep_learned          — baseline weight is already appropriate, no change needed\n"
        "  case_specific_exception — deviate from baseline because of description-level cues (use constraints)\n"
        "  fuzzy_only            — deviation applies only because baseline is in fuzzy zone [0.30–0.65]\n"
        "  explanation_only      — text informs the Reporter narrative but does NOT change weight\n"
        "  human_review          — flag for human review; do not change prediction\n\n"
        "For case_specific_exception or fuzzy_only, you may set constraints:\n"
        "  max_prediction_shift: maximum allowed change from baseline (e.g. 0.05)\n"
        "  min_text_confidence: minimum confidence required (e.g. 0.85)\n\n"
        + decision_heuristics +
        "Rules:\n"
        "- The Advocate will reason about grade-level historical text impact and may soft-veto; Green then arbitrates.\n"
        "- If you choose keep_learned, explanation_only, or human_review, proposed_weight should equal the baseline text weight (the value shown above).\n"
        + weight_rules +
        "\nStyle: make the rationale specific to this case; avoid generic phrases such as "
        "'text confidence is high' unless you connect them to the numeric context. "
        "Also write display_brief as a complete UI-ready sentence under 30 words; "
        "do not use ellipses or refer to hidden reasoning.\n\n"
        "CRITICAL OUTPUT FORMAT: respond with ONLY the JSON object. "
        "No reasoning, no markdown code fences, no commentary. "
        "Your response must start with { and end with }. Any text outside the JSON braces will break the parser.\n\n"
        '{"strategy_type": "<type>", "proposed_weight": <float>, '
        '"constraints": {"max_prediction_shift": <float or null>, "min_text_confidence": <float or null>, '
        '"only_if_fuzzy_zone": <true/false>}, "rationale": "<one sentence>", '
        '"display_brief": "<complete UI-ready sentence, max 30 words, no ellipsis>"}'
    )
    raw = call_llm(prompt, max_tokens=1200, temperature=0.2, response_format_json=True)
    try:
        result = _extract_json_object(raw)
    except Exception:
        result = {"strategy_type": "keep_learned", "proposed_weight": learned_w,
                  "constraints": {}, "rationale": "[parse error — defaulting to learned strategy]",
                  "display_brief": "Strategist output could not be parsed, so the learned grade-level text weight was kept."}

    # Safety clamp: keep the legacy upward-only ceiling of learned + 0.15 (cap [0, 0.2]).
    proposed = float(result.get("proposed_weight", learned_w))
    lo, hi = 0.0, min(0.20, learned_w + 0.15)
    result["proposed_weight"] = round(min(max(proposed, lo), hi), 3)
    return result, raw

def run_advocate_llm(grade, baseline_pred, text_score, confidence,
                     learned_w, proposed_w, hist_delta, hist_reason,
                     strategist_type, strategist_rationale, strategist_constraints,
                     desc_text, threshold=0.5):
    """
    Intelligent Advocate: reasons about subgroup reliability and current-case evidence.
    Deterministic severe-harm floors are enforced outside this function.
    """
    score_gap = text_score - baseline_pred
    baseline_fuzzy = 0.30 <= baseline_pred <= 0.65
    near_threshold = abs(baseline_pred - threshold) < 0.05
    text_extreme = abs(text_score - 0.5)
    history_summary = historical_text_impact(hist_delta, grade)
    text_evidence = logit(text_score) - logit(0.5)
    pred_learned = sigmoid(logit(baseline_pred) + learned_w * confidence * text_evidence)
    pred_proposed = sigmoid(logit(baseline_pred) + proposed_w * confidence * text_evidence)
    proposed_shift = pred_proposed - baseline_pred
    proposed_flip = (pred_proposed >= threshold) != (baseline_pred >= threshold)

    prompt = (
        "You are the Subgroup Advocate in a loan default prediction multi-agent system.\n"
        "Unlike a simple rule check, you must reason about whether this grade's text signal "
        "is reliable enough for this individual case. Your priority is subgroup reliability, "
        "not maximizing the Strategist's influence.\n\n"
        "=== Current case ===\n"
        f"Grade: {grade}\n"
        f"Baseline prediction: {baseline_pred:.3f}; threshold: {threshold:.2f}\n"
        f"Text risk score: {text_score:.3f}; confidence: {confidence:.3f}\n"
        f"Score gap (text - baseline): {score_gap:+.3f}\n"
        f"Text extremeness |score-0.5|: {text_extreme:.3f}\n"
        f"Baseline fuzzy zone [0.30, 0.65]: {'YES' if baseline_fuzzy else 'NO'}\n"
        f"Near threshold (+/-0.05): {'YES' if near_threshold else 'NO'}\n"
        f"Borrower text snippet: {desc_text[:350]}\n\n"
        "=== Strategist proposal ===\n"
        f"Strategy type: {strategist_type}\n"
        f"Weight: {learned_w:.3f} -> {proposed_w:.3f}\n"
        f"Rationale: {strategist_rationale}\n"
        f"Constraints: {json.dumps(strategist_constraints)}\n"
        f"At learned weight: pred={pred_learned:.3f}, shift={pred_learned - baseline_pred:+.3f}\n"
        f"At proposed weight: pred={pred_proposed:.3f}, shift={proposed_shift:+.3f}, "
        f"verdict_flip={'YES' if proposed_flip else 'NO'}\n\n"
        "=== Historical impact for similar cases ===\n"
        f"{history_summary}\n"
        f"Training note: {hist_reason}\n\n"
        "Decision options:\n"
        "  pass       — allow Strategist proposal; text history is acceptable for this case\n"
        "  soft_veto  — text may contribute, but only with stricter constraints or reduced weight\n"
        "  hard_veto  — text should not increase influence for this grade/case\n\n"
        "Guidance:\n"
        "- Your role is subgroup reliability, not final verdict governance; verdict flips are checked later by the Green Agent.\n"
        "- If historical impact is neutral or helpful, do NOT issue soft_veto or hard_veto. Pass the proposal, optionally noting that Green should inspect any verdict flip.\n"
        "- Historically harmful or unstable text is serious subgroup evidence; require strong current-case evidence to avoid veto.\n"
        "- High confidence alone is not enough if the description is generic or the proposed shift would flip the verdict.\n"
        "- If history is mildly negative but text is highly detailed, extreme, and baseline is fuzzy, prefer soft_veto over hard_veto.\n"
        "- If proposed weight creates a large shift or verdict flip, require stronger justification and constraints.\n"
        "- max_allowed_weight must be above learned_weight and below proposed_weight for soft_veto; "
        "soft_veto means reduced influence, not reverting to the learned weight. "
        "Use learned_weight only for hard_veto.\n\n"
        "Style: write the rationale in varied, case-specific language. Do not merely echo the historical-impact phrase. "
        "Also write display_brief as a complete UI-ready sentence under 30 words; do not use ellipses.\n\n"
        "CRITICAL OUTPUT FORMAT: respond with ONLY the JSON object. No reasoning, no markdown code fences, no commentary. Your response must start with { and end with }. Any text outside the JSON braces will break the parser.\n\n" + 'Output only JSON: {"decision": "<pass|soft_veto|hard_veto>", '
        '"severity": "<low|medium|high>", "max_allowed_weight": <float>, '
        '"max_prediction_shift": <float or null>, "rationale": "<one sentence using historical impact language, not AUC numbers>", '
        '"display_brief": "<complete UI-ready sentence, max 30 words, no ellipsis>", '
        '"constraints": ["<short constraint>", "..."]}'
    )
    raw = call_llm(prompt, max_tokens=1200, temperature=0.2, response_format_json=True)
    try:
        result = _extract_json_object(raw)
    except Exception as e:
        result = {
            "decision": "soft_veto" if hist_delta < 0 else "pass",
            "severity": "medium",
            "max_allowed_weight": min(proposed_w, learned_w + max(0.01, 0.5 * (proposed_w - learned_w))) if hist_delta < 0 else proposed_w,
            "max_prediction_shift": 0.05 if hist_delta < 0 else None,
            "rationale": f"[parse error: {e}] defaulting conservatively",
            "display_brief": "Advocate output could not be parsed, so subgroup reliability was handled conservatively.",
            "constraints": ["parse error fallback"],
        }

    decision = str(result.get("decision", "soft_veto")).strip().lower()
    if decision not in ("pass", "soft_veto", "hard_veto"):
        decision = "soft_veto" if hist_delta < 0 else "pass"

    # Role boundary: Advocate may veto only for subgroup reliability uncertainty.
    # Clear helpful history should pass; weak support near the threshold can still
    # justify soft-veto review when the Strategist is materially increasing weight.
    if hist_delta >= 0.01 and decision != "pass":
        decision = "pass"
        result["severity"] = "low"
        result["rationale"] = (
            f"For Grade {grade}, {historical_text_impact(hist_delta, grade)}, "
            "so subgroup reliability does not justify a veto; "
            "any verdict flip should be handled by Green final authority."
        )
        result["display_brief"] = (
            f"Grade {grade} history supports text use, so Advocate passes the proposal and leaves verdict flips to Green."
        )
        result["constraints"] = ["Green Agent should inspect any verdict flip"]
        result["max_prediction_shift"] = None
    elif 0 <= hist_delta < 0.01 and proposed_w > learned_w + 0.025 and near_threshold and decision == "pass":
        decision = "soft_veto"
        result["severity"] = "medium"
        result["rationale"] = (
            f"For Grade {grade}, {historical_text_impact(hist_delta, grade)}. "
            "Because this borrower sits close to the decision boundary, the Advocate "
            "does not reject the text but asks Green to arbitrate it down to a narrower influence."
        )
        result["display_brief"] = (
            f"Advocate allows text signal but asks Green to narrow its influence near the decision boundary."
        )
        result["constraints"] = [
            "arbitrator review near threshold",
            "limit incremental text weight",
            "avoid letting modest history become decisive",
        ]
        result["max_allowed_weight"] = min(proposed_w, learned_w + 0.02)
        result["max_prediction_shift"] = 0.04
    elif hist_delta < -0.01 and proposed_w > learned_w + 0.001 and decision == "pass":
        decision = "soft_veto"
        result["severity"] = "medium"
        result["rationale"] = (
            f"For Grade {grade}, {historical_text_impact(hist_delta, grade)}. "
            "The borrower narrative is usable, but the Advocate requires Green to arbitrate "
            "to narrow the extra text influence before it reaches final fusion."
        )
        result["display_brief"] = (
            f"Advocate treats the text as usable but requires Green to limit extra influence for this grade."
        )
        result["constraints"] = [
            "require arbitrator review",
            "limit extra text influence",
            "keep the text role case-specific",
        ]
        result["max_allowed_weight"] = min(proposed_w, learned_w + 0.02)
        result["max_prediction_shift"] = 0.04
    result["decision"] = decision

    if decision == "hard_veto":
        max_w = learned_w
    elif decision == "pass":
        max_w = proposed_w
    else:
        min_soft_increment = min(proposed_w - learned_w, max(0.01, 0.5 * (proposed_w - learned_w)))
        soft_floor = learned_w + min_soft_increment
        max_w = float(result.get("max_allowed_weight", proposed_w))
        max_w = min(max(max_w, soft_floor), proposed_w)
        result["constraints"] = [
            str(c) for c in result.get("constraints", [])
            if "learned weight" not in str(c).lower()
            and "learned_weight" not in str(c).lower()
            and "0.060" not in str(c)
        ]
    result["max_allowed_weight"] = round(max_w, 3)

    mps = result.get("max_prediction_shift")
    result["max_prediction_shift"] = None if mps in (None, "null", "") else round(max(float(mps), 0.0), 3)
    result["severity"] = str(result.get("severity", "medium")).strip().lower()
    result["rationale"] = result.get("rationale", "")
    result["constraints"] = result.get("constraints", [])
    return result, raw

def run_green_llm(baseline_pred, fused_before, fused_after, text_score,
                  confidence, grade, green_action, green_notes,
                  strat_rationale, hist_delta, threshold=0.5):
    """
    Green Agent LLM — final authority review for contested edge cases.
    Called only when hard rules fire on borderline evidence.
    Output is terminal: no further override possible.
    """
    flip_direction = ""
    if fused_before >= threshold and fused_after < threshold:
        flip_direction = "risky→safe (White Agents wanted to CLEAR the default prediction)"
    elif fused_before < threshold and fused_after >= threshold:
        flip_direction = "safe→risky (White Agents wanted to RAISE to default)"

    prompt = (
        "You are the Green Agent — the final authority in a loan default prediction system.\n"
        "White Agents (Text Analyst, Strategist, Advocate) finished their pipeline and produced a fused score; "
        "Green's flip gate then blocked the verdict change. You must decide whether to confirm or reverse "
        "that block.\n\n"
        f"=== Case context ===\n"
        f"Credit grade: {grade} | Decision threshold: {threshold}\n"
        f"Numeric baseline: {baseline_pred:.3f} ({'DEFAULT' if baseline_pred >= threshold else 'NON-DEFAULT'})\n"
        f"Fused after White Agents: {fused_before:.3f} ({'DEFAULT' if fused_before >= threshold else 'NON-DEFAULT'})"
        + (f" → flip detected ({flip_direction})\n" if flip_direction else "\n")
        + f"After Green hard rules: {fused_after:.3f} (rule fired: {green_action})\n"
        f"Rule note: {'; '.join(green_notes) if green_notes else 'none'}\n\n"
        f"=== Text evidence ===\n"
        f"Text risk score: {text_score:.3f}  (distance from 0.5: {abs(text_score - 0.5):.3f})\n"
        f"LLM confidence: {confidence:.3f}\n"
        f"Strategist rationale: {strat_rationale}\n"
        f"Training-time AUC delta at high text weight for Grade {grade}: {hist_delta:+.3f} "
        f"(positive = text historically helped this grade; negative = it has historically harmed)\n\n"
        "=== Your judgment ===\n"
        "You are the LAST checkpoint. The flip gate is a coarse safety check. "
        "Your job is to weigh the actual evidence:\n"
        "  • Was White Agents' shift backed by genuinely strong, internally consistent text evidence?\n"
        "  • Or was it driven by noisy/over-confident text that should not flip a numeric verdict?\n"
        "Use the numbers above to form your own conclusion. Do NOT default to one side; the data should drive you.\n\n"
        "Choose ONE decision and provide reasoning that REFERENCES THE SPECIFIC NUMBERS above:\n"
        "  accept_baseline — confirm the block; the flip should not happen\n"
        "  accept_fused    — overturn the block; the fused value (before hard rules) stands\n"
        "  custom          — settle on a value strictly between baseline and fused (rare)\n\n"
        f"For `custom`, final_pred must be inside [{min(baseline_pred, fused_before):.3f}, "
        f"{max(baseline_pred, fused_before):.3f}]. For accept_baseline use {baseline_pred:.3f}; "
        f"for accept_fused use {fused_before:.3f}.\n\n"
        "CRITICAL OUTPUT FORMAT: respond with ONLY the JSON object. No reasoning prose outside JSON, "
        "no markdown code fences. Your response must start with { and end with }.\n\n"
        f'{{"decision": "<accept_baseline|accept_fused|custom>", '
        f'"final_pred": <float>, "reasoning": "<one sentence citing the specific numbers from above>", '
        f'"display_brief": "<complete UI-ready sentence, max 30 words, no ellipsis>"}}'
    )
    raw = call_llm(prompt, max_tokens=1200, temperature=0.2, response_format_json=True)
    try:
        result = _extract_json_object(raw)
        lo, hi = min(baseline_pred, fused_before), max(baseline_pred, fused_before)
        result["final_pred"] = round(float(np.clip(float(result.get("final_pred", fused_after)), lo, hi)), 4)
        return result, raw
    except Exception as e:
        return {"decision": "accept_baseline", "final_pred": baseline_pred,
                "reasoning": f"[parse error: {e}]",
                "display_brief": "Green review output could not be parsed, so the baseline decision was retained."}, raw

def text_risk_label(score: float) -> str:
    if score < 0.25:   return "🟢 Low"
    if score < 0.40:   return "🟡 Below Average"
    if score < 0.55:   return "🟠 Average (C-G)"
    if score < 0.70:   return "🔴 Elevated"
    return              "🔴🔴 High"

# ── Preset demo cases ─────────────────────────────────────────────────────────
PRESETS = {
    # Case 1: Grade C, protective text → Strategist proposes weight increase → HARD VETO
    # Grade C delta=-0.054 < -0.02 → hard veto; Arbitrator enforces (sides with Advocate)
    # Narrative: borrower sounds trustworthy, but Grade C text historically hurts AUC badly
    "Case 1 · Grade C — Arbitrator sides with Advocate": {
        "grade": "C", "loan_amnt": 14000, "int_rate": 17.5, "installment": 350.0,
        "annual_inc": 68000, "dti": 14.0, "delinq_2yrs": 0, "fico_low": 705,
        "fico_high": 709, "inq_6mths": 1, "open_acc": 12, "pub_rec": 0,
        "revol_bal": 7000, "revol_util": 38.0, "total_acc": 22, "mort_acc": 1,
        "pub_rec_bk": 0,
        "desc": (
            "I am a registered nurse with 9 years of continuous employment at the same "
            "hospital. My annual salary is $68,000 and I have never missed a single "
            "payment on any account. I am consolidating two low-balance credit cards "
            "to simplify my finances and reduce my interest rate."
        ),
    },
    # Case 2: Grade E, protective text → Strategist proposes weight increase → SOFT VETO
    # Grade E delta=-0.018, in range [-0.02, 0) → soft veto; Arbitrator LLM negotiates (sides with Strategist)
    # Narrative: similar trustworthy borrower, but Grade E text marginally harmful — room for compromise
    "Case 2 · Grade E — Arbitrator sides with Strategist": {
        "grade": "E", "loan_amnt": 14000, "int_rate": 17.5, "installment": 350.0,
        "annual_inc": 68000, "dti": 14.0, "delinq_2yrs": 0, "fico_low": 705,
        "fico_high": 709, "inq_6mths": 1, "open_acc": 12, "pub_rec": 0,
        "revol_bal": 7000, "revol_util": 38.0, "total_acc": 22, "mort_acc": 1,
        "pub_rec_bk": 0,
        "desc": (
            "I have worked as a hospital nurse for 9 years with a stable annual salary "
            "of $68,000. I have no missed payments and am using this loan to consolidate "
            "credit card debt at a lower rate. My finances are well-managed and I am "
            "confident I can meet the monthly repayments without difficulty."
        ),
    },
    # Case 3: Grade D, protective text → Strategist proposes weight increase → NO VETO
    # Grade D delta=+0.018 ≥ 0 → Advocate passes; fusion lowers prediction (text helps)
    # Narrative: same trustworthy borrower, Grade D text historically improves AUC — approved
    "Case 3 · Grade D — Strategist passes cleanly": {
        "grade": "D", "loan_amnt": 14000, "int_rate": 17.5, "installment": 350.0,
        "annual_inc": 68000, "dti": 14.0, "delinq_2yrs": 0, "fico_low": 705,
        "fico_high": 709, "inq_6mths": 1, "open_acc": 12, "pub_rec": 0,
        "revol_bal": 7000, "revol_util": 38.0, "total_acc": 22, "mort_acc": 1,
        "pub_rec_bk": 0,
        "desc": (
            "I am a registered nurse with 9 years of continuous employment at the same "
            "hospital. My annual salary is $68,000 and I have never missed a single "
            "payment on any account. I am consolidating two low-balance credit cards "
            "to simplify my finances and reduce my interest rate."
        ),
    },
    # Case 4: Grade D, confidence too low — text analyst confidence < 0.65, text signal ignored
    "Case 4 · Grade D — Low confidence, text ignored": {
        "grade": "D", "loan_amnt": 14000, "int_rate": 17.5, "installment": 350.0,
        "annual_inc": 68000, "dti": 14.0, "delinq_2yrs": 0, "fico_low": 705,
        "fico_high": 709, "inq_6mths": 1, "open_acc": 12, "pub_rec": 0,
        "revol_bal": 7000, "revol_util": 38.0, "total_acc": 22, "mort_acc": 1,
        "pub_rec_bk": 0,
        "desc": "I need this loan.",
    },
}

def _extract_markdown_table(markdown: str, heading: str) -> pd.DataFrame | None:
    start = markdown.find(heading)
    if start < 0:
        return None
    lines = []
    in_table = False
    for line in markdown[start:].splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
            lines.append(stripped)
        elif in_table:
            break
    if len(lines) < 3:
        return None

    header = [part.strip() for part in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        values = [part.strip() for part in line.strip("|").split("|")]
        if len(values) == len(header):
            rows.append(values)
    return pd.DataFrame(rows, columns=header)

def render_offline_evaluation():
    st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-title">
  <div>
    <div class="eyebrow">Offline Evidence</div>
    <h2>Evaluation Snapshot</h2>
  </div>
  <span class="section-pill">held-out report</span>
</div>
        """,
        unsafe_allow_html=True,
    )
    if not REPORT_PATH.exists():
        st.info("No offline evaluation report found at `docs/eval_report_sample1.md`.")
        return

    report = REPORT_PATH.read_text(encoding="utf-8")
    generated = re.search(r"_generated_:\s*`([^`]+)`", report)
    if generated:
        st.caption(f"Source: `docs/eval_report_sample1.md` · generated `{generated.group(1)}`")
    else:
        st.caption("Source: `docs/eval_report_sample1.md`")

    pipeline_df = _extract_markdown_table(report, "## 2. Pipeline Head-to-Head")
    kpi_df = _extract_markdown_table(report, "## 3. Per-Agent KPI")
    ablation_df = _extract_markdown_table(report, "## 5. Ablation")
    trust_df = _extract_markdown_table(report, "### Confidence reliability bins")

    if pipeline_df is not None and not pipeline_df.empty:
        baseline = pipeline_df[pipeline_df["pipeline"] == "baseline"].iloc[0]
        strategist = pipeline_df[pipeline_df["pipeline"] == "strategist"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline AUC", baseline["auc"])
        c2.metric("Strategist AUC", strategist["auc"], delta=f"{float(strategist['auc']) - float(baseline['auc']):+.4f}")
        c3.metric("Strategist AUPRC", strategist["auprc"], delta=f"{float(strategist['auprc']) - float(baseline['auprc']):+.4f}")
        c4.metric("DeLong p", strategist["delong_p_vs_ref"])
        with st.expander("Pipeline head-to-head table"):
            st.dataframe(pipeline_df, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    veto = re.search(r"Advocate veto rate:\s*\*\*([^*]+)\*\*", report)
    slope = re.search(r"LLM-confidence-vs-accuracy slope:\s*\*\*([^*]+)\*\*", report)
    top_delta = re.search(r"Top-coverage ΔAUPRC .*?:\s*\*\*([^*]+)\*\*", report)
    c1.metric("Advocate veto rate", veto.group(1) if veto else "—")
    c2.metric("Confidence slope", slope.group(1) if slope else "—")
    c3.metric("Top-coverage ΔAUPRC", top_delta.group(1) if top_delta else "—")

    with st.expander("Agent KPI / ablation / trust calibration"):
        if kpi_df is not None:
            st.markdown("**Per-agent KPI**")
            st.dataframe(kpi_df, width="stretch", hide_index=True)
        if ablation_df is not None:
            st.markdown("**Ablation**")
            st.dataframe(ablation_df, width="stretch", hide_index=True)
        if trust_df is not None:
            st.markdown("**Confidence reliability bins**")
            st.dataframe(trust_df, width="stretch", hide_index=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary: #4E2A84;
    --primary-dark: #35185f;
    --primary-soft: #F3EEF9;
    --primary-soft-2: #E9DDF6;
    --ink: #1F1B2D;
    --muted: #6B6478;
    --border: #E5E2EA;
    --surface: #FFFFFF;
    --surface-soft: #F8F7FB;
    --risk: #B42318;
    --risk-soft: #FEEDEB;
    --safe: #15803D;
    --safe-soft: #ECFDF3;
    --warn: #B45309;
    --warn-soft: #FFF7E6;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2.5rem;
    max-width: 1320px;
}

div[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, #F7F4FB 0%, #FFFFFF 240px),
        #FFFFFF;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--ink);
    letter-spacing: 0;
}

.app-hero {
    background:
        linear-gradient(135deg, rgba(78,42,132,0.98), rgba(53,24,95,0.98));
    color: white;
    border-radius: 14px;
    padding: 22px 26px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 18px 45px rgba(78,42,132,0.18);
    margin-bottom: 18px;
}
.app-hero, .app-hero * {
    color: #FFFFFF !important;
}
.app-hero-row {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-end;
    flex-wrap: wrap;
}
.app-title {
    font-size: 2rem;
    line-height: 1.08;
    font-weight: 780;
    margin: 0;
    color: #FFFFFF !important;
}
.app-subtitle {
    color: rgba(255,255,255,0.84) !important;
    margin-top: 8px;
    font-size: 0.98rem;
}
.hero-kpis {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.hero-kpi {
    border: 1px solid rgba(255,255,255,0.24);
    background: rgba(255,255,255,0.10);
    color: #FFFFFF !important;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 0.78rem;
    font-weight: 650;
    white-space: nowrap;
}

.top-settings {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: -6px 2px 12px;
    color: var(--muted);
    font-size: 0.76rem;
}
.top-settings code {
    color: var(--primary);
    background: var(--primary-soft);
    border-radius: 5px;
    padding: 1px 5px;
}
.top-settings-status {
    color: var(--primary);
    font-weight: 700;
}

.section-title {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
    margin: 10px 0 12px;
}
.section-title h2 {
    margin: 0;
    font-size: 1.28rem;
    font-weight: 750;
}
.eyebrow {
    color: var(--primary);
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.section-pill {
    background: var(--primary-soft);
    color: var(--primary);
    border: 1px solid var(--primary-soft-2);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.75rem;
    font-weight: 680;
}
.section-break {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0 18px;
}

.preset-band {
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px 12px;
    margin-bottom: 18px;
}
.preset-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.preset-title strong {
    color: var(--ink);
    font-size: 0.96rem;
}
.preset-title span {
    color: var(--muted);
    font-size: 0.78rem;
}

div[data-testid="stHorizontalBlock"] button {
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 680;
    border: 1px solid var(--border);
    background: white;
    color: var(--ink);
    min-height: 2.35rem;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: var(--primary);
    color: var(--primary);
    background: var(--primary-soft);
}
button[kind="primary"], div.stButton > button[kind="primary"] {
    background: var(--primary);
    border-color: var(--primary);
    border-radius: 8px;
    font-weight: 720;
    color: #FFFFFF !important;
}
button[kind="primary"]:hover, div.stButton > button[kind="primary"]:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
    color: #FFFFFF !important;
}
button[kind="primary"] *, div.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
}
div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stExpander"] details {
    background: rgba(255,255,255,0.86);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 10px;
    box-shadow: 0 8px 22px rgba(31,27,45,0.04);
}
div[data-testid="stExpander"] summary {
    color: var(--ink);
    font-weight: 720;
}
div[data-testid="stExpander"] details:hover {
    border-color: #D6CBE5;
    border-left-color: var(--primary);
}
div[data-testid="stExpander"][data-agent-color="green"] details,
div[data-agent-color="green"][data-testid="stExpander"] details,
details[data-agent-color="green"] {
    background: #F7FFF9 !important;
    border-color: #B7E4C7 !important;
    border-left-color: var(--safe) !important;
}
div[data-testid="stExpander"][data-agent-color="green"] summary,
div[data-agent-color="green"][data-testid="stExpander"] summary,
details[data-agent-color="green"] summary {
    background: #F0FDF4 !important;
    color: #166534 !important;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stExpander"][data-agent-color="green"] summary *,
div[data-agent-color="green"][data-testid="stExpander"] summary *,
details[data-agent-color="green"] summary * {
    color: #166534 !important;
}
div[data-testid="stExpander"][data-agent-color="green"] details:hover,
div[data-agent-color="green"][data-testid="stExpander"] details:hover,
details[data-agent-color="green"]:hover {
    border-color: #86D39B !important;
    border-left-color: var(--safe) !important;
}
.green-expander-marker {
    height: 0;
    margin: 0;
    padding: 0;
    overflow: hidden;
}
.stMarkdown:has(.green-expander-marker),
div[data-testid="stMarkdown"]:has(.green-expander-marker),
div[data-testid="stMarkdownContainer"]:has(.green-expander-marker),
div.element-container:has(.green-expander-marker) {
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.green-expander-marker + div[data-testid="stExpander"] details {
    background: #F7FFF9;
    border-color: #B7E4C7;
    border-left-color: var(--safe);
}
.green-expander-marker + div[data-testid="stExpander"] summary {
    color: #166534;
}
.stMarkdown:has(.green-expander-marker) + div[data-testid="stExpander"] details,
div[data-testid="stMarkdown"]:has(.green-expander-marker) + div[data-testid="stExpander"] details,
div:has(> div[data-testid="stMarkdownContainer"] .green-expander-marker) + div[data-testid="stExpander"] details,
div.element-container:has(.green-expander-marker) + div.element-container div[data-testid="stExpander"] details {
    background: #F7FFF9;
    border-color: #B7E4C7;
    border-left-color: var(--safe);
}
.stMarkdown:has(.green-expander-marker) + div[data-testid="stExpander"] summary,
div[data-testid="stMarkdown"]:has(.green-expander-marker) + div[data-testid="stExpander"] summary,
div:has(> div[data-testid="stMarkdownContainer"] .green-expander-marker) + div[data-testid="stExpander"] summary,
div.element-container:has(.green-expander-marker) + div.element-container div[data-testid="stExpander"] summary {
    background: #F0FDF4;
    color: #166534;
    border-radius: 8px 8px 0 0;
}
div.element-container:has(.green-expander-marker) + div.element-container div[data-testid="stExpander"] details:hover {
    border-color: #86D39B;
    border-left-color: var(--safe);
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
textarea {
    border-radius: 8px !important;
}

/* pipeline status chips */
.pipe-chip {
    display: inline-block; padding: 4px 12px;
    border-radius: 16px; font-size: 0.78rem; font-weight: 600;
    margin: 2px 3px;
}
.chip-ok     { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.chip-warn   { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.chip-alert  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.chip-idle   { background:#f3f1f7; color:#6B6478; border:1px solid #DED8E8; }
.chip-arrow  { color:#8D829D; font-size:0.9rem; margin:0 1px; }

/* verdict box */
.verdict-default {
    background: linear-gradient(135deg, #FFF7F6, #FEEDEB);
    border: 1px solid #F4B8B2;
    border-left: 7px solid var(--risk);
    border-radius: 12px;
    padding: 18px 24px;
    text-align: left;
    box-shadow: 0 12px 28px rgba(180,35,24,0.08);
}
.verdict-safe {
    background: linear-gradient(135deg, #F5FFF8, #ECFDF3);
    border: 1px solid #B7E4C7;
    border-left: 7px solid var(--safe);
    border-radius: 12px;
    padding: 18px 24px;
    text-align: left;
    box-shadow: 0 12px 28px rgba(21,128,61,0.08);
}
.verdict-title {
    font-size: 1.65rem;
    font-weight: 820;
    margin: 0;
    color: var(--ink);
}
.verdict-sub {
    font-size: 0.92rem;
    margin-top: 5px;
    color: var(--muted);
}

.hint-panel {
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 10px;
    padding: 14px 16px;
    background: var(--surface-soft);
    color: var(--ink);
}
.hint-panel p {
    margin: 0 0 8px;
    font-weight: 720;
}
.hint-panel ol {
    margin: 0;
    padding-left: 1.2rem;
    color: var(--muted);
}

.evidence-map {
    margin-top: 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: #fff;
    padding: 10px 12px;
}
.evidence-map-title {
    color: var(--primary);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 780;
    margin-bottom: 6px;
}
.evidence-map-body {
    color: var(--ink);
    font-size: 0.86rem;
    line-height: 1.55;
    max-height: 210px;
    overflow: auto;
    white-space: pre-wrap;
}
.evidence-hit {
    color: #B42318;
    font-weight: 820;
    background: #FEEDEB;
    border-radius: 4px;
    padding: 0 2px;
}

.agent-brief {
    border-left: 3px solid var(--primary);
    background: var(--primary-soft);
    border-radius: 8px;
    padding: 9px 11px;
    margin: 8px 0;
    color: var(--ink);
    font-size: 0.9rem;
    line-height: 1.45;
}
.key-row [data-testid="stTextInput"] {
    margin-bottom: 0;
}
.key-ok {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 8px;
    width: 28px;
    height: 28px;
    border-radius: 999px;
    color: #FFFFFF;
    background: var(--safe);
    font-weight: 900;
}

@media (max-width: 760px) {
    .app-title { font-size: 1.45rem; }
    .app-hero { padding: 18px; }
    .hero-kpis { justify-content: flex-start; }
}
</style>
""", unsafe_allow_html=True)

def tag_green_expanders():
    components.html(
        """
<script>
const tagGreenExpanders = () => {
  const doc = window.parent.document;
  doc.querySelectorAll('div[data-testid="stExpander"]').forEach((expander) => {
    const summary = expander.querySelector('summary');
    const label = summary ? summary.textContent || '' : '';
    if (label.includes('Green Agent')) {
      expander.setAttribute('data-agent-color', 'green');
      const details = expander.querySelector('details');
      if (details) details.setAttribute('data-agent-color', 'green');
    }
  });
};
tagGreenExpanders();
setTimeout(tagGreenExpanders, 250);
setTimeout(tagGreenExpanders, 900);
const observer = new MutationObserver(tagGreenExpanders);
observer.observe(window.parent.document.body, {childList: true, subtree: true});
</script>
        """,
        height=0,
    )

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-hero">
  <div class="app-hero-row">
    <div>
      <h1 class="app-title">Loan Default Prediction</h1>
      <div class="app-subtitle">Multi-Agent Risk Assessment · Numeric baseline with audited text fusion</div>
    </div>
    <div class="hero-kpis">
      <span class="hero-kpi">XGBoost baseline</span>
      <span class="hero-kpi">5-scalar fusion</span>
      <span class="hero-kpi">threshold 0.50</span>
      <span class="hero-kpi">agent audit trail</span>
    </div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

key_status = "environment key loaded" if MINIMAX_API_KEY else (
    "session key loaded" if st.session_state.get("minimax_api_key") else "key required for LLM agents"
)
st.markdown(
    f"""
<div class="top-settings">
  <span>LLM model <code>{MINIMAX_MODEL}</code></span>
  <span class="top-settings-status">{key_status}</span>
</div>
    """,
    unsafe_allow_html=True,
)
if not MINIMAX_API_KEY:
    st.markdown('<div class="key-row">', unsafe_allow_html=True)
    key_col, ok_col = st.columns([0.96, 0.04], gap="small")
    with key_col:
        key_input = st.text_input(
            "MiniMax API key",
            type="password",
            label_visibility="collapsed",
            placeholder="MiniMax API key for this session",
            help="Used only in this Streamlit session if MINIMAX_API_KEY is not set in the environment.",
        )
    if key_input:
        st.session_state["minimax_api_key"] = key_input
    with ok_col:
        if st.session_state.get("minimax_api_key"):
            st.markdown('<span class="key-ok">✓</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Preset scenario buttons ───────────────────────────────────────────────────
st.markdown(
    """
<div class="preset-band">
  <div class="preset-title">
    <strong>Demo cases</strong>
    <span>Load one scenario, then run the full agent analysis</span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)
preset_short_labels = [
    "C · Advocate veto",
    "E · Arbitration",
    "D · Clean pass",
    "D · Low confidence",
]
preset_cols = st.columns(len(PRESETS))
for i, (label, vals) in enumerate(PRESETS.items()):
    button_label = preset_short_labels[i] if i < len(preset_short_labels) else label
    if preset_cols[i].button(button_label, width="stretch", key=f"preset_{i}", help=label):
        for k, v in vals.items():
            st.session_state[f"field_{k}"] = v

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_input, col_output = st.columns([0.82, 1.68], gap="large")

def _sv(key, default):
    """Read from session_state (set by preset buttons) or fall back to default."""
    return st.session_state.get(f"field_{key}", default)

with col_input:
    st.markdown(
        """
<div class="section-title">
  <div>
    <div class="eyebrow">Application Intake</div>
    <h2>Borrower Profile</h2>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    grade = st.selectbox("Credit Grade", ["C", "D", "E", "F", "G"],
                         index=["C","D","E","F","G"].index(_sv("grade","D")),
                         help="LendingClub credit grade (C=moderate risk, G=highest risk)")

    c1, c2 = st.columns(2)
    with c1:
        loan_amnt   = st.number_input("Loan Amount ($)",      1000,   40000,  _sv("loan_amnt",   12000), step=500)
        int_rate    = st.number_input("Interest Rate (%)",     5.0,    30.0,   _sv("int_rate",    18.5),  step=0.5)
        installment = st.number_input("Monthly Payment ($)",   10.0,  2000.0,  _sv("installment", 380.0), step=10.0)
        annual_inc  = st.number_input("Annual Income ($)",    5000,  500000,   _sv("annual_inc",  55000), step=1000)
        dti         = st.number_input("Debt-to-Income (%)",    0.0,    50.0,   _sv("dti",         18.0),  step=0.5)
        delinq_2yrs = st.number_input("Delinquencies (2yr)",   0,      20,     _sv("delinq_2yrs", 0))
        fico_low    = st.number_input("FICO Score (low)",      580,    850,    _sv("fico_low",    680),   step=5)
        fico_high   = st.number_input("FICO Score (high)",     580,    855,    _sv("fico_high",   684),   step=5)
    with c2:
        inq_6mths   = st.number_input("Inquiries (6mo)",       0,      20,     _sv("inq_6mths",   1))
        open_acc    = st.number_input("Open Accounts",         0,      60,     _sv("open_acc",    10))
        pub_rec     = st.number_input("Public Records",        0,      20,     _sv("pub_rec",     0))
        revol_bal   = st.number_input("Revolving Balance ($)", 0,    200000,   _sv("revol_bal",   8000),  step=500)
        revol_util  = st.number_input("Revolving Util (%)",    0.0,   120.0,   _sv("revol_util",  55.0),  step=1.0)
        total_acc   = st.number_input("Total Accounts",        0,     100,     _sv("total_acc",   20))
        mort_acc    = st.number_input("Mortgage Accounts",     0,      20,     _sv("mort_acc",    1))
        pub_rec_bk  = st.number_input("Bankruptcies",          0,       5,     _sv("pub_rec_bk",  0))

    st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
    desc_text = st.text_area(
        "📝 Borrower Description",
        value=_sv("desc", ""),
        placeholder="e.g. I need this loan to consolidate my credit card debts. I have been employed as a nurse for 8 years and make $55,000 per year...",
        height=230,
    )
    evidence_map_slot = st.empty()

    run_btn = st.button("Run Multi-Agent Analysis", type="primary", width="stretch")

# ── Prediction logic ──────────────────────────────────────────────────────────
with col_output:
    st.markdown(
        """
<div class="section-title">
  <div>
    <div class="eyebrow">Decision Workspace</div>
    <h2>Agent Audit Trail</h2>
  </div>
  <span class="section-pill">live run</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not run_btn:
        st.info("👈 Select a demo case or fill in the form, then click **Run Multi-Agent Analysis**.")
        st.markdown(
            """
<div class="hint-panel">
  <p>How the system decides whether to trust the text</p>
  <ol>
    <li>Is the borrower description informative?</li>
    <li>Is the LLM confidence above the strategy threshold?</li>
    <li>Does grade-level history support text influence?</li>
    <li>Would text fusion flip the final verdict?</li>
    <li>Is the case borderline enough to need a final review?</li>
  </ol>
</div>
            """,
            unsafe_allow_html=True,
        )

    if run_btn:
        # placeholder — filled with pipeline status after all agents complete
        _status_bar = st.empty()

        row = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
            "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
            "fico_range_low": fico_low, "fico_range_high": fico_high,
            "inq_last_6mths": inq_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
            "mort_acc": mort_acc, "pub_rec_bankruptcies": pub_rec_bk,
        }
        X = np.array([[row[f] for f in NUM_FEATURES]], dtype=float)

        # ── Agent 0: Green — baseline ─────────────────────────────────────────
        st.markdown('<div class="green-expander-marker"></div>', unsafe_allow_html=True)
        with st.expander("🟢 Green Agent — XGBoost Baseline", expanded=True):
            baseline_pred = float(model.predict_proba(X)[0, 1])
            risk_level = ("🔴 High" if baseline_pred > 0.65 else
                          "🟡 Medium" if baseline_pred > 0.40 else "🟢 Low")
            st.metric("Baseline Default Probability", f"{baseline_pred:.1%}", delta=None)
            st.caption(f"Risk level: {risk_level}")
            st.progress(min(baseline_pred, 1.0))
            st.caption("Source: XGBoost trained on 98k records, 16 numeric features. "
                       "Owns evaluation — White Agents cannot override.")

        # ── Agent 1: Text Analyst ─────────────────────────────────────────────
        text_result = None
        normalized_text = None
        text_expander = st.expander("📝 White Agent 1 — Text Analyst (LLM)", expanded=True)

        if not desc_text.strip():
            with text_expander:
                st.info("No description provided — text analysis skipped. "
                        "Fusion will rely on numeric model only.")
            fused_pred     = baseline_pred
            text_reasoning = "N/A"
            text_score_raw = None
        else:
            with text_expander:
                with st.spinner("Calling LLM to score borrower description..."):
                    t0 = time.time()
                    try:
                        text_result = text_analyst(desc_text)
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
                        text_result = {"text_risk_score": 0.45, "confidence": 0.5,
                                       "risk_signals": [], "protective_signals": [],
                                       "reasoning": f"[ERROR: {e}]",
                                       "_raw": f"EXCEPTION: {e}"}
                    elapsed = time.time() - t0

                raw_score  = text_result["text_risk_score"]
                confidence = text_result["confidence"]
                text_score_raw = raw_score

                with st.expander("🔍 Debug: raw LLM response", expanded=False):
                    st.code(text_result.get("_raw", "(not captured)"), language="json")

                c_left, c_right = st.columns(2)
                with c_left:
                    st.metric("Text Risk Score", f"{raw_score:.3f}")
                    st.markdown(f"Risk level: **{text_risk_label(raw_score)}**")
                    st.progress(raw_score)
                with c_right:
                    st.metric("LLM Confidence", f"{confidence:.3f}")
                    raw_conf = text_result.get("_raw_confidence")
                    conf_cap = text_result.get("_confidence_cap")
                    if raw_conf is not None and conf_cap is not None and confidence < raw_conf:
                        st.caption(
                            f"Calibrated from raw {raw_conf:.3f}; cap {conf_cap:.3f} "
                            f"({text_result.get('_confidence_cap_reason', 'text evidence quality')})."
                        )
                    st.caption(f"⏱ {elapsed:.1f}s")

                if text_result.get("risk_signals"):
                    st.markdown("**⚠️ Risk signals:** " +
                                " · ".join(f'"{s}"' for s in text_result["risk_signals"]))
                if text_result.get("protective_signals"):
                    st.markdown("**✅ Protective signals:** " +
                                " · ".join(f'"{s}"' for s in text_result["protective_signals"]))
                all_signals = list(text_result.get("risk_signals", [])) + list(text_result.get("protective_signals", []))
                render_highlighted_description(evidence_map_slot, desc_text, all_signals)
                st.markdown(
                    f'<div class="agent-brief"><strong>Brief:</strong> {html.escape(text_result.get("display_brief") or concise_text(text_result.get("reasoning", ""), 220))}</div>',
                    unsafe_allow_html=True,
                )

            text_reasoning = text_result.get("reasoning", "")
            text_score_for_fusion = raw_score

            # ── Agent 2: Feature Strategist (LLM) ────────────────────────────
            conf_threshold = BEST_STRATEGY["conf_threshold"]
            conf_ok        = confidence >= conf_threshold
            learned_w      = BEST_STRATEGY["grade_weights"].get(grade, 0.0)
            learned_w_src  = "5-scalar BEST_STRATEGY"
            train_stats    = GRADE_TRAIN_STATS.get(grade, {})

            # Trigger LLM Strategist only when it can plausibly add value beyond
            # the static grade policy. The Strategist's distinctive ability is reading
            # the borrower's prose, so it fires when there is text-model disagreement
            # or when the static weight is mismatched with extreme text content.
            score_gap       = abs(raw_score - baseline_pred)
            baseline_fuzzy  = 0.30 <= baseline_pred <= 0.65
            strategist_triggers = []
            if score_gap > 0.20:
                strategist_triggers.append("text-model disagreement")
            if baseline_fuzzy:
                strategist_triggers.append("fuzzy-zone review")
            if raw_score > 0.65 and learned_w == 0.0:
                strategist_triggers.append("extreme text risk with suppressed grade weight")
            if raw_score < 0.25 and learned_w > 0.10:
                strategist_triggers.append("protective text under a high learned grade weight")
            call_strategist = conf_ok and bool(strategist_triggers)

            strat_result = None
            strat_raw    = None
            with st.expander("⚙️ White Agent 2 — Feature Strategist (LLM)", expanded=True):
                st.markdown("**Learned baseline weights** (from iterative training loop, logit fusion):")
                cols = st.columns(5)
                for i, g in enumerate(["C", "D", "E", "F", "G"]):
                    with cols[i]:
                        lw = BEST_STRATEGY["grade_weights"].get(g, 0.0)
                        st.metric(f"Grade {g}", f"{lw:.2f}",
                                  delta="← this grade" if g == grade else None,
                                  delta_color="normal" if g == grade else "off")

                st.caption(f"Baseline weight source: **{learned_w_src}**.")

                st.markdown(f"**Confidence:** {confidence:.3f} {'✅' if conf_ok else '❌ below threshold → skip'}")
                st.markdown(f"**Score gap** (text − baseline): `{raw_score:.3f} − {baseline_pred:.3f} = {raw_score-baseline_pred:+.3f}`")
                st.markdown(f"**Baseline in fuzzy zone:** {'Yes' if baseline_fuzzy else 'No'}")

                if not call_strategist:
                    strat_result = {
                        "strategy_type": "keep_learned",
                        "proposed_weight": learned_w,
                        "constraints": {},
                        "rationale": "Text and numeric signals are consistent — learned strategy applies.",
                        "display_brief": "Text and numeric signals are consistent, so the learned grade-level weight is used.",
                    }
                    st.success(f"✅ No strategist review needed. Strategy: **keep_learned** (w = {learned_w:.2f})")
                else:
                    trigger_text = " · ".join(strategist_triggers) if strategist_triggers else "case-specific review"
                    st.warning(f"⚡ Strategist review triggered — {trigger_text}. Calling LLM Strategist...")
                    with st.spinner("Strategist reasoning about case context..."):
                        t_s = time.time()
                        strat_result, strat_raw = run_strategist_llm(
                            grade, baseline_pred, raw_score, confidence,
                            learned_w, train_stats, desc_text,
                        )
                        elapsed_s = time.time() - t_s

                    stype = strat_result.get("strategy_type", "keep_learned")
                    pw    = strat_result.get("proposed_weight", learned_w)
                    cst   = strat_result.get("constraints", {})
                    rat   = strat_result.get("rationale", "")

                    type_color = {"keep_learned": "✅", "case_specific_exception": "⚠️",
                                  "fuzzy_only": "🔵", "explanation_only": "📝",
                                  "human_review": "🚨"}.get(stype, "❓")

                    st.markdown(f"**Strategy type:** {type_color} `{stype}`  ⏱ {elapsed_s:.1f}s")
                    st.markdown(f"**Proposed weight:** `{learned_w:.2f}` → `{pw:.2f}`")
                    if cst:
                        cst_parts = []
                        if cst.get("max_prediction_shift"):
                            cst_parts.append(f"max_shift={cst['max_prediction_shift']}")
                        if cst.get("min_text_confidence"):
                            cst_parts.append(f"min_conf={cst['min_text_confidence']}")
                        if cst.get("only_if_fuzzy_zone"):
                            cst_parts.append("only_if_fuzzy_zone=true")
                        if cst_parts:
                            st.markdown(f"**Constraints:** `{' | '.join(cst_parts)}`")
                    st.markdown(
                        f'<div class="agent-brief"><strong>Rationale brief:</strong> {html.escape(strat_result.get("display_brief") or concise_text(rat, 240))}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔍 Raw LLM output (Strategist)", expanded=False):
                        st.code(strat_raw or "(none)", language="json")

            # Extract from strategy result for downstream use
            if strat_result is None:
                strat_result = {"strategy_type": "keep_learned", "proposed_weight": learned_w, "constraints": {}, "rationale": "", "display_brief": "Strategist kept the learned grade-level weight."}
            stype      = strat_result.get("strategy_type", "keep_learned")
            proposed_w = float(strat_result.get("proposed_weight", learned_w))
            cst        = strat_result.get("constraints", {})

            # Strategy types that propose a weight change go to Advocate
            strategist_proposes_change = (stype in ("case_specific_exception", "fuzzy_only") and
                                          proposed_w > learned_w + 0.001)

            # ── Agent 3: Subgroup Advocate ─────────────────────────────────────
            hist_delta     = train_stats.get("delta_at_high_weight", 0.0)
            history_summary = historical_text_impact(hist_delta, grade)
            extreme_signal = abs(raw_score - 0.5) > 0.35 and confidence > 0.85
            severe_harm    = hist_delta < -0.04
            exemption_applies = False
            adv_result = {
                "decision": "pass",
                "severity": "low",
                "max_allowed_weight": proposed_w,
                "max_prediction_shift": None,
                "rationale": "Strategist did not ask for extra text influence, so Advocate has no subgroup dispute to resolve.",
                "constraints": [],
            }
            adv_raw = None

            if strategist_proposes_change and severe_harm:
                adv_result = {
                    "decision": "hard_veto",
                    "severity": "high",
                    "max_allowed_weight": learned_w,
                    "max_prediction_shift": 0.0,
                    "rationale": (
                        f"For Grade {grade}, {history_summary}, so increased text influence is blocked."
                    ),
                    "constraints": ["severe subgroup harm hard floor", "no increased text weight"],
                }
            elif strategist_proposes_change:
                with st.spinner("Advocate reasoning about subgroup reliability..."):
                    adv_result, adv_raw = run_advocate_llm(
                        grade, baseline_pred, raw_score, confidence,
                        learned_w, proposed_w, hist_delta,
                        train_stats.get("reason", "unknown"),
                        stype, strat_result.get("rationale", ""), cst,
                        desc_text
                    )

            adv_decision = adv_result.get("decision", "pass")
            veto = strategist_proposes_change and adv_decision in ("soft_veto", "hard_veto")
            is_hard_veto = veto and adv_decision == "hard_veto"
            is_soft_veto = veto and adv_decision == "soft_veto"

            adv_label = ("🛡️ White Agent 3 — Subgroup Advocate (LLM)  🚨 HARD VETO" if is_hard_veto else
                         "🛡️ White Agent 3 — Subgroup Advocate (LLM)  ⚠️ SOFT VETO" if is_soft_veto else
                         "🛡️ White Agent 3 — Subgroup Advocate (LLM)  ✅ pass")
            with st.expander(adv_label, expanded=veto):
                st.caption("Hybrid intelligent advocate: LLM reasoning for subgroup reliability, with deterministic hard floor for severe historical harm.")
                if veto:
                    veto_type = "**HARD VETO**" if is_hard_veto else "**SOFT VETO**"
                    st.error(f"{veto_type}: {adv_result.get('display_brief') or concise_text(adv_result.get('rationale', ''), 240)}")
                    c1a, c2a, c3a, c4a = st.columns(4)
                    with c1a: st.metric("Learned weight", f"{learned_w:.2f}")
                    with c2a: st.metric("Proposed weight", f"{proposed_w:.2f}")
                    with c3a: st.metric("Historical read", "caution" if hist_delta < 0 else "support")
                    with c4a: st.metric("Max allowed", f"{adv_result.get('max_allowed_weight', learned_w):.2f}")
                    st.markdown(f"**Historical read:** {history_summary}.")
                    if adv_result.get("constraints"):
                        st.markdown("**Constraints:** " + " · ".join(str(x) for x in adv_result["constraints"]))
                    if adv_result.get("max_prediction_shift") not in (None, 0, 0.0):
                        st.markdown(f"**Max prediction shift:** `{adv_result['max_prediction_shift']:.3f}`")
                    if is_hard_veto:
                        st.markdown("**Hard veto — the reliability constraint is binding, so no compromise weight is allowed.**")
                    else:
                        st.markdown("**Soft veto — Green will arbitrate a narrower, case-specific weight.**")
                    st.markdown(f"→ Forwarding to **Green for arbitration**.")
                    if adv_raw:
                        with st.expander("🔍 Raw LLM output (Advocate)", expanded=False):
                            st.code(adv_raw, language="json")
                else:
                    if strategist_proposes_change:
                        st.success(f"Advocate passed: {adv_result.get('display_brief') or concise_text(adv_result.get('rationale', ''), 240)}")
                    else:
                        st.success("Advocate found no subgroup dispute to resolve.")
                    if hist_delta < 0 and not strategist_proposes_change:
                        st.markdown(
                            f"Grade {grade}: {history_summary}; "
                            f"but Strategist did not propose increasing the text weight, so no veto is needed."
                        )
                    elif hist_delta < 0:
                        st.markdown(
                            f"Grade {grade}: {history_summary}. Advocate allowed the proposal because the case-level constraints keep text influence contained."
                        )
                    else:
                        st.markdown(
                            f"Grade {grade}: {history_summary}; Advocate found no subgroup reason to block the proposal."
                        )
                    if adv_raw:
                        with st.expander("🔍 Raw LLM output (Advocate)", expanded=False):
                            st.code(adv_raw, language="json")

            # ── Green Agent — Arbitration (on-veto) ──────────────────────────────
            final_w    = learned_w
            arb_source = "learned"
            arb_label  = ("⚖️ Green Agent — Arbitration (on-veto)  🔔 called" if veto
                          else "⚖️ Green Agent — Arbitration (on-veto)  — idle")
            st.markdown('<div class="green-expander-marker"></div>', unsafe_allow_html=True)
            with st.expander(arb_label, expanded=veto):
                if veto:
                    st.caption("Called because Advocate vetoed. Classifies veto type, then mediates or enforces accordingly.")
                    if is_hard_veto:
                        # Hard veto: skip LLM compromise, enforce directly
                        final_w    = learned_w
                        arb_source = "hard-veto enforced"
                        st.error(
                            f"**Hard veto enforced.** For Grade {grade}, "
                            f"{history_summary}; Advocate judged extra text influence too risky here. "
                            f"{'The harm is severe enough that the LLM Advocate cannot override the hard floor. ' if severe_harm else 'The LLM Advocate rejected increased text influence for this case. '}"
                            f"The final weight stays at the learned policy level."
                        )
                        st.markdown(f"**Final weight for Grade {grade}: `{final_w:.2f}`** (no LLM call needed — constraint is absolute)")
                    else:
                        # Soft veto: call LLM for compromise
                        arb_threshold = 0.5
                        arb_text_evidence = logit(text_score_for_fusion) - logit(0.5)
                        arb_pred_learned = sigmoid(
                            logit(baseline_pred) + learned_w * confidence * arb_text_evidence
                        )
                        arb_pred_proposed = sigmoid(
                            logit(baseline_pred) + proposed_w * confidence * arb_text_evidence
                        )
                        arb_hi = min(proposed_w, float(adv_result.get("max_allowed_weight", proposed_w)))
                        arb_lo = learned_w
                        arb_flip_learned = (arb_pred_learned >= arb_threshold) != (baseline_pred >= arb_threshold)
                        arb_flip_proposed = (arb_pred_proposed >= arb_threshold) != (baseline_pred >= arb_threshold)
                        arb_constraints = []
                        if cst.get("max_prediction_shift"):
                            arb_constraints.append(f"max_prediction_shift={cst['max_prediction_shift']}")
                        if cst.get("min_text_confidence"):
                            arb_constraints.append(f"min_text_confidence={cst['min_text_confidence']}")
                        if cst.get("only_if_fuzzy_zone"):
                            arb_constraints.append("only_if_fuzzy_zone=true")
                        arb_constraint_text = ", ".join(arb_constraints) if arb_constraints else "none"
                        arb_risk_signals = text_result.get("risk_signals", []) if text_result else []
                        arb_protective_signals = text_result.get("protective_signals", []) if text_result else []
                        arb_numeric_profile = (
                            f"loan_amount=${row.get('loan_amnt', 0):,.0f}; "
                            f"interest_rate={row.get('int_rate', 0):.1f}%; "
                            f"installment=${row.get('installment', 0):,.0f}/mo; "
                            f"annual_income=${row.get('annual_inc', 0):,.0f}; "
                            f"DTI={row.get('dti', 0):.1f}%; "
                            f"FICO={row.get('fico_range_low', 0):.0f}-{row.get('fico_range_high', 0):.0f}; "
                            f"delinq_2yrs={row.get('delinq_2yrs', 0)}; "
                            f"inquiries_6mo={row.get('inq_last_6mths', 0)}; "
                            f"open_acc={row.get('open_acc', 0)}; "
                            f"revol_bal=${row.get('revol_bal', 0):,.0f}; "
                            f"revol_util={row.get('revol_util', 0):.1f}%; "
                            f"total_acc={row.get('total_acc', 0)}; "
                            f"mort_acc={row.get('mort_acc', 0)}; "
                            f"bankruptcies={row.get('pub_rec_bankruptcies', 0)}"
                        )
                        arb_text_reasoning = text_result.get("reasoning", "") if text_result else ""
                        arb_desc = re.sub(r"\s+", " ", desc_text.strip())[:900]
                        arb_strat_raw = re.sub(r"\s+", " ", str(strat_raw or ""))[:700]
                        arb_adv_raw = re.sub(r"\s+", " ", str(adv_raw or ""))[:700]
                        exemption_note = (
                            f"\nIMPORTANT: This soft veto reflects the Advocate's historical read for this grade: {history_summary}. "
                            f"but was downgraded because text score={raw_score:.2f} and confidence={confidence:.2f} "
                            f"meet the high-confidence extreme-signal exemption. "
                            f"The signal is unusually strong — you may allow a small weight (e.g. 0.05–0.10) "
                            f"to let this extreme signal partially influence the prediction."
                            if exemption_applies else ""
                        )
                        arb_prompt = (
                            f"You are the Green Agent acting as arbitrator in a loan default prediction system.\n"
                            f"Your role is to make a reasoned裁决, not to average the two sides.\n\n"
                            f"=== Dispute ===\n"
                            f"Strategist proposal: Grade {grade} text weight {learned_w:.3f} → {proposed_w:.3f}\n"
                            f"Strategist strategy: {stype}\n"
                            f"Strategist rationale: {rat}\n"
                            f"Strategist constraints: {arb_constraint_text}\n"
                            f"Advocate soft veto: {adv_result.get('rationale', '')}\n"
                            f"Advocate max allowed weight: {adv_result.get('max_allowed_weight', proposed_w):.3f}\n"
                            f"Advocate max prediction shift: {adv_result.get('max_prediction_shift')}\n"
                            f"Training-history reason: {train_stats.get('reason','unknown')}\n"
                            f"{exemption_note}\n\n"
                            f"=== Current case evidence ===\n"
                            f"Baseline numeric prediction: {baseline_pred:.3f} (decision threshold={arb_threshold})\n"
                            f"Borrower numeric profile: {arb_numeric_profile}\n"
                            f"Text risk score: {raw_score:.3f}; confidence: {confidence:.3f}\n"
                            f"Score gap (text - baseline): {raw_score - baseline_pred:+.3f}\n"
                            f"Baseline in fuzzy zone [0.30, 0.65]: {'YES' if baseline_fuzzy else 'NO'}\n"
                            f"Near decision threshold (+/-0.05): {'YES' if abs(baseline_pred - arb_threshold) < 0.05 else 'NO'}\n"
                            f"Risk signals: {arb_risk_signals[:3]}\n"
                            f"Protective signals: {arb_protective_signals[:3]}\n"
                            f"Text Analyst reasoning: {arb_text_reasoning}\n"
                            f"Borrower description: {arb_desc}\n\n"
                            f"=== Prior agent outputs to audit ===\n"
                            f"Strategist raw output excerpt: {arb_strat_raw}\n"
                            f"Advocate raw output excerpt: {arb_adv_raw}\n\n"
                            f"=== Consequences of candidate weights ===\n"
                            f"At learned weight {learned_w:.3f}: pred={arb_pred_learned:.3f}, "
                            f"shift={arb_pred_learned - baseline_pred:+.3f}, "
                            f"verdict_flip={'YES' if arb_flip_learned else 'NO'}\n"
                            f"At Strategist weight {proposed_w:.3f}: pred={arb_pred_proposed:.3f}, "
                            f"shift={arb_pred_proposed - baseline_pred:+.3f}, "
                            f"verdict_flip={'YES' if arb_flip_proposed else 'NO'}\n\n"
                            f"Decision task:\n"
                            f"- Make a reasoned case-level ruling, not a midpoint calculation.\n"
                            f"- First decide which side has the stronger argument for this borrower: "
                            f"strategist, advocate, or balanced.\n"
                            f"- If you side with Strategist, move close to the proposed weight.\n"
                            f"- If you side with Advocate, stay close to the learned weight, but explain why the soft veto is almost binding.\n"
                            f"- If both sides are partly right, choose a narrow compromise and explain what each side contributed.\n"
                            f"- Use the borrower text, confidence, score gap, threshold distance, historical read, and candidate-weight consequences.\n"
                            f"- Your evidence sentences must cite at least two concrete numeric features and at least two specific text details or phrases.\n"
                            f"- Do not write generic evidence such as 'confidence is high' unless you tie it to concrete borrower facts.\n"
                            f"- final_weight must stay within the safety range [{arb_lo:.3f}, {arb_hi:.3f}], inclusive.\n\n"
                            f"Style: write like a careful credit-risk adjudicator. Be specific to this case; do not sound like a generic compromise template.\n\n"
                            f"CRITICAL OUTPUT FORMAT: respond with ONLY the JSON object. No reasoning, no markdown code fences, no commentary. Your response must start with {{ and end with }}. Any text outside the JSON braces will break the parser.\n\n" + f'Respond with only JSON: {{"ruling": "<strategist|advocate|balanced>", '
                            f'"final_weight": <float>, '
                            f'"strategist_evidence": "<2 detailed sentences using numeric and text-specific facts>", '
                            f'"advocate_evidence": "<2 detailed sentences using numeric and text-specific facts>", '
                            f'"decision_basis": "<3-5 concrete factors used for the ruling>", '
                            f'"resolution": "<2 sentences explaining why this exact weight is justified for this borrower>"}}'
                        )
                        with st.spinner("Green arbitrating soft-veto compromise..."):
                            t0 = time.time()
                            try:
                                arb_raw     = call_llm(arb_prompt, max_tokens=1800, temperature=0.25, response_format_json=True)
                                arb_result  = _extract_json_object(arb_raw)
                                arb_w       = round(min(max(float(arb_result.get("final_weight", arb_lo)), arb_lo), arb_hi), 3)
                                arb_res     = arb_result.get("resolution", "")
                                arb_basis   = arb_result.get("decision_basis", "")
                                arb_ruling  = arb_result.get("ruling", "balanced")
                                arb_strat_ev = arb_result.get("strategist_evidence", "")
                                arb_adv_ev   = arb_result.get("advocate_evidence", "")
                            except Exception as e:
                                arb_w   = arb_lo
                                arb_res = f"[error: {e}] — using the lower soft-veto compromise bound"
                                arb_basis = ""
                                arb_ruling = "advocate"
                                arb_strat_ev = ""
                                arb_adv_ev = ""
                            elapsed_arb = time.time() - t0
                        final_w    = arb_w
                        arb_source = "soft-veto compromise"
                        st.markdown(f"**Ruling:** `{arb_ruling}`")
                        if arb_strat_ev:
                            st.markdown(f"**Strategist evidence:** {arb_strat_ev}")
                        if arb_adv_ev:
                            st.markdown(f"**Advocate evidence:** {arb_adv_ev}")
                        if arb_basis:
                            st.markdown(f"**Decision basis:** {arb_basis}")
                        st.markdown(f"**LLM resolution:** {arb_res}  ⏱ {elapsed_arb:.1f}s")
                        st.success(f"**Final weight for Grade {grade}: `{final_w:.3f}`** (soft compromise)")
                else:
                    st.caption("Called only when Advocate identifies a subgroup-reliability conflict.")
                    if strategist_proposes_change:
                        final_w    = min(proposed_w, float(adv_result.get("max_allowed_weight", proposed_w)))
                        arb_source = "strategist accepted"
                        st.success(
                            f"Advocate cleared the subgroup check; adopting Strategist proposed weight for Grade {grade}: "
                            f"`{learned_w:.2f}` → `{final_w:.3f}`."
                        )
                    else:
                        st.markdown("**Status:** Idle.")

            # Final effective weight — apply Strategist constraints
            effective_w = final_w if conf_ok else 0.0
            constraint_note = ""
            if stype in ("explanation_only", "human_review"):
                effective_w    = 0.0
                constraint_note = f"Strategy `{stype}` — text is evidence only, weight forced to 0."
            elif effective_w > 0 and cst:
                if cst.get("only_if_fuzzy_zone") and not baseline_fuzzy:
                    effective_w    = 0.0
                    constraint_note = "Strategist constraint `only_if_fuzzy_zone=true` — baseline not in fuzzy zone, weight suppressed."
                if effective_w > 0:
                    min_conf = cst.get("min_text_confidence")
                    if min_conf and confidence < float(min_conf):
                        effective_w    = 0.0
                        constraint_note = f"Strategist constraint `min_text_confidence={min_conf}` — LLM confidence {confidence:.2f} too low."
                if effective_w > 0:
                    max_shift = cst.get("max_prediction_shift")
                    if max_shift:
                        shift = abs(text_score_for_fusion - baseline_pred)
                        if shift > 0:
                            capped_w    = float(max_shift) / shift
                            effective_w = min(effective_w, round(capped_w, 3))
                            if effective_w < final_w:
                                constraint_note = (f"Strategist `max_prediction_shift={max_shift}` — "
                                                   f"weight capped from {final_w:.3f} to {effective_w:.3f}.")
                if effective_w > 0:
                    adv_max_shift = adv_result.get("max_prediction_shift")
                    if adv_max_shift:
                        shift = abs(text_score_for_fusion - baseline_pred)
                        if shift > 0:
                            capped_w = float(adv_max_shift) / shift
                            new_w = min(effective_w, round(capped_w, 3))
                            if new_w < effective_w:
                                constraint_note = (
                                    f"Advocate `max_prediction_shift={adv_max_shift}` — "
                                    f"weight capped from {effective_w:.3f} to {new_w:.3f}."
                                )
                                effective_w = new_w

            # ── Fusion (logit space) ──────────────────────────────────────────
            with st.expander("🔀 Fusion", expanded=True):
                if effective_w == 0:
                    fused_pred = baseline_pred
                    st.markdown(f"**Formula:** `fused = baseline = {baseline_pred:.3f}` (α = 0, text suppressed)")
                else:
                    text_evidence = logit(text_score_for_fusion) - logit(0.5)
                    fused_logit   = logit(baseline_pred) + effective_w * confidence * text_evidence
                    fused_pred    = sigmoid(fused_logit)
                    st.markdown(
                        f"**Formula (logit space):**\n\n"
                        f"`logit({baseline_pred:.3f}) = {logit(baseline_pred):+.3f}`\n\n"
                        f"`text evidence = logit({text_score_for_fusion:.3f}) − logit(0.5) = {text_evidence:+.3f}`\n\n"
                        f"`fused_logit = {logit(baseline_pred):+.3f} + {effective_w:.2f} × {confidence:.2f} × {text_evidence:+.3f} = {fused_logit:+.3f}`\n\n"
                        f"`fused_pred = sigmoid({fused_logit:+.3f}) = {fused_pred:.3f}`"
                    )
                if veto and is_hard_veto:
                    st.caption(f"α = 0 (hard veto: {history_summary}). "
                               f"Fused = baseline. Text is contextual evidence only.")
                if constraint_note:
                    st.caption(f"⚙️ Constraint applied: {constraint_note}")
                c1f, c2f, c3f = st.columns(3)
                with c1f:
                    st.metric("Baseline (numeric)", f"{baseline_pred:.3f}")
                with c2f:
                    st.metric("Text Risk Score", f"{text_score_for_fusion:.3f}",
                              delta="suppressed" if effective_w == 0 and text_score_raw else None,
                              delta_color="off")
                with c3f:
                    delta_val = fused_pred - baseline_pred
                    st.metric("Fused", f"{fused_pred:.3f}",
                              delta=f"{delta_val:+.3f} vs baseline" if delta_val != 0 else "= baseline (text suppressed)",
                              delta_color="inverse" if delta_val != 0 else "off")

        # conflict_fragile needed by both Green LLM trigger and Fragility Flags
        conflict_fragile = bool(desc_text.strip() and veto) if desc_text.strip() else False

        # ── Green Agent: Final Authority Check ───────────────────────────────
        # Green independently validates the fused result before verdict is issued.
        # The shift cap is a safety rail for the static 5-scalar text weights.
        THRESHOLD    = 0.5
        MAX_SHIFT    = 0.12

        _fused_before_green = fused_pred
        green_action  = "accepted"
        green_notes   = []

        if desc_text.strip() and effective_w > 0:
            shift = abs(fused_pred - baseline_pred)

            # Rule 1 — max shift cap
            if shift > MAX_SHIFT:
                direction  = 1 if fused_pred > baseline_pred else -1
                fused_pred = round(baseline_pred + direction * MAX_SHIFT, 4)
                green_action = "shift_capped"
                green_notes.append(
                    f"**Shift cap**: |{_fused_before_green:.3f} − {baseline_pred:.3f}| = {shift:.3f} "
                    f"> {MAX_SHIFT}. Capped to **{fused_pred:.3f}**."
                )

            # Rule 2 — verdict flip gate (asymmetric thresholds)
            baseline_verdict = baseline_pred >= THRESHOLD
            fused_verdict    = fused_pred    >= THRESHOLD
            if baseline_verdict != fused_verdict:
                flip_type = "risky→safe" if baseline_verdict else "safe→risky"
                # Clearing a default needs stronger evidence than raising one
                conf_req = 0.87 if baseline_verdict else 0.82
                text_req = 0.35 if baseline_verdict else 0.25
                text_gap = abs(text_score_for_fusion - 0.5)
                flip_ok  = (confidence >= conf_req) and (text_gap >= text_req)
                if not flip_ok:
                    fused_pred   = baseline_pred
                    green_action = "flip_blocked"
                    green_notes.append(
                        f"**Flip blocked** ({flip_type}): requires conf ≥ {conf_req} "
                        f"(got {confidence:.2f}) AND |text−0.5| ≥ {text_req} "
                        f"(got {text_gap:.2f}). Reverting to baseline **{fused_pred:.3f}**."
                    )
                else:
                    green_notes.append(
                        f"**Flip approved** ({flip_type}): conf={confidence:.2f} ≥ {conf_req}, "
                        f"|text−0.5|={text_gap:.2f} ≥ {text_req}."
                    )

        action_icon = {"accepted": "✅", "shift_capped": "⚠️", "flip_blocked": "🛑"}.get(green_action, "✅")
        green_label = f"🟢 Green Agent — Final Authority  {action_icon} {green_action.replace('_',' ')}"
        st.markdown('<div class="green-expander-marker"></div>', unsafe_allow_html=True)
        with st.expander(green_label, expanded=(green_action != "accepted")):
            st.caption("Independent of White Agents. Enforces hard rules on the fused result before verdict is issued.")
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1: st.metric("Fused (pre-check)", f"{_fused_before_green:.3f}")
            with col_g2: st.metric("Max allowed shift", f"±{MAX_SHIFT}")
            with col_g3: st.metric("Fused (post-check)", f"{fused_pred:.3f}",
                                   delta=f"{fused_pred-_fused_before_green:+.3f}" if fused_pred != _fused_before_green else "unchanged",
                                   delta_color="off")
            if green_notes:
                for note in green_notes:
                    if green_action == "flip_blocked":
                        st.error(note)
                    elif green_action == "shift_capped":
                        st.warning(note)
                    else:
                        st.success(note)
            else:
                st.success(f"Fused score {fused_pred:.3f} within bounds. No intervention needed.")
            st.markdown(
                "**Green's two hard rules:**\n"
                f"1. Shift cap: |fused − baseline| ≤ {MAX_SHIFT} — safety rail for text fusion\n"
                "2. Flip gate: verdict reversal requires high LLM confidence **and** extreme text signal "
                "(asymmetric: clearing a predicted default requires stronger evidence than raising one)"
            )

        # ── Green Agent LLM — edge-case terminal review ──────────────────────
        # Trigger 1: flip_blocked but evidence is near-miss
        # Trigger 2: Green rules overrode White Agent consensus (no inter-agent conflict)
        _strat_rationale  = strat_result.get("rationale", "") if desc_text.strip() and strat_result else ""
        _hist_delta_green = hist_delta if desc_text.strip() else 0.0

        call_green_llm_flag = False
        green_llm_reason    = ""

        if desc_text.strip() and effective_w > 0:
            if green_action == "flip_blocked":
                _bv  = baseline_pred >= THRESHOLD
                _cr  = 0.87 if _bv else 0.82
                _tr  = 0.35 if _bv else 0.25
                _gap = abs(text_score_for_fusion - 0.5)
                if (_cr - confidence) <= 0.08 and (_tr - _gap) <= 0.12:
                    call_green_llm_flag = True
                    green_llm_reason = (
                        f"Flip blocked but near threshold "
                        f"(conf {confidence:.2f} vs req {_cr}, gap {_gap:.2f} vs req {_tr})"
                    )
            if green_action in ("shift_capped", "flip_blocked") and not conflict_fragile:
                call_green_llm_flag = True
                green_llm_reason = green_llm_reason or "Green rules overrode White Agent consensus"

        _fused_post_hard_rules = fused_pred

        green_llm_label = (
            "🟢 Green Agent — LLM Review  🔔 called" if call_green_llm_flag
            else "🟢 Green Agent — LLM Review  — idle"
        )
        st.markdown('<div class="green-expander-marker"></div>', unsafe_allow_html=True)
        with st.expander(green_llm_label, expanded=call_green_llm_flag):
            st.caption("Called only when hard rules fire on contested borderline evidence. "
                       "Green LLM decision is terminal — no further override possible.")
            if not call_green_llm_flag:
                st.markdown("**Status:** Idle — hard rule outcome is uncontested.")
            else:
                st.info(f"**Trigger:** {green_llm_reason}")
                with st.spinner("Green Agent LLM reviewing contested outcome..."):
                    t0 = time.time()
                    green_llm_result, green_llm_raw = run_green_llm(
                        baseline_pred, _fused_before_green, _fused_post_hard_rules,
                        text_score_for_fusion if text_score_raw else 0.45,
                        confidence if text_score_raw else 0.5,
                        grade, green_action, green_notes,
                        _strat_rationale, _hist_delta_green
                    )
                    elapsed_g = time.time() - t0

                decision   = green_llm_result.get("decision", "accept_baseline")
                final_pred = float(green_llm_result.get("final_pred", _fused_post_hard_rules))
                reasoning  = green_llm_result.get("reasoning", "")
                reasoning_brief = green_llm_result.get("display_brief", "")

                decision_icon = {"accept_baseline": "🔵", "accept_fused": "🟠", "custom": "🟣"}.get(decision, "❓")
                st.markdown(f"**Decision:** {decision_icon} `{decision}`  ⏱ {elapsed_g:.1f}s")
                st.markdown(
                    f'<div class="agent-brief"><strong>Reasoning brief:</strong> {html.escape(reasoning_brief or concise_text(reasoning, 260))}</div>',
                    unsafe_allow_html=True,
                )

                col_gl1, col_gl2, col_gl3 = st.columns(3)
                with col_gl1: st.metric("After hard rules", f"{_fused_post_hard_rules:.3f}")
                with col_gl2: st.metric("Green LLM final", f"{final_pred:.3f}",
                                        delta=f"{final_pred - _fused_post_hard_rules:+.3f}",
                                        delta_color="off")
                with col_gl3: st.metric("Verdict", "DEFAULT" if final_pred >= THRESHOLD else "NO DEFAULT")

                with st.expander("🔍 Raw LLM output (Green)", expanded=False):
                    st.code(green_llm_raw, language="json")

                fused_pred = final_pred

        # ── Agent 4: Reporter ─────────────────────────────────────────────────
        FUZZY_LO, FUZZY_HI = 0.30, 0.65
        threshold      = THRESHOLD
        in_fuzzy       = FUZZY_LO <= fused_pred <= FUZZY_HI
        text_influence = abs(fused_pred - baseline_pred) if text_score_raw else 0.0
        # If Green hard rule or Green LLM reverted fused to baseline, report as suppressed
        _fused_eq_baseline    = (abs(fused_pred - baseline_pred) < 1e-6)
        _eff_w_for_reporter   = 0.0 if (not desc_text.strip() or _fused_eq_baseline) else effective_w
        _veto_for_reporter   = veto        if desc_text.strip() else False
        _hist_for_reporter   = hist_delta  if desc_text.strip() else 0.0

        reporter_exp = st.expander(
            f"📊 White Agent 4 — Reporter (LLM) {'🔔 triggered' if in_fuzzy else '— not triggered'}",
            expanded=in_fuzzy,
        )
        with reporter_exp:
            if in_fuzzy and desc_text.strip():
                with st.spinner("Generating faithful case narrative..."):
                    t0 = time.time()
                    try:
                        narrative = reporter(row, baseline_pred, fused_pred, grade,
                                            text_score_for_fusion, text_reasoning,
                                            effective_weight=_eff_w_for_reporter,
                                            confidence=confidence,
                                            veto_active=_veto_for_reporter,
                                            hist_delta=_hist_for_reporter,
                                            threshold=threshold)
                    except Exception as e:
                        narrative = (
                            f"Reporter LLM unavailable: {e}\n\n"
                            f"Fallback summary: numeric baseline={baseline_pred:.3f}, "
                            f"final fused score={fused_pred:.3f}, threshold={threshold:.2f}. "
                            f"Text score={text_score_for_fusion:.3f}, confidence={confidence:.3f}, "
                            f"effective text weight={_eff_w_for_reporter:.3f}."
                        )
                    narrative = strip_llm_thinking(narrative)
                    if not narrative:
                        narrative = (
                            f"Reporter returned hidden reasoning only. Fallback summary: "
                            f"baseline={baseline_pred:.3f}, fused={fused_pred:.3f}, "
                            f"text shift={fused_pred - baseline_pred:+.3f}."
                        )
                    elapsed = time.time() - t0
                st.info(narrative)
                st.caption(f"⏱ {elapsed:.1f}s | fused={fused_pred:.3f} in fuzzy zone [{FUZZY_LO}–{FUZZY_HI}] "
                           f"| text_weight={_eff_w_for_reporter:.2f}")
            elif in_fuzzy and not desc_text.strip():
                st.warning("Borderline numeric case — Reporter would generate narrative with text context.")
            else:
                st.caption(f"Fused {fused_pred:.3f} outside fuzzy zone [{FUZZY_LO}–{FUZZY_HI}]. Not called.")

        # ── Fragility Flags ────────────────────────────────────────────────────
        threshold_fragile   = abs(fused_pred - threshold) < 0.03
        # conflict_fragile already computed above (needed earlier by Green LLM trigger)
        suppression_fragile = (desc_text.strip() and
                               _eff_w_for_reporter == 0 and
                               text_score_raw is not None and
                               text_score_raw > 0.65)

        fragility_flags = []
        if threshold_fragile:
            margin = fused_pred - threshold
            fragility_flags.append(
                f"**Threshold fragility:** fused score is {abs(margin):.3f} "
                f"{'above' if margin >= 0 else 'below'} decision threshold — "
                f"small feature changes may flip the verdict.")
        if conflict_fragile:
            fragility_flags.append(
                f"**Agent conflict:** Strategist proposed text weight for Grade {grade}, "
                f"but Advocate vetoed ({'hard' if is_hard_veto else 'soft'} veto). "
                f"Agents disagreed about whether text should contribute.")
        if suppression_fragile:
            fragility_flags.append(
                f"**Evidence suppression:** Text risk score = {text_score_raw:.2f} (high) "
                f"but weight = 0.00 due to Grade {grade} subgroup policy. "
                f"Strong text signal is acknowledged but not used in prediction.")

        # ── Agent Conflict Summary ────────────────────────────────────────────
        if conflict_fragile:
            st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
            st.markdown(
                """
<div class="section-title">
  <div>
    <div class="eyebrow">Dispute Review</div>
    <h2>Agent Conflict Summary</h2>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            summary_md = (
                f"| Agent | Action |\n|-------|--------|\n"
                f"| 📝 Text Analyst | Score = **{text_score_raw:.2f}** — "
                f"{'severe' if text_score_raw > 0.8 else 'elevated'} risk signals |\n"
                f"| ⚙️ Strategist | Proposed weight `{learned_w:.2f} → {proposed_w:.2f}` |\n"
                f"| 🛡️ Advocate | **{'Hard' if is_hard_veto else 'Soft'} veto** — "
                f"{historical_text_impact(hist_delta, grade)} |\n"
                f"| ⚖️ Green Arbitrate | {'Hard veto enforced' if is_hard_veto else 'Soft compromise accepted'} |\n"
                f"| 🟢 Green | Numeric-driven. fused = baseline = **{fused_pred:.3f}** |"
            )
            st.markdown(summary_md)

        # ── Final Verdict ──────────────────────────────────────────────────────
        st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="section-title">
  <div>
    <div class="eyebrow">Final Decision</div>
    <h2>Verdict</h2>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        verdict = fused_pred >= threshold
        margin  = abs(fused_pred - threshold)
        risk_level = ("🔴 High" if fused_pred > 0.65 else
                      "🟡 Medium" if fused_pred > 0.40 else "🟢 Low")
        dec_conf = ("High"   if margin > 0.15 and not conflict_fragile else
                    "Medium" if margin > 0.05 and not conflict_fragile else "Low")

        if verdict:
            st.markdown(
                f'<div class="verdict-default">'
                f'<p class="verdict-title">🔴 PREDICTED: DEFAULT</p>'
                f'<p class="verdict-sub">Fused score {fused_pred:.3f} ≥ threshold {threshold} '
                f'| Risk: {risk_level} | Confidence: {dec_conf}</p>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="verdict-safe">'
                f'<p class="verdict-title">🟢 PREDICTED: NO DEFAULT</p>'
                f'<p class="verdict-sub">Fused score {fused_pred:.3f} &lt; threshold {threshold} '
                f'| Risk: {risk_level} | Confidence: {dec_conf}</p>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1: st.metric("Numeric baseline", f"{baseline_pred:.3f}")
        with col_v2: st.metric("Text risk score",  f"{text_score_raw:.3f}" if text_score_raw else "—")
        with col_v3: st.metric("Fused score",       f"{fused_pred:.3f}",
                               delta=f"{fused_pred-baseline_pred:+.3f}" if text_score_raw else None,
                               delta_color="inverse")
        with col_v4: st.metric("Margin from threshold", f"{margin:.3f}",
                               help="Distance from 0.5. < 0.03 = fragile.")

        if fragility_flags:
            st.markdown("**⚠️ Fragility flags:**")
            for ff in fragility_flags:
                st.markdown(f"- {ff}")

        # ── Pipeline status bar (fill placeholder at top) ──────────────────────
        has_text   = bool(desc_text.strip())
        ta_chip    = (f'<span class="pipe-chip chip-ok">📝 Analyst ✓</span>'   if has_text else
                      f'<span class="pipe-chip chip-idle">📝 Analyst —</span>')
        st_chip    = (f'<span class="pipe-chip chip-warn">⚙️ Strategist ⚡</span>' if (has_text and call_strategist) else
                      f'<span class="pipe-chip chip-ok">⚙️ Strategist ✓</span>'   if has_text else
                      f'<span class="pipe-chip chip-idle">⚙️ Strategist —</span>')
        adv_chip   = (f'<span class="pipe-chip chip-alert">🛡️ Advocate 🚨</span>'  if (has_text and veto and is_hard_veto) else
                      f'<span class="pipe-chip chip-warn">🛡️ Advocate ⚠️</span>'   if (has_text and veto) else
                      f'<span class="pipe-chip chip-ok">🛡️ Advocate ✓</span>'     if has_text else
                      f'<span class="pipe-chip chip-idle">🛡️ Advocate —</span>')
        arb_chip   = (f'<span class="pipe-chip chip-warn">⚖️ Green Arbitrate 🔔</span>' if (has_text and veto) else
                      f'<span class="pipe-chip chip-idle">⚖️ Green Arbitrate —</span>')
        grn_chip   = (f'<span class="pipe-chip chip-alert">🟢 Green 🛑</span>'    if green_action == "flip_blocked" else
                      f'<span class="pipe-chip chip-warn">🟢 Green ⚠️</span>'     if green_action == "shift_capped" else
                      f'<span class="pipe-chip chip-ok">🟢 Green ✓</span>')
        gllm_chip  = (f'<span class="pipe-chip chip-warn">🟢 LLM Review 🔔</span>' if call_green_llm_flag else
                      f'<span class="pipe-chip chip-idle">🟢 LLM Review —</span>')
        rep_chip   = (f'<span class="pipe-chip chip-ok">📊 Reporter 🔔</span>'  if in_fuzzy and has_text else
                      f'<span class="pipe-chip chip-idle">📊 Reporter —</span>')
        arrow = '<span class="chip-arrow">›</span>'

        _status_bar.markdown(
            f'<div style="margin-bottom:8px">'
            f'<span class="pipe-chip chip-ok">🟢 XGBoost ✓</span>{arrow}'
            f'{ta_chip}{arrow}{st_chip}{arrow}{adv_chip}{arrow}'
            f'{arb_chip}{arrow}{grn_chip}{arrow}{gllm_chip}{arrow}{rep_chip}'
            f'</div>',
            unsafe_allow_html=True
        )

render_offline_evaluation()
tag_green_expanders()
