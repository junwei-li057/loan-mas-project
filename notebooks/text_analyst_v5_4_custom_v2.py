# ============================================================
# Text Analyst v5.4 Compact Schema + Custom V2 Score
# Shareable Template
# ============================================================
# What this script does:
# 1. Reads a CSV containing borrower descriptions.
# 2. Calls MiniMax to extract compact structured credit evidence.
# 3. Computes both original score and custom_v2 score.
# 4. Saves results as a pickle file with resume support.
# 5. If quota/rate limit interrupts the run, rerun the same script later.
# 6. fallback_neutral rows are NOT counted as completed and will be retried.
#
# Fill in the CONFIG section before running.
# Do not hard-code API keys in shared notebooks/scripts.
# ============================================================

import os
import re
import json
import time
import pickle
import warnings
import requests
import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIG: fill in your own information here
# ============================================================

MINIMAX_URL = "https://api.minimaxi.com/v1/chat/completions"
MODEL_NAME = "MiniMax-M2.7"

# Leave empty to enter securely when prompted, or set environment variable MINIMAX_API_KEY.
MINIMAX_API_KEY = ""

# Required paths. Fill these in.
DATA_PATH = ""          # e.g. "/content/drive/MyDrive/.../loan_default_sample1.csv"
SAVE_PATH_FULL = ""    # e.g. "/content/drive/MyDrive/.../text_analysis_results_full.pkl"
OUTPUT_DIR = ""        # e.g. "/content/drive/MyDrive/.../text_analysis_outputs"

# Input column names.
DESC_COL = "desc"
LABEL_COL = "label"    # optional for diagnostics; 1 = default, 0 = non-default
GRADE_COL = "grade"    # optional for subgroup diagnostics

# Run all rows if None; otherwise run first N rows.
N_ROWS = None

# Runtime settings.
SAVE_EVERY = 50
SLEEP_BETWEEN_CALLS = 0.1
MAX_CONSECUTIVE_FALLBACK = 10
MAX_TOKENS = 1500
TEMPERATURE = 0.1
USE_RESPONSE_FORMAT_JSON = True


# ============================================================
# 1. Safety checks
# ============================================================

if not DATA_PATH:
    raise ValueError("Please fill DATA_PATH in CONFIG.")
if not SAVE_PATH_FULL:
    raise ValueError("Please fill SAVE_PATH_FULL in CONFIG.")
if not OUTPUT_DIR:
    raise ValueError("Please fill OUTPUT_DIR in CONFIG.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not MINIMAX_API_KEY:
    MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
if not MINIMAX_API_KEY:
    MINIMAX_API_KEY = getpass.getpass("Enter your MiniMax API key: ")


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(DATA_PATH)
if DESC_COL not in df.columns:
    raise ValueError(f"Description column '{DESC_COL}' not found in CSV.")
if "original_idx" not in df.columns:
    df["original_idx"] = df.index

df_run = df.copy() if N_ROWS is None else df.head(N_ROWS).copy()
print("Full data shape:", df.shape)
print("Rows to run:", len(df_run))


# ============================================================
# 3. Evidence fields and scoring weights
# ============================================================

class RateLimitError(Exception):
    pass

EVIDENCE_FIELDS = [
    "employment_stability",
    "income_or_savings",
    "repayment_plan",
    "repayment_history",
    "loan_purpose_specificity",
    "payment_burden_or_cashflow_pressure",
    "explicit_unemployment_or_no_income",
    "explicit_delinquency_or_collections",
    "financial_distress",
    "multiple_problem_debts_no_plan",
]

FIELD_ALIASES = {
    "employment": "employment_stability",
    "stable_employment": "employment_stability",
    "job_stability": "employment_stability",
    "income": "income_or_savings",
    "savings": "income_or_savings",
    "income_savings": "income_or_savings",
    "repayment": "repayment_plan",
    "payment_plan": "repayment_plan",
    "loan_purpose": "loan_purpose_specificity",
    "purpose": "loan_purpose_specificity",
    "cashflow_pressure": "payment_burden_or_cashflow_pressure",
    "cash_flow_pressure": "payment_burden_or_cashflow_pressure",
    "payment_burden": "payment_burden_or_cashflow_pressure",
    "monthly_payment_pressure": "payment_burden_or_cashflow_pressure",
    "unemployment": "explicit_unemployment_or_no_income",
    "no_income": "explicit_unemployment_or_no_income",
    "job_loss": "explicit_unemployment_or_no_income",
    "delinquency": "explicit_delinquency_or_collections",
    "collections": "explicit_delinquency_or_collections",
    "overdue": "explicit_delinquency_or_collections",
    "late_payments": "explicit_delinquency_or_collections",
    "distress": "financial_distress",
    "financial_pressure": "financial_distress",
    "problem_debts": "multiple_problem_debts_no_plan",
    "multiple_debts_no_plan": "multiple_problem_debts_no_plan",
}

FORBIDDEN_PATTERNS = re.compile(
    r"\b("
    r"grammar|spelling|typo|fluent|fluency|native language|non-native|"
    r"english level|poor english|education level|educated|uneducated|"
    r"cultural style|culture|race|ethnicity|nationality|gender|"
    r"borrower name|name-based|demographic proxy"
    r")\b",
    re.I,
)

ORIGINAL_WEIGHTS = {
    "employment_stability": -0.08,
    "income_or_savings": -0.03,
    "repayment_plan": -0.05,
    "repayment_history": -0.08,
    "purpose_high": -0.03,
    "purpose_medium": 0.00,
    "purpose_low": 0.00,
    "payment_burden": 0.06,
    "unemployment_or_no_income": 0.14,
    "delinquency_or_collections": 0.16,
    "financial_distress": 0.14,
    "problem_debts_no_plan": 0.10,
}

# Custom v2 weights, currently used as the main score.
CUSTOM_V2_WEIGHTS = {
    "employment_stability": -0.08,
    "income_or_savings": -0.01,
    "repayment_plan": -0.02,
    "repayment_history": -0.08,
    "purpose_high": -0.03,
    "purpose_medium": 0.00,
    "purpose_low": 0.03,
    "payment_burden": 0.04,
    "unemployment_or_no_income": 0.14,
    "delinquency_or_collections": 0.16,
    "financial_distress": 0.06,
    "problem_debts_no_plan": 0.10,
}


# ============================================================
# 4. Helper functions
# ============================================================

def _clean_desc(desc_text: str, max_chars: int = 500) -> str:
    return re.sub(r"<[^>]+>", " ", str(desc_text)).strip()[:max_chars]


def _extract_json(raw: str) -> dict:
    cleaned = str(raw or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.I).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

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

    raise ValueError(f"No valid JSON object found in LLM response: {str(raw)[:500]}")


def _normalize_status(status: str) -> str:
    s = str(status or "not_mentioned").strip().lower().replace(" ", "_")
    aliases = {
        "yes": "present",
        "no": "absent",
        "none": "absent",
        "not_present": "absent",
        "missing": "not_mentioned",
        "unknown": "unclear",
        "n/a": "not_mentioned",
        "na": "not_mentioned",
        "specific": "high",
        "generic": "medium",
        "vague": "low",
    }
    s = aliases.get(s, s)
    allowed = {"positive", "negative", "present", "absent", "not_mentioned", "unclear", "high", "medium", "low"}
    return s if s in allowed else "unclear"


def _normalize_field_name(field: str):
    f = str(field or "").strip().lower().replace(" ", "_")
    f = FIELD_ALIASES.get(f, f)
    return f if f in EVIDENCE_FIELDS else None


def _normalize_evidence(evidence_obj: dict) -> dict:
    evidence_obj = evidence_obj or {}
    out = {}
    for field in EVIDENCE_FIELDS:
        item = evidence_obj.get(field, {})
        if not isinstance(item, dict):
            item = {"status": str(item), "evidence": ""}
        out[field] = {
            "status": _normalize_status(item.get("status", "not_mentioned")),
            "evidence": str(item.get("evidence", "") or "").strip(),
        }
    return out


def normalize_evidence_from_items(parsed: dict) -> dict:
    evidence = {field: {"status": "not_mentioned", "evidence": ""} for field in EVIDENCE_FIELDS}
    items = parsed.get("evidence_items", [])
    if not isinstance(items, list):
        return evidence

    raw_items_kept = []
    for item in items:
        if not isinstance(item, dict):
            continue
        field = _normalize_field_name(item.get("field"))
        if field is None:
            continue
        status = _normalize_status(item.get("status", "not_mentioned"))
        phrase = str(item.get("evidence", "") or "").strip()
        if status in {"not_mentioned", "unclear"} and not phrase:
            continue

        old_status = evidence[field]["status"]
        old_phrase = evidence[field]["evidence"]
        if old_status == "not_mentioned":
            evidence[field] = {"status": status, "evidence": phrase}
        else:
            combined_phrase = old_phrase
            if phrase and phrase not in old_phrase:
                combined_phrase = old_phrase + "; " + phrase if old_phrase else phrase
            preferred_status = old_status
            if old_status in {"absent", "unclear", "not_mentioned"} and status not in {"absent", "unclear", "not_mentioned"}:
                preferred_status = status
            evidence[field] = {"status": preferred_status, "evidence": combined_phrase}

        raw_items_kept.append({"field": field, "status": status, "evidence": phrase})

    evidence["_compact_items_kept"] = raw_items_kept
    evidence["_compact_items_count"] = len(raw_items_kept)
    return evidence


def _is_positive(status: str) -> bool:
    return status in {"positive", "present", "high"}


def _is_risk_present(status: str) -> bool:
    return status in {"negative", "present", "high"}


def _trim_reasoning(reasoning: str, max_words: int = 25) -> str:
    reasoning = str(reasoning or "").strip()
    reasoning = re.sub(r"\s+", " ", reasoning)
    if not reasoning:
        return ""
    words = reasoning.split()
    return reasoning if len(words) <= max_words else " ".join(words[:max_words]) + "..."


# ============================================================
# 5. Evidence correction logic
# ============================================================

def _combined_evidence_text(evidence: dict, desc_text: str = "") -> str:
    parts = [str(desc_text or "")]
    for field in EVIDENCE_FIELDS:
        item = evidence.get(field, {})
        if isinstance(item, dict):
            parts.append(item.get("evidence", ""))
    return " ".join(parts).lower()


def correct_evidence_logic(evidence: dict, desc_text: str = "") -> dict:
    combined_text = _combined_evidence_text(evidence, desc_text)
    correction_flags = []

    concrete_plan_keywords = [
        "one monthly payment", "one payment", "single monthly payment", "fixed monthly payment",
        "automatic payment", "auto payment", "autopay", "payment plan", "repayment plan",
        "budget", "budgeting", "monthly payments", "make payments", "lower monthly payment",
        "reduce monthly payment", "combine payments", "consolidate into one monthly payment",
        "debt free in", "pay off in", "paid off in",
    ]
    generic_debt_only_patterns = [
        "debt consolidation", "consolidate my debt", "consolidate debt", "pay off debt",
        "pay off credit card", "pay off credit cards", "credit card payoff",
    ]
    has_concrete_plan = any(k in combined_text for k in concrete_plan_keywords)
    has_generic_debt_only = any(k in combined_text for k in generic_debt_only_patterns)

    if _is_positive(evidence["repayment_plan"]["status"]) and has_generic_debt_only and not has_concrete_plan:
        evidence["repayment_plan"] = {"status": "not_mentioned", "evidence": ""}
        correction_flags.append("repayment_plan:generic_debt_only_removed")

    payment_burden_keywords = [
        "lower my monthly payout", "lower monthly payout", "lower my monthly payment", "lower monthly payment",
        "reduce monthly obligations", "monthly obligations", "minimum payment", "minimum payments",
        "bills never stop", "stay on top of bills", "rent and utilities", "rent/utilities",
        "utility bills", "utilities", "recurring bills", "monthly bills", "payment burden",
        "cash flow", "cashflow", "one payment instead of several", "one payment instead of multiple",
    ]
    lower_interest_only_keywords = [
        "lower interest", "lower interest rate", "better interest rate", "reduce interest",
        "high interest", "high-interest", "high rate", "high-rate",
    ]
    has_payment_burden_language = any(k in combined_text for k in payment_burden_keywords)
    has_lower_interest_only = any(k in combined_text for k in lower_interest_only_keywords) and not has_payment_burden_language

    if _is_risk_present(evidence["payment_burden_or_cashflow_pressure"]["status"]) and has_lower_interest_only:
        evidence["payment_burden_or_cashflow_pressure"] = {"status": "absent", "evidence": ""}
        correction_flags.append("payment_burden:lower_interest_only_removed")

    problem_debt_keywords = [
        "maxed out", "maxed-out", "overdue", "past due", "behind on", "behind in",
        "collections", "collection", "delinquent", "delinquency", "late fee", "late fees",
        "can't keep up", "cannot keep up", "unable to keep up", "unable to pay",
        "struggling to pay", "desperate", "emergency", "urgent", "overwhelmed by debt",
        "drowning in debt", "bills are overdue", "accounts are overdue", "accounts in collections",
    ]
    has_problem_debt_language = any(k in combined_text for k in problem_debt_keywords)

    if _is_risk_present(evidence["multiple_problem_debts_no_plan"]["status"]):
        if has_concrete_plan or not has_problem_debt_language:
            evidence["multiple_problem_debts_no_plan"] = {"status": "absent", "evidence": ""}
            correction_flags.append("multiple_problem_debts_no_plan:removed_by_logic")

    crisis_keywords = [
        "emergency", "desperate", "urgent", "need immediately", "behind on", "past due",
        "late fee", "late fees", "cannot keep up", "can't keep up", "struggling to pay",
        "unable to pay", "collections", "collection", "job loss", "lost my job", "unemployed",
        "no income", "eviction", "foreclosure",
    ]
    ordinary_debt_keywords = [
        "debt consolidation", "consolidate", "credit card debt", "high interest",
        "lower interest", "pay off credit cards",
    ]
    has_crisis_language = any(k in combined_text for k in crisis_keywords)
    has_only_ordinary_debt_context = any(k in combined_text for k in ordinary_debt_keywords) and not has_crisis_language

    if _is_risk_present(evidence["financial_distress"]["status"]) and has_only_ordinary_debt_context:
        evidence["financial_distress"] = {"status": "absent", "evidence": ""}
        correction_flags.append("financial_distress:ordinary_debt_context_removed")

    evidence["_correction_flags"] = correction_flags
    return evidence


# ============================================================
# 6. Scoring and confidence
# ============================================================

def evidence_to_feature_dict(evidence: dict) -> dict:
    purpose_status = evidence["loan_purpose_specificity"]["status"]
    return {
        "employment_stability": int(_is_positive(evidence["employment_stability"]["status"])),
        "income_or_savings": int(_is_positive(evidence["income_or_savings"]["status"])),
        "repayment_plan": int(_is_positive(evidence["repayment_plan"]["status"])),
        "repayment_history": int(_is_positive(evidence["repayment_history"]["status"])),
        "purpose_high": int(purpose_status == "high"),
        "purpose_medium": int(purpose_status == "medium"),
        "purpose_low": int(purpose_status == "low"),
        "payment_burden": int(_is_risk_present(evidence["payment_burden_or_cashflow_pressure"]["status"])),
        "unemployment_or_no_income": int(_is_risk_present(evidence["explicit_unemployment_or_no_income"]["status"])),
        "delinquency_or_collections": int(_is_risk_present(evidence["explicit_delinquency_or_collections"]["status"])),
        "financial_distress": int(_is_risk_present(evidence["financial_distress"]["status"])),
        "problem_debts_no_plan": int(_is_risk_present(evidence["multiple_problem_debts_no_plan"]["status"])),
    }


def compute_score_from_weights(evidence: dict, weights: dict, base: float = 0.50) -> float:
    features = evidence_to_feature_dict(evidence)
    score = base + sum(value * weights[feature] for feature, value in features.items())
    return round(float(np.clip(score, 0.05, 0.95)), 3)


def compute_original_score(evidence: dict) -> float:
    return compute_score_from_weights(evidence, ORIGINAL_WEIGHTS)


def compute_custom_v2_score(evidence: dict) -> float:
    return compute_score_from_weights(evidence, CUSTOM_V2_WEIGHTS)


def compute_confidence(desc_text: str, evidence: dict):
    text = _clean_desc(desc_text)
    words = re.findall(r"[A-Za-z0-9$%]+", text)
    word_count = len(words)

    explicit_detail_count = 0
    for field in EVIDENCE_FIELDS:
        status = evidence[field]["status"]
        ev = evidence[field]["evidence"]
        if status not in {"not_mentioned", "absent", "unclear"} and ev:
            explicit_detail_count += 1

    has_money_or_rate = bool(re.search(r"\$[\d,]+|\d+%|\d+\s*(?:months?|years?)", text, re.I))

    if word_count <= 3:
        return 0.42, "extremely short description"
    if word_count <= 8 and explicit_detail_count == 0:
        return 0.50, "short vague description with no explicit credit evidence"
    if explicit_detail_count == 0:
        return 0.55, "no explicit credit-relevant evidence"
    if explicit_detail_count == 1:
        return 0.65, "one explicit credit-relevant detail"
    if explicit_detail_count == 2:
        return 0.75, "two explicit credit-relevant details"
    if explicit_detail_count >= 3 or has_money_or_rate:
        return 0.85, "sufficient explicit credit-relevant detail"
    return 0.65, "some explicit credit-relevant detail"


# ============================================================
# 7. Fairness audit
# ============================================================

def build_fairness_audit(reasoning: str) -> dict:
    forbidden_terms_in_reasoning = bool(FORBIDDEN_PATTERNS.search(str(reasoning or "")))
    return {
        "schema_excludes_forbidden_factors": True,
        "score_based_only_on_allowed_evidence": True,
        "missing_info_treated_as_low_confidence": True,
        "forbidden_terms_in_reasoning": forbidden_terms_in_reasoning,
    }


def apply_fairness_check(result: dict, desc_text: str) -> dict:
    reasoning = str(result.get("reasoning", ""))
    fairness_flags = []
    fairness_action = "pass"
    fairness_audit = build_fairness_audit(reasoning)

    if fairness_audit["forbidden_terms_in_reasoning"]:
        fairness_flags.append("forbidden_terms_in_reasoning")
        result["confidence"] = min(result.get("confidence", 0.5), 0.55)
        fairness_action = "lower_confidence"

    result["fairness_flags"] = fairness_flags
    result["fairness_action"] = fairness_action
    result["fairness_audit"] = fairness_audit
    return result


# ============================================================
# 8. Prompt and API call
# ============================================================

def build_text_analyst_prompt(desc_text: str) -> str:
    return (
        "You are the Text Analyst in a loan default prediction system.\n"
        "Extract explicit credit-relevant evidence from the borrower description.\n"
        "Return one valid JSON object only. Do not infer missing facts or invent evidence.\n\n"
        "Fairness rule:\n"
        "Do not use grammar, spelling, fluency, non-native English, cultural expression, name, education inference, "
        "gender, race, ethnicity, or nationality as credit evidence. Missing income/employment is not negative evidence.\n\n"
        "Allowed fields:\n"
        "- employment_stability: stable job, occupation, same company, full-time work, or years employed.\n"
        "- income_or_savings: stated income, salary, savings, or financial resources.\n"
        "- repayment_plan: concrete repayment mechanics, such as one monthly payment, automatic payments, fixed payments, budget, or payoff schedule. Generic debt consolidation alone is not a repayment plan.\n"
        "- repayment_history: no missed payments, never late, paid on time, good standing.\n"
        "- loan_purpose_specificity: high=specific details; medium=generic debt consolidation/payoff; low=vague or unclear use.\n"
        "- payment_burden_or_cashflow_pressure: monthly payment burden, recurring bills, rent/utilities pressure, minimum-payment pressure, or bills never stop. Not generic consolidation or lower interest alone.\n"
        "- explicit_unemployment_or_no_income: explicit unemployment, job loss, or no current income.\n"
        "- explicit_delinquency_or_collections: behind, overdue, past due, delinquency, collections, late rent/bills.\n"
        "- financial_distress: emergency, urgent cash need, unable to pay, eviction, foreclosure, severe pressure.\n"
        "- multiple_problem_debts_no_plan: problem-debt language such as maxed out, overdue, cannot keep up, collections, late fees, and no concrete plan. Multiple debts alone are not risk evidence.\n\n"
        "Status values: positive, negative, present, high, medium, low.\n"
        "Only output fields that are explicitly supported by the description.\n"
        "Do not output fields that are not mentioned. Do not output not_mentioned fields.\n"
        "Reasoning must be one sentence and under 25 words. Do not include step-by-step analysis.\n\n"
        "Compact output schema:\n"
        "{\n"
        '  "evidence_items": [\n'
        '    {"field": "<allowed field>", "status": "<status>", "evidence": "<short phrase>"}\n'
        "  ],\n"
        '  "reasoning": "<one concise sentence under 25 words>"\n'
        "}\n\n"
        "Examples:\n"
        "A. Description: I want to consolidate my credit card debt.\n"
        "Correct JSON:\n"
        "{\n"
        '  "evidence_items": [\n'
        '    {"field": "loan_purpose_specificity", "status": "medium", "evidence": "consolidate my credit card debt"}\n'
        "  ],\n"
        '  "reasoning": "The text gives a generic debt consolidation purpose."\n'
        "}\n\n"
        "B. Description: I want to consolidate my bills and lower my monthly payout because the bills never stop.\n"
        "Correct JSON:\n"
        "{\n"
        '  "evidence_items": [\n'
        '    {"field": "loan_purpose_specificity", "status": "medium", "evidence": "consolidate my bills"},\n'
        '    {"field": "payment_burden_or_cashflow_pressure", "status": "present", "evidence": "lower my monthly payout; bills never stop"}\n'
        "  ],\n"
        '  "reasoning": "The text shows debt consolidation and monthly payment pressure."\n'
        "}\n\n"
        "C. Description: I lost my job, am behind on rent and bills, and need emergency cash.\n"
        "Correct JSON:\n"
        "{\n"
        '  "evidence_items": [\n'
        '    {"field": "explicit_unemployment_or_no_income", "status": "present", "evidence": "lost my job"},\n'
        '    {"field": "explicit_delinquency_or_collections", "status": "present", "evidence": "behind on rent and bills"},\n'
        '    {"field": "financial_distress", "status": "present", "evidence": "need emergency cash"}\n'
        "  ],\n"
        '  "reasoning": "The text shows job loss, overdue bills, and emergency cash need."\n'
        "}\n\n"
        "D. Description: I have worked at the same hospital for 8 years and have never missed a payment.\n"
        "Correct JSON:\n"
        "{\n"
        '  "evidence_items": [\n'
        '    {"field": "employment_stability", "status": "positive", "evidence": "worked at the same hospital for 8 years"},\n'
        '    {"field": "repayment_history", "status": "positive", "evidence": "never missed a payment"}\n'
        "  ],\n"
        '  "reasoning": "The text shows stable employment and positive repayment history."\n'
        "}\n\n"
        f"Now analyze this description:\n{desc_text}\n\n"
        "Return only the compact JSON object."
    )


def call_minimax(prompt: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if USE_RESPONSE_FORMAT_JSON:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        MINIMAX_URL,
        headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Non-JSON API response: status={resp.status_code}, text={resp.text[:1000]}")

    if resp.status_code == 429:
        raise RateLimitError(f"API rate limit: {data}")
    if resp.status_code != 200:
        raise RuntimeError(f"API HTTP error {resp.status_code}: {data}")
    if "choices" not in data:
        raise RuntimeError(f"API response missing choices: {data}")

    choice = data["choices"][0]
    content = choice["message"]["content"]
    return {
        "content": content,
        "usage": data.get("usage", {}),
        "finish_reason": choice.get("finish_reason", None),
        "raw_char_len": len(content),
        "raw_approx_tokens": round(len(content) / 4, 1),
    }


# ============================================================
# 9. Text Analyst agent
# ============================================================

def text_analyst_agent(desc_text: str) -> dict:
    start_time = time.time()
    desc_text = _clean_desc(desc_text)
    prompt = build_text_analyst_prompt(desc_text)
    raw = ""
    api_result = {}

    try:
        api_start = time.time()
        api_result = call_minimax(prompt)
        api_elapsed = time.time() - api_start

        raw = api_result["content"]
        parsed = _extract_json(raw)

        if "evidence_items" in parsed:
            evidence = normalize_evidence_from_items(parsed)
        else:
            evidence = _normalize_evidence(parsed.get("credit_relevant_evidence", {}))

        compact_items_kept = evidence.pop("_compact_items_kept", [])
        compact_items_count = evidence.pop("_compact_items_count", 0)

        evidence = correct_evidence_logic(evidence, desc_text)
        correction_flags = evidence.pop("_correction_flags", [])

        original_score = compute_original_score(evidence)
        custom_v2_score = compute_custom_v2_score(evidence)
        confidence, confidence_reason = compute_confidence(desc_text, evidence)

        reasoning_raw = str(parsed.get("reasoning", ""))
        reasoning = _trim_reasoning(reasoning_raw, max_words=25)
        total_elapsed = time.time() - start_time

        result = {
            "desc_preview": desc_text[:500],
            "credit_relevant_evidence": evidence,
            "original_text_risk_score": original_score,
            "original_evidence_based_score": original_score,
            "text_risk_score": custom_v2_score,
            "final_text_risk_score": custom_v2_score,
            "evidence_based_score": custom_v2_score,
            "custom_v2_text_risk_score": custom_v2_score,
            "score_weights_version": "custom_v2",
            "score_delta_custom_v2_minus_original": round(custom_v2_score - original_score, 3),
            "confidence": confidence,
            "_evidence_confidence": confidence,
            "_confidence_reason": confidence_reason,
            "reasoning": reasoning,
            "_llm_reasoning_raw": reasoning_raw,
            "_compact_items_kept": compact_items_kept,
            "_compact_items_count": compact_items_count,
            "_correction_flags": correction_flags,
            "_raw_preview": str(raw)[:1000],
            "_raw_char_len": api_result.get("raw_char_len"),
            "_raw_approx_tokens": api_result.get("raw_approx_tokens"),
            "_api_usage": api_result.get("usage"),
            "_finish_reason": api_result.get("finish_reason"),
            "_api_elapsed_seconds": round(api_elapsed, 2),
            "_total_elapsed_seconds": round(total_elapsed, 2),
            "_attempt": 1,
        }
        return apply_fairness_check(result, desc_text)

    except RateLimitError:
        raise

    except Exception as e:
        return {
            "desc_preview": desc_text[:500],
            "credit_relevant_evidence": _normalize_evidence({}),
            "original_text_risk_score": 0.50,
            "original_evidence_based_score": 0.50,
            "text_risk_score": 0.50,
            "final_text_risk_score": 0.50,
            "evidence_based_score": 0.50,
            "custom_v2_text_risk_score": 0.50,
            "score_weights_version": "custom_v2",
            "score_delta_custom_v2_minus_original": 0.00,
            "confidence": 0.40,
            "fairness_flags": [],
            "fairness_action": "fallback_neutral",
            "fairness_audit": {
                "schema_excludes_forbidden_factors": True,
                "score_based_only_on_allowed_evidence": True,
                "missing_info_treated_as_low_confidence": True,
                "forbidden_terms_in_reasoning": False,
            },
            "reasoning": "Fallback neutral score because the LLM response could not be parsed reliably.",
            "_llm_reasoning_raw": "",
            "_error": str(e)[:500],
            "_compact_items_kept": [],
            "_compact_items_count": 0,
            "_correction_flags": [],
            "_raw_preview": str(raw)[:1000],
            "_raw_char_len": api_result.get("raw_char_len") if isinstance(api_result, dict) else None,
            "_raw_approx_tokens": api_result.get("raw_approx_tokens") if isinstance(api_result, dict) else None,
            "_api_usage": api_result.get("usage") if isinstance(api_result, dict) else {},
            "_finish_reason": api_result.get("finish_reason") if isinstance(api_result, dict) else None,
            "_api_elapsed_seconds": None,
            "_total_elapsed_seconds": round(time.time() - start_time, 2),
            "_attempt": 1,
        }


# ============================================================
# 10. Batch runner with fallback retry on resume
# ============================================================

def run_text_analysis(df_subset, save_path):
    """
    Resume behavior:
    - Successful rows are treated as completed.
    - fallback_neutral rows are NOT treated as completed.
    - Therefore fallback rows will be retried in the next run.
    """
    try:
        with open(save_path, "rb") as f:
            previous_results = pickle.load(f)

        valid_results = [r for r in previous_results if r.get("fairness_action") != "fallback_neutral"]
        fallback_results = [r for r in previous_results if r.get("fairness_action") == "fallback_neutral"]
        results = valid_results
        done_idx = {r["idx"] for r in valid_results if "idx" in r}
        remaining = len(df_subset) - len(done_idx)

        print(
            f"Resuming {os.path.basename(save_path)}: "
            f"{len(valid_results)} valid done, "
            f"{len(fallback_results)} fallback will be retried, "
            f"{remaining} remaining"
        )

    except FileNotFoundError:
        results, done_idx = [], set()
        print(f"Starting fresh: {len(df_subset)} records")

    errors = []
    consecutive_fallback = 0

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        if idx in done_idx:
            continue

        try:
            result = text_analyst_agent(row[DESC_COL])
            result["idx"] = idx
            result["original_idx"] = row.get("original_idx", idx)
            results.append(result)

            if result.get("fairness_action") == "fallback_neutral":
                consecutive_fallback += 1
            else:
                consecutive_fallback = 0

            if consecutive_fallback >= MAX_CONSECUTIVE_FALLBACK:
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                print("\nToo many consecutive fallback results. Saving and stopping.")
                break

            if len(results) % SAVE_EVERY == 0:
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

        except RateLimitError as e:
            print("\nRate limit reached. Saving current results and stopping.")
            print(str(e)[:500])
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            break

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            time.sleep(1)

        time.sleep(SLEEP_BETWEEN_CALLS)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    fallback_count = sum(r.get("fairness_action") == "fallback_neutral" for r in results)
    print(f"Done: {len(results)}, failed: {len(errors)}, fallback currently saved: {fallback_count}")
    return results


# ============================================================
# 11. Diagnostics
# ============================================================

def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2 or len(np.unique(score)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)


def ks_statistic(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    thresholds = np.sort(np.unique(score))
    return float(max(abs(np.mean(pos <= t) - np.mean(neg <= t)) for t in thresholds))


def evaluate_scores(y_true, score, name):
    score = np.asarray(score)
    score_clip = np.clip(score, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "n": len(y_true),
        "default_rate": float(np.mean(y_true)),
        "auc": safe_auc(y_true, score),
        "brier": brier_score_loss(y_true, score_clip),
        "log_loss": log_loss(y_true, score_clip, labels=[0, 1]),
        "ks": ks_statistic(y_true, score),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "score_min": float(np.min(score)),
        "score_median": float(np.median(score)),
        "score_max": float(np.max(score)),
    }


def threshold_table(y_true, score, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70)):
    rows = []
    for th in thresholds:
        mask = score >= th
        n = int(mask.sum())
        rows.append({
            "threshold": th,
            "n": n,
            "share": n / len(score),
            "default_rate": float(np.mean(y_true[mask])) if n > 0 else np.nan,
            "avg_score": float(np.mean(score[mask])) if n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def bin_table(y_true, score, q=5):
    tmp = pd.DataFrame({"label": y_true, "score": score})
    try:
        tmp["score_bin"] = pd.qcut(tmp["score"], q=q, duplicates="drop")
    except Exception:
        tmp["score_bin"] = pd.cut(tmp["score"], bins=q)
    return tmp.groupby("score_bin").agg(
        n=("label", "size"),
        default_rate=("label", "mean"),
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()


def summarize_full_results(results, df):
    res_df = pd.DataFrame(results)

    if "fairness_action" in res_df.columns:
        valid = res_df[res_df["fairness_action"] != "fallback_neutral"].copy()
    else:
        valid = res_df.copy()

    print("\n===== Basic Summary =====")
    print("Total saved results:", len(results))
    print("Valid rows:", len(valid))

    res_df.to_csv(os.path.join(OUTPUT_DIR, "full_custom_v2_results_flat.csv"), index=False)

    if LABEL_COL not in df.columns:
        print(f"Label column '{LABEL_COL}' not found. Skipping label-based diagnostics.")
        return valid, None, None

    keep_cols = ["original_idx", LABEL_COL]
    if GRADE_COL in df.columns:
        keep_cols.append(GRADE_COL)

    merged = valid.merge(df[keep_cols], on="original_idx", how="left")
    merged = merged.dropna(subset=[LABEL_COL]).copy()
    merged[LABEL_COL] = merged[LABEL_COL].astype(int)

    print("Valid rows for evaluation:", len(merged))
    print("Default rate:", merged[LABEL_COL].mean())

    print("\nFairness action counts:")
    print(res_df.get("fairness_action", pd.Series(dtype=object)).value_counts(dropna=False))

    y = merged[LABEL_COL].values
    model_scores = {
        "original_score": merged["original_text_risk_score"].values,
        "custom_v2_score": merged["custom_v2_text_risk_score"].values,
    }

    metrics_df = pd.DataFrame([evaluate_scores(y, score, name) for name, score in model_scores.items()])
    print("\n===== Overall Metrics =====")
    print(metrics_df.sort_values("auc", ascending=False))

    threshold_df = pd.concat([
        threshold_table(y, score).assign(model=name)
        for name, score in model_scores.items()
    ], ignore_index=True)
    threshold_df = threshold_df[["model", "threshold", "n", "share", "default_rate", "avg_score"]]
    print("\n===== Threshold Tables =====")
    print(threshold_df)

    print("\n===== Score Bin Tables =====")
    bin_tables = {}
    for name, score in model_scores.items():
        bt = bin_table(y, score, q=5)
        bin_tables[name] = bt
        print(f"\n--- {name} ---")
        print(bt)

    grade_auc_df = pd.DataFrame()
    if GRADE_COL in merged.columns:
        grade_rows = []
        for grade, sub in merged.groupby(GRADE_COL):
            if len(sub) < 10 or sub[LABEL_COL].nunique() < 2:
                continue
            for model_name, score_col in [("original_score", "original_text_risk_score"), ("custom_v2_score", "custom_v2_text_risk_score")]:
                grade_rows.append({
                    "grade": grade,
                    "model": model_name,
                    "n": len(sub),
                    "default_rate": sub[LABEL_COL].mean(),
                    "auc": safe_auc(sub[LABEL_COL].values, sub[score_col].values),
                    "avg_score": sub[score_col].mean(),
                    "score_std": sub[score_col].std(),
                })
        grade_auc_df = pd.DataFrame(grade_rows)
        print("\n===== Grade-Level AUC =====")
        print(grade_auc_df.sort_values(["grade", "auc"], ascending=[True, False]))

    merged.to_csv(os.path.join(OUTPUT_DIR, "full_custom_v2_eval_merged.csv"), index=False)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "full_custom_v2_metrics.csv"), index=False)
    threshold_df.to_csv(os.path.join(OUTPUT_DIR, "full_custom_v2_threshold_tables.csv"), index=False)
    if len(grade_auc_df) > 0:
        grade_auc_df.to_csv(os.path.join(OUTPUT_DIR, "full_custom_v2_grade_auc.csv"), index=False)
    for name, bt in bin_tables.items():
        bt.to_csv(os.path.join(OUTPUT_DIR, f"full_custom_v2_bin_table_{name}.csv"), index=False)

    print("\nSaved diagnostics to:", OUTPUT_DIR)
    return merged, metrics_df, threshold_df


# ============================================================
# 12. Run
# ============================================================

print("\n--- Running Full-Data Text Analyst v5.4 Compact + Custom V2 Score ---")
results_full = run_text_analysis(df_run, SAVE_PATH_FULL)
merged_eval, metrics_df, threshold_df = summarize_full_results(results_full, df)
print("\nDone.")
