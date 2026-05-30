# Multi-Agent Loan-Default System — Evaluation Report

_generated_: `2026-05-30T15:57:37`
_data_: `combined.pkl (16801 train, 4199 test; id-joined to loan_default.csv)`
_model_artifact_: `models/loan_default_model.pkl`
_decision_log_: `real (2-iter, from decision_log_sample1.pkl)`

## 1. Executive Summary

- **Discrimination**: Strategist AUC = 0.6147 vs baseline 0.6138 (Δ = 0.0009, DeLong p = 0.0895; not significant).
- **Agent contribution**: overall KPI = 0.303 over 2 loop iter(s), advocate veto rate = 0.000 (converged).
- **MAS vs grid search**: Δ test AUC = 0.0006 (MAS beats brute-force).
- **Trust calibration**: confidence is a real trust signal; selective fusion helps.

## 2. Pipeline Head-to-Head

Bootstrap 95% CIs on AUC and AUPRC. `delong_p_vs_ref` is the two-sided p-value vs the reference pipeline (NaN for the reference itself).

| pipeline | n | default_rate | auc | auc_ci | auprc | auprc_ci | log_loss | brier | delong_p_vs_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 4199 | 0.2189 | 0.6138 | (0.5961, 0.6359) | 0.3001 | (0.2739, 0.3281) | 0.7812 | 0.2904 | — |
| strategist | 4199 | 0.2189 | 0.6147 | (0.5974, 0.6362) | 0.3015 | (0.2755, 0.3296) | 0.7801 | 0.2899 | 0.0895 |

## 3. Per-Agent KPI (MultiAgentBench §3.3)

Iterations recorded: **2**. Milestones fired: **17** / 19.

| agent | n_hit | n_total | kpi |
| --- | --- | --- | --- |
| TextAnalyst | 6.0000 | 19 | 0.3158 |
| Strategist | 6.0000 | 19 | 0.3158 |
| Advocate | 5.0000 | 19 | 0.2632 |
| Green | 6.0000 | 19 | 0.3158 |
| overall_kpi | — | 19 | 0.3026 |

## 4. Loop Dynamics

- Iterations: **2**  
- Converged: **True**  
- Advocate veto rate: **0.000**  
- Advocate modes: `{'rule': 1, 'LLM': 1}`  
- Event counts: `{'strategy_update': 1, 'converged': 1}`  
- Arbitration outcomes: `{}`  

### Per-iter trace

| iter | event | veto | advocate_mode | overall_baseline_auc | overall_fused_auc | overall_baseline_auprc | overall_fused_auprc | advocate_opportunities | surprise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | strategy_update | False | rule | 0.7408 | 0.7401 | 0.4561 | 0.4553 |  | 0.0165 |
| 1 | converged | False | LLM | 0.7408 | 0.7404 | 0.4561 | 0.4555 | [Grade E: Consistently stable or positive deltas (AUPRC and AUC) with high recall., Grade F: High precision and recall, though sample size is small.] | 0.0016 |

## 5. Ablation — Is MAS Worth Its Complexity?

Each row is an alternative method. `vs_baseline_*` columns are absolute differences (positive = better than the XGBoost-only baseline).

| method | test_auc | test_auprc | vs_baseline_auc | vs_baseline_auprc | note |
| --- | --- | --- | --- | --- | --- |
| baseline-only | 0.6138 | 0.3001 | 0.0000 | 0.0000 | XGBoost on numeric features |
| LR stacking | 0.6094 | 0.3080 | -0.0045 | 0.0078 | LogReg, 22 features |
| Grid search (best) | 0.6142 | 0.3032 | 0.0003 | 0.0031 | 1620-combo brute force |
| MAS Strategist | 0.6147 | 0.3015 | 0.0009 | 0.0014 | Multi-agent LLM-tuned 5-scalar weights |

## 6. Trust Calibration

**Verdict**: confidence is a real trust signal; selective fusion helps  
- LLM-confidence-vs-accuracy slope: **1.4027** (positive ⇒ higher claimed confidence really is more reliable)  
- Top-coverage ΔAUPRC (fused − baseline at the most-trusted 0.05): **0.0129**

### Confidence reliability bins

| index | conf_bin | conf_bin_lo | conf_bin_hi | n | mean_conf | mean_text_risk | default_rate | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [0.40, 0.46] | 0.4000 | 0.4563 | 39 | 0.4021 | 0.5526 | 0.1026 | 0.1026 |
| 1 | [0.46, 0.51] | 0.4563 | 0.5125 | 29 | 0.5000 | 0.5534 | 0.1724 | 0.1724 |
| 2 | [0.51, 0.57] | 0.5125 | 0.5687 | 108 | 0.5500 | 0.5537 | 0.2222 | 0.2222 |
| 3 | [0.62, 0.68] | 0.6250 | 0.6813 | 2529 | 0.6500 | 0.5461 | 0.2357 | 0.2985 |
| 4 | [0.74, 0.79] | 0.7375 | 0.7937 | 965 | 0.7500 | 0.4872 | 0.1927 | 0.6062 |
| 5 | [0.79, 0.85] | 0.7937 | 0.8500 | 529 | 0.8500 | 0.3065 | 0.1966 | 0.6843 |

### Selective-fusion curve

| index | coverage | n_kept | baseline_auc | fused_auc | delta_auc | baseline_auprc | fused_auprc | delta_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0500 | 210 | 0.6109 | 0.6235 | 0.0126 | 0.2466 | 0.2595 | 0.0129 |
| 1 | 0.1000 | 420 | 0.5776 | 0.5824 | 0.0048 | 0.2405 | 0.2443 | 0.0038 |
| 2 | 0.1500 | 630 | 0.6010 | 0.6046 | 0.0037 | 0.2666 | 0.2720 | 0.0054 |
| 3 | 0.2000 | 840 | 0.6145 | 0.6171 | 0.0026 | 0.2785 | 0.2811 | 0.0026 |
| 4 | 0.2500 | 1050 | 0.6117 | 0.6133 | 0.0016 | 0.2838 | 0.2858 | 0.0021 |
| 5 | 0.3000 | 1260 | 0.6094 | 0.6108 | 0.0015 | 0.2667 | 0.2683 | 0.0016 |
| 6 | 0.3500 | 1470 | 0.6161 | 0.6170 | 0.0009 | 0.2778 | 0.2798 | 0.0020 |
| 7 | 0.4000 | 1680 | 0.6201 | 0.6211 | 0.0010 | 0.2836 | 0.2855 | 0.0018 |
| 8 | 0.4500 | 1890 | 0.6114 | 0.6126 | 0.0012 | 0.2803 | 0.2817 | 0.0014 |
| 9 | 0.5000 | 2099 | 0.6147 | 0.6158 | 0.0011 | 0.2878 | 0.2892 | 0.0014 |
| 10 | 0.5500 | 2309 | 0.6188 | 0.6201 | 0.0013 | 0.2940 | 0.2955 | 0.0015 |
| 11 | 0.6000 | 2519 | 0.6180 | 0.6193 | 0.0013 | 0.2948 | 0.2960 | 0.0012 |
| 12 | 0.6500 | 2729 | 0.6177 | 0.6190 | 0.0012 | 0.2971 | 0.2986 | 0.0015 |
| 13 | 0.7000 | 2939 | 0.6191 | 0.6202 | 0.0011 | 0.2990 | 0.3004 | 0.0014 |
| 14 | 0.7500 | 3149 | 0.6153 | 0.6165 | 0.0012 | 0.2972 | 0.2986 | 0.0014 |
| 15 | 0.8000 | 3359 | 0.6153 | 0.6165 | 0.0011 | 0.2958 | 0.2970 | 0.0012 |
| 16 | 0.8500 | 3569 | 0.6159 | 0.6169 | 0.0011 | 0.2991 | 0.3006 | 0.0015 |
| 17 | 0.9000 | 3779 | 0.6152 | 0.6163 | 0.0010 | 0.3019 | 0.3034 | 0.0015 |
| 18 | 0.9500 | 3989 | 0.6146 | 0.6156 | 0.0010 | 0.3034 | 0.3048 | 0.0015 |
| 19 | 1.0000 | 4199 | 0.6138 | 0.6147 | 0.0009 | 0.3001 | 0.3015 | 0.0014 |
