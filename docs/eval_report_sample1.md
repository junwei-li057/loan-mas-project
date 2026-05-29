# Multi-Agent Loan-Default System — Evaluation Report

_generated_: `2026-05-28T14:50:09`
_data_: `sample1 (train/test)`
_model_artifact_: `models/loan_default_model.pkl`
_decision_log_: `synthetic (4-iter)`

## 1. Executive Summary

- **Discrimination**: Strategist AUC = 0.6726 vs baseline 0.6756 (Δ = -0.0030, DeLong p = 0.6747; not significant).
- **Agent contribution**: overall KPI = 0.258 over 4 loop iter(s), advocate veto rate = 0.250 (converged).
- **MAS vs grid search**: Δ test AUC = 0.0074 (MAS beats brute-force).
- **Trust calibration**: confidence tracks accuracy but fusion gains are flat at the top.

## 2. Pipeline Head-to-Head

Bootstrap 95% CIs on AUC and AUPRC. `delong_p_vs_ref` is the two-sided p-value vs the reference pipeline (NaN for the reference itself).

| pipeline | n | default_rate | auc | auc_ci | auprc | auprc_ci | log_loss | brier | delong_p_vs_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 60 | 0.2500 | 0.6756 | (0.5064, 0.8198) | 0.4334 | (0.2432, 0.6883) | 0.7481 | 0.2766 | — |
| strategist | 60 | 0.2500 | 0.6726 | (0.5044, 0.8109) | 0.4318 | (0.2416, 0.6650) | 0.7464 | 0.2759 | 0.6747 |

## 3. Per-Agent KPI (MultiAgentBench §3.3)

Iterations recorded: **4**. Milestones fired: **22** / 30.

| agent | n_hit | n_total | kpi |
| --- | --- | --- | --- |
| TextAnalyst | 5.0000 | 30 | 0.1667 |
| Strategist | 10.0000 | 30 | 0.3333 |
| Advocate | 7.0000 | 30 | 0.2333 |
| Green | 9.0000 | 30 | 0.3000 |
| overall_kpi | — | 30 | 0.2583 |

## 4. Loop Dynamics

- Iterations: **4**  
- Converged: **True**  
- Advocate veto rate: **0.250**  
- Advocate modes: `{'rule': 3, 'LLM': 1}`  
- Event counts: `{'strategy_update': 2, 'green_arbitrate': 1, 'converged': 1}`  
- Arbitration outcomes: `{'green_arbitrate': 1}`  

### Per-iter trace

| iter | event | veto | advocate_mode | overall_baseline_auc | overall_fused_auc | overall_baseline_auprc | overall_fused_auprc | advocate_opportunities | surprise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | strategy_update | False | rule | 0.7200 | 0.7300 | 0.4500 | 0.4700 | [grade_C_underweighted] | 0.0300 |
| 1 | green_arbitrate | True | LLM | 0.7200 | 0.7250 | 0.4500 | 0.4600 |  | 0.0800 |
| 2 | strategy_update | False | rule | 0.7200 | 0.7350 | 0.4500 | 0.4800 | [grade_D_boost] | 0.0200 |
| 3 | converged | False | rule | 0.7200 | 0.7350 | 0.4500 | 0.4800 |  | 0.0100 |

## 5. Ablation — Is MAS Worth Its Complexity?

Each row is an alternative method. `vs_baseline_*` columns are absolute differences (positive = better than the XGBoost-only baseline).

| method | test_auc | test_auprc | vs_baseline_auc | vs_baseline_auprc | note |
| --- | --- | --- | --- | --- | --- |
| baseline-only | 0.6756 | 0.4334 | 0.0000 | 0.0000 | XGBoost on numeric features |
| LR stacking | 0.5585 | 0.3318 | -0.1170 | -0.1016 | LogReg, 22 features |
| Grid search (best) | 0.6652 | 0.4299 | -0.0104 | -0.0035 | 180-combo brute force |
| MAS Strategist | 0.6726 | 0.4318 | -0.0030 | -0.0016 | Multi-agent LLM-tuned 5-scalar weights |

## 6. Trust Calibration

**Verdict**: confidence tracks accuracy but fusion gains are flat at the top  
- LLM-confidence-vs-accuracy slope: **0.9198** (positive ⇒ higher claimed confidence really is more reliable)  
- Top-coverage ΔAUPRC (fused − baseline at the most-trusted 0.35): **-0.0046**

### Confidence reliability bins

| index | conf_bin | conf_bin_lo | conf_bin_hi | n | mean_conf | mean_text_risk | default_rate | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [0.46, 0.52] | 0.4580 | 0.5195 | 8 | 0.4725 | 0.6381 | 0.2500 | 0.2500 |
| 1 | [0.52, 0.58] | 0.5195 | 0.5810 | 10 | 0.5468 | 0.4770 | 0.3000 | 0.5000 |
| 2 | [0.58, 0.64] | 0.5810 | 0.6425 | 17 | 0.6169 | 0.5831 | 0.1765 | 0.4706 |
| 3 | [0.64, 0.70] | 0.6425 | 0.7040 | 9 | 0.6856 | 0.4823 | 0.2222 | 0.3333 |
| 4 | [0.70, 0.77] | 0.7040 | 0.7655 | 9 | 0.7384 | 0.4430 | 0.3333 | 0.5556 |
| 5 | [0.77, 0.83] | 0.7655 | 0.8270 | 6 | 0.7903 | 0.3812 | 0.3333 | 0.6667 |

### Selective-fusion curve

| index | coverage | n_kept | baseline_auc | fused_auc | delta_auc | baseline_auprc | fused_auprc | delta_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3500 | 21 | 0.6020 | 0.5918 | -0.0102 | 0.4152 | 0.4106 | -0.0046 |
| 1 | 0.4000 | 24 | 0.6555 | 0.6471 | -0.0084 | 0.4090 | 0.4044 | -0.0046 |
| 2 | 0.4500 | 27 | 0.6143 | 0.6000 | -0.0143 | 0.3491 | 0.3342 | -0.0149 |
| 3 | 0.5000 | 30 | 0.6364 | 0.6250 | -0.0114 | 0.3748 | 0.3622 | -0.0126 |
| 4 | 0.5500 | 33 | 0.6550 | 0.6450 | -0.0100 | 0.3669 | 0.3543 | -0.0126 |
| 5 | 0.6000 | 36 | 0.6741 | 0.6652 | -0.0089 | 0.3615 | 0.3489 | -0.0126 |
| 6 | 0.6500 | 39 | 0.6855 | 0.6774 | -0.0081 | 0.3546 | 0.3422 | -0.0124 |
| 7 | 0.7000 | 42 | 0.7344 | 0.7281 | -0.0062 | 0.5033 | 0.4918 | -0.0116 |
| 8 | 0.7500 | 45 | 0.7514 | 0.7429 | -0.0086 | 0.5016 | 0.4886 | -0.0130 |
| 9 | 0.8000 | 48 | 0.7494 | 0.7396 | -0.0098 | 0.4968 | 0.4832 | -0.0136 |
| 10 | 0.8500 | 51 | 0.7222 | 0.7137 | -0.0085 | 0.4805 | 0.4680 | -0.0125 |
| 11 | 0.9000 | 54 | 0.7017 | 0.6942 | -0.0075 | 0.4533 | 0.4453 | -0.0080 |
| 12 | 0.9500 | 57 | 0.6993 | 0.6960 | -0.0033 | 0.4455 | 0.4436 | -0.0019 |
| 13 | 1.0000 | 60 | 0.6756 | 0.6726 | -0.0030 | 0.4334 | 0.4318 | -0.0016 |
