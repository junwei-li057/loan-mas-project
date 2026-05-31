# Green Arbitration Ablation

This is a lightweight arbitration-layer comparison, not a full XGBoost replay.
It compares two policy proxies on a joined LendingClub/text-analysis sample:

- **Old proxy, separate Arbitrator:** high-confidence, nontrivial text signals are allowed to act broadly.
- **New Green-as-Arbitrator:** the same signals act only when grade-level text history supports them, or when borrower-level evidence is extreme enough for an exception.

## Data

- Loan CSV: `loan_default_sample2.csv`
- Text analysis: `text_ana_results/text_analysis_combined.pkl`
- Joined rows: `7000`
- Confidence threshold: `0.65`
- Text-gap threshold: `0.10`
- Grade support threshold: text AUC >= `0.51`

## Result Summary

| design                        | acted_n | acted_share | action_accuracy | default_precision_when_text_says_default | nondefault_precision_when_text_says_safe |
| ----------------------------- | ------- | ----------- | --------------- | ---------------------------------------- | ---------------------------------------- |
| old_proxy_separate_arbitrator | 578     | 0.0826      | 0.7474          | 0.2549                                   | 0.7951                                   |
| new_green_as_arbitrator       | 291     | 0.0416      | 0.7423          | 0.2917                                   | 0.7828                                   |

## Event Counts

| event                       | count |
| --------------------------- | ----- |
| old_design_acted            | 578   |
| new_green_acted             | 291   |
| green_suppressed_old_action | 287   |
| old_wrong_new_suppressed    | 71    |
| old_right_new_suppressed    | 216   |
| extreme_exception_allowed   | 0     |

## Grade Text Reliability

| grade | n    | default_rate | text_auc | high_conf_n | high_conf_text_direction_accuracy |
| ----- | ---- | ------------ | -------- | ----------- | --------------------------------- |
| C     | 3548 | 0.1807       | 0.5066   | 3436        | 0.4144                            |
| D     | 2020 | 0.2228       | 0.5230   | 1947        | 0.4474                            |
| E     | 922  | 0.2701       | 0.5335   | 888         | 0.4696                            |
| F     | 399  | 0.3860       | 0.4945   | 385         | 0.4779                            |
| G     | 111  | 0.4054       | 0.5640   | 109         | 0.5229                            |

## Interpretation

Green-as-Arbitrator is a selective-trust mechanism, not a universal metric booster.
In this run it changed action coverage from `578` rows to `291` rows.
Action accuracy changed by `-0.0051`, while precision when text says default changed by `+0.0368`.

The key tradeoff is visible in the case files:

- `cases_new_green_better.csv`: old proxy would trust text and be wrong; Green suppresses it.
- `cases_old_proxy_better.csv`: old proxy would trust text and be right; Green suppresses it.

Presentation-safe takeaway:

> Green arbitration trades coverage for selective trust. Its job is not to let text influence every confident case; its job is to decide when text is reliable enough to deserve influence.
