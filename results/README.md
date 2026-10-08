# ArthBodh evaluation results

Test set 1: 1600 auto-built sentences (IITB parallel corpus). Test set 2: 20 hand-labelled Hindi sentences.

| Variant | Test set 1: auto-built (%) | Test set 2: hand-labelled Hindi (%) |
|---|---|---|
| Baseline: first sense | 38.8 | 50.0 |
| Simplified Lesk | 43.4 | 80.0 |
| Embedding Lesk, Hindi mode | 56.9 | 95.0 |
| Embedding Lesk, English mode | 61.4 | not applicable |
| Simplified Lesk (no POS filter) | 40.5 | 85.0 |
| Embedding Lesk, Hindi mode (no POS filter) | 55.3 | 95.0 |

## Details

| test_set   | variant                                    |    n |   accuracy |   top2_accuracy |   gold_in_candidates |   acc_high_conf |   acc_low_conf |
|:-----------|:-------------------------------------------|-----:|-----------:|----------------:|---------------------:|----------------:|---------------:|
| auto       | Baseline: first sense                      | 1600 |       38.8 |            58.8 |                 94.4 |           nan   |          nan   |
| auto       | Simplified Lesk                            | 1600 |       43.4 |            63.9 |                 94.4 |            49.5 |           40.4 |
| auto       | Embedding Lesk, Hindi mode                 | 1600 |       56.9 |            74.9 |                 94.4 |            68.1 |           45.9 |
| auto       | Embedding Lesk, English mode               | 1600 |       61.4 |            77.8 |                 94.4 |            73.5 |           49.5 |
| auto       | Simplified Lesk (no POS filter)            | 1600 |       40.5 |            58.8 |                 94.8 |            47.6 |           37   |
| auto       | Embedding Lesk, Hindi mode (no POS filter) | 1600 |       55.3 |            73.6 |                 94.8 |            66.5 |           44.1 |
| manual     | Baseline: first sense                      |   20 |       50   |            80   |                100   |           nan   |          nan   |
| manual     | Simplified Lesk                            |   20 |       80   |            90   |                100   |           100   |           77.8 |
| manual     | Embedding Lesk, Hindi mode                 |   20 |       95   |           100   |                100   |           100   |           90   |
| manual     | Simplified Lesk (no POS filter)            |   20 |       85   |            90   |                100   |           100   |           83.3 |
| manual     | Embedding Lesk, Hindi mode (no POS filter) |   20 |       95   |           100   |                100   |           100   |           90   |

## English weight sweep (Embedding Lesk, English mode, test set 1)

|   w_en |   accuracy |   top2_accuracy |
|-------:|-----------:|----------------:|
|    0.3 |       60.7 |            77   |
|    0.5 |       61.4 |            77.8 |
|    0.7 |       61.4 |            78.1 |

Notes: English mode on test set 1 looks strong because the English side contains the answer keyword; Hindi mode and test set 2 are the honest measure. `gold_in_candidates` is how often the gold sense survives lookup + POS filter (the upper bound for any method). Confidence split is at the median score gap.

Charts: `accuracy.png`, `per_word.png`.