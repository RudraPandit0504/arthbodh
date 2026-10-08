# ArthBodh evaluation results

Test set 1: 1600 auto-built sentences (IITB parallel corpus). Test set 2: 0 hand-labelled Hindi sentences.

| Variant | Test set 1: auto-built (%) | Test set 2: hand-labelled Hindi (%) |
|---|---|---|
| Baseline: first sense | 38.8 | pending |
| Simplified Lesk | 43.4 | pending |
| Embedding Lesk, Hindi mode | 65.6 | pending |
| Embedding Lesk, English mode | 69.7 | pending |
| Simplified Lesk (no POS filter) | 40.5 | pending |
| Embedding Lesk, Hindi mode (no POS filter) | 63.7 | pending |
| Embedding Lesk, Hindi mode (LaBSE only, no prior) | 56.9 | pending |
| Embedding Lesk, English mode (LaBSE only, no prior) | 61.4 | pending |

## Details

| test_set   | variant                                             |    n |   accuracy |   top2_accuracy |   gold_in_candidates |   acc_high_conf |   acc_low_conf |
|:-----------|:----------------------------------------------------|-----:|-----------:|----------------:|---------------------:|----------------:|---------------:|
| auto       | Baseline: first sense                               | 1600 |       38.8 |            58.8 |                 94.4 |           nan   |          nan   |
| auto       | Simplified Lesk                                     | 1600 |       43.4 |            63.9 |                 94.4 |            49.5 |           40.4 |
| auto       | Embedding Lesk, Hindi mode                          | 1600 |       65.6 |            80.8 |                 94.4 |            78.1 |           53.2 |
| auto       | Embedding Lesk, English mode                        | 1600 |       69.7 |            83.9 |                 94.4 |            82.7 |           56.8 |
| auto       | Simplified Lesk (no POS filter)                     | 1600 |       40.5 |            58.8 |                 94.8 |            47.6 |           37   |
| auto       | Embedding Lesk, Hindi mode (no POS filter)          | 1600 |       63.7 |            79.2 |                 94.8 |            75.1 |           52.2 |
| auto       | Embedding Lesk, Hindi mode (LaBSE only, no prior)   | 1600 |       56.9 |            74.9 |                 94.4 |            68.1 |           45.9 |
| auto       | Embedding Lesk, English mode (LaBSE only, no prior) | 1600 |       61.4 |            77.7 |                 94.4 |            73.5 |           49.5 |

## English weight sweep (Embedding Lesk, English mode, test set 1)

|   w_en |   accuracy |   top2_accuracy |
|-------:|-----------:|----------------:|
|    0.3 |       67.3 |            82.9 |
|    0.5 |       69.7 |            83.9 |
|    0.7 |       70.6 |            84.2 |

Notes: English mode on test set 1 looks strong because the English side contains the answer keyword; Hindi mode and test set 2 are the honest measure. `gold_in_candidates` is how often the gold sense survives lookup + POS filter (the upper bound for any method). Confidence split is at the median score gap.

Embedding Lesk averages LaBSE and L3Cube HindSBERT scores and adds a small sense-order prior (src/embed_lesk.py). Models and prior were chosen on the even rows of test set 1 and checked on the odd rows; the 'LaBSE only, no prior' rows are the original method.

Charts: `accuracy.png`, `per_word.png`.