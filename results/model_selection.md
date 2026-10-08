# Choosing the Embedding Lesk configuration

Data: test set 1, split into **dev** (even rows, 800) for choosing and **held-out** (odd rows, 800) for checking.
Hindi mode (Hindi sentence only). Scripts: `experiments/model_selection.py`, `experiments/compare_configs.py`.

## What we tried

- **Encoders**: LaBSE (original), paraphrase-multilingual-mpnet-base-v2, multilingual-e5-base,
  L3Cube HindSBERT (`l3cube-pune/hindi-sentence-similarity-sbert`), BGE-M3.
- **Sense text**: `gloss` = gloss + examples (original); `syn` = synonyms + gloss + examples;
  `ext` = syn + hypernym gloss + hyponym words (Extended Lesk).
- **Sense-order prior**: subtract `beta × i / n` from the score of sense *i* of *n*, because IndoWordNet lists
  common senses first.

## Single encoders (dev accuracy, best beta on dev → held-out)

| Encoder | gloss | syn | ext |
|---|---|---|---|
| LaBSE | 53.6 → 60.5 (beta .2), held 58.8 | 51.5 → 57.2, held 58.5 | 48.9 → 56.0, held 53.1 |
| multilingual-mpnet | 54.0 → 59.9, held 60.4 | 52.5 → 58.1, held 63.1 | 56.0 → 59.8, held 64.4 |
| multilingual-e5-base | 45.4 → 52.2, held 54.0 | 41.1 → 52.2, held 51.8 | 48.5 → 56.5, held 51.6 |
| **HindSBERT** | 55.6 → 59.0, held 63.7 | **55.4 → 60.8, held 64.9** | 55.6 → 62.0, held 63.6 |
| BGE-M3 | 49.5 → 57.9, held 54.0 | 46.8 → 58.5, held 57.5 | 53.9 → 60.9, held 60.1 |

(First number: beta 0 on dev; second: best beta on dev.)

## Combinations (full set, dev, held-out; paired sign test vs LaBSE only)

| Configuration | All | Dev | Held-out | Net gain | p |
|---|---|---|---|---|---|
| LaBSE gloss (original) | 57.0 | 53.6 | 60.4 | 0 | |
| LaBSE gloss + prior 0.1 | 61.0 | 59.4 | 62.6 | +64 | <0.001 |
| HindSBERT syn + prior 0.2 | 62.8 | 60.8 | 64.9 | +93 | <0.001 |
| LaBSE gloss + HindSBERT syn, no prior | 64.2 | 59.8 | 68.6 | +115 | <0.001 |
| **LaBSE gloss + HindSBERT syn + prior 0.1** | **65.8** | **64.0** | **67.5** | **+140** | <0.001 |
| LaBSE gloss + HindSBERT ext + prior 0.1 | 66.2 | 64.1 | 68.4 | +147 | <0.001 |

The chosen configuration (bold) is the best on dev among the simpler options; the Extended Lesk version is
only 0.1 better on dev and needs extra relation lookups, so we kept the simpler one. In the full evaluation
(`results/README.md`) the chosen method scores **65.6%** in Hindi mode and **69.7%** in English mode (w_en 0.5),
versus 56.9% and 61.4% for LaBSE only.

## Lessons

- The **sense-order prior** helps every encoder: IndoWordNet's order carries frequency information.
- **Synonyms and Extended Lesk** hurt LaBSE but help HindSBERT and mpnet: LaBSE was trained for translation
  matching, not for comparing a sentence with a list of words.
- **Combining two different encoders** helps most: their errors are partly independent.
- Caveat: test set 1 is auto-labelled and noisy (see `error_analysis.md`); test set 2 is the honest check.
