# ArthBodh: context for Claude Code

NLP mini project: **ArthBodh**, knowledge-based Hindi Word Sense Disambiguation (WSD).
The source of truth for scope, architecture, evaluation plan and viva prep is
`Hindi Word Sense Disambiguation – NLP Mini Project.pdf` (18 pages). Read it before
proposing design changes. The README covers setup and usage.

## Setting up on a fresh machine

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12 .venv
pip install -r requirements.txt                          # or: uv pip install --python .venv/bin/python -r requirements.txt
python setup_models.py    # Stanza hi + IndoWordNet (~/iwn_data) + LaBSE + HindSBERT (~3 GB HF cache)
python -m src.engine      # smoke test: फल -> 2035 (result) and 662 (fruit)
streamlit run app.py
```

- Use Python 3.10–3.12. Python 3.13+ may lack torch/stanza wheels.
- Always run modules as `python -m src.<name>` from the repo root, because they use relative imports.
- Not in git and recreated on demand: `.venv/`, `data/raw/` (the IITB corpus parquet, ~180 MB, which
  `build_testset.py` re-downloads only if you rebuild test set 1), plus the model caches in the home directory.
- No GPU is required. LaBSE uses CUDA if present (full evaluation ~5 min on a GTX 1650 Ti), otherwise CPU (~10–20 min).

## Code map

`src/translate.py` (Google, then MyMemory fallback; returns None on failure) → `src/nlp.py`
(Stanza, cached) → `src/wordnet.py` (pyiwn; `lookup()` tries lemma → surface → nukta-free → suffix-stripped, preferring the first form with a sense of the tagged POS;
POS filter falls back to all senses) → `src/lesk.py` (Simplified Lesk) / `src/embed_lesk.py`
(LaBSE + HindSBERT averaged, sense-order prior 0.1, gloss vectors cached per (model, synset id); fp16 on GPU) → `src/engine.py` `analyse()` → `app.py` (Streamlit).
Evaluation: `src/build_testset.py`, `src/evaluate.py` (`--limit N`, `--charts-only`), `src/label_tool.py` (`--review/--merge/--handcheck`), `src/error_analysis.py`, `src/collect_sentences.py`.
Model selection: `experiments/` → `results/model_selection.md`.

Facts that aren't obvious from the code:
- In pyiwn, `synset.pos()` returns a plain string (`'noun'`, `'verb'`, `'adjective'`, `'adverb'`).
- IndoWordNet duplicates meanings across synsets (e.g. कल "yesterday" = 617 noun + 22207 adverb).
  Gold labels therefore allow `|`-separated id groups, and a prediction is correct if it is in the group.
  `data/sense_keywords.json` keys use the same `id|id` format.
- Google Translate's free endpoint rate-limits easily (`TooManyRequests`), which is why the MyMemory fallback exists.
- Hindi mode works offline once models are downloaded; English mode needs internet.
- Matplotlib Devanagari fonts have no Latin glyphs, so charts use `["DejaVu Sans", <devanagari font>]`.
- The 4 GB GTX 1650 Ti runs out of memory with both encoders in fp32, hence `model.half()` on CUDA.
- hi.wikipedia rate-limits (HTTP 429); `collect_sentences.py` retries with Retry-After and sleeps 1 s per request.
- The unsure threshold `UNSURE_GAP = 0.02` (engine.py) was calibrated on test set 1: below it, answers are ~49% right.

## Status (as of 2026-10-08)

Done (PDF steps 1–9):
- Engine, both methods, Streamlit app (tested via `streamlit.testing.v1.AppTest` in both modes).
- Test set 1: `data/test_auto.csv`, 1,600 rows, 32 words × 50, senses balanced round-robin.
- `data/test_auto_handcheck.csv`: 100 random rows for the human auto-label check.
- Evaluation results in `results/` (see `results/README.md`).
  Test set 1: first sense 38.8 · Simplified Lesk 43.4 · Embedding Lesk HI 65.6 · EN 69.7
  (LaBSE-only original: 56.9 / 61.4, kept as comparison rows). `w_en` 0.7 scores 70.6 but stays 0.5,
  since English mode on test set 1 is inflated by the answer keyword anyway.
- Accuracy improvement: LaBSE + HindSBERT (synonyms in sense text) + sense-order prior, chosen on even rows of
  test set 1 and confirmed on odd rows (`results/model_selection.md`).
- Error analysis (PDF §8): `results/error_analysis.md`, 10 explained cases + cause table; browse with `python -m src.error_analysis`.
  It led to a POS-aware `lookup()` fix (Stanza lemmatizes the noun मान as the verb मानना): मान 18% → 60%.
  It also showed test set 1 label noise: most कर and मूल "errors" are auto-label mistakes ("hand", "root cause").

To do (needs humans or is still open):
1. **Test set 2 labelling (humans only)**: `data/test_manual_draft.csv` has 199 real hi.wikipedia sentences,
   unlabelled. Two members run `python -m src.label_tool --review data/test_manual_draft.csv --labeller a` / `b`,
   then `--merge`, settle disagreements with `--labeller final`, `--merge` again, and `python -m src.evaluate`.
   Never fill labeller columns on the team's behalf. The 20 `source=seed` rows in test_manual.csv were written by
   Claude for demos; `evaluate.py` excludes them unless `--with-seeds`.
2. **Auto-label check (humans only)**: `python -m src.label_tool --handcheck` fills `label_ok` in
   `data/test_auto_handcheck.csv`. Error analysis suggests कर/मूल labels are often wrong.
3. Report and presentation (PDF Step 10). Possible extensions are in PDF §10 (Extended Lesk with hypernyms, Marathi).

## Conventions

- Keep the code simple and explainable for the viva. The PDF's code is the reference style.
- Never invent synset ids. Verify them with `src.wordnet._all_senses(word)` before writing labels or keywords.
- Commit as `RudraPandit0504 <rudra.pandit0504@gmail.com>`; messages end with the Claude co-author line.
  Repo: github.com/RudraPandit0504/arthbodh (public on GitHub).
