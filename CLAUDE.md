# ArthBodh: context for Claude Code

NLP mini project: **ArthBodh**, knowledge-based Hindi Word Sense Disambiguation (WSD).
The source of truth for scope, architecture, evaluation plan and viva prep is
`Hindi Word Sense Disambiguation – NLP Mini Project.pdf` (18 pages). Read it before
proposing design changes. The README covers setup and usage.

## Setting up on a fresh machine

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12 .venv
pip install -r requirements.txt                          # or: uv pip install --python .venv/bin/python -r requirements.txt
python setup_models.py    # Stanza hi + IndoWordNet (~/iwn_data) + LaBSE (~2 GB HF cache)
python -m src.engine      # smoke test: फल -> 2035 (result) and 662 (fruit)
streamlit run app.py
```

- Use Python 3.10–3.12. Python 3.13+ may lack torch/stanza wheels.
- Always run modules as `python -m src.<name>` from the repo root, because they use relative imports.
- Not in git and recreated on demand: `.venv/`, `data/raw/` (the IITB corpus parquet, ~180 MB, which
  `build_testset.py` re-downloads only if you rebuild test set 1), plus the model caches in the home directory.
- No GPU is required. LaBSE uses CUDA if present, otherwise CPU (the full evaluation then takes ~10–20 min).

## Code map

`src/translate.py` (Google, then MyMemory fallback; returns None on failure) → `src/nlp.py`
(Stanza, cached) → `src/wordnet.py` (pyiwn; `lookup()` tries lemma → surface → nukta-free → suffix-stripped;
POS filter falls back to all senses) → `src/lesk.py` (Simplified Lesk) / `src/embed_lesk.py`
(LaBSE, gloss vectors cached per synset id) → `src/engine.py` `analyse()` → `app.py` (Streamlit).
Evaluation: `src/build_testset.py`, `src/evaluate.py` (`--limit N`, `--charts-only`), `src/label_tool.py`.

Facts that aren't obvious from the code:
- In pyiwn, `synset.pos()` returns a plain string (`'noun'`, `'verb'`, `'adjective'`, `'adverb'`).
- IndoWordNet duplicates meanings across synsets (e.g. कल "yesterday" = 617 noun + 22207 adverb).
  Gold labels therefore allow `|`-separated id groups, and a prediction is correct if it is in the group.
  `data/sense_keywords.json` keys use the same `id|id` format.
- Google Translate's free endpoint rate-limits easily (`TooManyRequests`), which is why the MyMemory fallback exists.
- Hindi mode works offline once models are downloaded; English mode needs internet.
- Matplotlib Devanagari fonts have no Latin glyphs, so charts use `["DejaVu Sans", <devanagari font>]`.

## Status (as of 2026-10-08)

Done (PDF steps 1–9):
- Engine, both methods, Streamlit app (tested via `streamlit.testing.v1.AppTest` in both modes).
- Test set 1: `data/test_auto.csv`, 1,600 rows, 32 words × 50, senses balanced round-robin.
- `data/test_auto_handcheck.csv`: 100 random rows for the human auto-label check.
- Evaluation results in `results/` (see `results/README.md`).
  Test set 1: first sense 37.9 · Simplified Lesk 43.1 · Embedding Lesk HI 55.6 · EN 60.1 (`w_en` 0.5 is best).

To do (needs humans or is still open):
1. **Test set 2**: `data/test_manual.csv` holds only 20 *seed* rows (`source=seed`) that Claude
   wrote for demo and testing. They are not from news or Wikipedia. The team must hand-label 150–200 real
   sentences with `python -m src.label_tool` (two labellers, resolve disagreements), then decide whether
   to drop the seed rows, and re-run `python -m src.evaluate`. The 95% on test set 2 is from the seeds only
   and must not be reported as a real result.
2. **Auto-label check**: fill a `label_ok` column (1/0) in `data/test_auto_handcheck.csv` and report the share that is correct.
3. **Error analysis** (PDF §8): pick 10 wrong answers from `results/predictions.csv` and explain each.
   Weak words to study: मान, जाल, चाल, मूल, काल, कर (see `results/per_word.png`).
4. Report and presentation (PDF Step 10). Possible extensions are in PDF §10 (Extended Lesk with hypernyms, Marathi).

## Conventions

- Keep the code simple and explainable for the viva. The PDF's code is the reference style.
- Never invent synset ids. Verify them with `src.wordnet._all_senses(word)` before writing labels or keywords.
- Commit messages end with the Claude co-author line. The repo is private: github.com/RudraPandit0504/arthbodh.
