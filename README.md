# ArthBodh · अर्थबोध

**Context-Aware Meaning Finder for Hindi Words using Knowledge-Based Word Sense Disambiguation**

Type any English or Hindi sentence, click any Hindi word, and ArthBodh shows the one meaning of
that word that fits the sentence. It uses IndoWordNet as its dictionary, so nothing is trained per
word. Two methods are used: **Simplified Lesk** (baseline) and **Embedding Lesk** (main), which averages
two pretrained sentence encoders, LaBSE and L3Cube HindSBERT, and prefers common senses when scores are close.

Example: in *"बाज़ार में ताज़े फल बिक रहे हैं।"* the word फल means **fruit**; in
*"किसानों को उनकी मेहनत का अच्छा फल मिला।"* it means **result**.

The full design spec is in `Hindi Word Sense Disambiguation – NLP Mini Project.pdf`.

## Quick start (new laptop)

If you use Claude Code, `CLAUDE.md` holds the project status and next steps.

Requires Python 3.10–3.12 (3.13+ may not have torch/stanza wheels yet) and ~4 GB free disk.

```bash
git clone https://github.com/RudraPandit0504/arthbodh.git
cd arthbodh
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py             # one time, needs good Wi-Fi: Stanza Hindi, IndoWordNet, LaBSE + HindSBERT (~3 GB)
streamlit run app.py
```

With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`.

The first app load takes ~30 s while models load; after that each word is answered in about a second.
English mode and the English meanings need internet (Google Translate, with MyMemory as fallback).
**Hindi mode works fully offline** once `setup_models.py` has run, which makes it the safe choice if the network fails during the demo.

## Architecture

```
English sentence ──► translate.py (EN → HI) ──┐
                                              ▼
Hindi sentence ─────────────────────► nlp.py (Stanza: tokens, lemma, POS)
                                              ▼
                                    user picks a word (app.py)
                                              ▼
                     wordnet.py (IndoWordNet senses, filtered by POS)
                                              ▼
                     0 senses: not found · 1 sense: only meaning · 2+ senses:
                     ├─ lesk.py        Simplified Lesk (word overlap, baseline)
                     └─ embed_lesk.py  Embedding Lesk (LaBSE + HindSBERT, sense-order prior; + English sentence in EN mode)
                                              ▼
                  engine.py → meaning (Hindi + English), confidence, other meanings, comparison
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI: mode switch, word picker, result, "Compare methods" |
| `src/translate.py` | English → Hindi sentence, Hindi gloss → English |
| `src/nlp.py` | Stanza tokenization, lemmatization, POS tagging |
| `src/wordnet.py` | IndoWordNet lookup, POS filter, lemma → surface → nukta-free → suffix-stripped fallbacks |
| `src/lesk.py` | Simplified Lesk (our implementation) |
| `src/embed_lesk.py` | Embedding Lesk: LaBSE + HindSBERT scores, sense-order prior, cross-lingual context, cached gloss vectors |
| `src/engine.py` | `analyse(hi_sentence, word, en_sentence)`, the full pipeline for one word |
| `src/build_testset.py` | Builds test set 1 from the IIT Bombay English–Hindi parallel corpus |
| `src/collect_sentences.py` | Collects real Hindi Wikipedia sentences as unlabelled candidates for test set 2 |
| `src/label_tool.py` | CLI for the human labelling: blind review by two labellers, merge + Cohen's kappa, auto-label hand-check |
| `src/error_analysis.py` | Prints wrong answers with gold and predicted glosses |
| `experiments/` | Model selection scripts for Embedding Lesk (`results/model_selection.md`) |
| `src/evaluate.py` | Accuracy of all methods on both test sets, with charts in `results/` |
| `data/sense_keywords.json` | Sense → English keywords for ~30 ambiguous words |
| `data/stopwords_hi.txt` | Hindi stop words removed before Simplified Lesk |

**Confidence** is the gap between the best and second-best Embedding Lesk score. A gap below 0.02 is shown as "unsure" (the lowest quarter of answers, which are right only about half the time).

## Evaluation

```bash
python -m src.build_testset          # downloads the IITB corpus (~1.6 M pairs) and writes data/test_auto.csv
python -m src.evaluate               # writes results/results.csv, results/*.png (--limit N for a quick run)
python -m src.evaluate --charts-only # redraw charts from the saved CSVs
python -m src.label_tool             # add hand-labelled rows to data/test_manual.csv
```

### Test set 2: what the team needs to do

`data/test_manual_draft.csv` holds 199 real sentences from Hindi Wikipedia (39 ambiguous words), collected with
`python -m src.collect_sentences`. They are **not labelled**. Two team members label them independently:

```bash
python -m src.label_tool --review data/test_manual_draft.csv --labeller a   # member 1 (resumable, q to stop)
python -m src.label_tool --review data/test_manual_draft.csv --labeller b   # member 2, on their own
python -m src.label_tool --merge data/test_manual_draft.csv                 # agreement %, Cohen's kappa, disagreements
python -m src.label_tool --review data/test_manual_draft.csv --labeller final  # settle the disagreements together
python -m src.label_tool --merge data/test_manual_draft.csv                 # copies the final labels to test_manual.csv
python -m src.evaluate                                                      # test set 2 results appear
python -m src.label_tool --handcheck                                        # 100-row check of test set 1's auto labels
```

The 20 `source=seed` rows in `data/test_manual.csv` are demo sentences written for testing; `evaluate.py` leaves them
out unless `--with-seeds` is given.

Gold labels may list several synset ids separated by `|`. IndoWordNet often stores the same meaning
twice (e.g. कल "yesterday" as both noun 617 and adverb 22207), and a prediction counts as correct if it matches any of them.

Current results on test set 1 (1,600 auto-labelled sentences, 32 ambiguous words):

| Variant | Accuracy (%) |
|---|---|
| Baseline: first listed sense | 38.8 |
| Simplified Lesk | 43.4 |
| Embedding Lesk, Hindi mode | **65.6** |
| Embedding Lesk, English mode | **69.7** |
| Embedding Lesk, Hindi mode, LaBSE only (original method) | 56.9 |
| Embedding Lesk, English mode, LaBSE only (original method) | 61.4 |

Test set 2 results are pending until the team has labelled the draft (see above).
Full tables and charts: `results/README.md`, `results/*.png`. How the configuration was chosen: `results/model_selection.md`.
Error analysis of 10 wrong answers: `results/error_analysis.md` (`python -m src.error_analysis` to browse more).

## Smoke test

```bash
python -m src.engine
```

## Tech stack

pyiwn (IndoWordNet) · Stanza · sentence-transformers (LaBSE, L3Cube HindSBERT) · deep-translator · pandas · scikit-learn · matplotlib · Streamlit

## References

- Lesk (1986). Automatic sense disambiguation using machine readable dictionaries.
- Jurafsky & Martin, *Speech and Language Processing*: Word Senses and WordNet.
- Yusuf et al. (2022). HindiWSD: A package for word sense disambiguation in Hinglish & Hindi. WILDRE.
- Kunchukuttan et al. (2018). The IIT Bombay English-Hindi Parallel Corpus. LREC.
- IndoWordNet, CFILT, IIT Bombay; `pyiwn` Python interface.
- Feng et al. (2022). Language-agnostic BERT Sentence Embedding (LaBSE).
- Joshi et al. (2022). L3Cube-MahaSBERT and HindSBERT: Sentence BERT models and benchmarking BERT sentence representations for Hindi and Marathi.
- Qi et al. (2020). Stanza: A Python NLP toolkit for many human languages.
