# ArthBodh · अर्थबोध

**Context-Aware Meaning Finder for Hindi Words using Knowledge-Based Word Sense Disambiguation**

Type any English or Hindi sentence, click any Hindi word, and ArthBodh shows the one meaning of
that word that fits the sentence. It uses IndoWordNet as its dictionary, so nothing is trained per
word. Two methods are used: **Simplified Lesk** (baseline) and **Embedding Lesk** with LaBSE (main).

Example: in *"बाज़ार में ताज़े फल बिक रहे हैं।"* the word फल means **fruit**; in
*"किसानों को उनकी मेहनत का अच्छा फल मिला।"* it means **result**.

The full design spec is in `Hindi Word Sense Disambiguation – NLP Mini Project.pdf`.

## Quick start (new laptop)

Requires Python 3.10–3.12 (3.13+ may not have torch/stanza wheels yet) and ~3 GB free disk.

```bash
git clone https://github.com/RudraPandit0504/arthbodh.git
cd arthbodh
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py             # one time, needs good Wi-Fi: Stanza Hindi, IndoWordNet, LaBSE (~2 GB)
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
                     └─ embed_lesk.py  Embedding Lesk (LaBSE; + English sentence in EN mode)
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
| `src/embed_lesk.py` | Embedding Lesk with LaBSE, cross-lingual context, cached gloss vectors |
| `src/engine.py` | `analyse(hi_sentence, word, en_sentence)`, the full pipeline for one word |
| `src/build_testset.py` | Builds test set 1 from the IIT Bombay English–Hindi parallel corpus |
| `src/label_tool.py` | CLI helper for hand-labelling test set 2 |
| `src/evaluate.py` | Accuracy of all methods on both test sets, with charts in `results/` |
| `data/sense_keywords.json` | Sense → English keywords for ~30 ambiguous words |
| `data/stopwords_hi.txt` | Hindi stop words removed before Simplified Lesk |

**Confidence** is the gap between the best and second-best Embedding Lesk score. A gap below 0.03 is shown as "unsure".

## Evaluation

```bash
python -m src.build_testset          # downloads the IITB corpus (~1.6 M pairs) and writes data/test_auto.csv
python -m src.evaluate               # writes results/results.csv, results/*.png
python -m src.label_tool             # add hand-labelled rows to data/test_manual.csv
```

Gold labels may list several synset ids separated by `|`. IndoWordNet often stores the same meaning
twice (e.g. कल "yesterday" as both noun 617 and adverb 22207), and a prediction counts as correct if it matches any of them.

See `results/README.md` for the latest numbers.

## Smoke test

```bash
python -m src.engine
```

## Tech stack

pyiwn (IndoWordNet) · Stanza · sentence-transformers (LaBSE) · deep-translator · pandas · scikit-learn · matplotlib · Streamlit

## References

- Lesk (1986). Automatic sense disambiguation using machine readable dictionaries.
- Jurafsky & Martin, *Speech and Language Processing*: Word Senses and WordNet.
- Yusuf et al. (2022). HindiWSD: A package for word sense disambiguation in Hinglish & Hindi. WILDRE.
- Kunchukuttan et al. (2018). The IIT Bombay English-Hindi Parallel Corpus. LREC.
- IndoWordNet, CFILT, IIT Bombay; `pyiwn` Python interface.
- Feng et al. (2022). Language-agnostic BERT Sentence Embedding (LaBSE).
- Qi et al. (2020). Stanza: A Python NLP toolkit for many human languages.
