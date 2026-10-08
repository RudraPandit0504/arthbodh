"""Step 8: build test set 1 automatically from the IIT Bombay English-Hindi parallel corpus.

For each word in data/sense_keywords.json, keep sentence pairs where the Hindi side
contains the word as a token and the English side contains keywords of exactly one
sense. The English translation is the free answer key.

    python -m src.build_testset [--per-word 50] [--seed 13]
"""
import argparse
import json
import random
import re

import pandas as pd

from . import DATA

CORPUS_URL = ("https://huggingface.co/api/datasets/cfilt/iitb-english-hindi/"
              "parquet/default/train/0.parquet")
RAW = DATA / "raw" / "iitb_train.parquet"
TOKEN_SPLIT = re.compile(r"[\s,.;:!?\"'()\[\]{}।॥\-–—/]+")


def load_corpus():
    """Download the corpus once into data/raw/ and return (hi, en) lists."""
    if not RAW.exists():
        import requests
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading IITB corpus to {RAW} ...")
        with requests.get(CORPUS_URL, stream=True, allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(RAW, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    df = pd.read_parquet(RAW)
    pairs = df["translation"].tolist()
    return [p["hi"] for p in pairs], [p["en"] for p in pairs]


def keyword_patterns(senses):
    return {gold: re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b", re.I)
            for gold, kws in senses.items()}


def label(en, patterns):
    """Return the single sense whose keywords appear in `en`, or None if zero/many match."""
    hits = [gold for gold, pat in patterns.items() if pat.search(en)]
    return hits[0] if len(hits) == 1 else None


def build(per_word=50, seed=13, min_len=4, max_len=40):
    keywords = json.loads((DATA / "sense_keywords.json").read_text(encoding="utf-8"))
    hi_all, en_all = load_corpus()
    print(f"Corpus: {len(hi_all):,} sentence pairs")
    candidates = {w: [] for w in keywords}
    patterns = {w: keyword_patterns(s) for w, s in keywords.items()}

    for hi, en in zip(hi_all, en_all):
        if not hi or not en:
            continue
        tokens = set(TOKEN_SPLIT.split(hi))
        hits = [w for w in keywords if w in tokens]
        if not hits or not (min_len <= len(hi.split()) <= max_len):
            continue
        for w in hits:
            gold = label(en, patterns[w])
            if gold:
                candidates[w].append((hi.strip(), en.strip(), w, gold))

    rng = random.Random(seed)
    rows = []
    for w, items in candidates.items():
        items = list(dict.fromkeys(items))  # drop duplicate pairs
        # Balance senses: round-robin over senses so the common sense doesn't swamp the rest.
        by_sense = {}
        for it in items:
            by_sense.setdefault(it[3], []).append(it)
        for lst in by_sense.values():
            rng.shuffle(lst)
        picked = []
        while len(picked) < per_word and any(by_sense.values()):
            for g in list(by_sense):
                if by_sense[g] and len(picked) < per_word:
                    picked.append(by_sense[g].pop())
        print(f"  {w}: {len(items):5d} candidates -> {len(picked)} kept "
              f"({', '.join(f'{g}:{sum(p[3] == g for p in picked)}' for g in keywords[w])})")
        rows += picked

    df = pd.DataFrame(rows, columns=["hindi_sentence", "english_sentence", "target_word",
                                     "gold_sense_id"])
    df["source"] = "auto"
    out = DATA / "test_auto.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out}")

    check = df.sample(min(100, len(df)), random_state=seed)
    check.to_csv(DATA / "test_auto_handcheck.csv", index=False, encoding="utf-8")
    print(f"Wrote 100 random rows to {DATA / 'test_auto_handcheck.csv'} for the manual "
          "auto-label accuracy check (add a column 'label_ok' with 1/0).")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-word", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    build(a.per_word, a.seed)
