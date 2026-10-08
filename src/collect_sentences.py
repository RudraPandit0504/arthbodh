"""Collect real Hindi Wikipedia sentences as candidates for test set 2 (to be hand-labelled).

    python -m src.collect_sentences [--per-word 5] [--words आम हार ...]

For each word, search hi.wikipedia.org with the word alone and with the synonyms of each of its
IndoWordNet senses (so that rarer senses also show up), and keep sentences that contain the word
as a token. Writes data/test_manual_draft.csv with empty labeller columns; the team then labels
it with `python -m src.label_tool --review data/test_manual_draft.csv --labeller a` (and b).
No labels are produced here.
"""
import argparse
import csv
import json
import random
import re
import time

import requests

from . import DATA
from .build_testset import TOKEN_SPLIT
from .wordnet import _all_senses

API = "https://hi.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ArthBodh-student-project/1.0 (NLP mini project; test set collection)"}
OUT = DATA / "test_manual_draft.csv"
HEADER = ["hindi_sentence", "english_sentence", "target_word", "source",
          "labeller_a", "labeller_b", "final_gold"]
EXTRA_WORDS = ["हल", "पर", "कलम", "तारा", "नाग", "बैठक", "आँख"]


def api(**params):
    params.update(format="json", formatversion=2)
    for attempt in range(5):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))  # rate-limited: wait
    r.raise_for_status()
    time.sleep(1.0)  # be polite to Wikipedia
    return r.json()


def search(query, limit=10):
    data = api(action="query", list="search", srsearch=query, srlimit=limit, srnamespace=0)
    return [hit["title"] for hit in data["query"]["search"]]


def page_text(title):
    data = api(action="query", prop="extracts", explaintext=1, titles=title)
    return data["query"]["pages"][0].get("extract", "")


def sentences_with(text, word):
    """Clean sentences of 6-35 tokens that contain `word` as a whole token."""
    for line in text.split("\n"):
        if line.startswith("=") or len(line) < 20:
            continue
        for s in re.split(r"(?<=[।?!])\s+", line):
            s = s.strip()
            tokens = [t for t in TOKEN_SPLIT.split(s) if t]
            if word in tokens and 6 <= len(tokens) <= 35 and s.endswith(("।", "?", "!")) \
                    and not re.search(r"[A-Za-z]{4,}|\d{3,}.*\d{3,}", s):
                yield s


def queries(word, max_queries=5):
    """The word alone, then the word with one synonym of each sense (to reach rarer senses)."""
    yield word
    for s in _all_senses(word)[: max_queries - 1]:
        syn = [l for l in s["lemmas"] if l != word and " " not in l]
        if syn:
            yield f"{word} {syn[0]}"


def collect(word, per_word, rng):
    found, seen_pages = [], set()
    for q in queries(word):
        for title in search(q, limit=4):
            if title in seen_pages:
                continue
            seen_pages.add(title)
            cands = list(sentences_with(page_text(title), word))
            if cands:
                found.append((rng.choice(cands), title))  # one sentence per page for variety
        if len(found) >= 3 * per_word:
            break
    rng.shuffle(found)
    return found[:per_word]


def main(words, per_word, seed):
    rng = random.Random(seed)
    existing = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            existing = {(r["hindi_sentence"], r["target_word"]) for r in csv.DictReader(f)}
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for word in words:
            if len(_all_senses(word)) < 2:
                print(f"{word}: fewer than 2 senses in IndoWordNet, skipped")
                continue
            rows = [(s, t) for s, t in collect(word, per_word, rng) if (s, word) not in existing]
            for s, title in rows:
                w.writerow([s, "", word, f"wikipedia:{title}", "", "", ""])
            f.flush()
            print(f"{word}: {len(rows)} sentences")
    print(f"Candidates written to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-word", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--words", nargs="*")
    a = ap.parse_args()
    default = list(json.loads((DATA / "sense_keywords.json").read_text(encoding="utf-8"))) + EXTRA_WORDS
    main(a.words or default, a.per_word, a.seed)
