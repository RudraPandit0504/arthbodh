"""Helper for hand-labelling test set 2 (data/test_manual.csv).

    python -m src.label_tool [--source manual]

Type a Hindi sentence (from Hindi news / Wikipedia) and the target word; the tool lists
every IndoWordNet sense of that word and you type the number of the correct one (several
numbers separated by spaces if senses are duplicates). The row is appended to the CSV.
Each team member should label independently, then resolve disagreements.
"""
import argparse
import csv

from . import DATA
from .evaluate import find_word
from .wordnet import lookup

OUT = DATA / "test_manual.csv"
HEADER = ["hindi_sentence", "english_sentence", "target_word", "gold_sense_id", "source"]


def main(source):
    if not OUT.exists():
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
    print("Empty sentence quits.\n")
    while True:
        sent = input("Hindi sentence: ").strip()
        if not sent:
            break
        target = input("Target word (as it appears in the sentence): ").strip()
        word = find_word(sent, target)
        form, senses = lookup(word, use_pos=False)
        if not senses:
            print(f"  '{target}' not found in IndoWordNet, skipped.\n")
            continue
        print(f"  lemma={word['lemma']} POS={word['upos']}  senses of {form}:")
        for i, s in enumerate(senses):
            print(f"  [{i}] {s['id']:>6} ({s['pos']}) {', '.join(s['lemmas'][:3])}: {s['gloss']}")
        choice = input("Correct sense number(s), blank to skip: ").split()
        try:
            ids = [str(senses[int(c)]["id"]) for c in choice]
        except (ValueError, IndexError):
            print("  invalid choice, skipped.\n")
            continue
        if not ids:
            continue
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([sent, "", target, "|".join(ids), source])
        print(f"  saved: {target} -> {'|'.join(ids)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="manual")
    main(ap.parse_args().source)
