"""Error analysis helper (PDF section 8): print wrong answers with the gold and predicted glosses.

    python -m src.error_analysis [--variant "Embedding Lesk, Hindi mode"] [--word मान] [--n 10]

Reads results/predictions.csv (written by src.evaluate), so run the evaluation first.
The explained cases themselves are written up by hand in results/error_analysis.md.
"""
import argparse

import pandas as pd

from . import RESULTS
from .wordnet import _all_senses


def gloss_of(word, sense_id):
    """Gloss of a synset id, looked up among the senses of the target word."""
    for s in _all_senses(word):
        if s["id"] == sense_id:
            return f"{s['id']} ({s['pos']}) {', '.join(s['lemmas'][:3])}: {s['gloss']}"
    return f"{sense_id} (not a sense of {word}; found via another form)"


def wrong_answers(variant, test_set="auto", word=None):
    p = pd.read_csv(RESULTS / "predictions.csv")
    p = p[(p.variant == variant) & (p.test_set == test_set) & (~p.correct)]
    return p[p.target_word == word] if word else p


def main(variant, test_set, word, n, seed):
    wrong = wrong_answers(variant, test_set, word)
    print(f"{len(wrong)} wrong answers for '{variant}' on test set '{test_set}'"
          f"{' for ' + word if word else ''}; gold unreachable in "
          f"{(~wrong.gold_reachable).sum()} of them.\n")
    for r in wrong.sample(min(n, len(wrong)), random_state=seed).itertuples():
        gold = [int(x) for x in str(r.gold).split("|")]
        print(f"[{r.target_word}] POS={r.upos} senses={r.n_senses} confidence={r.confidence:.3f}"
              f" gold_reachable={r.gold_reachable}")
        print(f"  sentence: {r.hindi_sentence}")
        for g in gold:
            print(f"  gold: {gloss_of(r.target_word, g)}")
        if pd.notna(r.pred):
            print(f"  pred: {gloss_of(r.target_word, int(r.pred))}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="Embedding Lesk, Hindi mode")
    ap.add_argument("--test-set", default="auto", choices=["auto", "manual"])
    ap.add_argument("--word")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.variant, a.test_set, a.word, a.n, a.seed)
