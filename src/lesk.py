"""Step 4: Simplified Lesk (baseline), written by us."""
from functools import lru_cache

from . import DATA
from .nlp import analyse_words

STOP = set((DATA / "stopwords_hi.txt").read_text(encoding="utf-8").split())


@lru_cache(maxsize=16384)
def words_of(text):
    """Set of content-word lemmas in `text` (cached: glosses repeat a lot)."""
    return frozenset(w["lemma"] for w in analyse_words(text) if w["lemma"] not in STOP)


def sense_text(sense):
    return sense["gloss"] + " " + " ".join(sense["examples"])


def simplified_lesk(sentence, target_lemma, senses):
    """Pick the sense whose gloss + examples share the most lemmas with the sentence.

    Returns (best_index, overlaps). Ties go to the earlier (more common) sense."""
    context = words_of(sentence) - {target_lemma}
    overlaps = [len(context & words_of(sense_text(s))) for s in senses]
    best = max(range(len(senses)), key=lambda i: (overlaps[i], -i))
    return best, overlaps
