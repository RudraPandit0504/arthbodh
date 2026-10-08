"""Step 2: tokens, lemmas and POS tags for Hindi sentences (Stanza)."""
import logging
from functools import lru_cache

import stanza

_nlp = None


def get_pipeline():
    """Build the Stanza Hindi pipeline once (run setup_models.py first)."""
    global _nlp
    if _nlp is None:
        logging.getLogger("stanza").setLevel(logging.WARNING)
        _nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma",
                               download_method=None, verbose=False)
    return _nlp


@lru_cache(maxsize=8192)
def _analyse(sentence):
    doc = get_pipeline()(sentence)
    return tuple((w.text, w.lemma or w.text, w.upos)
                 for s in doc.sentences for w in s.words if w.upos != "PUNCT")


def analyse_words(sentence):
    """Return [{'text', 'lemma', 'upos'}] for every non-punctuation word."""
    return [{"text": t, "lemma": l, "upos": u} for t, l, u in _analyse(sentence)]
