"""Step 3: IndoWordNet sense lookup with POS filter and cache."""
import pyiwn

POS_MAP = {"NOUN": "noun", "PROPN": "noun", "VERB": "verb", "AUX": "verb",
           "ADJ": "adjective", "ADV": "adverb"}

# Common Hindi inflection endings, longest first, for the suffix-stripping fallback.
SUFFIXES = ["ियों", "ाओं", "ुओं", "ियाँ", "ियां", "ाएँ", "ाएं", "ों", "ें", "ीं",
            "ाँ", "ां", "ता", "ती", "ते", "ना", "नी", "ने", "ा", "े", "ी", "ो"]

_iwn = None
CACHE = {}


def get_iwn():
    global _iwn
    if _iwn is None:
        _iwn = pyiwn.IndoWordNet()  # Hindi by default
    return _iwn


def _all_senses(word):
    if word not in CACHE:
        try:
            CACHE[word] = [{"id": s.synset_id(), "pos": str(s.pos()).lower(),
                            "gloss": s.gloss(), "examples": s.examples(),
                            "lemmas": s.lemma_names()}
                           for s in get_iwn().synsets(word)]
        except Exception:  # pyiwn raises KeyError for unknown words
            CACHE[word] = []
    return CACHE[word]


def get_senses(word, upos=None, use_pos=True):
    """All senses of `word`; if use_pos, keep those matching the POS (fall back to all)."""
    senses = _all_senses(word)
    wanted = POS_MAP.get(upos) if use_pos else None
    same_pos = [s for s in senses if wanted and wanted in s["pos"]]
    return same_pos or senses


NUKTA = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य"}


def remove_nukta(word):
    """IndoWordNet often spells बाज़ार as बाजार; try the form without nukta too."""
    word = word.replace("\u093c", "")
    return "".join(NUKTA.get(ch, ch) for ch in word)


def strip_suffix(word):
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            yield word[: -len(suf)]


def lookup(word, use_pos=True):
    """Senses for an analysed word dict: lemma -> surface form -> suffix-stripped forms.

    Returns (form_found, senses)."""
    candidates = [word["lemma"], word["text"], remove_nukta(word["lemma"]),
                  remove_nukta(word["text"]), *strip_suffix(word["text"])]
    seen = set()
    for form in candidates:
        if form in seen:
            continue
        seen.add(form)
        senses = get_senses(form, word.get("upos"), use_pos)
        if senses:
            return form, senses
    return word["lemma"], []
