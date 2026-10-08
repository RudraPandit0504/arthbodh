"""Step 6: the engine. Runs every step for one sentence + word and packages the result."""
from .embed_lesk import embedding_lesk
from .lesk import simplified_lesk
from .translate import hi_to_en
from .wordnet import lookup

METHODS = ["first_sense", "simplified_lesk", "embedding_lesk"]
UNSURE_GAP = 0.02  # confidence below this is shown as "unsure" (lowest quarter, ~49% accurate)


def disambiguate(method, hi_sentence, lemma, senses, en_sentence=None, w_en=0.5):
    """Run one method. Returns (best_index, scores) where higher score = better."""
    if method == "first_sense":
        return 0, [1.0] + [0.0] * (len(senses) - 1)
    if method == "simplified_lesk":
        return simplified_lesk(hi_sentence, lemma, senses)
    if method == "embedding_lesk":
        return embedding_lesk(hi_sentence, senses, en_sentence, w_en)
    raise ValueError(f"unknown method {method}")


def confidence(scores):
    """Gap between the best and second-best score."""
    top2 = sorted(scores, reverse=True)[:2]
    return top2[0] - top2[1] if len(top2) == 2 else 1.0


def analyse(hi_sentence, word, en_sentence=None, use_pos=True, w_en=0.5, translate=True):
    """word: {'text', 'lemma', 'upos'} from nlp.analyse_words."""
    form, senses = lookup(word, use_pos)
    if not senses:
        return {"status": "not_found", "word": word, "form": form}
    if len(senses) == 1:
        s = senses[0]
        return {"status": "single", "word": word, "form": form, "sense": s,
                "meaning_en": hi_to_en(s["gloss"]) if translate else None}

    e_idx, e_scores = embedding_lesk(hi_sentence, senses, en_sentence, w_en)
    l_idx, overlaps = simplified_lesk(hi_sentence, form, senses)
    conf = confidence(e_scores)
    return {"status": "ambiguous", "word": word, "form": form,
            "sense": senses[e_idx],
            "meaning_en": hi_to_en(senses[e_idx]["gloss"]) if translate else None,
            "confidence": conf,
            "unsure": conf < UNSURE_GAP,
            "scores": e_scores,
            "senses": senses,
            "other_senses": [s for i, s in enumerate(senses) if i != e_idx],
            "lesk_choice": senses[l_idx], "lesk_overlaps": overlaps,
            "first_sense": senses[0]}


if __name__ == "__main__":
    from .nlp import analyse_words
    from .translate import en_to_hi

    demos = [("किसानों को उनकी मेहनत का अच्छा फल मिला।",
              "The farmers got a good result for their hard work."),
             ("बाज़ार में ताज़े फल बिक रहे हैं।", None)]
    for hi, en in demos:
        words = analyse_words(hi)
        word = next(w for w in words if w["text"] == "फल")
        r = analyse(hi, word, en)
        print(hi, "| EN:", en)
        print("  ->", r["sense"]["id"], r["sense"]["gloss"][:60], "|", r["meaning_en"])
        print("  confidence %.3f  lesk ->" % r["confidence"], r["lesk_choice"]["id"])
    print("en_to_hi:", en_to_hi("The farmers got a good result for their hard work."))
