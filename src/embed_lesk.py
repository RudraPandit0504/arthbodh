"""Step 5: Embedding Lesk with LaBSE, plus cross-lingual context in English mode."""
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/LaBSE"
_model = None
_gloss_cache = {}  # synset id -> gloss vector


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def gloss_text(sense):
    return sense["gloss"] + " । " + " ".join(sense["examples"])


def gloss_vectors(senses):
    """Encode glosses once per synset and reuse them (LaBSE is the slow part)."""
    missing = [s for s in senses if s["id"] not in _gloss_cache]
    if missing:
        vecs = get_model().encode([gloss_text(s) for s in missing], convert_to_tensor=True)
        for s, v in zip(missing, vecs):
            _gloss_cache[s["id"]] = v
    return [_gloss_cache[s["id"]] for s in senses]


def encode(text):
    return get_model().encode(text, convert_to_tensor=True)


def embedding_lesk(hi_sentence, senses, en_sentence=None, w_en=0.5):
    """Return (best_index, scores): cosine similarity of the sentence with each gloss.

    In English mode the original English sentence is a second clue:
    score = (1 - w_en) * sim(hindi, gloss) + w_en * sim(english, gloss)."""
    import torch
    G = torch.stack(gloss_vectors(senses))
    scores = util.cos_sim(encode(hi_sentence), G)[0]
    if en_sentence:
        en_scores = util.cos_sim(encode(en_sentence), G)[0]
        scores = (1 - w_en) * scores + w_en * en_scores
    return int(scores.argmax()), scores.tolist()
