"""Step 5: Embedding Lesk, plus cross-lingual context in English mode.

Two pretrained sentence encoders score every sense, and their scores are averaged:
- LaBSE (multilingual, places English and Hindi in one space) compares the sentence with gloss + examples.
- L3Cube HindSBERT (trained for Hindi sentence similarity) compares it with synonyms + gloss + examples.
A small sense-order prior then prefers IndoWordNet's earlier (more common) senses when scores are close.
The models and the prior weight were chosen on half of test set 1 and checked on the other half
(results/README.md); `models=("labse",), prior=0` gives the original LaBSE-only method.
"""
from sentence_transformers import SentenceTransformer, util

MODEL_NAMES = {"labse": "sentence-transformers/LaBSE",
               "hindi": "l3cube-pune/hindi-sentence-similarity-sbert"}
DEFAULT_MODELS = ("labse", "hindi")
PRIOR = 0.1  # score penalty for the last-listed sense; 0 for the first

_models = {}
_gloss_cache = {}  # (model, synset id) -> gloss vector


def get_model(name="labse"):
    if name not in _models:
        model = SentenceTransformer(MODEL_NAMES[name])
        if model.device.type == "cuda":
            model.half()  # both models fit in a 4 GB laptop GPU in half precision
        _models[name] = model
    return _models[name]


def gloss_text(sense, with_synonyms=False):
    text = sense["gloss"] + " । " + " ".join(sense["examples"])
    if with_synonyms:  # other words for the same sense, e.g. मान -> सम्मान, आदर
        text = ", ".join(sense["lemmas"]) + " : " + text
    return text


def gloss_vectors(senses, name):
    """Encode glosses once per synset and model, then reuse them (encoding is the slow part)."""
    missing = [s for s in senses if (name, s["id"]) not in _gloss_cache]
    if missing:
        texts = [gloss_text(s, with_synonyms=(name == "hindi")) for s in missing]
        vecs = get_model(name).encode(texts, convert_to_tensor=True)
        for s, v in zip(missing, vecs):
            _gloss_cache[(name, s["id"])] = v
    return [_gloss_cache[(name, s["id"])] for s in senses]


def encode(text, name="labse"):
    return get_model(name).encode(text, convert_to_tensor=True)


def embedding_lesk(hi_sentence, senses, en_sentence=None, w_en=0.5,
                   models=DEFAULT_MODELS, prior=PRIOR):
    """Return (best_index, scores): cosine similarity of the sentence with each gloss.

    In English mode the original English sentence is a second clue:
    score = (1 - w_en) * sim(hindi, gloss) + w_en * sim(english, gloss).
    With several models the scores are averaged; then prior * i / n is subtracted for sense i."""
    import torch
    total = 0
    for name in models:
        G = torch.stack(gloss_vectors(senses, name))
        scores = util.cos_sim(encode(hi_sentence, name), G)[0]
        if en_sentence:
            en_scores = util.cos_sim(encode(en_sentence, name), G)[0]
            scores = (1 - w_en) * scores + w_en * en_scores
        total = total + scores / len(models)
    total = total.cpu() - prior * torch.arange(len(senses)) / len(senses)
    return int(total.argmax()), total.tolist()
