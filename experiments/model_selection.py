"""Model selection for Embedding Lesk on test set 1 (dev = even rows, held-out = odd rows).

    python experiments/model_selection.py                      # all five encoders (~3 GB of downloads)
    python experiments/model_selection.py sentence-transformers/LaBSE
    DUMP_ONLY=1 python experiments/model_selection.py sentence-transformers/LaBSE \
        l3cube-pune/hindi-sentence-similarity-sbert           # save scores for compare_configs.py

Run from the repo root. For each encoder it scores every sense with three sense texts
(gloss = gloss + examples, syn = synonyms + gloss, ext = syn + hypernym gloss + hyponym words)
and a range of sense-order prior weights (beta). Caches go to experiments/*.pkl.
Results are summarised in results/model_selection.md.
"""
import sys, pickle, os
sys.path.insert(0, os.getcwd())
import pandas as pd, torch
from pyiwn import SynsetRelations as R
from sentence_transformers import SentenceTransformer, util
from src.evaluate import find_word
from src.wordnet import lookup, get_iwn

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "rows.pkl")

if os.path.exists(CACHE):
    rows = pickle.load(open(CACHE, "rb"))
else:
    df = pd.read_csv("data/test_auto.csv")
    iwn, syn_obj, rows = get_iwn(), {}, []
    for k, r in enumerate(df.itertuples(index=False)):
        word = find_word(r.hindi_sentence, r.target_word)
        form, senses = lookup(word, True)
        for s in iwn.synsets(form) if senses else []:
            syn_obj[s.synset_id()] = s
        ext = {}
        for s in senses:
            o = syn_obj.get(s["id"])
            hyper = [h.gloss() for h in iwn.synset_relation(o, R.HYPERNYMY)] if o else []
            hypo = [l for h in (iwn.synset_relation(o, R.HYPONYMY) if o else [])[:5] for l in h.lemma_names()[:1]]
            ext[s["id"]] = (hyper, hypo)
        rows.append(dict(hi=r.hindi_sentence, en=r.english_sentence, form=form, senses=senses, ext=ext,
                         gold={int(x) for x in str(r.gold_sense_id).split("|")}, split=k % 2, word=r.target_word))
    pickle.dump(rows, open(CACHE, "wb"))
print("rows", len(rows))


def text(s, form, ext, kind):
    syn = ", ".join(l for l in s["lemmas"] if l != form)
    base = s["gloss"] + " । " + " ".join(s["examples"])
    if kind == "gloss":
        return base
    if kind == "syn":
        return syn + " : " + base
    hyper, hypo = ext[s["id"]]
    return syn + " : " + base + " । " + " ".join(hyper) + " । " + ", ".join(hypo)


def run(model_name, prefix=""):
    m = SentenceTransformer(model_name, device="cuda")
    m.half()
    enc = lambda xs: m.encode([prefix + x for x in xs], convert_to_tensor=True, batch_size=64,
                              normalize_embeddings=True, show_progress_bar=False)
    S = enc([r["hi"] for r in rows])
    E = enc([r["en"] for r in rows])
    res = {}
    for kind in ["gloss", "syn", "ext"]:
        cache = {}
        allt = {(s["id"], text(s, r["form"], r["ext"], kind)) for r in rows for s in r["senses"]}
        keys = list(allt)
        V = enc([t for _, t in keys])
        for (i, t), v in zip(keys, V):
            cache[(i, t)] = v
        sc = []
        for k, r in enumerate(rows):
            if not r["senses"]:
                sc.append(None); continue
            G = torch.stack([cache[(s["id"], text(s, r["form"], r["ext"], kind))] for s in r["senses"]])
            sc.append(((S[k] @ G.T).float().cpu(), (E[k] @ G.T).float().cpu()))
        res[kind] = sc
    del m; torch.cuda.empty_cache()
    return res


def acc(sc, beta=0.0, w_en=0.0, split=None):
    ok = n = 0
    for r, s in zip(rows, sc):
        if split is not None and r["split"] != split:
            continue
        n += 1
        if s is None:
            continue
        hi, en = s
        x = (1 - w_en) * hi + w_en * en
        x = x - beta * torch.arange(len(x)) / len(x)  # prefer earlier (more common) senses
        ok += r["senses"][int(x.argmax())]["id"] in r["gold"]
    return 100 * ok / n


MODELS = [("sentence-transformers/LaBSE", ""),
          ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
          ("intfloat/multilingual-e5-base", "query: "),
          ("l3cube-pune/hindi-sentence-similarity-sbert", ""),
          ("BAAI/bge-m3", "")]
BETAS = (0, 0.05, 0.1, 0.15, 0.2, 0.3)
names = sys.argv[1:] or [m for m, _ in MODELS]
for name, pre in MODELS:
    if name not in names:
        continue
    res = run(name, pre)
    pickle.dump(res, open(os.path.join(OUT, name.split('/')[-1] + '.scores.pkl'), 'wb'))
    if os.environ.get('DUMP_ONLY'): continue
    for kind, sc in res.items():
        line = [f"{acc(sc, b, 0, 0):.1f}" for b in BETAS]
        best_b = max(BETAS, key=lambda b: acc(sc, b, 0, 0))
        print(f"{name.split('/')[-1]:45s} {kind:5s} dev HI beta 0/.05/.1/.15/.2/.3: {' '.join(line)} | "
              f"held-out HI beta0 {acc(sc, 0, 0, 1):.1f} beta{best_b} {acc(sc, best_b, 0, 1):.1f} | "
              f"held-out EN(w.5) {acc(sc, best_b, 0.5, 1):.1f}", flush=True)
