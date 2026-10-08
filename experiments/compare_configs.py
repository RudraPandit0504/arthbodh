"""Full-set accuracy and a paired sign test (vs LaBSE only) for candidate configurations.

    python experiments/compare_configs.py experiments

Needs the score caches written by `DUMP_ONLY=1 python experiments/model_selection.py ...`.
"""
import pickle, sys, torch, numpy as np
S = sys.argv[1]
rows = pickle.load(open(f"{S}/rows.pkl", "rb"))
L = pickle.load(open(f"{S}/LaBSE.scores.pkl", "rb")); H = pickle.load(open(f"{S}/hindi-sentence-similarity-sbert.scores.pkl", "rb"))
def correct(sc_list, beta=0, w_en=0, mix=None):
    out = []
    for k, r in enumerate(rows):
        parts = []
        for sc, wt in sc_list:
            s = sc[k]
            if s is None: break
            parts.append(wt * ((1 - w_en) * s[0] + w_en * s[1]))
        if not parts: out.append(False); continue
        x = sum(parts); x = x - beta * torch.arange(len(x)) / len(x)
        out.append(r["senses"][int(x.argmax())]["id"] in r["gold"])
    return np.array(out)
split = np.array([r["split"] for r in rows])
base = correct([(L["gloss"], 1)])
cfgs = {"LaBSE gloss (current)": ([(L["gloss"],1)],0)}
for b in (0.1, 0.2): cfgs[f"LaBSE gloss prior{b}"] = ([(L["gloss"],1)], b)
for kind in ("gloss","syn","ext"):
    for b in (0, 0.1, 0.2): cfgs[f"HindSBERT {kind} prior{b}"] = ([(H[kind],1)], b)
for b in (0, 0.1, 0.2): cfgs[f"LaBSE gloss + HindSBERT syn prior{b}"] = ([(L["gloss"],.5),(H["syn"],.5)], b)
def mcn(a, b):
    from scipy.stats import binomtest
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return n10 - n01, binomtest(min(n01, n10), n01 + n10).pvalue if n01 + n10 else 1
print(f"{'config':42s} {'all':>5} {'dev':>5} {'held':>5}  vs current (net, p)")
for name, (sl, b) in cfgs.items():
    c = correct(sl, b)
    net, p = mcn(base, c)
    print(f"{name:42s} {100*c.mean():5.1f} {100*c[split==0].mean():5.1f} {100*c[split==1].mean():5.1f}  {net:+4d}  p={p:.3f}")
print("EN mode (w_en .5):")
for name, (sl, b) in [("LaBSE gloss", ([(L['gloss'],1)],0)), ("HindSBERT syn prior0.2", ([(H['syn'],1)],0.2)), ("LaBSE+HindSBERT syn prior0.1", ([(L['gloss'],.5),(H['syn'],.5)],0.1))]:
    print(f"  {name:40s} {100*correct(sl,b,0.5).mean():5.1f}")
print("--- ensemble grid (choose on dev)")
for kind in ("gloss", "syn", "ext"):
    for b in (0.05, 0.1, 0.15):
        c = correct([(L["gloss"], .5), (H[kind], .5)], b)
        print(f"  LaBSE + HindSBERT {kind:5s} prior{b:<5} all {100*c.mean():5.1f} dev {100*c[split==0].mean():5.1f} held {100*c[split==1].mean():5.1f}")
c = correct([(L["gloss"], .5), (H["syn"], .5)], 0.1)
top2 = []
print("EN variants for best:", [round(100*correct([(L['gloss'],.5),(H['syn'],.5)],0.1,w).mean(),1) for w in (0.3,0.5,0.7)])
