"""Step 9: evaluate every method on both test sets and produce tables and charts.

    python -m src.evaluate [--limit N]

Writes results/results.csv (summary), results/per_word.csv, results/predictions.csv,
results/w_en_sweep.csv, results/README.md and bar charts.
"""
import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import DATA, RESULTS  # noqa: E402
from .embed_lesk import embedding_lesk  # noqa: E402
from .lesk import simplified_lesk  # noqa: E402
from .nlp import analyse_words  # noqa: E402
from .wordnet import lookup  # noqa: E402

VARIANTS = [  # name, method, use English sentence, use POS filter
    ("Baseline: first sense", "first", False, True),
    ("Simplified Lesk", "lesk", False, True),
    ("Embedding Lesk, Hindi mode", "embed", False, True),
    ("Embedding Lesk, English mode", "embed", True, True),
    ("Simplified Lesk (no POS filter)", "lesk", False, False),
    ("Embedding Lesk, Hindi mode (no POS filter)", "embed", False, False),
]
W_EN_SWEEP = [0.3, 0.5, 0.7]


def find_word(sentence, target):
    """Locate the target word in the analysed sentence (by surface form, then lemma)."""
    words = analyse_words(sentence)
    for key in ("text", "lemma"):
        for w in words:
            if w[key] == target:
                return w
    for w in words:  # target glued to punctuation or split differently
        if target in w["text"]:
            return {**w, "lemma": target}
    return {"text": target, "lemma": target, "upos": None}


def run_method(method, hi, form, senses, en, w_en=0.5):
    """Return (ranked sense indices, confidence or None)."""
    if method == "first":
        return list(range(len(senses))), None
    if method == "lesk":
        _, scores = simplified_lesk(hi, form, senses)
    else:
        _, scores = embedding_lesk(hi, senses, en, w_en)
    order = sorted(range(len(senses)), key=lambda i: (-scores[i], i))
    top2 = sorted(scores, reverse=True)[:2]
    conf = top2[0] - top2[1] if len(top2) == 2 else None
    return order, conf


def evaluate_set(df, name, variants, w_en=0.5):
    rows = []
    for k, r in enumerate(df.itertuples(index=False)):
        en = r.english_sentence if isinstance(r.english_sentence, str) and r.english_sentence else None
        gold = {int(x) for x in str(r.gold_sense_id).split("|")}
        word = find_word(r.hindi_sentence, r.target_word)
        for vname, method, use_en, use_pos in variants:
            if use_en and not en:
                continue
            form, senses = lookup(word, use_pos)
            if not senses:
                order, conf, ids = [], None, []
            else:
                order, conf = run_method(method, r.hindi_sentence, form, senses,
                                         en if use_en else None, w_en)
                ids = [senses[i]["id"] for i in order]
            rows.append({"test_set": name, "variant": vname, "target_word": r.target_word,
                         "gold": r.gold_sense_id, "pred": ids[0] if ids else None,
                         "correct": bool(ids) and ids[0] in gold,
                         "top2": any(i in gold for i in ids[:2]),
                         "gold_reachable": any(i in gold for i in ids),
                         "n_senses": len(ids), "confidence": conf,
                         "upos": word["upos"], "hindi_sentence": r.hindi_sentence})
        if (k + 1) % 200 == 0:
            print(f"  {name}: {k + 1}/{len(df)}")
    return pd.DataFrame(rows)


def confidence_split(p):
    """Accuracy on high- vs low-confidence answers (split at the median confidence)."""
    p = p.dropna(subset=["confidence"])
    if p.empty:
        return None, None
    med = p["confidence"].median()
    return p[p.confidence > med]["correct"].mean(), p[p.confidence <= med]["correct"].mean()


def summarise(pred):
    out = []
    for (ts, v), p in pred.groupby(["test_set", "variant"], sort=False):
        hi_c, lo_c = confidence_split(p)
        out.append({"test_set": ts, "variant": v, "n": len(p),
                    "accuracy": p.correct.mean() * 100, "top2_accuracy": p.top2.mean() * 100,
                    "gold_in_candidates": p.gold_reachable.mean() * 100,
                    "acc_high_conf": None if hi_c is None else hi_c * 100,
                    "acc_low_conf": None if lo_c is None else lo_c * 100})
    return pd.DataFrame(out)


def chart(summary, path):
    main = summary[~summary.variant.str.contains("no POS")]
    piv = main.pivot(index="variant", columns="test_set", values="accuracy")
    piv = piv.reindex([v[0] for v in VARIANTS if v[0] in piv.index])
    piv = piv.rename(columns={"auto": "Test set 1: auto-built",
                              "manual": "Test set 2: hand-labelled Hindi"})
    ax = piv.plot.barh(figsize=(8, 4.6), color=["#3b6ea8", "#e0913a"][: len(piv.columns)])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.set_title("ArthBodh: WSD accuracy by method")
    ax.legend(title=None, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    for c, col in zip(ax.containers, piv.columns):
        ax.bar_label(c, labels=["" if pd.isna(v) else f"{v:.1f}" for v in piv[col]],
                     padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def devanagari_font():
    """First installed font that can draw Hindi labels, else matplotlib's default."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans Devanagari", "Lohit Devanagari", "Mangal", "Nirmala UI",
                 "Kohinoor Devanagari", "Droid Sans Devanagari"]:
        if name in installed:
            return name
    return "DejaVu Sans"


def per_word_chart(per_word, path):
    piv = per_word.pivot(index="target_word", columns="variant", values="accuracy")
    cols = [v for v in ["Baseline: first sense", "Simplified Lesk", "Embedding Lesk, Hindi mode"]
            if v in piv.columns]
    piv = piv[cols].sort_values(cols[-1])
    # Latin text in DejaVu Sans, Hindi labels fall back to a Devanagari font.
    plt.rcParams["font.family"] = ["DejaVu Sans", devanagari_font()]
    ax = piv.plot.barh(figsize=(8, 10), color=["#9aa3ad", "#e0913a", "#3b6ea8"])
    ax.set_xlabel("Accuracy (%) on test set 1")
    ax.set_ylabel("")
    ax.set_title("Per-word accuracy (test set 1)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fmt(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1f}"


def write_report(summary, sweep, n_auto, n_manual, path):
    main = [v[0] for v in VARIANTS[:4]]
    lines = ["# ArthBodh evaluation results", "",
             f"Test set 1: {n_auto} auto-built sentences (IITB parallel corpus). "
             f"Test set 2: {n_manual} hand-labelled Hindi sentences.", "",
             "| Variant | Test set 1: auto-built (%) | Test set 2: hand-labelled Hindi (%) |",
             "|---|---|---|"]
    for v in main + [v[0] for v in VARIANTS[4:]]:
        cells = []
        for ts in ["auto", "manual"]:
            m = summary[(summary.variant == v) & (summary.test_set == ts)]
            cells.append(fmt(m.accuracy.iloc[0]) if len(m) else "not applicable")
        lines.append(f"| {v} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Details", "", summary.round(1).to_markdown(index=False), "",
              "## English weight sweep (Embedding Lesk, English mode, test set 1)", "",
              sweep.round(1).to_markdown(index=False), "",
              "Notes: English mode on test set 1 looks strong because the English side contains "
              "the answer keyword; Hindi mode and test set 2 are the honest measure. "
              "`gold_in_candidates` is how often the gold sense survives lookup + POS filter "
              "(the upper bound for any method). Confidence split is at the median score gap.",
              "", "Charts: `accuracy.png`, `per_word.png`."]
    path.write_text("\n".join(lines), encoding="utf-8")


def charts_only():
    """Redraw charts from saved CSVs without re-running the evaluation."""
    chart(pd.read_csv(RESULTS / "results.csv"), RESULTS / "accuracy.png")
    per_word_chart(pd.read_csv(RESULTS / "per_word.csv"), RESULTS / "per_word.png")
    print(f"Charts redrawn in {RESULTS}")


def main(limit=None):
    RESULTS.mkdir(exist_ok=True)
    auto = pd.read_csv(DATA / "test_auto.csv")
    manual = pd.read_csv(DATA / "test_manual.csv")
    if limit:
        auto = auto.sample(min(limit, len(auto)), random_state=0)
    print(f"Test set 1: {len(auto)} rows, test set 2: {len(manual)} rows")

    pred = pd.concat([evaluate_set(auto, "auto", VARIANTS),
                      evaluate_set(manual, "manual", VARIANTS)], ignore_index=True)
    pred.to_csv(RESULTS / "predictions.csv", index=False, encoding="utf-8")

    summary = summarise(pred)
    summary.to_csv(RESULTS / "results.csv", index=False)
    print(summary.round(1).to_string(index=False))

    per_word = (pred[pred.test_set == "auto"].groupby(["target_word", "variant"])
                .correct.mean().mul(100).reset_index(name="accuracy"))
    per_word.to_csv(RESULTS / "per_word.csv", index=False, encoding="utf-8")

    sweep_rows = []
    en_variant = [("Embedding Lesk, English mode", "embed", True, True)]
    for w in W_EN_SWEEP:
        p = evaluate_set(auto, "auto", en_variant, w_en=w)
        sweep_rows.append({"w_en": w, "accuracy": p.correct.mean() * 100,
                           "top2_accuracy": p.top2.mean() * 100})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(RESULTS / "w_en_sweep.csv", index=False)
    print(sweep.round(1).to_string(index=False))

    chart(summary, RESULTS / "accuracy.png")
    per_word_chart(per_word, RESULTS / "per_word.png")
    write_report(summary, sweep, len(auto), len(manual), RESULTS / "README.md")
    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="evaluate on a random subset of test set 1")
    ap.add_argument("--charts-only", action="store_true",
                    help="redraw charts from results/*.csv without re-evaluating")
    args = ap.parse_args()
    charts_only() if args.charts_only else main(args.limit)
