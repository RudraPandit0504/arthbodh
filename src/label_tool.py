"""Helpers for the human labelling work (test set 2 and the auto-label check).

    python -m src.label_tool                       # type your own sentences into data/test_manual.csv
    python -m src.label_tool --review data/test_manual_draft.csv --labeller a
    python -m src.label_tool --review data/test_manual_draft.csv --labeller b
    python -m src.label_tool --merge data/test_manual_draft.csv
    python -m src.label_tool --handcheck           # fill label_ok in data/test_auto_handcheck.csv

Review: each team member labels the draft on their own (blind: you never see the other column).
For every sentence the tool lists all IndoWordNet senses of the word; type the number of the
correct one (several numbers separated by spaces if senses are duplicates), `n` if no sense fits,
blank to skip, `q` to quit. Progress is saved after every row, so you can stop and resume.

Merge: rows where both labellers agree go into final_gold; disagreements are listed so you can
discuss them and type the agreed answer into final_gold (or run --review --labeller final).
Rows with a final_gold are copied into data/test_manual.csv. Agreement and Cohen's kappa are
printed for the report.
"""
import argparse
import csv

from . import DATA
from .evaluate import find_word
from .wordnet import _all_senses, lookup

MANUAL = DATA / "test_manual.csv"
HANDCHECK = DATA / "test_auto_handcheck.csv"
HEADER = ["hindi_sentence", "english_sentence", "target_word", "gold_sense_id", "source"]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def show_senses(sentence, target):
    """Print the numbered senses of `target`; returns the sense list (empty if not found)."""
    word = find_word(sentence, target)
    form, senses = lookup(word, use_pos=False)
    if not senses:
        print(f"  '{target}' not found in IndoWordNet.")
        return []
    print(f"  lemma={word['lemma']} POS={word['upos']}  senses of {form}:")
    for i, s in enumerate(senses):
        print(f"  [{i}] {s['id']:>6} ({s['pos']}) {', '.join(s['lemmas'][:3])}: {s['gloss']}")
    return senses


def ask_sense(senses, allow_none=False):
    """Returns '|'-joined ids, 'none', '' (skip) or 'q' (quit)."""
    hint = ", n = no sense fits" if allow_none else ""
    choice = input(f"Correct sense number(s){hint}, blank to skip, q to quit: ").strip()
    if choice in ("q", ""):
        return choice
    if allow_none and choice == "n":
        return "none"
    try:
        return "|".join(str(senses[int(c)]["id"]) for c in choice.split())
    except (ValueError, IndexError):
        print("  invalid choice, skipped.")
        return ""


def add_sentences(source):
    """Original mode: type a sentence and a word, pick the sense, append to test_manual.csv."""
    if not MANUAL.exists():
        with open(MANUAL, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
    print("Empty sentence quits.\n")
    while True:
        sent = input("Hindi sentence: ").strip()
        if not sent:
            break
        target = input("Target word (as it appears in the sentence): ").strip()
        senses = show_senses(sent, target)
        if not senses:
            continue
        ids = ask_sense(senses)
        if ids in ("", "q"):
            continue
        with open(MANUAL, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([sent, "", target, ids, source])
        print(f"  saved: {target} -> {ids}\n")


def review(path, labeller):
    col = "final_gold" if labeller == "final" else f"labeller_{labeller}"
    rows = read_rows(path)
    todo = [r for r in rows if not r[col]]
    print(f"{len(rows) - len(todo)}/{len(rows)} rows already labelled in '{col}'.\n")
    for n, r in enumerate(todo, 1):
        print(f"({n}/{len(todo)}) [{r['target_word']}] {r['hindi_sentence']}")
        if labeller == "final":
            print(f"  labeller a: {r['labeller_a']}   labeller b: {r['labeller_b']}")
        senses = show_senses(r["hindi_sentence"], r["target_word"])
        ans = ask_sense(senses, allow_none=True) if senses else "none"
        if ans == "q":
            break
        if ans:
            r[col] = ans
            write_rows(path, rows)
        print()
    print(f"Saved to {path}")


def merge(path):
    from sklearn.metrics import cohen_kappa_score

    rows = read_rows(path)
    both = [r for r in rows if r["labeller_a"] and r["labeller_b"]]
    agree = [r for r in both if set(r["labeller_a"].split("|")) == set(r["labeller_b"].split("|"))]
    for r in agree:
        r["final_gold"] = r["final_gold"] or r["labeller_a"]
    write_rows(path, rows)
    if both:
        # kappa on the first chosen id of each labeller (labels are categorical per sentence)
        a = [r["labeller_a"].split("|")[0] for r in both]
        b = [r["labeller_b"].split("|")[0] for r in both]
        print(f"Labelled by both: {len(both)}  agreement: {100 * len(agree) / len(both):.1f}%  "
              f"Cohen's kappa: {cohen_kappa_score(a, b):.2f}")
    open_rows = [r for r in both if r not in agree and not r["final_gold"]]
    if open_rows:
        print(f"\n{len(open_rows)} disagreements to resolve "
              f"(edit final_gold or run --review {path} --labeller final):")
        for r in open_rows:
            print(f"  [{r['target_word']}] a={r['labeller_a']} b={r['labeller_b']}  {r['hindi_sentence'][:70]}")

    done = [r for r in rows if r["final_gold"] and r["final_gold"] != "none"]
    manual = read_rows(MANUAL)
    have = {(m["hindi_sentence"], m["target_word"]) for m in manual}
    new = [{"hindi_sentence": r["hindi_sentence"], "english_sentence": "",
            "target_word": r["target_word"], "gold_sense_id": r["final_gold"], "source": r["source"]}
           for r in done if (r["hindi_sentence"], r["target_word"]) not in have]
    if new:
        write_rows(MANUAL, manual + new)
    print(f"\n{len(new)} new rows added to {MANUAL} ({len(done)} final labels in the draft).")


def gloss(word, ids):
    by_id = {s["id"]: s for s in _all_senses(word)}
    return " / ".join(by_id[int(i)]["gloss"] if int(i) in by_id else i for i in ids.split("|"))


def handcheck():
    rows = read_rows(HANDCHECK)
    for r in rows:
        r.setdefault("label_ok", "")
    todo = [r for r in rows if r["label_ok"] == ""]
    print(f"{len(rows) - len(todo)}/{len(rows)} rows already checked. "
          "Is the auto label right? 1 = yes, 0 = no, blank = skip, q = quit.\n")
    for n, r in enumerate(todo, 1):
        print(f"({n}/{len(todo)}) [{r['target_word']}]")
        print(f"  HI: {r['hindi_sentence']}\n  EN: {r['english_sentence']}")
        print(f"  auto label {r['gold_sense_id']}: {gloss(r['target_word'], r['gold_sense_id'])}")
        ans = input("  correct? ").strip()
        if ans == "q":
            break
        if ans in ("0", "1"):
            r["label_ok"] = ans
            write_rows(HANDCHECK, rows)
        print()
    checked = [r for r in rows if r["label_ok"] in ("0", "1")]
    if checked:
        ok = sum(r["label_ok"] == "1" for r in checked)
        print(f"Auto labels correct: {ok}/{len(checked)} = {100 * ok / len(checked):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="manual", help="source tag when typing your own sentences")
    ap.add_argument("--review", metavar="CSV", help="label a draft file (e.g. data/test_manual_draft.csv)")
    ap.add_argument("--labeller", choices=["a", "b", "final"], help="your column in the draft")
    ap.add_argument("--merge", metavar="CSV", help="combine both labellers and add to test_manual.csv")
    ap.add_argument("--handcheck", action="store_true", help="check the auto labels of test set 1")
    a = ap.parse_args()
    if a.review:
        if not a.labeller:
            ap.error("--review needs --labeller a, b or final")
        review(a.review, a.labeller)
    elif a.merge:
        merge(a.merge)
    elif a.handcheck:
        handcheck()
    else:
        add_sentences(a.source)
