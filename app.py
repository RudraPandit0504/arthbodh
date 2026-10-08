"""Step 7: ArthBodh Streamlit app.

Run:  streamlit run app.py
"""
import streamlit as st

from src import embed_lesk, nlp, wordnet
from src.engine import analyse
from src.translate import en_to_hi, hi_to_en

st.set_page_config(page_title="ArthBodh", page_icon="📖", layout="centered")


@st.cache_resource(show_spinner="Loading Stanza, IndoWordNet and LaBSE (first run only)...")
def load_models():
    nlp.get_pipeline()
    wordnet.get_iwn()
    embed_lesk.get_model()
    return True


@st.cache_data(show_spinner=False)
def cached_en_to_hi(text):
    return en_to_hi(text)


@st.cache_data(show_spinner=False)
def cached_words(hi):
    return nlp.analyse_words(hi)


@st.cache_data(show_spinner=False)
def cached_gloss_en(gloss):
    return hi_to_en(gloss)


def sense_line(sense, show_en=True):
    lemmas = ", ".join(sense["lemmas"][:4])
    text = f"**{lemmas}** ({sense['pos']}) — {sense['gloss']}"
    if show_en:
        en = cached_gloss_en(sense["gloss"])
        if en:
            text += f"  \n*{en}*"
    return text


st.title("ArthBodh · अर्थबोध")
st.caption("Context-aware meaning finder for Hindi words — knowledge-based Word Sense "
           "Disambiguation with IndoWordNet, Simplified Lesk and Embedding Lesk (LaBSE).")

load_models()

EXAMPLES = {"English": "The farmers got a good result for their hard work.",
            "Hindi": "बाज़ार में ताज़े फल बिक रहे हैं।"}

mode = st.radio("Input language", ["English", "Hindi"], horizontal=True)
text = st.text_input("Type a sentence", placeholder=EXAMPLES[mode])
show_en = st.toggle("Show English translations of meanings", value=True,
                    help="Needs internet. Turn off if the translator is down.")

if not text:
    st.info(f"Try: *{EXAMPLES[mode]}*")
    st.stop()

if mode == "English":
    en = text
    hi = cached_en_to_hi(text)
    if not hi:
        st.error("Translation is unavailable (no internet or translator blocked). "
                 "Switch to Hindi mode and type the Hindi sentence directly.")
        st.stop()
else:
    en, hi = None, text

st.subheader(hi)
words = cached_words(hi)
if not words:
    st.warning("No words found in the sentence.")
    st.stop()

labels = [f"{i}:{w['text']}" for i, w in enumerate(words)]
pick = st.pills("Pick a word", labels, format_func=lambda s: s.split(":", 1)[1],
                selection_mode="single")
if pick is None:
    st.caption("Click any Hindi word above to see its meaning in this sentence.")
    st.stop()

word = words[int(pick.split(":", 1)[0])]
with st.spinner("Disambiguating..."):
    r = analyse(hi, word, en, translate=False)

st.markdown(f"#### {word['text']}  ·  lemma **{word['lemma']}**  ·  POS **{word['upos']}**")

if r["status"] == "not_found":
    st.warning("Meaning not available: this word was not found in IndoWordNet "
               "(tried the lemma, the surface form and suffix-stripped forms).")
    st.stop()

if r["status"] == "single":
    st.success("Only one meaning in IndoWordNet:")
    st.markdown(sense_line(r["sense"], show_en))
    st.stop()

st.markdown("**Meaning in this sentence**")
st.success(sense_line(r["sense"], show_en))

conf = r["confidence"]
c1, c2 = st.columns(2)
c1.metric("Confidence (score gap)", f"{conf:.3f}")
c2.metric("Senses considered", len(r["senses"]))
if r["unsure"]:
    st.warning("The top two meanings scored almost the same, so the system is unsure.")

with st.expander(f"Other meanings of {r['form']}"):
    order = sorted(range(len(r["senses"])), key=lambda i: -r["scores"][i])
    for i in order:
        s = r["senses"][i]
        if s["id"] == r["sense"]["id"]:
            continue
        st.markdown(f"- `{r['scores'][i]:.3f}` " + sense_line(s, show_en))

with st.expander("Compare methods"):
    st.markdown("**Embedding Lesk (main)** — LaBSE similarity between the sentence"
                + (" (Hindi + English)" if en else "") + " and each gloss:")
    st.markdown(sense_line(r["sense"], False))
    st.markdown("**Simplified Lesk (baseline)** — shared content words between the "
                "sentence and each gloss + examples:")
    st.markdown(sense_line(r["lesk_choice"], False))
    st.markdown("**First listed sense** — no context used:")
    st.markdown(sense_line(r["first_sense"], False))
    st.table({"sense": [", ".join(s["lemmas"][:3]) for s in r["senses"]],
              "embedding score": [round(x, 3) for x in r["scores"]],
              "lesk overlap": r["lesk_overlaps"]})
