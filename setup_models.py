"""One-time downloads: Stanza Hindi model, IndoWordNet data (pyiwn), LaBSE, HindSBERT.

Run once on good Wi-Fi:  python setup_models.py
"""
import stanza
import pyiwn
from sentence_transformers import SentenceTransformer

print("Downloading Stanza Hindi model ...")
stanza.download("hi", processors="tokenize,pos,lemma")

print("Loading IndoWordNet (downloads data on first use) ...")
pyiwn.IndoWordNet()

print("Downloading LaBSE (~1.8 GB) ...")
SentenceTransformer("sentence-transformers/LaBSE")

print("Downloading L3Cube HindSBERT (~1 GB) ...")
SentenceTransformer("l3cube-pune/hindi-sentence-similarity-sbert")

print("All models ready.")
