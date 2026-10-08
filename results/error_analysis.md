# ArthBodh error analysis (PDF section 8)

Main method at the time of this analysis: **Embedding Lesk, Hindi mode** with LaBSE only, on test set 1
(1,600 auto-built sentences). It was wrong on 689 sentences (accuracy 56.9%). The analysis led to two
improvements (the lookup fix below and the two-model method in the last section), which raised Hindi mode to 65.6%. Helper to reproduce or browse more cases:

```bash
python -m src.error_analysis                 # 10 random wrong answers with gold and predicted glosses
python -m src.error_analysis --word चाल --n 5
python -m src.error_analysis --variant "Embedding Lesk, English mode"
```

## The big picture

| Where the error comes from | Share of the 689 errors | What it means |
|---|---|---|
| Gold sense never reaches the method (lookup / POS filter / bad auto-label) | 90 (13%) | No WSD method can be right; the problem is earlier in the pipeline or in the test data |
| Gold sense is the method's **second** choice | 287 (42%) | Close call; this is why top-2 accuracy is 74.9% |
| Gold sense ranked lower | 312 (45%) | Fine-grained or short glosses, missing context cues |

- Confidence works as intended: the median confidence (score gap) is 0.051 for right answers and 0.027 for wrong ones,
  and accuracy is 68% on high-confidence vs 46% on low-confidence answers.
- Weakest words: जाल and चाल (26%), मूल (32%), काल and गति (34%), कर (36%). Strongest: डाक (88%), तीर (86%), राशि (82%).
  The weak words all have 9–18 senses in IndoWordNet, many of them near-duplicates.

## Fix found through this analysis: wrong lemma (मान)

Stanza lemmatizes the **noun** मान (respect, value) as the **verb** मानना (to believe). The lookup tried the lemma first,
found only verb senses, and the POS filter fell back to those, so the right sense could never be chosen
(मान: 18% accuracy, gold reachable in only 22% of sentences).

Fix in `src/wordnet.py` `lookup()`: among the candidate forms (lemma, surface form, nukta-free, suffix-stripped),
prefer the first one that has a sense of the tagged POS. Result on test set 1:

| | Before | After |
|---|---|---|
| मान accuracy (Embedding Lesk, HI) | 18.0% | 60.0% |
| मान gold reachable | 22% | 86% |
| Embedding Lesk, Hindi mode (all words) | 55.6% | 56.9% |
| Embedding Lesk, English mode (all words) | 60.1% | 61.4% |
| Baseline: first sense | 37.9% | 38.8% |
| Simplified Lesk | 43.1% | 43.4% |

## Ten wrong answers explained

Notation: *gold* = correct IndoWordNet synset, *pred* = what ArthBodh chose, conf = confidence (score gap).

### 1. मान: still wrong after the fix, now a close call
> आप अपनी देह भले त्याग दें, लेकिन आप अपना मान नष्ट न होने दें।

Gold 46 *सम्मान, आदर (respect)* · pred 11339 *मान, परिमाण (value / measure)* · conf 0.057.
After the lemma fix the right sense is a candidate, but the sentence has no word close to "respect"
(देह, त्याग, नष्ट), so the vectors cannot separate "lose your honour" from "lose your value".
**Cause:** context too weak; needs world knowledge about the idiom.

### 2. आम: wrong POS tag removes the right sense
> जैसे ही बंदर आगे बढ़ा, उसने कुछ लड़कों को परस्पर हरे हरे आम बांटते हुए देखा।

Gold 3462|3463 *mango (noun)* · pred 3469 *सामूहिक, आम (common, adjective)*.
Stanza tags आम as ADJ (it is usually "common" in its training data), so the POS filter keeps only adjective senses
and the mango sense is thrown away before disambiguation (12 such cases for आम).
**Cause:** wrong POS from Stanza. The "no POS filter" variant does not suffer from this, which is why the POS
filter adds only ~1.6 points overall.

### 3. कर: auto-label error in test set 1, not a system error
> लेकिन गहरी नींद में भी मॉँ नन्हें पुत्र को छाती से चिपकाये अपने हाथ से उसकी रक्षा कर रही थी।

Gold 491|13314|13322 *hand* · ArthBodh looks up the verb करना (to do).
Here कर is the verb "do" (रक्षा कर रही थी = was protecting). The auto-labeller saw "hand" in the English
sentence (it belongs to हाथ) and labelled कर as "hand". ArthBodh's reading is actually right.
**Cause:** test set 1 label noise. 25 of the 32 कर errors are of this kind (another 6 are "tax" sentences where Stanza tagged कर as a verb), so कर's 36% is far too pessimistic.

### 4. मूल: auto-label error ("root cause")
> इन आंकड़ों का मूल कारण उच्च साक्षरता स्तर है।

Gold 2020 *जड़ (plant root)* · pred 7726 *आधारभूत, बुनियादी (basic, fundamental)*.
The English side says "root cause", so the keyword "root" labelled it as a plant root.
ArthBodh's answer "fundamental" is the correct one. 10 मूल errors are of this kind.
**Cause:** test set 1 label noise (keyword matching cannot tell literal and figurative "root" apart).

### 5. कल: tense decides the meaning
> पुरुषोत्तम और मेनन कल आए थे।

Gold 617 *yesterday* · pred 1589 *भविष्य (future)* · conf 0.011.
Only the past tense of आए थे says "yesterday". LaBSE sentence vectors hardly encode tense, and the gloss
of "future" shares the time-related meaning with the gold. Very low confidence, so the app shows "unsure".
**Cause:** verb tense is not modelled (as predicted in the PDF).

### 6. कल: date in brackets, still wrong
> कल (16 सितंबर, 2019), राष्ट्रपति ने लुबल्याना में 'भारत-स्लोवेनिया व्यापार-मंच' को संबोधित किया।

Gold 617 *yesterday* · pred 632 *tomorrow* · conf 0.034.
"Yesterday" and "tomorrow" glosses differ in a single phrase (आज से एक दिन पहले vs आज के बाद), so their vectors are
almost identical. The past verb संबोधित किया is the only cue.
**Cause:** near-identical glosses + tense not modelled.

### 7. काल: fine-grained senses (the prediction is arguably right)
> प्राचीन वैदिक काल में मन्दिर नहीं होते थे।

Gold 878 *समय (time in general)* · pred 4212 *युग (era of history)* · conf 0.015.
"The ancient Vedic period" is really an *era*, so the prediction is defensible. The auto-label used the English
word "time". काल has 9 noun senses, including three for "period/era/time".
**Cause:** IndoWordNet senses finer than humans can reliably separate, and the auto-label picked one of them.

### 8. जाल: gloss-word bias
> इतिहास गवाह है कि जब भी किसी राष्ट्र ने बेहतर सड़कों का जाल बिछाया है, वहां प्रगति के पहिए बड़ी तेजी से घूमे हैं।

Gold 12880 *network, a woven group of many things* · pred 26732 *net used in tennis etc.* · conf 0.053.
The gold gloss is abstract and short, while the sports-net gloss is long and concrete. The "network of roads"
meaning is figurative, and LaBSE is pulled towards the concrete object.
**Cause:** short, abstract gloss; figurative use.

### 9. चाल: near-duplicate senses
> यदि तुम्हारी संचिका में रंगीन ग्राफिक्स हैं, तो मुद्रक चाल धीमी हो जाएगी।

Gold 10368 *गति, रफ़्तार (speed: distance per unit time)* · pred 11193 *गति, चाल, रफ़्तार (the act of moving)* · conf 0.032.
Both synsets share the lemmas गति and रफ़्तार, and a human would accept either answer. चाल has 15 senses.
**Cause:** duplicated / very fine-grained senses in IndoWordNet.

### 10. हार: English mode misled by an English idiom
> खुद बंगाल में हिंदु शासन, … स्थानीय शासक लक्ष्मण सेन की हार के साथ खत्म हो गया।
> *English: … with the defeat of the native ruler Lakshman Sen at the hands of the Turks.*

Hindi mode: correct (679 *पराजय, defeat*). English mode: pred 1206 *माला (garland)* · conf 0.000.
Adding the English sentence makes it worse: "at the hands of" and "Lakshman Sen" pull the vector towards a
physical, ornamental reading. Overall, English mode fixes 117 Hindi-mode errors but introduces 45 new ones.
**Cause:** cross-lingual context is a mixed signal; the English sentence can carry its own distractors.

## Summary of causes

| Cause | Cases above | Fixable how |
|---|---|---|
| Wrong lemma | 1 (fixed) | POS-aware lookup (done) |
| Wrong POS tag | 2 | Fall back to all senses when confidence is low; better tagger |
| Test set 1 label noise | 3, 4 (and partly 7) | Hand-check (data/test_auto_handcheck.csv); stricter keywords |
| Tense / time words | 5, 6 | Add a tense feature for कल; out of scope for Lesk |
| Fine-grained or duplicate senses | 7, 9 | Merge near-duplicate synsets; report top-2 accuracy |
| Short / abstract gloss | 8 | Extended Lesk: add glosses of related synsets |
| Misleading English context | 10 | Lower `w_en`; only use English when Hindi confidence is low |

## After the improvement: LaBSE + HindSBERT + sense-order prior

The error analysis suggested that glosses alone are a weak signal (cases 1, 8) and that close calls are
common (42% of errors had the gold sense in second place). We therefore tried, on the even rows of
test set 1 only, (a) four other sentence encoders, (b) adding synonyms and related-synset glosses
(Extended Lesk) to each sense, and (c) a small prior for IndoWordNet's earlier (more common) senses.
The best combination averages **LaBSE** (gloss + examples) and **L3Cube HindSBERT** (synonyms + gloss +
examples) and subtracts `0.1 × i / n` from sense *i* of *n*. Model selection details: `results/model_selection.md`.

| Test set 1 | LaBSE only | LaBSE + HindSBERT + prior |
|---|---|---|
| Hindi mode | 56.9% | **65.6%** |
| English mode | 61.4% | **69.7%** |
| Hindi mode, top-2 | 74.9% | 80.8% |
| Wrong answers | 689 | 550 |

The ten cases above with the new method:

| # | Word | Now | Comment |
|---|---|---|---|
| 1 | मान | ✅ correct | Synonyms सम्मान, आदर in the sense text match the context of honour |
| 2 | आम | ❌ | Still a POS-tag error: the mango sense is filtered out before scoring |
| 3 | कर | ❌ (label wrong) | ArthBodh says "do"; the auto-label "hand" is wrong |
| 4 | मूल | ❌ (label wrong) | ArthBodh says "fundamental", which is right; the auto-label "plant root" is wrong |
| 5 | कल | ❌ | Now says "tomorrow" for आए थे; tense is still not modelled |
| 6 | कल | ❌ | Same, confidence 0.023 |
| 7 | काल | ❌ | "era" vs "time"; confidence 0.000, shown as unsure |
| 8 | जाल | ❌ | Now "an old kind of cannon"; the abstract "network" gloss is still too weak |
| 9 | चाल | ❌ | Now "a move in chess or cards"; 15 near-duplicate senses |
| 10 | हार | ✅ correct (English mode) | HindSBERT is less distracted by "at the hands of" |

Remaining errors: 90 of 550 have an unreachable gold sense (lemma, POS or label problems), 242 have the gold
sense in second place. English mode now fixes 103 Hindi-mode errors and introduces 38. The weakest words are
गति (36%), कर and मूल (38%, mostly label noise), जाल (40%), चाल (42%) and काल (48%).
