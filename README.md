# AI Powered Dark Pattern E-Commerce Manipulative Text Detector

Final-year B.Tech project - AI & Data Science, Yashoda Technical Campus, Satara (DBATU, 2026-27).
Team: _(add member names and guide here)_

A Chrome (Manifest V3) extension that reads the visible text of an e-commerce page, classifies each
segment with a fine-tuned **DistilBERT** model plus a small **regex rule layer**, and highlights wording that
*may* be manipulative ("dark patterns"). Each category has its own colour, a "Dark Pattern" badge, and a hover
tooltip with a plain-language explanation and the model's confidence. All text is processed on the user's own
computer; nothing is sent to the internet.

> It is an assistant, not a legal verdict. It marks wording that may be manipulative and never blocks anything.

## Status
- [x] Phase 1 - data preparation (cleaning, label mapping, page-grouped split)
- [x] Phase 2 - TF-IDF + Logistic Regression baseline
- [x] Phase 3 - DistilBERT fine-tuning (Google Colab, T4 GPU)
- [x] Phase 4 - local FastAPI server, Chrome extension, rule layer, false-alarm guards, demo page
- [ ] Phase 5 - testing on real shopping sites and error analysis (in progress)
- [ ] Phase 6 - final report, slides, backup demo video

## Categories
| Category | Colour | Typical wording |
|---|---|---|
| Urgency | red | "Sale ends in 02:15:40", "Today only" |
| Scarcity | orange | "Only 2 left in stock", "Selling fast" |
| Social Proof | blue | "27 people are looking at this right now" |
| Misdirection | purple | "No thanks, I don't like saving money" |
| Other | green | auto-renewing trials, forced marketing consent, hidden add-ons |

"Other" merges Sneaking, Obstruction and Forced Action, which have too few examples in the dataset to train
separately (12, 27 and 4 texts).

## Model results (test split, 343 texts, split by web page)
| Model | Dark vs not dark - accuracy | Dark-class F1 | 6-way accuracy | 6-way macro F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.962 | 0.960 | 0.950 | 0.895 |
| DistilBERT (fine-tuned) | 0.965 | 0.963 | 0.953 | 0.900 |

6-way = Benign + the five categories above. The two models are statistically close on this data.

DistilBERT per class:
| Class | Precision | Recall | F1 | Test texts |
|---|---|---|---|---|
| Benign | 0.956 | 0.978 | 0.967 | 178 |
| Misdirection | 0.829 | 0.829 | 0.829 | 35 |
| Other | 0.750 | 0.600 | 0.667 | 5 |
| Scarcity | 1.000 | 0.958 | 0.979 | 48 |
| Social Proof | 0.978 | 1.000 | 0.989 | 45 |
| Urgency | 1.000 | 0.938 | 0.968 | 32 |

"Other" has only 5 test texts, so its score is not reliable. Full metrics: `results/`.

## How it works
```
page text -> content.js (extract, split into sentences, skip hidden text and customer reviews)
          -> background.js -> local server http://127.0.0.1:8000 (FastAPI)
                                 DistilBERT model + src/rules.py wording rules
          <- label + confidence
          -> content.js guards (sensitivity threshold, ignore prices, counters, IDs, policy text,
                                product titles, half-sentences) -> highlights, tooltip, summary card
```

## Run it (Windows)
1. `pip install -r requirements.txt`
2. Put the trained model folder at `models/final/` (about 250 MB, **not** stored in this repo).
   If it is missing, the server falls back to the TF-IDF baseline.
3. Double-click `start_server.bat` and wait for `Ready on http://127.0.0.1:8000 (backend: distilbert)`.
   Keep that window open while using the extension.
4. Chrome: `chrome://extensions` -> turn on Developer mode -> **Load unpacked** -> choose the `extension` folder.
5. Open a shopping page, refresh it, click the extension icon and switch it on.

## Demo page
`demo/demo_shop.html` is a fictional test shop with every pattern category plus normal text.
In the `demo` folder run `python -m http.server 8080`, then open `http://localhost:8080/demo_shop.html`
with the extension on. Expected result: **14** flagged elements and none in the "normal text" section.

## Live testing notes
First tests on live shopping pages (Flipkart, Myntra, Amazon.in) showed that the dataset-trained model
over-flags ordinary page text: page counters, registration numbers, review counts and dates, return policies,
product titles, customer reviews and half-sentences split by links. A guard layer in `extension/content.js`
now ignores these. A quantitative real-site evaluation is in progress; results will be added here.

## Limitations
- Text only: it cannot see pre-ticked checkboxes, button size or colour, or costs revealed at checkout.
- It cannot tell whether a review or a "sold in the last 24 hours" claim is genuine.
- Rare categories (sneaking, obstruction, forced action) are merged into "Other" and are weakly covered.
- Training data comes from 2019 (mostly US shopping sites) and is English only.
- Needs the local server to be running. May show false alarms and misses.

## Repository layout
```
extension/   Chrome extension (manifest, content script, popup, icons)
src/         prepare_data.py, baseline.py, train_distilbert.py, rules.py, server.py
notebooks/   Colab notebook used to fine-tune DistilBERT
data/        raw dataset and train / val / test splits
results/     baseline and DistilBERT metrics, error list
demo/        demo_shop.html
```

## Credits and licence
- Dataset: Yada et al., *Dark patterns in e-commerce: a dataset and its baseline evaluations* (2022),
  https://github.com/yamanalab/ec-darkpattern (Apache-2.0). Its dark-pattern texts come from
  Mathur et al., *Dark Patterns at Scale* (2019).
- This project is released under the MIT licence.
