# AI Powered Dark Pattern E-Commerce Manipulative Text Detector

Final-year B.Tech project (AI & DS, Yashoda Technical Campus, Satara, DBATU 2026-27).
A browser extension that reads the text of an e-commerce page, classifies each
segment with a fine-tuned DistilBERT model, and highlights dark patterns by category.

## Status
- [x] Phase 1: data preparation (cleaning, label mapping, page-grouped split)
- [x] Phase 2: TF-IDF + Logistic Regression baseline
- [ ] Phase 3: DistilBERT fine-tuning (Colab) – notebook ready: `notebooks/train_distilbert_colab.ipynb`
- [ ] Phase 4: Manifest V3 extension
- [ ] Phase 5: testing on real shopping sites

## Quick start
```bash
pip install -r requirements.txt
python src/prepare_data.py --scheme 5   # or --scheme 7
python src/baseline.py
```

## Data
`data/raw/ec_darkpattern_dataset.tsv` is the dataset of Yada et al. (2022), from
https://github.com/yamanalab/ec-darkpattern (Apache-2.0). Its dark-pattern texts come
from Mathur et al. (2019). Splits are grouped by `page_id` so no web page appears in
both train and test.

## Baseline results (TF-IDF + LR, test split, 343 texts)
| Task | Accuracy | Macro F1 |
|---|---|---|
| Dark vs not dark | 0.962 | 0.962 |
| 5-class category (scheme 5) | 0.950 | 0.895 |

Rare classes (Sneaking 12, Obstruction 27, Forced Action 4 samples in total) are too
small to evaluate reliably; see `results/`.
