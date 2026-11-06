"""TF-IDF + Logistic Regression baseline (binary and multi-class)."""
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union


def load(name):
    return pd.read_csv(f"data/processed/{name}.csv")


def make_model():
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
    )
    return make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced", C=5))


def run(target):
    tr, va, te = load("train"), load("val"), load("test")
    m = make_model().fit(tr["text"], tr[target])
    out = {}
    for nm, d in [("val", va), ("test", te)]:
        p = m.predict(d["text"])
        out[nm] = {"macro_f1": f1_score(d[target], p, average="macro")}
        if nm == "test":
            print(f"\n=== {target} : TEST ===")
            print(classification_report(d[target], p, digits=3, zero_division=0))
            out["report"] = classification_report(d[target], p, output_dict=True, zero_division=0)
    return out


if __name__ == "__main__":
    res = {t: run(t) for t in ["is_dark", "category"]}
    json.dump(res, open("results/baseline_metrics.json", "w"), indent=2)
