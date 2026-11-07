"""Local classification server for the Dark Pattern Detector extension.

Start (from the repo root):
    python src/server.py                         # uses models/final if present
    python src/server.py --model_dir models/final --port 8000

Backends
    distilbert : fine-tuned DistilBERT in --model_dir (needs torch + transformers)
    tfidf      : fallback trained at start-up from data/processed (needs only scikit-learn)
Rule layer (src/rules.py) is applied on top of either backend.
All text stays on this computer; nothing is stored.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import match_rules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_TEXTS, MAX_CHARS = 500, 1000


class TfidfBackend:
    name = "tfidf"

    def __init__(self):
        import pandas as pd
        from baseline import make_model
        df = pd.concat([pd.read_csv(ROOT / f"data/processed/{n}.csv") for n in ["train", "val"]])
        self.model = make_model().fit(df["text"], df["category"])
        self.labels = list(self.model.classes_)

    def predict(self, texts):
        proba = self.model.predict_proba(texts)
        idx = proba.argmax(axis=1)
        return [(self.labels[i], float(proba[k, i])) for k, i in enumerate(idx)]


class BertBackend:
    name = "distilbert"

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        meta = json.load(open(Path(model_dir) / "label_map.json"))
        self.labels, self.max_len = meta["labels"], meta.get("max_len", 128)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.predict(["warm up"])

    def predict(self, texts, batch=64):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt")
                p = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
                out += [(self.labels[j], float(p[k, j])) for k, j in enumerate(p.argmax(axis=1))]
        return out


def load_backend(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        try:
            b = BertBackend(model_dir)
            print(f"[server] DistilBERT model loaded from {model_dir}")
            return b
        except Exception as e:  # missing torch, corrupt files, ...
            print(f"[server] WARNING: could not load DistilBERT ({type(e).__name__}: {e})")
    else:
        print(f"[server] No model found in {model_dir}")
    print("[server] Falling back to the TF-IDF backend")
    return TfidfBackend()


class Req(BaseModel):
    texts: List[str]


app = FastAPI(title="Dark Pattern Detector (local)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BACKEND = None


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND.name, "labels": BACKEND.labels}


@app.post("/classify")
def classify(req: Req):
    t0 = time.perf_counter()
    texts = [t[:MAX_CHARS] for t in req.texts[:MAX_TEXTS]]
    preds = BACKEND.predict(texts) if texts else []
    results = []
    for text, (label, conf) in zip(texts, preds):
        item = {"label": label, "confidence": round(conf, 4), "source": BACKEND.name,
                "model_label": label, "model_confidence": round(conf, 4)}
        rule = match_rules(text)
        if rule:
            item.update(label=rule[0], confidence=rule[1], source="rule")
        results.append(item)
    return {"backend": BACKEND.name, "ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": results}


def main():
    global BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "models" / "final"))
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    BACKEND = load_backend(a.model_dir)
    print(f"[server] Ready on http://127.0.0.1:{a.port}  (backend: {BACKEND.name})")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
