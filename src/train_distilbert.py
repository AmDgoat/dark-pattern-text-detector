"""Fine-tune DistilBERT on the prepared dark-pattern splits.

Run from the repo root (after `python src/prepare_data.py --scheme 5`):
    python src/train_distilbert.py --out_dir /content/dp_model

One 6-way classifier is trained (Benign + 5 dark-pattern classes). The binary
"dark vs not dark" score is derived from it (anything not Benign = dark).
Class-weighted loss handles the imbalance; the best epoch is picked on the
validation macro-F1 and only then evaluated once on the test split.
"""
import argparse
import dataclasses
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def make_args(out_dir, a, n_train):
    """Build TrainingArguments that work across transformers 4.x and 5.x."""
    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    steps_per_epoch = math.ceil(n_train / a.batch_size)
    kw = dict(output_dir=os.path.join(out_dir, "checkpoints"), num_train_epochs=a.epochs,
              per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64,
              learning_rate=a.lr, weight_decay=0.01,
              warmup_steps=int(0.1 * steps_per_epoch * a.epochs),   # ~10% warm-up
              save_strategy="epoch", load_best_model_at_end=True,
              metric_for_best_model="macro_f1", greater_is_better=True,
              save_total_limit=1, fp16=torch.cuda.is_available(), seed=a.seed,
              report_to="none", logging_steps=25)
    kw["eval_strategy" if "eval_strategy" in names else "evaluation_strategy"] = "epoch"
    dropped = [k for k in kw if k not in names]
    if dropped:
        print("WARNING: this transformers version does not support:", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--out_dir", default="/content/dp_model")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs("results", exist_ok=True)

    tr, va, te = (pd.read_csv(f"data/processed/{n}.csv") for n in ["train", "val", "test"])
    labels = ["Benign"] + sorted(c for c in tr["category"].unique() if c != "Benign")
    l2i = {l: i for i, l in enumerate(labels)}
    for d in (tr, va, te):
        d["y"] = d["category"].map(l2i)
        assert d["y"].notna().all(), "unknown label in split"
    print("labels:", labels)

    counts = tr["y"].value_counts().reindex(range(len(labels))).values.astype(float)
    weights = (len(tr) / (len(labels) * counts)) ** 0.5   # softened inverse frequency
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float)
    print("class weights:", dict(zip(labels, weights.tolist())))

    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=len(labels),
        id2label={i: l for l, i in l2i.items()}, label2id=l2i)

    ds = {n: TextDataset(d["text"], d["y"], tok, a.max_len)
          for n, d in [("train", tr), ("val", va), ("test", te)]}
    trainer = WeightedTrainer(
        model=model, args=make_args(a.out_dir, a, len(tr)), train_dataset=ds["train"],
        eval_dataset=ds["val"], data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, class_weights=weights,
        processing_class=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    pred = trainer.predict(ds["test"])
    y_true, y_pred = te["y"].values, np.argmax(pred.predictions, axis=-1)
    rep = classification_report(y_true, y_pred, target_names=labels, digits=3,
                                output_dict=True, zero_division=0)
    print("\n=== 6-way category report (TEST) ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3, zero_division=0))
    bt, bp = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    brep = classification_report(bt, bp, target_names=["Not dark", "Dark"], digits=3,
                                 output_dict=True, zero_division=0)
    print("=== Dark vs not dark (TEST) ===")
    print(classification_report(bt, bp, target_names=["Not dark", "Dark"], digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    print("confusion matrix (rows=true, cols=pred), order:", labels)
    print(cm)

    metrics = {"model": a.model_name, "labels": labels,
               "test_accuracy": float(accuracy_score(y_true, y_pred)),
               "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
               "binary_accuracy": float(accuracy_score(bt, bp)),
               "binary_f1_dark": float(brep["Dark"]["f1-score"]),
               "category_report": rep, "binary_report": brep,
               "confusion_matrix": cm.tolist(), "hyperparameters": vars(a)}
    json.dump(metrics, open("results/distilbert_metrics.json", "w"), indent=2)
    err = te.assign(pred=[labels[i] for i in y_pred])
    err[err["category"] != err["pred"]][["text", "category", "pred"]].to_csv(
        "results/distilbert_test_errors.csv", index=False)

    os.makedirs(a.out_dir, exist_ok=True)
    final = os.path.join(a.out_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    json.dump({"labels": labels, "max_len": a.max_len},
              open(os.path.join(final, "label_map.json"), "w"), indent=2)
    print("\nSaved model to", final)
    print("Saved results/distilbert_metrics.json and results/distilbert_test_errors.csv")


if __name__ == "__main__":
    main()
