"""Clean the ec-darkpattern dataset, map labels, and make train/val/test splits.

Usage:
    python src/prepare_data.py --scheme 5     # recommended (rare classes merged)
    python src/prepare_data.py --scheme 7     # all 7 synopsis categories
Splits are grouped by page_id so texts from one web page never appear in
both train and test (avoids leakage from repeated page templates).
"""
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW = "data/raw/ec_darkpattern_dataset.tsv"
RARE = ["Sneaking", "Obstruction", "Forced Action"]


def build_labels(df, scheme):
    cat = df["Pattern Category"].replace({"Not Dark Pattern": "Benign"})
    if scheme == 5:
        cat = cat.replace({c: "Other" for c in RARE})
    return cat


def group_split(df, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(gss.split(df, groups=df["page_id"]))
    return df.iloc[a], df.iloc[b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", type=int, choices=[5, 7], default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(RAW, sep="\t")
    n0 = len(df)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 1]
    df = df.drop_duplicates(subset="text")
    df["_k"] = df["text"].str.lower()
    df = df.drop_duplicates(subset="_k").drop(columns="_k")
    df["category"] = build_labels(df, args.scheme)
    df["is_dark"] = (df["category"] != "Benign").astype(int)
    print(f"rows: {n0} -> {len(df)} after cleaning/dedup")

    trainval, test = group_split(df, 0.15, args.seed)
    train, val = group_split(trainval, 0.176, args.seed)  # ~15% of total
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["page_id", "text", "category", "is_dark"]].to_csv(
            f"data/processed/{name}.csv", index=False)
        print(f"\n{name}: {len(part)} rows")
        print(part["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
