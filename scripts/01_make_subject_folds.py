import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature CSV path")
    parser.add_argument("--out", required=True, help="Output folds JSON")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--subject_col", default="subject_id")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    required = {args.subject_col, args.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    subject_labels = (
        df.groupby(args.subject_col)[args.label_col]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    subjects = subject_labels[args.subject_col].values
    labels = subject_labels[args.label_col].values

    try:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        splits = splitter.split(subjects, labels)
    except ValueError:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        splits = splitter.split(subjects)

    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_subjects = subjects[train_idx].tolist()
        test_subjects = subjects[test_idx].tolist()
        folds.append({
            "fold": fold_id,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(folds, indent=2), encoding="utf-8")

    print(f"Saved folds to {out}")
    print(f"Number of folds: {len(folds)}")
    for f in folds:
        print(f"Fold {f['fold']}: train subjects={len(f['train_subjects'])}, test subjects={len(f['test_subjects'])}")

if __name__ == "__main__":
    main()
