import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


ID_COLS_DEFAULT = ["subject_id", "trial_id", "window_id", "label", "pairing"]


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    return models


def get_feature_columns(df, id_cols):
    return [c for c in df.columns if c not in id_cols]


def safe_auroc(y_true, proba_or_score):
    try:
        if proba_or_score.ndim == 2:
            if proba_or_score.shape[1] == 2:
                score = proba_or_score[:, 1]
                return roc_auc_score(y_true, score)
            return roc_auc_score(y_true, proba_or_score, multi_class="ovr")
        return roc_auc_score(y_true, proba_or_score)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--pairing_filter", default=None, help="Optional pairing value: correct/eeg_only/ecg_only/etc.")
    args = parser.parse_args()

    df = pd.read_csv(args.features)

    if args.pairing_filter is not None:
        df = df[df["pairing"] == args.pairing_filter].copy()
        if df.empty:
            raise ValueError(f"No rows left after pairing_filter={args.pairing_filter}")

    required = set(ID_COLS_DEFAULT)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature CSV missing required columns: {missing}")

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    feature_cols = get_feature_columns(df, ID_COLS_DEFAULT)

    if not feature_cols:
        raise ValueError("No feature columns found.")

    models = build_models()
    rows = []

    for fold in folds:
        fold_id = fold["fold"]
        train_subjects = set(fold["train_subjects"])
        test_subjects = set(fold["test_subjects"])

        train_df = df[df[args.subject_col].isin(train_subjects)].copy()
        test_df = df[df[args.subject_col].isin(test_subjects)].copy()

        if train_df.empty or test_df.empty:
            print(f"Skipping fold {fold_id}: empty train/test after filtering")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[args.label_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[args.label_col].values

        for model_name, model in models.items():
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - start

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
            elif hasattr(model, "decision_function"):
                proba = model.decision_function(X_test)
            else:
                proba = None

            auroc = safe_auroc(y_test, proba) if proba is not None else np.nan

            rows.append({
                "fold": fold_id,
                "model": model_name,
                "pairing": args.pairing_filter or "all",
                "input": "features",
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "auroc": auroc,
                "train_time_sec": train_time,
                "infer_time_sec": infer_time,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_features": len(feature_cols),
            })

            print(f"Fold {fold_id} | {model_name}: macro_f1={rows[-1]['macro_f1']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(f"Saved results to {out}")

    summary = (
        res.groupby(["model", "pairing"])
        [["accuracy", "macro_f1", "balanced_accuracy", "mcc", "auroc", "train_time_sec"]]
        .agg(["mean", "std"])
    )
    print(summary)

if __name__ == "__main__":
    main()
