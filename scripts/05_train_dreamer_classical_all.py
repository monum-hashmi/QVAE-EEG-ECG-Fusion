import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

META_COLS = ["subject_id", "trial_id", "window_id", "label", "pairing"]


def build_models():
    models = {
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        "KNN": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM-RBF": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale"))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", n_jobs=-1))]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1))])
    return models


def safe_auroc(y_true, model, X_test):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return roc_auc_score(y_true, proba[:, 1])
            return roc_auc_score(y_true, proba, multi_class="ovr")
    except Exception:
        return np.nan
    return np.nan


def make_subject_folds(df, n_splits=5):
    subject_labels = df.groupby("subject_id")["label"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    subjects = subject_labels["subject_id"].to_numpy()
    labels = subject_labels["label"].to_numpy()
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(subjects, labels)
    except ValueError:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(subjects)
    return [(fid, set(subjects[tr]), set(subjects[te])) for fid, (tr, te) in enumerate(split_iter)]


def infer_input_name(path):
    name = path.stem.lower()
    if "eeg_only" in name:
        return "EEG"
    if "ecg_only" in name:
        return "ECG"
    return "EEG--ECG"


def evaluate_file(csv_path, n_splits):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    models = build_models()
    rows = []
    input_name = infer_input_name(csv_path)
    folds = make_subject_folds(df, n_splits=n_splits)
    for pairing, pdf in df.groupby("pairing"):
        for fold_id, train_subjects, test_subjects in folds:
            train_df = pdf[pdf.subject_id.isin(train_subjects)].copy()
            test_df = pdf[pdf.subject_id.isin(test_subjects)].copy()
            if train_df.empty or test_df.empty:
                continue
            X_train = train_df[feature_cols].to_numpy()
            y_train = train_df["label"].to_numpy()
            X_test = test_df[feature_cols].to_numpy()
            y_test = test_df["label"].to_numpy()
            for model_name, model in models.items():
                start = time.time(); model.fit(X_train, y_train); train_time = time.time() - start
                start = time.time(); y_pred = model.predict(X_test); infer_time = time.time() - start
                rows.append({
                    "source_file": csv_path.name, "fold": fold_id, "model": model_name, "input": input_name, "pairing": pairing,
                    "accuracy": accuracy_score(y_test, y_pred), "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred), "mcc": matthews_corrcoef(y_test, y_pred),
                    "auroc": safe_auroc(y_test, model, X_test), "train_time_sec": train_time, "infer_time_sec": infer_time,
                    "n_train": len(train_df), "n_test": len(test_df), "n_features": len(feature_cols),
                })
                print(f"{csv_path.name} | {pairing} | fold {fold_id} | {model_name}: macro_f1={rows[-1]['macro_f1']:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/dreamer_features")
    parser.add_argument("--label", default="valence", choices=["valence", "arousal", "dominance"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    features_dir = Path(args.features_dir)
    out = Path(args.out) if args.out else Path("results") / f"dreamer_{args.label}_classical_all_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [
        features_dir / f"dreamer_{args.label}_eeg_only.csv",
        features_dir / f"dreamer_{args.label}_ecg_only.csv",
        features_dir / f"dreamer_{args.label}_correct.csv",
        features_dir / f"dreamer_{args.label}_paired_all.csv",
    ]
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir}. Run 04_build_dreamer_features.py first.")
    all_rows = []
    for p in files:
        all_rows.extend(evaluate_file(p, n_splits=args.n_splits))
    res = pd.DataFrame(all_rows)
    res.to_csv(out, index=False)
    print("\nSaved:", out)
    print("\nSummary:")
    print(res.groupby(["model", "input", "pairing"])[["accuracy", "macro_f1", "balanced_accuracy", "mcc", "auroc"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
