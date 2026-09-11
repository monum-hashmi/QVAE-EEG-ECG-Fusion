import argparse
from pathlib import Path
import pandas as pd

ORDER = {
    "eeg_only": 0,
    "ecg_only": 1,
    "correct": 2,
    "wrong_trial": 3,
    "cross_subject_label_matched": 4,
    "cross_subject_unrestricted": 5,
}

READABLE = {
    "eeg_only": "EEG only",
    "ecg_only": "ECG only",
    "correct": "Correct",
    "wrong_trial": "Wrong-trial",
    "cross_subject_label_matched": "Cross-subject label-matched",
    "cross_subject_unrestricted": "Cross-subject unrestricted",
}

def clean(x):
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    return str(x).strip()

def esc(x):
    return clean(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

def fmt(mean, std):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"

def flatten_columns(df):
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                new_cols.append(c[0])
            else:
                new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="overleaf/dreamer_results_tables_clean.tex")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "auroc"]

    g = df.groupby(["model", "input", "pairing"], as_index=False)[metrics].agg(["mean", "std"]).reset_index()
    g = flatten_columns(g)
    g["pair_order"] = g["pairing"].map(ORDER).fillna(99)
    g = g.sort_values(["pair_order", "input", "model"]).reset_index(drop=True)

    lines = []

    # Compact best-condition table for main paper
    best_rows = []
    for (input_name, pairing), sub in g.groupby(["input", "pairing"]):
        best_rows.append(sub.sort_values("macro_f1_mean", ascending=False).iloc[0])
    best = pd.DataFrame(best_rows)
    best["pair_order"] = best["pairing"].map(ORDER).fillna(99)
    best = best.sort_values(["pair_order", "input"]).reset_index(drop=True)

    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best classical baseline for each DREAMER valence input and pairing condition under subject-independent evaluation. Selection is based on mean Macro-F1.}")
    lines.append(r"\label{tab:dreamer_valence_best_classical}")
    lines.append(r"\begin{tabular}{lllllll}")
    lines.append(r"\toprule")
    lines.append(r"Input & Pairing & Best Model & Accuracy & Macro-F1 & MCC & AUROC \\")
    lines.append(r"\midrule")

    for _, row in best.iterrows():
        input_name = esc(row["input"])
        pairing = esc(READABLE.get(clean(row["pairing"]), clean(row["pairing"])))
        model = esc(row["model"])
        acc = fmt(row["accuracy_mean"], row["accuracy_std"])
        mf1 = fmt(row["macro_f1_mean"], row["macro_f1_std"])
        mcc = fmt(row["mcc_mean"], row["mcc_std"])
        auroc = fmt(row["auroc_mean"], row["auroc_std"])
        lines.append(f"{input_name} & {pairing} & {model} & {acc} & {mf1} & {mcc} & {auroc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # Pair-validity table
    pair_df = g[(g["input"] == "EEG--ECG") & (g["pairing"].isin(["correct", "wrong_trial", "cross_subject_label_matched", "cross_subject_unrestricted"]))].copy()
    pair_best_rows = []
    for pairing, sub in pair_df.groupby("pairing"):
        pair_best_rows.append(sub.sort_values("macro_f1_mean", ascending=False).iloc[0])
    pair_best = pd.DataFrame(pair_best_rows)
    pair_best["pair_order"] = pair_best["pairing"].map(ORDER).fillna(99)
    pair_best = pair_best.sort_values("pair_order")

    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pair-validity comparison on DREAMER valence classification. For each pairing condition, the best classical model is selected by mean Macro-F1.}")
    lines.append(r"\label{tab:dreamer_valence_pair_validity}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"Pairing & Best Model & Macro-F1 & MCC \\")
    lines.append(r"\midrule")

    for _, row in pair_best.iterrows():
        pairing = esc(READABLE.get(clean(row["pairing"]), clean(row["pairing"])))
        model = esc(row["model"])
        mf1 = fmt(row["macro_f1_mean"], row["macro_f1_std"])
        mcc = fmt(row["mcc_mean"], row["mcc_std"])
        lines.append(f"{pairing} & {model} & {mf1} & {mcc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append("% Detailed full table for appendix")
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Detailed classical baseline performance on DREAMER valence classification. Values are mean $\pm$ standard deviation across folds.}")
    lines.append(r"\label{tab:dreamer_valence_classical_detailed}")
    lines.append(r"\begin{tabular}{llllllll}")
    lines.append(r"\toprule")
    lines.append(r"Model & Input & Pairing & Accuracy & Macro-F1 & Bal. Acc. & MCC & AUROC \\")
    lines.append(r"\midrule")

    for _, row in g.iterrows():
        model = esc(row["model"])
        input_name = esc(row["input"])
        pairing = esc(READABLE.get(clean(row["pairing"]), clean(row["pairing"])))
        acc = fmt(row["accuracy_mean"], row["accuracy_std"])
        mf1 = fmt(row["macro_f1_mean"], row["macro_f1_std"])
        bal = fmt(row["balanced_accuracy_mean"], row["balanced_accuracy_std"])
        mcc = fmt(row["mcc_mean"], row["mcc_std"])
        auroc = fmt(row["auroc_mean"], row["auroc_std"])
        lines.append(f"{model} & {input_name} & {pairing} & {acc} & {mf1} & {bal} & {mcc} & {auroc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved clean LaTeX tables to {out}")

if __name__ == "__main__":
    main()
