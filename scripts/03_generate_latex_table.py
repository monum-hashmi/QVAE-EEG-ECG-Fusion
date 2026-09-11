import argparse
from pathlib import Path

import pandas as pd

def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.results)

    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "auroc"]
    grouped = df.groupby(["model", "pairing"])[metrics].agg(["mean", "std"]).reset_index()

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Classical baseline performance under subject-independent evaluation. Values are mean $\pm$ standard deviation across folds.}")
    lines.append(r"\label{tab:classical_results}")
    lines.append(r"\begin{tabular}{lllllll}")
    lines.append(r"\toprule")
    lines.append(r"Model & Pairing & Accuracy & Macro-F1 & Bal. Acc. & MCC & AUROC \\")
    lines.append(r"\midrule")

    for _, row in grouped.iterrows():
        model = row["model"]
        pairing = row["pairing"]
        vals = []
        for m in metrics:
            vals.append(fmt_mean_std(row[(m, "mean")], row[(m, "std")]))
        lines.append(f"{model} & {pairing} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table to {out}")

if __name__ == "__main__":
    main()
