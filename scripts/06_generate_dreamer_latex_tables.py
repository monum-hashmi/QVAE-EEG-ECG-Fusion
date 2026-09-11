import argparse
from pathlib import Path

import pandas as pd


def fmt(mean, std):
    if pd.isna(mean):
        return '--'
    if pd.isna(std):
        return f'{mean:.3f}'
    return f'{mean:.3f} $\\pm$ {std:.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', default='overleaf/dreamer_results_tables.tex')
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = ['accuracy', 'macro_f1', 'balanced_accuracy', 'mcc', 'auroc']
    g = df.groupby(['model', 'input', 'pairing'])[metrics].agg(['mean', 'std']).reset_index()

    lines = []
    lines.append('\\begin{table*}[!t]')
    lines.append('\\centering')
    lines.append('\\caption{Classical baseline performance on DREAMER under subject-independent evaluation. Values are mean $\\pm$ standard deviation across folds.}')
    lines.append('\\label{tab:dreamer_classical_results}')
    lines.append('\\begin{tabular}{llllllll}')
    lines.append('\\toprule')
    lines.append('Model & Input & Pairing & Accuracy & Macro-F1 & Bal. Acc. & MCC & AUROC \\\\')
    lines.append('\\midrule')

    order = {
        'eeg_only': 0,
        'ecg_only': 1,
        'correct': 2,
        'wrong_trial': 3,
        'cross_subject_label_matched': 4,
        'cross_subject_unrestricted': 5,
    }
    g['pair_order'] = g['pairing'].map(order).fillna(99)
    g = g.sort_values(['pair_order', 'model'])

    for _, row in g.iterrows():
        vals = [fmt(row[(m, 'mean')], row[(m, 'std')]) for m in metrics]
        lines.append(
            f"{row['model']} & {row['input']} & {row['pairing']} & "
            f"{vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\\\"
        )

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    pair_df = g[g['input'] == 'EEG--ECG'].copy()
    if not pair_df.empty:
        lines.append('\\begin{table}[!t]')
        lines.append('\\centering')
        lines.append('\\caption{Pair-validity comparison on DREAMER. For each pairing condition, the best classical model is selected by mean Macro-F1.}')
        lines.append('\\label{tab:dreamer_pair_validity_results}')
        lines.append('\\begin{tabular}{llll}')
        lines.append('\\toprule')
        lines.append('Pairing & Best Model & Macro-F1 & MCC \\\\')
        lines.append('\\midrule')

        pair_df['macro_f1_mean'] = pair_df[('macro_f1', 'mean')]
        for pairing, sub in pair_df.groupby('pairing'):
            best = sub.sort_values('macro_f1_mean', ascending=False).iloc[0]
            mf1 = fmt(best[('macro_f1', 'mean')], best[('macro_f1', 'std')])
            mcc = fmt(best[('mcc', 'mean')], best[('mcc', 'std')])
            lines.append(f"{pairing} & {best['model']} & {mf1} & {mcc} \\\\")

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')

    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved LaTeX tables to {out}')


if __name__ == '__main__':
    main()
