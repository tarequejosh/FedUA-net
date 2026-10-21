"""
JOURNAL EXTENSION -- reporting only, does not train or touch model weights.

Aggregates the new fedconformal_{strategy}_seed{seed}.csv and
ood_{strategy}_seed{seed}.csv files (written by experiment.py when run with
--federated_conformal / --ood_eval) across seeds, and produces:
  - reports/federated_conformal_comparison.csv : per-client, per-alpha,
    per-mode (group / marginal_federated / shrunk) coverage + mean set size,
    aggregated (mean +/- std) across seeds -- the direct comparison table for
    "legacy independent per-client conformal" (mode='group', which is exactly
    what the conference version computed, now on the fixed calib split) vs.
    the two genuinely federated modes.
  - reports/ood_robustness_summary.csv : accuracy / ECE / conformal coverage
    on out-of-distribution brain-MRI cohorts (brisc2025, PMRAM), aggregated
    across seeds, for the domain-shift stress test.

Usage:
    python scripts/report_federated_conformal.py --out ./outputs_experiments
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./outputs_experiments')
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / 'raw'
    reports = out / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    fed_files = sorted(raw.glob('fedconformal_*_seed*.csv'))
    if fed_files:
        df = pd.concat([pd.read_csv(f) for f in fed_files], ignore_index=True)
        agg = (df.groupby(['strategy', 'client_name', 'alpha', 'mode'])
                 .agg(coverage_mean=('coverage', 'mean'),
                      coverage_std=('coverage', 'std'),
                      set_size_mean=('mean_set_size', 'mean'),
                      set_size_std=('mean_set_size', 'std'),
                      n_seeds=('coverage', 'count'),
                      n_calib_mean=('n_calib', 'mean'),
                      shrinkage_weight_mean=('shrinkage_weight', 'mean'))
                 .reset_index())
        agg.to_csv(reports / 'federated_conformal_comparison.csv', index=False)
        print(f'[OK] wrote {reports / "federated_conformal_comparison.csv"} '
              f'({len(agg)} rows from {len(fed_files)} seed files)')
        print(agg.to_string(index=False))
    else:
        print('[WARN] no fedconformal_*_seed*.csv files found under', raw,
              '-- re-run experiment.py with --federated_conformal first.')

    ood_files = sorted(raw.glob('ood_*_seed*.csv'))
    if ood_files:
        df = pd.concat([pd.read_csv(f) for f in ood_files], ignore_index=True)
        agg = (df.groupby(['strategy', 'cohort', 'alpha'])
                 .agg(acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
                      ece_mean=('ece', 'mean'),
                      coverage_mean=('coverage', 'mean'), coverage_std=('coverage', 'std'),
                      set_size_mean=('mean_set_size', 'mean'),
                      n_ood=('n_ood', 'first'), n_seeds=('acc', 'count'))
                 .reset_index())
        agg.to_csv(reports / 'ood_robustness_summary.csv', index=False)
        print(f'\n[OK] wrote {reports / "ood_robustness_summary.csv"} '
              f'({len(agg)} rows from {len(ood_files)} seed files)')
        print(agg.to_string(index=False))
    else:
        print('[WARN] no ood_*_seed*.csv files found under', raw,
              '-- re-run experiment.py with --ood_eval brisc2025 pmram first.')


if __name__ == '__main__':
    main()
