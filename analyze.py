# ============================================================
# FedUA-Net Comprehensive Publication Analysis Suite
#
# Computes:
#   1. Cross-seed means and standard deviations for all metrics
#   2. Per-client and pooled breakdowns
#   3. Rigorous paired statistical significance tests (Wilcoxon + Paired t-test + 95% CI)
#   4. Uncertainty calibration ladder (Raw ECE vs Calibrated ECE, Brier score)
#   5. Conformal Prediction APS efficiency (Coverage vs Mean Set Size at alpha=0.05, 0.10, 0.20)
#   6. Selective classification risk-coverage area under curve (AUC-RC)
#   7. Formatted LaTeX and Markdown tables for manuscript
# ============================================================
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

_ap = argparse.ArgumentParser()
_ap.add_argument('--out', default='./outputs_experiments')
OUT = Path(_ap.parse_args().out)
RAW = OUT / 'raw'
REP = OUT / 'reports'
REP.mkdir(parents=True, exist_ok=True)

REF = 'fedua'

def load_raw():
    frames = []
    for f in sorted(RAW.glob('raw_*_seed*.csv')):
        if 'smoke' in f.name:
            continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit('no raw files found in ' + str(RAW))
    return pd.concat(frames, ignore_index=True)

def main():
    df = load_raw()
    df['client'] = df['client'].astype(str)
    
    # Filter out pooled row from per-client average for centralized
    df_clients_only = df[df['client'].isin(['0', '1', '2'])]
    
    # Per-seed, per-strategy mean over the 3 clients
    mean_entry = df_clients_only.groupby(['seed', 'strategy']).agg(
        mean_acc=('acc', 'mean'),
        mean_f1=('f1', 'mean'),
        mean_mcc=('mcc', 'mean'),
        mean_auc=('auc', 'mean'),
        mean_ece=('ece', 'mean'),
        mean_brier=('brier', 'mean')
    ).reset_index()

    print('=' * 85)
    print('                    FEDUA-NET COMPREHENSIVE BENCHMARK REPORT')
    print('=' * 85)
    
    # Per-client breakdown
    clients = sorted(df_clients_only['client'].unique())
    client_names = {
        '0': 'Hospital A (Brain-Tumor MRI)',
        '1': 'Hospital B (Breast Ultrasound)',
        '2': 'Hospital C (COVID-19 X-Ray)'
    }
    
    met_rows = []
    for strat in sorted(df['strategy'].unique()):
        for c in sorted(df['client'].unique()):
            sub = df[(df['strategy'] == strat) & (df['client'] == c)]
            if sub.empty:
                continue
            row = {'strategy': strat, 'client': c, 'client_name': client_names.get(c, 'Pooled (11 cls)')}
            for k in ('acc', 'f1', 'mcc', 'auc', 'ece', 'brier'):
                row[f'{k}_mean'] = sub[k].mean()
                row[f'{k}_std'] = sub[k].std()
            met_rows.append(row)
    per_client_df = pd.DataFrame(met_rows)
    per_client_df.to_csv(REP / 'per_client_metrics.csv', index=False)
    
    # Summary across all seeds
    sum_rows = []
    for s in mean_entry['strategy'].unique():
        g = mean_entry[mean_entry['strategy'] == s]
        sum_rows.append({
            'strategy': s,
            'acc_mean': g['mean_acc'].mean(),
            'acc_std': g['mean_acc'].std(),
            'f1_mean': g['mean_f1'].mean(),
            'f1_std': g['mean_f1'].std(),
            'mcc_mean': g['mean_mcc'].mean(),
            'mcc_std': g['mean_mcc'].std(),
            'auc_mean': g['mean_auc'].mean(),
            'auc_std': g['mean_auc'].std(),
            'ece_mean': g['mean_ece'].mean(),
            'ece_std': g['mean_ece'].std(),
            'brier_mean': g['mean_brier'].mean(),
            'brier_std': g['mean_brier'].std(),
            'n_seeds': len(g)
        })
    summary = pd.DataFrame(sum_rows).set_index('strategy').sort_values('acc_mean', ascending=False)
    summary.to_csv(REP / 'summary.csv')
    
    print('\n[1] Overall Performance Summary (Mean +/- Std across seeds):')
    print('-' * 85)
    disp_summary = summary.copy()
    for col in ['acc', 'f1', 'mcc', 'auc']:
        disp_summary[f'{col}'] = disp_summary.apply(lambda r: f"{r[f'{col}_mean']*100:.2f} +/- {r[f'{col}_std']*100:.2f}%", axis=1)
    for col in ['ece', 'brier']:
        disp_summary[f'{col}'] = disp_summary.apply(lambda r: f"{r[f'{col}_mean']:.4f} +/- {r[f'{col}_std']:.4f}", axis=1)
    print(disp_summary[['acc', 'f1', 'mcc', 'auc', 'ece', 'brier', 'n_seeds']].to_string())
    
    # Statistical Significance Testing
    print('\n[2] Paired Statistical Significance Tests (vs Reference: FedUA-Net):')
    print('-' * 85)
    ref_data = mean_entry[mean_entry['strategy'] == REF].set_index('seed')
    stat_rows = []
    
    def compute_bootstrap_ci(diff, n_boot=10000, ci=0.95, seed=42):
        if len(diff) < 2:
            return float(np.mean(diff)), float(np.mean(diff))
        rng = np.random.default_rng(seed)
        boot_means = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(n_boot)]
        low = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100))
        high = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100))
        return low, high
    
    for s in summary.index:
        if s == REF:
            continue
        other_data = mean_entry[mean_entry['strategy'] == s].set_index('seed')
        common = ref_data.index.intersection(other_data.index)
        if len(common) < 2:
            continue
        a = ref_data.loc[common, 'mean_acc'].values
        b = other_data.loc[common, 'mean_acc'].values
        diff = a - b
        
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(a, b)
        
        # Wilcoxon
        if np.all(diff == 0):
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                w_stat, w_pval = stats.wilcoxon(a, b, zero_method='wilcox')
            except Exception:
                w_stat, w_pval = np.nan, np.nan
                
        # 95% Parametric Confidence Interval of delta
        se = stats.sem(diff) if len(diff) > 1 else 0.0
        ci95 = se * stats.t.ppf((1 + 0.95) / 2., len(diff) - 1) if len(diff) > 1 else 0.0
        
        # 95% Bootstrap Confidence Interval (10,000 resamples)
        boot_low, boot_high = compute_bootstrap_ci(diff, n_boot=10000, ci=0.95)
        
        stat_rows.append({
            'reference': REF,
            'baseline': s,
            'delta_acc_pct': float(np.mean(diff) * 100),
            'ci95_low': float((np.mean(diff) - ci95) * 100),
            'ci95_high': float((np.mean(diff) + ci95) * 100),
            'boot_ci95_low': float(boot_low * 100),
            'boot_ci95_high': float(boot_high * 100),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pval),
            'wilcoxon_stat': float(w_stat),
            'wilcoxon_pvalue': float(w_pval)
        })

    # Apply Holm-Bonferroni correction across all pairwise comparisons
    if stat_rows:
        t_pvals = [r['t_pvalue'] if not np.isnan(r['t_pvalue']) else 1.0 for r in stat_rows]
        w_pvals = [r['wilcoxon_pvalue'] if not np.isnan(r['wilcoxon_pvalue']) else 1.0 for r in stat_rows]
        
        _, t_pvals_holm, _, _ = multipletests(t_pvals, method='holm')
        _, w_pvals_holm, _, _ = multipletests(w_pvals, method='holm')
        
        for i, row in enumerate(stat_rows):
            row['t_pvalue_holm'] = float(t_pvals_holm[i])
            row['wilcoxon_pvalue_holm'] = float(w_pvals_holm[i])
            s = row['baseline']
            delta = row['delta_acc_pct']
            ci_l, ci_h = row['boot_ci95_low'], row['boot_ci95_high']
            t_p, t_ph = row['t_pvalue'], row['t_pvalue_holm']
            w_p, w_ph = row['wilcoxon_pvalue'], row['wilcoxon_pvalue_holm']
            print(f"  FedUA-Net vs {s:12s}: Delta Acc = {delta:+6.2f}% [Boot 95% CI: {ci_l:+5.2f}%, {ci_h:+5.2f}%] | t-pval={t_p:.4f} (Holm: {t_ph:.4f}) | Wilc-pval={w_p:.4f} (Holm: {w_ph:.4f})")
        
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(REP / 'statistical_significance.csv', index=False)
    
    # Calibration & Conformal Analysis
    cal_files = sorted(RAW.glob('cal_*.csv'))
    conf_summary = None
    if cal_files:
        cal = pd.concat([pd.read_csv(f) for f in cal_files if 'smoke' not in f.name], ignore_index=True)
        cb = cal[cal['metric'] == 'calib']
        if not cb.empty:
            print('\n[3] Uncertainty Calibration Comparison (Raw vs Temperature-Scaled):')
            print('-' * 85)
            cal_agg = cb.groupby('strategy').agg(
                ece_raw=('ece_raw', 'mean'),
                ece_cal=('ece_cal', 'mean'),
                brier_raw=('brier_raw', 'mean'),
                brier_cal=('brier_cal', 'mean'),
                temp=('temp', 'mean')
            ).reset_index()
            cal_agg['ece_reduction_pct'] = (cal_agg['ece_raw'] - cal_agg['ece_cal']) / cal_agg['ece_raw'] * 100
            print(cal_agg.to_string(index=False))
            cal_agg.to_csv(REP / 'calibration_comparison.csv', index=False)
            
        conf = cal[cal['alpha'].notna()]
        if not conf.empty:
            print('\n[4] Conformal Prediction APS Performance (Target Coverage vs Set Efficiency):')
            print('-' * 85)
            conf_agg = conf.groupby(['strategy', 'alpha']).agg(
                target_coverage=('alpha', lambda x: f"{(1-x.iloc[0])*100:.0f}%"),
                empirical_coverage=('coverage', 'mean'),
                coverage_std=('coverage', 'std'),
                mean_set_size=('mean_set_size', 'mean'),
                set_size_std=('mean_set_size', 'std'),
                qhat_mean=('qhat', 'mean')
            ).reset_index()
            print(conf_agg.to_string(index=False))
            conf_agg.to_csv(REP / 'conformal_results.csv', index=False)
            conf_summary = conf_agg
            
        pv = cal[cal['metric'] == 'risk_cov']
        if not pv.empty:
            print('\n[5] Risk-Coverage (Selective Classification AUC):')
            print('-' * 85)
            cols = [c for c in pv.columns if 'acc_at_cov' in c or 'acc_cov_auc' in c]
            rc = pv.groupby('strategy')[cols].agg('mean').reset_index()
            print(rc.to_string(index=False))
            rc.to_csv(REP / 'risk_coverage_summary.csv', index=False)

    # Generate Markdown Table for Paper
    print('\n' + '=' * 85)
    print('                      TABLE I: PUBLICATION MARKDOWN TABLE')
    print('=' * 85)
    print("| Method | Accuracy (%) | Macro F1 (%) | MCC | ECE (Uncal / Cal) | Brier Score | APS Set Size (α=0.10) |")
    print("|---|---|---|---|---|---|---|")
    
    set_sizes_10 = {}
    if conf_summary is not None:
        sub10 = conf_summary[conf_summary['alpha'] == 0.10]
        set_sizes_10 = dict(zip(sub10['strategy'], sub10['mean_set_size']))
        
    cal_mapping = {}
    if cal_files and not cb.empty:
        cal_mapping = dict(zip(cal_agg['strategy'], zip(cal_agg['ece_raw'], cal_agg['ece_cal'], cal_agg['brier_cal'])))
        
    for strat, row in summary.iterrows():
        acc_s = f"{row['acc_mean']*100:.2f} ± {row['acc_std']*100:.2f}"
        f1_s = f"{row['f1_mean']*100:.2f} ± {row['f1_std']*100:.2f}"
        mcc_s = f"{row['mcc_mean']:.3f} ± {row['mcc_std']:.3f}"
        
        if strat in cal_mapping:
            er, ec, br = cal_mapping[strat]
            ece_s = f"{er:.4f} → **{ec:.4f}**"
            brier_s = f"{br:.4f}"
        else:
            ece_s = f"{row['ece_mean']:.4f}"
            brier_s = f"{row['brier_mean']:.4f}"
            
        ss = set_sizes_10.get(strat, np.nan)
        ss_s = f"{ss:.2f}" if pd.notna(ss) else "—"
        
        strat_display = "FedUA-Net (Proposed)" if strat == 'fedua' else strat.capitalize()
        print(f"| {strat_display} | {acc_s} | {f1_s} | {mcc_s} | {ece_s} | {brier_s} | {ss_s} |")
        
    # Generate LaTeX Table
    with open(REP / 'table1_publication.tex', 'w', encoding='utf-8') as f:
        f.write("% Table I: Multi-Strategy Federated Learning Performance Comparison\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Quantitative comparison across 3 clinical sites (3-seed mean $\\pm$ std). Uncertainty metrics reported with Temperature Scaling; Conformal APS set size evaluated at 90\\% target coverage ($\\alpha=0.10$).}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Method & Accuracy (\\%) & Macro F1 (\\%) & MCC & Raw ECE & Calibrated ECE & APS Set Size (\\alpha=0.10) \\\\\n\\midrule\n")
        for strat, row in summary.iterrows():
            strat_display = "\\textbf{FedUA-Net (Ours)}" if strat == 'fedua' else strat.capitalize()
            acc_s = f"{row['acc_mean']*100:.2f} \\pm {row['acc_std']*100:.2f}"
            f1_s = f"{row['f1_mean']*100:.2f} \\pm {row['f1_std']*100:.2f}"
            mcc_s = f"{row['mcc_mean']:.3f}"
            if strat in cal_mapping:
                er, ec, br = cal_mapping[strat]
                er_s = f"{er:.4f}"
                ec_s = f"\\textbf{{{ec:.4f}}}" if strat == 'fedua' else f"{ec:.4f}"
            else:
                er_s = f"{row['ece_mean']:.4f}"
                ec_s = "—"
            ss = set_sizes_10.get(strat, np.nan)
            ss_s = f"{ss:.2f}" if pd.notna(ss) else "—"
            f.write(f"{strat_display} & {acc_s} & {f1_s} & {mcc_s} & {er_s} & {ec_s} & {ss_s} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
        
    print(f"\n[OK] Comprehensive reports generated in: {REP}")

if __name__ == '__main__':
    main()