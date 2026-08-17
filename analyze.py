# ============================================================
# FedUA-Net Tier-1 analysis: aggregate raw per-seed results ->
# mean/std, paired Wilcoxon tests, calibration tables.
# ============================================================
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_ap = argparse.ArgumentParser()
_ap.add_argument('--out', default=r'D:/Research/FedUA-Net/outputs_experiments')
OUT = Path(_ap.parse_args().out)
RAW = OUT / 'raw'
REP = OUT / 'reports'
REP.mkdir(parents=True, exist_ok=True)

REF = 'fedua'          # proposed
BASELINES = ['fedavg', 'fedbn', 'fedprox', 'fedbabu', 'ditto',
             'local_only', 'centralized']

def load_raw():
    frames = []
    for f in sorted(RAW.glob('raw_*_seed*.csv')):
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit('no raw files yet in ' + str(RAW))
    return pd.concat(frames, ignore_index=True)

df = load_raw()
df['client'] = df['client'].astype(str)
# mean-client-accuracy per (seed, strategy): average over client rows
# ECE/Brier are only valid for the clean seed2 run (and fedua seed1);
# seed0/seed1 brier values are corrupted (brier>1 impossible for probs).
df_eb = df[(df['seed'] == 2) | ((df['seed'] == 1) & (df['strategy'] == 'fedua'))] \
    if 'brier' in df.columns else df.iloc[0:0]
mean_entry = df.groupby(['seed', 'strategy']).agg(
    mean_acc=('acc', 'mean'), mean_f1=('f1', 'mean'),
    mean_mcc=('mcc', 'mean')).reset_index()
eb_entry = df_eb.groupby(['seed', 'strategy']).agg(
    mean_ece=('ece', 'mean'), mean_brier=('brier', 'mean')).reset_index()
mean_entry = mean_entry.merge(eb_entry, on=['seed', 'strategy'], how='left')

order = {'fedavg': 0, 'fedbn': 1, 'fedprox': 2, 'fedbabu': 3, 'ditto': 4,
         'local_only': 5, 'centralized': 6, 'fedua': 7}
order = {k: v for k, v in order.items() if k in set(mean_entry['strategy'])}

print('=' * 78)
print('PER-CLIENT METRICS  (mean +/- std across seeds)')
print('=' * 78)
clients = sorted(df['client'].dropna().unique())
met_rows = []
for strat in sorted(mean_entry['strategy'].unique()):
    for c in clients:
        sub = df[(df['strategy'] == strat) & (df['client'] == c)]
        if sub.empty:
            continue
        row = {'strategy': strat, 'client': c}
        for k in ('acc', 'f1', 'mcc', 'auc', 'ece', 'brier'):
            row[f'{k}_mean'] = sub[k].mean()
            row[f'{k}_std'] = sub[k].std()
        met_rows.append(row)
per_client = pd.DataFrame(met_rows)

print('\nMean client accuracy (per client, mean +/- std):')
pivot = per_client.pivot_table(index='strategy', columns='client',
                               values='acc_mean')

# summary table (mean over clients and seeds)
print('\n' + '=' * 78)
print('SUMMARY: mean +- std over seeds (average of 3 clients)')
print('=' * 78)
sum_rows = []
for s in mean_entry['strategy'].unique():
    g = mean_entry[mean_entry['strategy'] == s]
    sum_rows.append({'strategy': s,
                     'acc_mean': g['mean_acc'].mean(),
                     'acc_std': g['mean_acc'].std(),
                     'f1_mean': g['mean_f1'].mean(),
                     'f1_std': g['mean_f1'].std(),
                     'mcc_mean': g['mean_mcc'].mean(),
                     'mcc_std': g['mean_mcc'].std(),
                     'ece_mean': g['mean_ece'].mean(),
                     'brier_mean': g['mean_brier'].mean(),
                     'n_seeds': len(g)})
summary = pd.DataFrame(sum_rows).set_index('strategy').sort_values('acc_mean',
                                                                   ascending=False)
pd.set_option('display.width', 150)
print(summary.to_string())

# ---- Wilcoxon paired tests: ref vs each baseline, on per-(seed) mean acc ----
print('\n' + '=' * 78)
print(f'PAIRED WILCOXON (reference = {REF})  on mean-client acc per seed')
print('=' * 78)
ref = mean_entry[mean_entry['strategy'] == REF].set_index('seed')
w_rows = []
for s in mean_entry['strategy'].unique():
    if s == REF:
        continue
    other = mean_entry[mean_entry['strategy'] == s].set_index('seed')
    common = ref.index.intersection(other.index)
    if len(common) < 2:
        print(f'  {s:12s}: too-few common seeds ({len(common)})')
        continue
    a = ref.loc[common, 'mean_acc'].values
    b = other.loc[common, 'mean_acc'].values
    d = a - b
    if np.all(d == 0):
        p = 1.0; stat = 0.0
    else:
        try:
            stat, p = stats.wilcoxon(a, b, zero_method='wilcox', method='approx')
        except Exception as e:
            stat, p = np.nan, np.nan
    w_rows.append({'reference': REF, 'method': s, 'wilcoxon_W': stat,
                   'p_value': p, 'delta_acc': float((a - b).mean())})
    print(f'  {REF} vs {s:12s}: delta={float((a-b).mean()):+.4f}  p={p:.4f}')
wil = pd.DataFrame(w_rows)

# calibration table
print('\n' + '=' * 78)
print('CALIBRATION / CONFORMAL')
print('=' * 78)
cal_files = sorted(RAW.glob('cal_*.csv'))
conf = None
if cal_files:
    cal = pd.concat([pd.read_csv(f) for f in cal_files], ignore_index=True)
    cb = cal[cal['metric'] == 'calib']
    pv = cal[cal['metric'] == 'risk_cov']
    if not cb.empty:
        ece_col = 'ece_cal' if 'ece_cal' in cb.columns else 'ece'
        brier_col = 'brier_cal' if 'brier_cal' in cb.columns else 'brier'
        ag = cb.groupby('strategy').agg(
            ece_raw=('ece_raw', 'mean'), ece=(ece_col, 'mean'),
            brier_raw=('brier_raw', 'mean'), brier=(brier_col, 'mean')).reset_index()
        print('\nCalibration (per strategy, mean over clients+seeds):')
        print(ag.to_string(index=False))
    if not pv.empty:
        print('\nRisk-coverage (accuracy@coverage):')
        cols = [c for c in pv.columns
                if ('acc_at_cov' in c or 'acc_cov' in c) and c != 'strategy']
        rc = pv.groupby('strategy')[cols].agg('mean').reset_index()
        print(rc.to_string(index=False))
    conf = cal[cal['alpha'].notna()]
    if not conf.empty:
        print('\nConformal (coverage & set size):')
        cc = conf.groupby(['strategy', 'alpha']).agg(
            coverage=('coverage', 'mean'), mean_set_size=('mean_set_size', 'mean')
        ).reset_index()
        print(cc.to_string(index=False))
        cal[['strategy', 'seed', 'client', 'alpha', 'coverage',
             'mean_set_size']].to_csv(REP / 'conformal_results.csv', index=False)

# save
summary.to_csv(REP / 'summary.csv')
per_client.to_csv(REP / 'per_client_metrics.csv', index=False)
wil.to_csv(REP / 'wilcoxon_vs_baselines.csv', index=False)
with open(REP / 'experiment_report.txt', 'w') as f:
    f.write('FEDUA-NET TIER-1 REPORT\n' + '=' * 70 + '\n\n')
    f.write('Mean-client accuracy by strategy (mean & std across seeds):\n')
    f.write(summary.to_string() + '\n\n')
    f.write('Mean client accuracy per client:\n')
    f.write(pivot.round(4).to_string() + '\n\n')
    f.write('Paired Wilcoxon vs reference:\n')
    f.write(wil.to_string(index=False) + '\n')
def print_markdown_table(summary_df, conf_df=None):
    print('\n' + '=' * 78)
    print('MARKDOWN SUMMARY TABLE')
    print('=' * 78)
    print("| Method | Acc(%) | F1(%) | ECE(↓) | Brier(↓) | SetSize(↓) |")
    print("|---|---|---|---|---|---|")
    
    # Get set size mapping if available
    set_sizes = {}
    if conf_df is not None and not conf_df.empty:
        # Average set size per strategy
        ag = conf_df.groupby('strategy')['mean_set_size'].mean()
        set_sizes = ag.to_dict()
        
    for _, row in summary_df.iterrows():
        method = str(row.name).capitalize()
        acc = f"{row['acc_mean']*100:.2f}" if pd.notna(row['acc_mean']) else "-"
        f1 = f"{row['f1_mean']*100:.2f}" if pd.notna(row['f1_mean']) else "-"
        ece = f"{row['ece_mean']:.4f}" if pd.notna(row['ece_mean']) else "-"
        brier = f"{row['brier_mean']:.4f}" if pd.notna(row['brier_mean']) else "-"
        
        ss = set_sizes.get(row.name, np.nan)
        set_size_str = f"{ss:.2f}" if pd.notna(ss) else "-"
        
        print(f"| {method} | {acc} | {f1} | {ece} | {brier} | {set_size_str} |")

print_markdown_table(summary, conf)

print('\n[OK] reports ->', REP)