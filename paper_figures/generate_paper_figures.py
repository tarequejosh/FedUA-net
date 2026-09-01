# ============================================================
# FedUA-Net Publication-Grade Figure Generation Suite
# High-Resolution (300 DPI), Publication-Quality Aesthetics
# ============================================================
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'outputs_experiments' / 'raw'
REP = ROOT / 'outputs_experiments' / 'reports'
OUT = ROOT / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)

def setup_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10.5,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

STRAT_COLORS = {
    'fedua': '#C44E52',        # Crimson Red (Proposed)
    'ditto': '#8172B3',        # Purple
    'centralized': '#6C757D',  # Slate Grey
    'local_only': '#4C72B0',   # Steel Blue
    'fedbabu': '#937860',      # Brown
    'fedprox': '#DD8452',      # Orange
    'fedavg': '#55A868',       # Green
    'fedbn': '#CCB974',        # Gold
}

STRAT_NAMES = {
    'fedua': 'FedUA-Net (Ours)',
    'ditto': 'Ditto',
    'centralized': 'Centralized',
    'local_only': 'Local-Only',
    'fedbabu': 'FedBABU',
    'fedprox': 'FedProx',
    'fedavg': 'FedAvg',
    'fedbn': 'FedBN',
}

CLIENT_NAMES = ['Brain MRI\n(Hospital A)', 'Breast US\n(Hospital B)', 'Chest X-Ray\n(Hospital C)', 'Mean Client\nAccuracy']

def plot_main_benchmark():
    setup_academic_style()
    pm_file = REP / 'per_client_metrics.csv'
    sum_file = REP / 'summary.csv'
    if not pm_file.exists() or not sum_file.exists():
        print("Waiting for summary files...")
        return
        
    pm = pd.read_csv(pm_file)
    pm['client'] = pm['client'].astype(str)
    sm = pd.read_csv(sum_file).set_index('strategy')
    
    strategies = [s for s in ['fedua', 'ditto', 'centralized', 'local_only', 'fedbabu', 'fedprox', 'fedavg', 'fedbn'] if s in pm['strategy'].unique()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLIENT_NAMES))
    n_strats = len(strategies)
    width = 0.85 / n_strats
    
    for i, s in enumerate(strategies):
        sub = pm[pm['strategy'] == s]
        c0 = sub[sub['client'] == '0']['acc_mean'].values[0] * 100 if len(sub[sub['client'] == '0']) else 0
        c1 = sub[sub['client'] == '1']['acc_mean'].values[0] * 100 if len(sub[sub['client'] == '1']) else 0
        c2 = sub[sub['client'] == '2']['acc_mean'].values[0] * 100 if len(sub[sub['client'] == '2']) else 0
        mean_acc = sm.loc[s, 'acc_mean'] * 100 if s in sm.index else 0
        
        c0_std = sub[sub['client'] == '0']['acc_std'].values[0] * 100 if len(sub[sub['client'] == '0']) else 0
        c1_std = sub[sub['client'] == '1']['acc_std'].values[0] * 100 if len(sub[sub['client'] == '1']) else 0
        c2_std = sub[sub['client'] == '2']['acc_std'].values[0] * 100 if len(sub[sub['client'] == '2']) else 0
        mean_std = sm.loc[s, 'acc_std'] * 100 if s in sm.index else 0
        
        vals = [c0, c1, c2, mean_acc]
        errs = [c0_std, c1_std, c2_std, mean_std]
        
        offset = (i - n_strats / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, yerr=errs, capsize=2.5,
                       label=STRAT_NAMES.get(s, s), color=STRAT_COLORS.get(s, '#333333'),
                       edgecolor='black', linewidth=0.8, alpha=0.92)
        
    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLIENT_NAMES, fontweight='bold')
    ax.set_ylim(45, 102)
    ax.set_title('Cross-Silo Multi-Task Medical Image Classification Performance', fontweight='bold', pad=12)
    ax.legend(loc='lower left', ncol=4, frameon=True, edgecolor='gray', facecolor='white', framealpha=0.95)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    plt.tight_layout()
    fig.savefig(OUT / 'fig2_main_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] Generated fig2_main_benchmark.png')

def plot_calibration_comparison():
    setup_academic_style()
    cal_file = REP / 'calibration_comparison.csv'
    if not cal_file.exists():
        return
        
    cal = pd.read_csv(cal_file)
    strategies = [s for s in ['fedua', 'fedbn', 'fedavg', 'fedprox', 'fedbabu', 'ditto', 'local_only', 'centralized'] if s in cal['strategy'].unique()]
    cal = cal.set_index('strategy').loc[strategies].reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subplot A: Raw vs Calibrated ECE
    ax1 = axes[0]
    x = np.arange(len(strategies))
    width = 0.38
    
    rects1 = ax1.bar(x - width/2, cal['ece_raw'], width, label='Uncalibrated (Raw Softmax)',
                     color='#4C72B0', edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, cal['ece_cal'], width, label='Calibrated (Temperature Scaling)',
                     color='#C44E52', edgecolor='black', linewidth=0.8)
                     
    ax1.set_ylabel('Expected Calibration Error (ECE ↓)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([STRAT_NAMES.get(s, s) for s in strategies], rotation=35, ha='right', fontweight='bold')
    ax1.set_title('(A) Expected Calibration Error (ECE)', fontweight='bold')
    ax1.legend(frameon=True, edgecolor='gray')
    ax1.yaxis.grid(True); ax1.xaxis.grid(False)
    
    # Subplot B: Brier Score
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, cal['brier_raw'], width, label='Uncalibrated',
                     color='#55A868', edgecolor='black', linewidth=0.8)
    rects4 = ax2.bar(x + width/2, cal['brier_cal'], width, label='Calibrated',
                     color='#DD8452', edgecolor='black', linewidth=0.8)
                     
    ax2.set_ylabel('Brier Score (↓)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([STRAT_NAMES.get(s, s) for s in strategies], rotation=35, ha='right', fontweight='bold')
    ax2.set_title('(B) Probability Brier Score', fontweight='bold')
    ax2.legend(frameon=True, edgecolor='gray')
    ax2.yaxis.grid(True); ax2.xaxis.grid(False)
    
    plt.tight_layout()
    fig.savefig(OUT / 'fig3_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] Generated fig3_calibration.png')

def plot_conformal_efficiency():
    setup_academic_style()
    conf_file = REP / 'conformal_results.csv'
    if not conf_file.exists():
        return
        
    conf = pd.read_csv(conf_file)
    strategies = [s for s in ['fedua', 'ditto', 'local_only', 'fedbabu', 'fedbn', 'fedavg', 'centralized'] if s in conf['strategy'].unique()]
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    for s in strategies:
        sub = conf[conf['strategy'] == s].sort_values('alpha')
        if sub.empty:
            continue
        marker = 'D' if s == 'fedua' else 'o'
        lw = 2.5 if s == 'fedua' else 1.6
        ms = 7.5 if s == 'fedua' else 6.0
        ax.plot(sub['alpha'], sub['mean_set_size'], marker=marker, ms=ms, lw=lw,
                label=STRAT_NAMES.get(s, s), color=STRAT_COLORS.get(s, '#333333'))
                
    ax.set_xlabel(r'Target Error Rate ($\alpha$)', fontweight='bold')
    ax.set_ylabel('Mean Conformal Prediction Set Size (Classes)', fontweight='bold')
    ax.set_title('Conformal Prediction Set Efficiency vs Error Tolerance', fontweight='bold', pad=10)
    ax.legend(frameon=True, edgecolor='gray', facecolor='white', framealpha=0.95)
    ax.yaxis.grid(True); ax.xaxis.grid(True)
    
    plt.tight_layout()
    fig.savefig(OUT / 'fig4_conformal_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] Generated fig4_conformal_efficiency.png')

def plot_risk_coverage():
    setup_academic_style()
    rc_file = REP / 'risk_coverage_summary.csv'
    if not rc_file.exists():
        return
        
    rc = pd.read_csv(rc_file)
    strategies = [s for s in ['fedua', 'ditto', 'local_only', 'fedbabu', 'fedbn', 'fedprox', 'fedavg'] if s in rc['strategy'].unique()]
    
    cov_points = [0.50, 0.70, 0.80, 0.90, 0.95]
    cols = [f'acc_at_cov_{t:.2f}' for t in cov_points]
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s in strategies:
        sub = rc[rc['strategy'] == s]
        if sub.empty:
            continue
        vals = [sub[c].values[0] * 100 for c in cols]
        marker = 'D' if s == 'fedua' else 's'
        lw = 2.5 if s == 'fedua' else 1.6
        ms = 7.5 if s == 'fedua' else 5.5
        ax.plot(cov_points, vals, marker=marker, ms=ms, lw=lw,
                label=f"{STRAT_NAMES.get(s, s)} (AUC={sub['acc_cov_auc'].values[0]:.3f})",
                color=STRAT_COLORS.get(s, '#333333'))
                
    ax.set_xlabel('Coverage (Fraction of Evaluated Samples)', fontweight='bold')
    ax.set_ylabel('Selective Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Selective Classification Risk-Coverage Curves (Clinical Triage)', fontweight='bold', pad=10)
    ax.legend(loc='lower left', frameon=True, edgecolor='gray', facecolor='white', framealpha=0.95)
    ax.yaxis.grid(True); ax.xaxis.grid(True)
    
    plt.tight_layout()
    fig.savefig(OUT / 'fig5_risk_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] Generated fig5_risk_coverage.png')

def main():
    plot_main_benchmark()
    plot_calibration_comparison()
    plot_conformal_efficiency()
    plot_risk_coverage()

if __name__ == '__main__':
    main()