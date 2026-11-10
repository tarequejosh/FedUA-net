"""Publication figures for the IWAIT 2027 camera-ready (FedUA-Net).

Fig. 2  (a) Site-B accuracy vs. training-set size, (b) server-weighting ablation,
        (c) reliability diagram of FedUA-Net before/after temperature scaling.
Fig. 3  Conformal triage: coverage vs. target and referral rate vs. target per site.

All numbers are read from results/main_uniform, results/scarcity_*, results/sample_weighted
and results/rerun_v2/logits; nothing is typed by hand.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]          # project root (FedUA-Net-IWAIT2027)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2 as A  # noqa: E402  (softmax, fit_temperature, ece, SCORES, ...)

OUT = ROOT / 'paper' / 'camera_ready' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ---- style: reference categorical palette (fixed order), print-safe secondary encoding
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, MUTED, GRID = '#8a8986', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def per_client(path):
    d = pd.read_csv(path)
    d['client'] = d['client'].astype(str)
    return d


def site_b(path, strat):
    d = per_client(path)
    r = d[(d.strategy == strat) & (d.client == '1')].iloc[0]
    return 100 * r.acc_mean, 100 * r.acc_std


VER = ROOT / 'results'
SW = ROOT / 'results' / 'sample_weighted' / 'reports' / 'per_client_metrics.csv'
MAIN = VER / 'main_uniform' / 'reports' / 'per_client_metrics.csv'


def panel_scarcity(ax):
    sizes = [100, 200, 546]
    paths = {100: VER / 'scarcity_100/reports/per_client_metrics.csv',
             200: VER / 'scarcity_200/reports/per_client_metrics.csv', 546: MAIN}
    spec = [('local_only', 'Local only (ref.)', GREY, '^', '--'),
            ('fedbn', 'FedBN', ORANGE, 's', '-'),
            ('fedua', 'FedUA-Net', BLUE, 'o', '-')]
    ends = {}
    for strat, lab, col, mk, ls in spec:
        m, s = np.array([site_b(paths[n], strat) for n in sizes]).T
        ax.fill_between(sizes, m - s, m + s, color=col, alpha=0.12, lw=0)
        ax.plot(sizes, m, ls=ls, marker=mk, ms=4.5, color=col, label=lab,
                mec='white', mew=0.8)
        ends[lab] = m[-1]
    # direct labels at the right end, de-collided (min 2.4 points apart)
    labs = sorted(ends, key=ends.get)
    ys = [ends[labs[0]]]
    for lab in labs[1:]:
        ys.append(max(ends[lab], ys[-1] + 2.4))
    for lab, y in zip(labs, ys):
        ax.annotate(lab, (sizes[-1] * 1.08, y), va='center', fontsize=7, color=INK,
                    annotation_clip=False)
    ax.set_xscale('log')
    ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(85, 2600)
    ax.set_xlabel('Site-B training images $N_B$')
    ax.set_ylabel('Site-B test accuracy (%)')
    ax.set_title('(a) Data scarcity at the ultrasound site', loc='left')


def panel_weighting(ax):
    rows = [('fedavg', 'FedAvg'), ('fedbn', 'FedBN'), ('ditto', 'Ditto'), ('fedua', 'FedUA-Net')]
    for i, (strat, lab) in enumerate(rows):
        a, _ = site_b(SW, strat)
        b, _ = site_b(MAIN, strat)
        ax.plot([a, b], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle='round')
        ax.scatter([a], [i], s=26, facecolor='white', edgecolor=MUTED, lw=1.0, zorder=3,
                   label='$w_k\\propto N_k$' if i == 0 else None)
        ax.scatter([b], [i], s=26, color=BLUE, edgecolor='white', lw=0.8, zorder=3,
                   label='$w_k=1/K$' if i == 0 else None)
        ax.annotate(f'+{b - a:.1f}', ((a + b) / 2, i), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=6.8, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(48, 95)
    ax.set_xlabel('Site-B test accuracy (%)')
    ax.set_title('(b) Server weighting', loc='left')
    ax.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.1)


def load_fedua(nb=0):
    recs = A.load(ROOT / 'results' / 'rerun_v2' / 'logits')
    out = {}
    for s in range(3):
        for c in range(3):
            r = recs.get(('fedua', s, s, nb, c))
            if r is None:
                raise SystemExit(f'missing fedua logits split{s} c{c}')
            T = A.fit_temperature(r['val_logits'], r['val_y'])
            out[(s, c)] = dict(raw=A.softmax(r['test_logits']),
                               ts=A.softmax(r['test_logits'] / T), y=r['test_y'], T=T)
    return out


def reliability(conf, corr, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        msk = (conf >= edges[i]) & ((conf <= edges[i + 1]) if i == n_bins - 1 else (conf < edges[i + 1]))
        if msk.sum() >= 20:
            xs.append(conf[msk].mean()); ys.append(corr[msk].mean()); ns.append(msk.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def panel_reliability(ax, axh, F):
    ax.plot([0, 1], [0, 1], ls=':', color=GREY, lw=1.0)
    edges = np.linspace(0.3, 1.0, 15)
    spec = [('raw', 'Before TS', ORANGE, 's', -0.25), ('ts', 'After TS', BLUE, 'o', 0.25)]
    w = edges[1] - edges[0]
    for key, lab, col, mk, off in spec:
        # pool top-1 confidence and correctness over sites and seeds
        conf = np.concatenate([F[k][key].max(1) for k in sorted(F)])
        corr = np.concatenate([(F[k][key].argmax(1) == F[k]['y']) for k in sorted(F)]).astype(float)
        x, yv, _ = reliability(conf, corr)
        e = np.mean([A.ece(F[k][key], F[k]['y']) for k in sorted(F)]) * 100
        ax.plot(x, yv, marker=mk, ms=3.8, color=col, mec='white', mew=0.7,
                label=f'{lab} (ECE {e:.1f}%)')
        h, _ = np.histogram(conf, bins=edges)
        axh.bar(edges[:-1] + w * (0.5 + off / 1.4), 100 * h / len(conf), width=w * 0.34,
                color=col, lw=0)
    for a in (ax, axh):
        a.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel('Accuracy')
    ax.tick_params(labelbottom=False)
    ax.set_title('(c) Reliability of FedUA-Net', loc='left')
    ax.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.1)
    axh.set_yscale('log'); axh.set_ylim(0.05, 100)
    axh.set_yticks([0.1, 10]); axh.set_yticklabels(['0.1', '10'])
    axh.minorticks_off()
    axh.set_ylabel('% scans', fontsize=7)
    axh.set_xlabel('Confidence (top-1 probability)')
    axh.grid(axis='x', visible=False)


def fig2():
    F = load_fedua()
    fig = plt.figure(figsize=(7.0, 2.08))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 0.92], height_ratios=[2.6, 1.0],
                          hspace=0.08, wspace=0.42)
    panel_scarcity(fig.add_subplot(gs[:, 0]))
    panel_weighting(fig.add_subplot(gs[:, 1]))
    axr = fig.add_subplot(gs[0, 2]); axh = fig.add_subplot(gs[1, 2], sharex=axr)
    panel_reliability(axr, axh, F)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig2_results.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    temps = [F[k]['T'] for k in sorted(F)]
    print('fig2 written; FedUA-Net re-run temperatures: min %.2f max %.2f' % (min(temps), max(temps)))


def fig3(n_rep=300):
    F = load_fedua()
    targets = np.array([0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975])
    scores = [('LAC', 'LAC', BLUE, 'o', '-'), ('RAPS', 'RAPS', ORANGE, 's', '-'),
              ('APS', 'APS (deterministic)', AQUA, '^', '--')]
    res = {}
    for sc, *_ in scores:
        for c in range(3):
            for t in targets:
                vals = [A.conformal_eval(F[(s, c)]['ts'], F[(s, c)]['y'], 1 - t, sc, n_rep, seed=s)
                        for s in range(3)]
                res[(sc, c, t)] = (np.mean([v['coverage'] for v in vals]),
                                   1 - np.mean([v['singleton'] for v in vals]))
    rows = [dict(score=sc, site='ABC'[c], target=t, coverage=v[0], referral=v[1])
            for (sc, c, t), v in res.items()]
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'rerun_v2' / 'reports' / 'conformal_sweep.csv', index=False)

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.72), sharex=True)
    ax = axs[0]
    ax.plot([0.8, 0.975], [0.8, 0.975], ls=':', color=GREY, lw=1.0)
    for sc, lab, col, mk, ls in scores:
        cov = [np.mean([res[(sc, c, t)][0] for c in range(3)]) for t in targets]
        ax.plot(targets, cov, ls=ls, marker=mk, ms=3.5, color=col, mec='white', mew=0.6, label=lab)
    ax.set_ylabel('Empirical coverage'); ax.set_title('(a) Coverage (site mean)', loc='left')
    ax.set_ylim(0.78, 1.005)
    names = ['MRI', 'ultrasound', 'X-ray']
    for c in range(3):
        ax = axs[c + 1]
        for sc, lab, col, mk, ls in scores:
            ref = [100 * res[(sc, c, t)][1] for t in targets]
            ax.plot(targets, ref, ls=ls, marker=mk, ms=3.5, color=col, mec='white', mew=0.6, label=lab)
        ax.set_ylim(-3, 103)
        ax.set_title(f'({"bcd"[c]}) Referral, site {"ABC"[c]} ({names[c]})', loc='left')
        if c == 0:
            ax.set_ylabel('Referred scans (%)')
    for ax in axs:
        ax.set_xticks([0.80, 0.85, 0.90, 0.95]); ax.set_xticklabels(['80', '85', '90', '95'])
        ax.set_xlabel('Target coverage $1-\\alpha$ (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=2.2)
    fig.tight_layout(w_pad=0.9)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig3_conformal.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('fig3 written')


if __name__ == '__main__':
    fig2()
    fig3()
