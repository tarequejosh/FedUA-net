# ============================================================
# FedUA-Net v2 post-hoc analysis (no training).
#
# Inputs : results/rerun_v2/logits/{job}_split{s}_train{t}_nb{n}_c{c}.npz
# Outputs: results/rerun_v2/reports/*.csv
#
# Experts per (split seed s, Hospital-B size nb, client c):
#   L  = local expert            (job=local, train seed s)
#   L' = second local expert     (job=local, train seed s+1000)   -> ensemble control
#   F  = federated expert        (job=fedua, train seed s)        -> FedUA federated branch
# For nb>0, clients A and C are untouched by subsampling, so their local experts are
# the nb=0 ones (identical data, identical training).
#
# Methods compared (all calibrated on the client's validation split only):
#   local, local_ts, fed, fed_ts,
#   ens_LL   : deep-ensemble control, mean of temperature-scaled probs of L and L'
#   ens_LF   : mean of temperature-scaled probs of L and F (uncalibrated fusion)
#   cef_LL   : calibrated expert fusion of L and L' (control)
#   cef_LF   : calibrated expert fusion of L and F  (PROPOSED)
# CEF: p(y|x) ∝ exp( w * log p_L^T(y|x) + (1-w) * log p_F^T(y|x) ) / tau,
#      per-client temperatures T and (w, tau) chosen on validation NLL.
# ============================================================
import argparse, itertools, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]          # repository root
CLIENT = {0: 'A (Brain MRI)', 1: 'B (Breast US)', 2: 'C (Chest X-ray)'}


# ---------------- basic helpers ----------------
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))

def fit_temperature(logits, y):
    """1-D NLL minimisation over log T (bounded, never degenerates)."""
    f = lambda lt: nll(softmax(logits / np.exp(lt)), y)
    r = minimize_scalar(f, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    return float(np.exp(r.x))

def ece(p, y, n_bins=15):
    conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        msk = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if msk.any():
            e += msk.mean() * abs(conf[msk].mean() - corr[msk].mean())
    return float(e)

def brier(p, y):
    oh = np.eye(p.shape[1])[y]; return float(np.mean(np.sum((p - oh) ** 2, 1)))

def macro_f1(pred, y, k):
    f = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def aurc(p, y):
    """Area under the risk-coverage curve (lower is better), confidence = max prob."""
    order = np.argsort(-p.max(1)); err = (p.argmax(1) != y)[order].astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))

def metrics(p, y):
    pred = p.argmax(1)
    return {'acc': float(np.mean(pred == y)), 'f1': macro_f1(pred, y, p.shape[1]),
            'nll': nll(p, y), 'ece': ece(p, y), 'brier': brier(p, y), 'aurc': aurc(p, y)}


# ---------------- fusion ----------------
def log_pool(lpa, lpb, w, tau):
    return softmax((w * lpa + (1 - w) * lpb) / tau)

def fit_cef(lpa_v, lpb_v, y_v):
    """Grid over w in [0,1], 1-D search over tau; minimise validation NLL."""
    best = (np.inf, 0.5, 1.0)
    for w in np.linspace(0, 1, 21):
        f = lambda lt: nll(log_pool(lpa_v, lpb_v, w, np.exp(lt)), y_v)
        r = minimize_scalar(f, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
        if r.fun < best[0] - 1e-9:
            best = (r.fun, float(w), float(np.exp(r.x)))
    return best[1], best[2]


# ---------------- conformal ----------------
def scores_lac(p, y):                       # 1 - p_y
    return 1 - p[np.arange(len(y)), y]

def aps_cum(p, rng=None):
    order = np.argsort(-p, 1); ps = np.take_along_axis(p, order, 1); cum = np.cumsum(ps, 1)
    return order, ps, cum

def scores_aps(p, y, rng=None, lam=0.0, kreg=0):
    order, ps, cum = aps_cum(p)
    rank = np.argmax(order == y[:, None], 1)
    s = cum[np.arange(len(y)), rank]
    if lam > 0: s = s + lam * np.maximum(rank + 1 - kreg, 0)
    if rng is not None:                     # randomised APS / RAPS
        s = s - rng.uniform(size=len(y)) * ps[np.arange(len(y)), rank]
    return s

def sets_lac(p, q):
    return p >= 1 - q

def sets_aps(p, q, rng=None, lam=0.0, kreg=0):
    order, ps, cum = aps_cum(p); n, k = p.shape
    pen = lam * np.maximum(np.arange(1, k + 1) - kreg, 0)[None, :]
    s_all = cum + pen
    if rng is not None:
        s_all = s_all - rng.uniform(size=(n, 1)) * ps
        inc_sorted = s_all <= q
    else:
        # deterministic APS: include labels until cumulative mass reaches q (as in experiment.py)
        inc_sorted = np.zeros_like(cum, bool)
        kk = np.minimum((s_all < q).sum(1) + 1, k)
        inc_sorted[np.arange(k)[None, :] < kk[:, None]] = True
    inc_sorted[:, 0] |= ~inc_sorted.any(1)  # never empty
    inc = np.zeros_like(inc_sorted)
    np.put_along_axis(inc, order, inc_sorted, 1)
    return inc

SCORES = {
    'APS':      (lambda p, y, r: scores_aps(p, y),                 lambda p, q, r: sets_aps(p, q)),
    'APS-rand': (lambda p, y, r: scores_aps(p, y, r),              lambda p, q, r: sets_aps(p, q, r)),
    'RAPS':     (lambda p, y, r: scores_aps(p, y, r, 0.1, 1),      lambda p, q, r: sets_aps(p, q, r, 0.1, 1)),
    'LAC':      (lambda p, y, r: scores_lac(p, y),                 lambda p, q, r: sets_lac(p, q)),
}

def conformal_eval(p, y, alpha, score, n_rep=500, seed=0, mondrian=False):
    rng = np.random.default_rng(seed); n = len(y); k = p.shape[1]
    sf, setf = SCORES[score]; out = []
    for r in range(n_rep):
        perm = rng.permutation(n); ci, ti = perm[: n // 2], perm[n // 2:]
        s = sf(p[ci], y[ci], rng)
        if mondrian:                         # class-conditional quantiles (LAC only)
            qs = np.ones(k)
            for c in range(k):
                sc = np.sort(s[y[ci] == c]); m = len(sc)
                lvl = int(np.ceil((m + 1) * (1 - alpha)))
                qs[c] = sc[lvl - 1] if 0 < lvl <= m else 1.0
            inc = p[ti] >= 1 - qs[None, :]
            inc[np.arange(len(ti)), p[ti].argmax(1)] |= ~inc.any(1)
        else:
            m = len(s); lvl = int(np.ceil((m + 1) * (1 - alpha)))
            q = np.sort(s)[lvl - 1] if lvl <= m else np.inf
            inc = setf(p[ti], q, rng)
        cov = inc[np.arange(len(ti)), y[ti]]
        size = inc.sum(1)
        ccov = [cov[y[ti] == c].mean() for c in range(k) if (y[ti] == c).any()]
        single = size == 1
        # triage view: singleton -> automatic decision, otherwise refer to a clinician
        s_acc = cov[single].mean() if single.any() else np.nan
        out.append((cov.mean(), size.mean(), single.mean(), np.min(ccov), np.mean(size == 0), s_acc))
    o = np.array(out)
    return {'coverage': o[:, 0].mean(), 'coverage_sd': o[:, 0].std(), 'set_size': o[:, 1].mean(),
            'singleton': o[:, 2].mean(), 'worst_class_cov': o[:, 3].mean(),
            'empty': o[:, 4].mean(), 'singleton_acc': np.nanmean(o[:, 5])}


# ---------------- loading ----------------
def load(logdir):
    recs = {}
    for f in logdir.glob('*_c[0-9].npz'):
        mt = re.match(r'(.+)_split(\d+)_train(\d+)_nb(\d+)_c(\d)\.npz', f.name)
        job, s, t, nb, c = mt.group(1), *map(int, mt.groups()[1:])
        recs[(job, s, t, nb, c)] = dict(np.load(f))
    return recs

def get(recs, job, s, t, nb, c):
    if c != 1 and job == 'local':
        nb = 0                                  # A/C local experts do not depend on nb
    return recs.get((job, s, t, nb, c))


def logp_ts(r, split, T, mc=False):
    if mc:
        return np.log(np.clip(r[f'{split}_mc'], 1e-12, 1))
    return np.log(softmax(r[f'{split}_logits'] / T))


def build_methods(L, L2, Fd, use_mc=False):
    """Return {method: (test_probs, extra)} for one (seed, nb, client)."""
    yv, yt = L['val_y'], L['test_y']
    out = {}
    TL = fit_temperature(L['val_logits'], yv)
    out['local'] = (softmax(L['test_logits']), {})
    out['local_ts'] = (softmax(L['test_logits'] / TL), {'T': TL})
    if Fd is not None:
        assert np.array_equal(Fd['test_y'], yt) and np.array_equal(Fd['val_y'], yv)
        TF = fit_temperature(Fd['val_logits'], yv)
        out['fed'] = (softmax(Fd['test_logits']), {})
        out['fed_ts'] = (softmax(Fd['test_logits'] / TF), {'T': TF})
        a_v, a_t = logp_ts(L, 'val', TL, use_mc), logp_ts(L, 'test', TL, use_mc)
        b_v, b_t = logp_ts(Fd, 'val', TF, use_mc), logp_ts(Fd, 'test', TF, use_mc)
        out['ens_LF'] = ((np.exp(a_t) + np.exp(b_t)) / 2, {})
        w, tau = fit_cef(a_v, b_v, yv)
        out['cef_LF'] = (log_pool(a_t, b_t, w, tau), {'w_local': w, 'tau': tau})
    if L2 is not None:
        T2 = fit_temperature(L2['val_logits'], yv)
        a_v, a_t = logp_ts(L, 'val', TL, use_mc), logp_ts(L, 'test', TL, use_mc)
        c_v, c_t = logp_ts(L2, 'val', T2, use_mc), logp_ts(L2, 'test', T2, use_mc)
        out['ens_LL'] = ((np.exp(a_t) + np.exp(c_t)) / 2, {})
        w, tau = fit_cef(a_v, c_v, yv)
        out['cef_LL'] = (log_pool(a_t, c_t, w, tau), {'w_local': w, 'tau': tau})
    return out, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/rerun_v2')
    ap.add_argument('--n_rep', type=int, default=500)
    ap.add_argument('--mc', action='store_true', help='fuse MC-dropout probabilities instead of TS logits')
    a = ap.parse_args()
    base = ROOT / a.dir; rep = base / 'reports'; rep.mkdir(parents=True, exist_ok=True)
    recs = load(base / 'logits')
    seeds = sorted({k[1] for k in recs}); nbs = sorted({k[3] for k in recs})
    rows, crows, wrows = [], [], []
    dump = {}                                   # per-sample test probabilities, for paired tests
    for s, nb, c in itertools.product(seeds, nbs, range(3)):
        L = get(recs, 'local', s, s, nb, c); L2 = get(recs, 'local', s, s + 1000, nb, c)
        Fd = get(recs, 'fedua', s, s, nb, c)
        if L is None and Fd is not None:          # fed-only available (e.g. checkpoints)
            yt = Fd['test_y']; TF = fit_temperature(Fd['val_logits'], Fd['val_y'])
            meths = {'fed': (softmax(Fd['test_logits']), {}),
                     'fed_ts': (softmax(Fd['test_logits'] / TF), {'T': TF})}
        elif L is None:
            continue
        else:
            meths, yt = build_methods(L, L2, Fd, a.mc)
        Fc = get(recs, 'fedua_cka', s, s, nb, c)
        if Fc is not None:
            T = fit_temperature(Fc['val_logits'], Fc['val_y'])
            meths['fed_cka'] = (softmax(Fc['test_logits']), {}); meths['fed_cka_ts'] = (softmax(Fc['test_logits'] / T), {'T': T})
        dump[f'y|{s}|{nb}|{c}'] = yt
        for name, (p, extra) in meths.items():
            dump[f'p|{s}|{nb}|{c}|{name}'] = p.astype(np.float32)
            rows.append({'seed': s, 'nb': nb, 'client': c, 'method': name, 'n_test': len(yt),
                         **metrics(p, yt), **extra})
            if name in ('local_ts', 'fed_ts', 'ens_LL', 'cef_LL', 'cef_LF', 'local', 'fed'):
                for alpha, score in itertools.product((0.05, 0.10), SCORES):
                    crows.append({'seed': s, 'nb': nb, 'client': c, 'method': name, 'alpha': alpha,
                                  'score': score, **conformal_eval(p, yt, alpha, score, a.n_rep, seed=s)})
                if name in ('local_ts', 'cef_LL', 'cef_LF'):
                    crows.append({'seed': s, 'nb': nb, 'client': c, 'method': name, 'alpha': 0.10,
                                  'score': 'LAC-Mondrian',
                                  **conformal_eval(p, yt, 0.10, 'LAC', a.n_rep, seed=s, mondrian=True)})
    df = pd.DataFrame(rows); cf = pd.DataFrame(crows)
    sfx = '_mc' if a.mc else ''
    df.to_csv(rep / f'per_run_metrics{sfx}.csv', index=False)
    cf.to_csv(rep / f'conformal{sfx}.csv', index=False)
    np.savez_compressed(rep / f'test_probs{sfx}.npz', **dump)

    # ---- summaries ----
    pc = df.groupby(['nb', 'method', 'client'])[['acc', 'f1', 'nll', 'ece', 'brier', 'aurc']].agg(['mean', 'std'])
    pc.to_csv(rep / f'summary_per_client{sfx}.csv')
    mean_c = df.groupby(['nb', 'method', 'seed'])[['acc', 'f1', 'nll', 'ece', 'brier', 'aurc']].mean()
    worst = df.groupby(['nb', 'method', 'seed'])['acc'].min().rename('worst_acc')
    agg = mean_c.join(worst).groupby(['nb', 'method']).agg(['mean', 'std'])
    agg.to_csv(rep / f'summary_mean_client{sfx}.csv')
    if len(cf):
        cs = cf.groupby(['nb', 'client', 'method', 'alpha', 'score'])[
            ['coverage', 'coverage_sd', 'set_size', 'singleton', 'worst_class_cov',
             'empty', 'singleton_acc']].mean()
        cs.to_csv(rep / f'summary_conformal{sfx}.csv')
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 400)
    print((df.groupby(['nb', 'client', 'method'])[['acc', 'f1', 'ece', 'nll']].mean() * [100, 100, 100, 1]).round(2))
    if 'w_local' in df:
        print(df[df.method.isin(['cef_LF', 'cef_LL'])].groupby(['nb', 'client', 'method'])[['w_local', 'tau']].agg(['mean', 'std']).round(3))


if __name__ == '__main__':
    main()
