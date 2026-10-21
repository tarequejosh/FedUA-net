"""
REVIEW FIX v3 -- post-hoc and inference-only: no training, no weight updates.

Reloads every final_models/{strategy}_seed{seed}_client{c}.pt checkpoint
written by `experiment.py --save_final_models`, rebuilds the exact
seed-dependent splits (experiment.build_data), and:

  1. Reproduction check: refits T on 'temp' and recomputes the legacy
     deterministic-APS conformal on 'conf' -> 'test', and compares against
     raw/cal_{strategy}_seed{seed}.csv. Everything below is only trustworthy
     if this matches.
  2. Conformal re-scoring: LAC and randomized APS (see
     src/uncertainty/federated_conformal.py, REVIEW FIX v3) next to the
     original deterministic APS, for the exact per-client threshold and for
     the three federated modes (group / marginal_federated / shrunk).
  3. Two evaluation protocols (see REVIEW_PROVENANCE.md):
       'standard' -- the splits exactly as experiment.py used them.
       'dedup'    -- every 'temp' / 'conf' / 'test' image with a near-
                     duplicate in the SAME client's 'train' split is dropped
                     before T is fit, before the conformal threshold is fit,
                     and before evaluation. OOD-cohort images with a near-
                     duplicate ANYWHERE in Hospital A are dropped as well.
     Near-duplicate = 16x16 difference-hash Hamming distance <=
     --dup_threshold (of 256 bits); a sample of flagged pairs' pixel
     correlations is reported so the threshold can be audited.

Logits are cached under <out>/rescore/logits/ and image hashes in
<out>/rescore/dhash_store.npz, so re-running the analysis needs no GPU.

Usage:
    python scripts/rescore_from_checkpoints.py --out ./outputs_journal_5client_v2 \
        --num_clients 5 --strategies fedbn fedprox ditto local_only fedua \
        --seeds 0 1 2 3 4 --ood_eval brisc2025 pmram
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import experiment as ex          # parses default args at import; patched in main()
import fedua_net as m
from src.uncertainty.federated_conformal import (
    SCORES, ClientCalibrationData, federated_conformal_calibrate, evaluate_federated_modes)

ALPHAS = (0.05, 0.10, 0.20)
PROTOCOLS = ('standard', 'dedup')
EXACT = 10 ** 9   # digest_delta large enough that the digest is the exact score list


# ------------------------------------------------------------
# Near-duplicate detection
# ------------------------------------------------------------
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def _gray(path, size):
    img = m.load_image_tensor(path)                    # (C,H,W) uint8
    arr = img.float().mean(0).numpy().astype(np.uint8)
    return Image.fromarray(arr).resize(size, Image.BILINEAR)

def dhash(path, n=16):
    a = np.asarray(_gray(path, (n + 1, n)), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())

class HashStore:
    """path -> 256-bit dHash, persisted so each image is hashed once."""
    def __init__(self, file):
        self.file, self.h = file, {}
        if file.exists():
            z = np.load(file, allow_pickle=True)
            self.h = dict(zip(z['paths'].tolist(), z['h']))
        self.dirty = False
    def get(self, paths):
        for p in paths:
            if p not in self.h:
                self.h[p] = dhash(p); self.dirty = True
        return np.stack([self.h[p] for p in paths]) if paths else np.zeros((0, 32), np.uint8)
    def save(self):
        if self.dirty:
            np.savez(self.file, paths=np.array(list(self.h), dtype=object), h=np.stack(list(self.h.values())))
            self.dirty = False

def nearest(query, ref, chunk=64):
    """(min Hamming distance, argmin index into ref) for each query hash."""
    dist = np.empty(len(query), dtype=np.int32); idx = np.empty(len(query), dtype=np.int64)
    for s in range(0, len(query), chunk):
        d = _POPCOUNT[query[s:s + chunk, None, :] ^ ref[None, :, :]].sum(-1)
        dist[s:s + chunk] = d.min(1); idx[s:s + chunk] = d.argmin(1)
    return dist, idx

def pixel_corr(p, q, n=64):
    a = np.asarray(_gray(p, (n, n)), np.float64).ravel(); b = np.asarray(_gray(q, (n, n)), np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(a @ b / a.size)

def dup_mask(query_paths, ref_paths, store, threshold, key, audit, n_audit=200):
    """True where a query image has a near-duplicate in ref_paths."""
    if not query_paths or not ref_paths:
        return np.zeros(len(query_paths), bool)
    dist, idx = nearest(store.get(query_paths), store.get(ref_paths))
    mask = dist <= threshold
    flagged = np.flatnonzero(mask)
    sample = np.random.default_rng(0).choice(flagged, min(n_audit, len(flagged)), replace=False)
    corrs = [pixel_corr(query_paths[i], ref_paths[idx[i]]) for i in sample]
    audit[key] = {'n': int(len(mask)), 'n_dup': int(mask.sum()), 'frac_dup': float(mask.mean()),
                  'n_exact_hash': int((dist == 0).sum()), 'threshold_bits': threshold,
                  'audited_pairs': len(corrs),
                  'pixel_corr_min': float(min(corrs)) if corrs else None,
                  'pixel_corr_median': float(np.median(corrs)) if corrs else None}
    return mask


# ------------------------------------------------------------
# Inference with logit cache
# ------------------------------------------------------------
class PathDS(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.y, self.tf = list(paths), list(labels), m.eval_transforms()
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = m.load_image_tensor(self.paths[i])
        if img.shape[0] == 1: img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3: img = img[:3]
        return self.tf(img), self.y[i]

def ood_frame(cohort):
    """Same cohort loading and label mapping as experiment.run_ood_eval."""
    df = m.load_ood_brain_shift(cohort).copy()
    cls_names = m.client_meta(0)[2]
    df['local'] = df['label'].map(lambda lbl: sorted(cls_names).index(lbl))
    return df

def get_logits(cache, net_fn, loaders, ood):
    if cache.exists():
        L = dict(np.load(cache))
        if all(f'ood_{k}_logits' in L for k in ood):
            return L
    net = net_fn()
    out = {}
    for split in ('temp', 'conf', 'test'):
        lg, y = ex.collect_logits(net, loaders[split], ex.DEV)
        out[f'{split}_logits'], out[f'{split}_y'] = lg.numpy(), y.numpy()
    for cohort, df in ood.items():
        dl = DataLoader(PathDS(df['path'], df['local']), batch_size=ex.ARGS.batch, shuffle=False)
        lg, y = ex.collect_logits(net, dl, ex.DEV)
        out[f'ood_{cohort}_logits'], out[f'ood_{cohort}_y'] = lg.numpy(), y.numpy()
    np.savez(cache, **out)
    del net; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------
# Conformal helpers
# ------------------------------------------------------------
def softmax(logits, T):
    return torch.softmax(torch.as_tensor(logits) / T, dim=-1).numpy()

def single_client_conformal(cal_p, cal_y, te_p, te_y, alpha, score, seed):
    """Exact split conformal (one client, undigested scores) for any score."""
    r = federated_conformal_calibrate([ClientCalibrationData('c', cal_p, cal_y)], alpha=alpha,
                                      score=score, seed=seed, digest_delta=EXACT)
    v = evaluate_federated_modes(r, [ClientCalibrationData('c', te_p, te_y)])['c']['group']
    return {'qhat': v['q_hat'], 'coverage': v['coverage'],
            'mean_set_size': v['mean_set_size'], 'frac_empty': v['frac_empty']}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_clients', type=int, required=True, choices=[3, 4, 5])
    ap.add_argument('--strategies', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--ood_eval', nargs='*', default=[], choices=['brisc2025', 'pmram'])
    ap.add_argument('--ood_strategies', nargs='*', default=['fedua'],
                    help='strategies whose Hospital A model is evaluated on the OOD cohorts '
                         '(experiment.py only ever did fedua)')
    ap.add_argument('--dup_threshold', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    ex.ARGS.num_clients = args.num_clients
    ex.ARGS.batch = args.batch
    m.cfg.BATCH_SIZE = args.batch

    out = Path(args.out); raw = out / 'raw'; ckpts = out / 'final_models'
    res = out / 'rescore'; cache_dir = res / 'logits'
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = HashStore(res / 'dhash_store.npz')
    print(f'device={ex.DEV} | out={out} | scores={SCORES} | protocols={PROTOCOLS}')

    audit, repro, acc_rows, per_client, fed_rows, ood_rows = {}, [], [], [], [], []
    ood = {c: ood_frame(c) for c in args.ood_eval}

    for seed in args.seeds:
        all_df, classes, c2i, client_dfs = ex.build_data(seed)
        cids = list(range(args.num_clients))
        loaders = {c: ex.loaders_for(client_dfs[c], client_id=c) for c in cids}
        ncls = {c: int(client_dfs[c]['local'].max()) + 1 for c in cids}
        names = {c: m.client_meta(c)[3] for c in cids}

        # rows to KEEP per (protocol, client, split); loader order == df order
        keep = {'standard': {}, 'dedup': {}}
        for c in cids:
            df = client_dfs[c]
            train = df[df['split'] == 'train']['path'].tolist()
            keep['standard'][c], keep['dedup'][c] = {}, {}
            for split in ('temp', 'conf', 'test'):
                q = df[df['split'] == split]['path'].tolist()
                keep['standard'][c][split] = np.ones(len(q), bool)
                keep['dedup'][c][split] = ~dup_mask(q, train, store, args.dup_threshold,
                                                    f'seed{seed}|{names[c]}|{split}', audit)
            store.save()
            print(f'[leakage] seed={seed} {names[c]}: ' + ', '.join(
                f"{s} {audit[f'seed{seed}|{names[c]}|{s}']['n_dup']}/{audit[f'seed{seed}|{names[c]}|{s}']['n']}"
                for s in ('temp', 'conf', 'test')))
        a_all = client_dfs[0]['path'].tolist()
        ood_keep = {'standard': {c: np.ones(len(df), bool) for c, df in ood.items()},
                    'dedup': {c: ~dup_mask(df['path'].tolist(), a_all, store, args.dup_threshold,
                                           f'seed{seed}|ood_{c}|vs_hospitalA_all', audit)
                              for c, df in ood.items()}}
        store.save()

        for name in args.strategies:
            cal_csv = raw / f'cal_{name}_seed{seed}.csv'
            orig = pd.read_csv(cal_csv) if cal_csv.exists() else None
            fed_cal = {(p, s): [] for p in PROTOCOLS for s in SCORES}
            fed_te = {(p, s): [] for p in PROTOCOLS for s in SCORES}
            for c in cids:
                ck = ckpts / f'{name}_seed{seed}_client{c}.pt'
                if not ck.exists():
                    print(f'[WARN] missing checkpoint {ck}'); continue
                def net_fn(c=c, ck=ck):
                    net = ex.new_client_net(ncls[c], seed)
                    net.load_state_dict(torch.load(ck, map_location=ex.DEV)); net.eval()
                    return net
                want_ood = ood if (c == 0 and name in args.ood_strategies) else {}
                L = get_logits(cache_dir / f'{name}_seed{seed}_client{c}.npz', net_fn, loaders[c], want_ood)

                for proto in PROTOCOLS:
                    k = keep[proto][c]
                    T = ex.fit_temp(torch.as_tensor(L['temp_logits'][k['temp']]),
                                    torch.as_tensor(L['temp_y'][k['temp']]))
                    cal_p, cal_y = softmax(L['conf_logits'][k['conf']], T), L['conf_y'][k['conf']]
                    te_p, te_y = softmax(L['test_logits'][k['test']], T), L['test_y'][k['test']]
                    met = ex.full_metrics(te_y, te_p)
                    met_raw = ex.full_metrics(te_y, softmax(L['test_logits'][k['test']], 1.0))
                    acc_rows.append({'protocol': proto, 'strategy': name, 'seed': seed, 'client': c,
                                     'client_name': names[c], 'n_temp': int(k['temp'].sum()),
                                     'n_conf': int(k['conf'].sum()), 'n_test': int(k['test'].sum()),
                                     'temp': T, 'acc': met['acc'], 'f1': met['f1'], 'mcc': met['mcc'],
                                     'ece_raw': met_raw['ece'], 'ece_cal': met['ece']})

                    # 1. reproduction check vs. the original run
                    if proto == 'standard' and orig is not None:
                        o = orig[orig['client'] == c]
                        oc = o[o['metric'] == 'calib'].iloc[0]
                        row = {'strategy': name, 'seed': seed, 'client': c,
                               'd_temp': abs(T - oc['temp']), 'd_acc': abs(met['acc'] - oc['acc_cal']),
                               'd_ece_cal': abs(met['ece'] - oc['ece_cal']),
                               'd_n_calib': abs(len(cal_y) - oc['n_calib'])}
                        for alpha in ALPHAS:
                            sets, _ = ex.conformal_aps(cal_p, cal_y, te_p, alpha=alpha)
                            cov = float(np.mean([te_y[i] in sets[i] for i in range(len(te_y))]))
                            size = float(np.mean([len(s) for s in sets]))
                            oo = o[(o['metric'] == 'conformal') & np.isclose(o['alpha'], alpha)].iloc[0]
                            row[f'd_cov_{alpha}'] = abs(cov - oo['coverage'])
                            row[f'd_size_{alpha}'] = abs(size - oo['mean_set_size'])
                        repro.append(row)

                    # 2. per-client exact conformal, every score
                    for score in SCORES:
                        for alpha in ALPHAS:
                            r = single_client_conformal(cal_p, cal_y, te_p, te_y, alpha, score, seed * 100 + c)
                            per_client.append({'protocol': proto, 'strategy': name, 'seed': seed,
                                               'client': c, 'client_name': names[c], 'score': score,
                                               'alpha': alpha, 'n_calib': len(cal_y),
                                               'n_test': len(te_y), **r})
                        fed_cal[(proto, score)].append(ClientCalibrationData(names[c], cal_p, cal_y))
                        fed_te[(proto, score)].append(ClientCalibrationData(names[c], te_p, te_y))

                    # 3. OOD: Hospital A's model + its in-distribution T and 'conf' threshold
                    for cohort in want_ood:
                        km = ood_keep[proto][cohort]
                        o_p, o_y = softmax(L[f'ood_{cohort}_logits'][km], T), L[f'ood_{cohort}_y'][km]
                        mo = ex.full_metrics(o_y, o_p)
                        for score in SCORES:
                            for alpha in ALPHAS:
                                r = single_client_conformal(cal_p, cal_y, o_p, o_y, alpha, score, seed * 100 + 99)
                                ood_rows.append({'protocol': proto, 'strategy': name, 'seed': seed,
                                                 'cohort': cohort, 'n_ood': len(o_y), 'acc': mo['acc'],
                                                 'f1': mo['f1'], 'ece': mo['ece'], 'score': score,
                                                 'alpha': alpha, **r})

            # federated modes (default digest, as in experiment.run_calibration)
            for (proto, score), cal in fed_cal.items():
                if len(cal) < 2:
                    continue
                for alpha in ALPHAS:
                    result = federated_conformal_calibrate(cal, alpha=alpha, score=score, seed=seed * 100 + 50)
                    for cname, modes in evaluate_federated_modes(result, fed_te[(proto, score)]).items():
                        for mode, vals in modes.items():
                            fed_rows.append({'protocol': proto, 'strategy': name, 'seed': seed,
                                             'client_name': cname, 'score': score, 'alpha': alpha,
                                             'mode': mode, 'n_calib': result.n_cal[cname],
                                             'shrinkage_weight': result.shrinkage_weight[cname], **vals})
            print(f'[{name}] seed={seed} done')

    write_reports(res, audit, repro, acc_rows, per_client, fed_rows, ood_rows)


def _agg(df, keys, extra=None):
    spec = dict(coverage_mean=('coverage', 'mean'), coverage_std=('coverage', 'std'),
                coverage_min=('coverage', 'min'),
                set_size_mean=('mean_set_size', 'mean'), set_size_std=('mean_set_size', 'std'),
                frac_empty_mean=('frac_empty', 'mean'), n=('coverage', 'count'))
    spec.update(extra or {})
    return df.groupby(keys).agg(**spec).reset_index()

def write_reports(res, audit, repro, acc_rows, per_client, fed_rows, ood_rows):
    (res / 'leakage_audit.json').write_text(json.dumps(audit, indent=2))
    au = pd.DataFrame([{'key': k, **v} for k, v in audit.items()])
    au[['seed', 'set', 'split']] = au['key'].str.split('|', expand=True)
    (au.groupby(['set', 'split'])
       .agg(n_mean=('n', 'mean'), n_dup_mean=('n_dup', 'mean'), frac_dup_mean=('frac_dup', 'mean'),
            pixel_corr_min=('pixel_corr_min', 'min'), pixel_corr_median=('pixel_corr_median', 'median'))
       .reset_index().to_csv(res / 'leakage_summary.csv', index=False))

    if repro:
        rp = pd.DataFrame(repro); rp.to_csv(res / 'repro_check.csv', index=False)
        print('\n[repro] max abs diff vs original run:\n'
              + rp.drop(columns=['strategy', 'seed', 'client']).max().to_string())

    # accuracy / calibration per protocol; headline = mean over clients, then over seeds
    # (analyze.py's convention), with paired t-tests of fedua vs. each baseline
    ac = pd.DataFrame(acc_rows); ac.to_csv(res / 'accuracy_by_protocol_rows.csv', index=False)
    (ac.groupby(['protocol', 'strategy', 'client_name'])
       .agg(n_test=('n_test', 'mean'), acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
            ece_raw_mean=('ece_raw', 'mean'), ece_cal_mean=('ece_cal', 'mean'), temp_mean=('temp', 'mean'))
       .reset_index().to_csv(res / 'accuracy_by_protocol_per_client.csv', index=False))
    head = ac.groupby(['protocol', 'strategy', 'seed'])[['acc', 'f1', 'mcc', 'ece_raw', 'ece_cal']].mean()
    summ = head.groupby(['protocol', 'strategy']).agg(['mean', 'std'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ.reset_index().sort_values(['protocol', 'acc_mean'], ascending=[True, False]) \
        .to_csv(res / 'accuracy_by_protocol.csv', index=False)
    sig = []
    for proto in PROTOCOLS:
        h = head.loc[proto]['acc'].unstack('seed')
        if 'fedua' not in h.index:
            continue
        for b in h.index.drop('fedua'):
            d = (h.loc['fedua'] - h.loc[b]).dropna()
            t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
            sig.append({'protocol': proto, 'reference': 'fedua', 'baseline': b,
                        'delta_acc_pct': 100 * d.mean(), 'n_seeds': len(d),
                        't_pvalue': float(t.pvalue) if t else np.nan})
    if sig:
        sg = pd.DataFrame(sig)
        for proto, g in sg.groupby('protocol'):    # Holm within protocol
            p = g['t_pvalue'].values; order = np.argsort(p); adj = np.empty_like(p); run = 0.0
            for rank, i in enumerate(order):
                run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
            sg.loc[g.index, 't_pvalue_holm'] = adj
        sg.to_csv(res / 'significance_by_protocol.csv', index=False)

    pc = pd.DataFrame(per_client); pc.to_csv(res / 'conformal_per_client_rows.csv', index=False)
    _agg(pc, ['protocol', 'strategy', 'score', 'alpha']).to_csv(res / 'conformal_by_score.csv', index=False)
    _agg(pc, ['protocol', 'strategy', 'client_name', 'score', 'alpha'],
         {'n_calib_mean': ('n_calib', 'mean'), 'n_test_mean': ('n_test', 'mean')}) \
        .to_csv(res / 'conformal_by_score_per_client.csv', index=False)

    if fed_rows:
        fd = pd.DataFrame(fed_rows); fd.to_csv(res / 'fedconformal_rows.csv', index=False)
        _agg(fd, ['protocol', 'strategy', 'client_name', 'score', 'alpha', 'mode'],
             {'n_calib_mean': ('n_calib', 'mean'), 'shrinkage_weight_mean': ('shrinkage_weight', 'mean')}) \
            .to_csv(res / 'federated_conformal_comparison_by_score.csv', index=False)

    if ood_rows:
        od = pd.DataFrame(ood_rows); od.to_csv(res / 'ood_rows.csv', index=False)
        _agg(od, ['protocol', 'strategy', 'cohort', 'score', 'alpha'],
             {'n_ood': ('n_ood', 'first'), 'acc_mean': ('acc', 'mean'), 'acc_std': ('acc', 'std'),
              'ece_mean': ('ece', 'mean')}).to_csv(res / 'ood_summary.csv', index=False)
    print(f'\n[OK] reports -> {res}')


if __name__ == '__main__':
    main()
