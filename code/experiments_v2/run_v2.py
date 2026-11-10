# ============================================================
# FedUA-Net v2 experiment runner.
#
# Reuses the canonical harness (experiment.py / fedua_net.py) unchanged and adds:
#   - decoupled split seed (data partition) vs. train seed (init / order / dropout)
#     so that two independently trained local experts share one data split;
#   - per-sample logits + MC-dropout probabilities on val and test, saved to .npz,
#     so that calibration, expert fusion and conformal analysis run post hoc
#     without retraining;
#   - Hospital-B data-scarcity levels;
#   - client subsets (e.g. local-only training of Hospital B alone).
#
# Usage:
#   python experiments_v2/run_v2.py --job fedua --split_seed 0 --train_seed 0 --nb 0
#   python experiments_v2/run_v2.py --job local --split_seed 0 --train_seed 1000 --nb 100 --clients 1
# ============================================================
import os, sys, time, json, argparse, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('TORCH_HOME', str(ROOT / '.torch_cache'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as E
import fedua_net as m

JOB_TO_STRATEGY = {'fedua': 'fedua', 'local': 'local_only', 'fedbn': 'fedbn',
                   'fedavg': 'fedavg', 'fedprox': 'fedprox', 'fedbabu': 'fedbabu',
                   'ditto': 'ditto'}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, choices=list(JOB_TO_STRATEGY))
    ap.add_argument('--split_seed', type=int, required=True)
    ap.add_argument('--train_seed', type=int, required=True)
    ap.add_argument('--nb', type=int, default=0, help='Hospital-B train subset size (0 = all)')
    ap.add_argument('--clients', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--mc_iters', type=int, default=30)
    ap.add_argument('--out', default='outputs_v2')
    return ap.parse_args()


def make_loaders(df, cid, nb, split_seed, workers, batch):
    """Same splits / transforms / subsampling as m.build_loaders, with persistent workers."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        tf = m.train_transforms(ultrasound=False) if split == 'train' else m.eval_transforms()
        ds = m.MedImgDataset(sub, tf)
        if split == 'train' and nb and cid == 1:
            rng = np.random.default_rng(split_seed)
            idx = rng.choice(len(ds), size=min(nb, len(ds)), replace=False).tolist()
            ds = torch.utils.data.Subset(ds, idx)
        is_tr = split == 'train'
        # train: persistent workers; val: 2 persistent workers; test: in-process (read once).
        # Few worker processes and a timeout: a hung worker spawn raises instead of stalling.
        nw = workers if is_tr else (min(2, workers) if split == 'val' else 0)
        loaders[split] = DataLoader(
            ds, batch_size=batch, shuffle=is_tr, num_workers=nw,
            pin_memory=True, drop_last=(is_tr and len(ds) >= batch),
            persistent_workers=nw > 0, timeout=(900 if nw > 0 else 0),
            prefetch_factor=(4 if nw > 0 else None))
    return loaders


@torch.no_grad()
def dump_outputs(net, loader, mc_iters, device):
    """Deterministic logits, MC-dropout mean probabilities and embeddings."""
    net.eval()
    L, P, Y = [], [], []
    for x, y in loader:
        v = net.body(x.to(device), mc=False)            # dropout off
        L.append(net.head(v).float().cpu())
        mc = torch.stack([torch.softmax(net.head(F.dropout(v, p=net.body.dropout_rate,
                                                              training=True)).float(), -1)
                          for _ in range(mc_iters)]).mean(0)
        P.append(mc.cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(P).numpy(), torch.cat(Y).numpy()


def main():
    a = parse()
    dev = E.DEV
    strategy = JOB_TO_STRATEGY[a.job]
    out = ROOT / a.out
    (out / 'logits').mkdir(parents=True, exist_ok=True)
    tag = f'{a.job}_split{a.split_seed}_train{a.train_seed}_nb{a.nb}'
    done_flag = out / 'logits' / f'{tag}.done'
    if done_flag.exists():
        print(f'[{tag}] already done, skipping'); return

    # configure harness exactly as experiment.py does
    E.ARGS.hospital_b_subset_size = a.nb
    E.ARGS.rounds = a.rounds
    m.cfg.HOSPITAL_B_SUBSET_SIZE = a.nb if a.nb > 0 else None
    m.cfg.COMM_ROUNDS = a.rounds
    m.cfg.PERSONALIZE_DEEP = False
    m.cfg.ULTRASOUND_AUG = False
    m.cfg.ULTRASOUND_AUG_MILD = False

    torch.manual_seed(a.train_seed); np.random.seed(a.train_seed); random.seed(a.train_seed)
    all_df, classes, class_to_idx, client_dfs = E.build_data(a.split_seed)   # split by split_seed
    cids = list(range(m.cfg.NUM_CLIENTS))
    m.SEED = a.split_seed   # class_weights subsampling rng uses the split seed
    cw = {c: m.class_weights(client_dfs[c], hospital_b_subset_size=a.nb or None) for c in cids}
    ncls = {c: int(client_dfs[c]['local'].max()) + 1 for c in cids}
    agg = E.agg_weights(client_dfs, weight_type='uniform')
    ids = cids if strategy != 'local_only' else a.clients
    mk = lambda c: make_loaders(client_dfs[c], c, a.nb, a.split_seed, a.workers, E.ARGS.batch)
    npz = lambda c: out / 'logits' / f'{tag}_c{c}.npz'
    rows, t0 = [], time.time()

    def finish(c, net, ld, n_train, train_time):
        rec = {}
        for split in ('val', 'test'):
            lg, mc, y = dump_outputs(net, ld[split], a.mc_iters, dev)
            rec[f'{split}_logits'] = lg; rec[f'{split}_mc'] = mc; rec[f'{split}_y'] = y
        np.savez_compressed(npz(c), **rec)
        ev = E.full_metrics(rec['test_y'], E._to_probs(rec['test_logits']))
        row = {'job': a.job, 'split_seed': a.split_seed, 'train_seed': a.train_seed,
               'nb': a.nb, 'client': c, 'n_train': n_train,
               'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'], 'auc': ev['auc'],
               'ece': ev['ece'], 'brier': ev['brier'], 'train_time_s': train_time}
        pd.DataFrame([row]).to_csv(out / 'logits' / f'{tag}_c{c}_metrics.csv', index=False)
        print(f'  c{c}: acc={ev["acc"]*100:.2f} f1={ev["f1"]*100:.2f} ece={ev["ece"]:.4f} '
              f'({train_time/60:.1f} min)', flush=True)

    if strategy == 'local_only':
        # independent per client: train, dump and free loaders one client at a time (resumable)
        for c in ids:
            if npz(c).exists() and (out / 'logits' / f'{tag}_c{c}_metrics.csv').exists():
                print(f'[{tag}] c{c} already done, skipping', flush=True); continue
            ld = mk(c); t1 = time.time()
            print(f'[{tag}] local c{c} n_train={len(ld["train"].dataset)}', flush=True)
            nets = E.run_strategy('local_only', client_dfs, [c], ncls, cw, {c: ld},
                                  a.train_seed, a.rounds, agg, dev)
            finish(c, nets[c], ld, len(ld['train'].dataset), time.time() - t1)
            del nets, ld; torch.cuda.empty_cache()
    else:
        loaders = {c: mk(c) for c in cids}
        n_train = {c: len(loaders[c]['train'].dataset) for c in ids}
        print(f'[{tag}] strategy={strategy} clients={ids} n_train={n_train}', flush=True)
        nets = E.run_strategy(strategy, client_dfs, ids, ncls, cw, loaders,
                              a.train_seed, a.rounds, agg, dev)
        for c in ids:
            finish(c, nets[c], loaders[c], n_train[c], time.time() - t0)
    train_time = time.time() - t0
    rows = [pd.read_csv(out / 'logits' / f'{tag}_c{c}_metrics.csv') for c in ids]
    pd.concat(rows).to_csv(out / 'logits' / f'{tag}_metrics.csv', index=False)
    done_flag.write_text(json.dumps({'train_time_s': train_time}))
    print(f'[{tag}] done in {train_time/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
