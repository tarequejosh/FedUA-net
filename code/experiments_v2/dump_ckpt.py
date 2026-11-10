# Dump val/test logits + MC-dropout probabilities from saved final client checkpoints
# (produced by experiment.py --save_final_models) into the v2 logits format.
#   python code/experiments_v2/dump_ckpt.py            (defaults: models/fedua_uniform_3seed -> results/rerun_v2)
# Paths given on the command line are relative to the repository root.
import os, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # code/
REPO = ROOT.parent                                  # repository root
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments_v2'))
os.chdir(ROOT)
os.environ.setdefault('TORCH_HOME', str(ROOT / '.torch_cache'))

import numpy as np, pandas as pd, torch
import experiment as E
import fedua_net as m
from run_v2 import make_loaders, dump_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='models/fedua_uniform_3seed')
    ap.add_argument('--job', default='fedua')
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--out', default='results/rerun_v2')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = REPO / a.out / 'logits'; out.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        tag = f'{a.job}_split{s}_train{s}_nb0'
        _, _, _, client_dfs = E.build_data(s)
        rows = []
        for c in range(m.cfg.NUM_CLIENTS):
            ncls = int(client_dfs[c]['local'].max()) + 1
            net = m.ClientNet(ncls, m.cfg.BACKBONE, m.cfg.EMB, m.cfg.DROPOUT, attention='cbam').to(E.DEV)
            net.load_state_dict(torch.load(REPO / a.ckpt_dir / 'final_models' / f'fedua_seed{s}_client{c}.pt',
                                           map_location=E.DEV))
            ld = make_loaders(client_dfs[c], c, 0, s, a.workers, 64)
            rec = {}
            for split in ('val', 'test'):
                lg, mc, y = dump_outputs(net, ld[split], 30, E.DEV)
                rec[f'{split}_logits'] = lg; rec[f'{split}_mc'] = mc; rec[f'{split}_y'] = y
            np.savez_compressed(out / f'{tag}_c{c}.npz', **rec)
            ev = E.full_metrics(rec['test_y'], E._to_probs(rec['test_logits']))
            rows.append({'job': a.job, 'split_seed': s, 'train_seed': s, 'nb': 0, 'client': c,
                         'n_train': int((client_dfs[c]['split'] == 'train').sum()),
                         'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'], 'auc': ev['auc'],
                         'ece': ev['ece'], 'brier': ev['brier'], 'train_time_s': np.nan})
            print(f'{tag} c{c}: acc={ev["acc"]*100:.2f}')
        pd.DataFrame(rows).to_csv(out / f'{tag}_metrics.csv', index=False)
        (out / f'{tag}.done').write_text('from checkpoint ' + a.ckpt_dir)


if __name__ == '__main__':
    main()
