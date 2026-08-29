# ============================================================
# FedUA-Net Ablation Study
#
# Runs systematic ablations to isolate contributions:
#   1. Attention ablation: none / channel / spatial / cbam
#   2. Fine-tuning ablation: fedbn (no finetune) vs fedua (with finetune)
#   3. BN ablation: fedavg (shared BN) vs fedbn (local BN)
#
# Produces: outputs_experiments/raw/ablation_*.csv
# ============================================================
import os, sys, time, copy, random, argparse
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r'D:/Research/FedUA-Net')
os.environ.setdefault('TORCH_HOME', r'D:/Research/FedUA-Net/.torch_cache')

import experiment as exp
import fedua_net as m

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=[0, 1, 2], nargs='*')
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--out', default=r'D:/Research/FedUA-Net/outputs_experiments')
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    OUT = Path(args.out)
    RAW = OUT / 'raw'
    RAW.mkdir(parents=True, exist_ok=True)

    exp.ARGS.batch = args.batch
    exp.ARGS.rounds = args.rounds
    m.cfg.BATCH_SIZE = args.batch
    m.cfg.COMM_ROUNDS = args.rounds

    print(f'PyTorch {torch.__version__} | device={DEV}')
    if DEV == 'cuda':
        print(f'GPU={torch.cuda.get_device_name(0)}  '
              f'free VRAM {torch.cuda.mem_get_info()[0] / 2**30:.1f}GB')

    # ---- ABLATION 1: Attention variants (all under fedua strategy) ----
    attention_variants = ['none', 'channel', 'spatial', 'cbam']
    # fedua = FedBN body + local heads + fine-tune + attention
    # fedbn = FedBN body + local heads (no fine-tune)

    all_rows = []
    for seed in args.seeds:
        print(f'\n{"="*60}')
        print(f'  SEED {seed}')
        print(f'{"="*60}')

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        all_df, classes, class_to_idx, client_dfs = exp.build_data(seed)
        cids = list(range(m.cfg.NUM_CLIENTS))
        loaders = {c: exp.loaders_for(client_dfs[c]) for c in cids}
        cw = {c: m.class_weights(client_dfs[c]) for c in cids}
        ncls_local = {c: int(client_dfs[c]['local'].max()) + 1 for c in cids}
        agg = exp.agg_weights(client_dfs)

        # --- Attention ablation (with fine-tuning = fedua strategy) ---
        for attn in attention_variants:
            tag = f'ablation_attn_{attn}'
            out_csv = RAW / f'ablation_{tag}_seed{seed}.csv'
            if args.resume and out_csv.exists():
                print(f'  [{tag}] seed={seed} already done — skipping')
                existing = pd.read_csv(out_csv)
                all_rows.extend(existing.to_dict('records'))
                continue

            t0 = time.time()
            print(f'  [{tag}] training (attention={attn}) ...')
            nets = exp.run_strategy('fedua', client_dfs, cids, ncls_local,
                                    cw, loaders, seed, args.rounds, agg,
                                    DEV, attention=attn)
            rows = []
            for c in cids:
                ev = exp.full_metrics(*exp._eval_arrays(nets[c], loaders[c]['test']))
                row = {'seed': seed, 'ablation': tag, 'attention': attn,
                       'strategy': 'fedua', 'finetune': True,
                       'client': c, 'client_name': m.client_meta(c)[3],
                       'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'],
                       'auc': ev['auc'], 'ece': ev['ece'], 'brier': ev['brier']}
                rows.append(row)
                all_rows.append(row)
                print(f'    C{c} acc={ev["acc"]:.3f} f1={ev["f1"]:.3f}')

            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'  [{tag}] done in {time.time()-t0:.1f}s')

            # cleanup GPU
            del nets
            torch.cuda.empty_cache() if DEV == 'cuda' else None

        # --- Fine-tuning ablation: fedbn (no fine-tune) with cbam ---
        tag = 'ablation_no_finetune'
        out_csv = RAW / f'ablation_{tag}_seed{seed}.csv'
        if args.resume and out_csv.exists():
            print(f'  [{tag}] seed={seed} already done — skipping')
            existing = pd.read_csv(out_csv)
            all_rows.extend(existing.to_dict('records'))
        else:
            t0 = time.time()
            print(f'  [{tag}] training (fedbn + cbam, NO fine-tune) ...')
            nets = exp.run_strategy('fedbn', client_dfs, cids, ncls_local,
                                    cw, loaders, seed, args.rounds, agg,
                                    DEV, attention='cbam')
            rows = []
            for c in cids:
                ev = exp.full_metrics(*exp._eval_arrays(nets[c], loaders[c]['test']))
                row = {'seed': seed, 'ablation': tag, 'attention': 'cbam',
                       'strategy': 'fedbn', 'finetune': False,
                       'client': c, 'client_name': m.client_meta(c)[3],
                       'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'],
                       'auc': ev['auc'], 'ece': ev['ece'], 'brier': ev['brier']}
                rows.append(row)
                all_rows.append(row)
                print(f'    C{c} acc={ev["acc"]:.3f} f1={ev["f1"]:.3f}')

            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'  [{tag}] done in {time.time()-t0:.1f}s')
            del nets
            torch.cuda.empty_cache() if DEV == 'cuda' else None

    # Save combined ablation results
    abl_df = pd.DataFrame(all_rows)
    abl_df.to_csv(RAW / 'ablation_all.csv', index=False)

    # Print summary table
    print('\n' + '='*80)
    print('  ABLATION STUDY RESULTS')
    print('='*80)
    for tag in abl_df['ablation'].unique():
        sub = abl_df[abl_df['ablation'] == tag]
        mean_acc = sub.groupby('seed')['acc'].mean()
        print(f'  {tag:30s}: acc = {mean_acc.mean()*100:.2f} ± {mean_acc.std()*100:.2f}%')

    print(f'\n[OK] Ablation results saved to {RAW}')


if __name__ == '__main__':
    main()
