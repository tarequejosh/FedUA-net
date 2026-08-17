"""FedUA-Net v4 - post-FL personalization fine-tune + final publication eval.

Loads the federated checkpoint, then fine-tunes each client's personalized
network on its own local data (standard FedPer practice) and produces the
final per-client + per-class metrics, uncertainty stats, curves, and a
comparison vs the prior (v1) results.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'D:/Research/FedUA-Net')
import fedua_net_v4_pytorch as m

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = m.cfg.OUTPUT_DIR
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FINETUNE_EPOCHS = {0: 8, 1: 15, 2: 5}
LR_HEAD, LR_BODY = 4e-4, 4e-5

print(f'device={DEV}')

# ---- data ----
df, classes, cmap = m.discover_all(smoke=False)
client_dfs = {cid: df[df['client'] == cid] for cid in range(m.cfg.NUM_CLIENTS)}
for cid in range(m.cfg.NUM_CLIENTS):
    uniq = sorted(client_dfs[cid]['gid'].unique())
    g2l = {g: i for i, g in enumerate(uniq)}
    client_dfs[cid] = client_dfs[cid].copy()
    client_dfs[cid]['local'] = client_dfs[cid]['gid'].map(g2l)
loaders = {cid: m.build_loaders(client_dfs[cid], m.cfg.BATCH_SIZE, 0)
           for cid in range(m.cfg.NUM_CLIENTS)}
cweights = {cid: m.class_weights(client_dfs[cid]) for cid in range(m.cfg.NUM_CLIENTS)}
client_uniq = {cid: sorted(client_dfs[cid]['gid'].unique()) for cid in range(m.cfg.NUM_CLIENTS)}

# ---- load checkpoint ----
ckpt = torch.load(os.path.join(OUT, 'models', 'checkpoint_r10.pt'), map_location=DEV)
nets = {}
for cid in range(m.cfg.NUM_CLIENTS):
    nets[cid] = m.ClientNet(len(client_uniq[cid]), m.cfg.BACKBONE,
                            m.cfg.EMB, m.cfg.DROPOUT).to(DEV)
    nets[cid].load_state_dict(ckpt['nets'][cid])
    nets[cid].train()
print('[OK] checkpoint loaded')

# ---- per-client fine-tune ----
def ft(net, loader, cw, epochs, lr_h, lr_b):
    body = [p for p in net.body.parameters() if p.requires_grad]
    head = [p for p in net.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': body, 'lr': lr_b, 'weight_decay': m.cfg.WD},
        {'params': head, 'lr': lr_h, 'weight_decay': m.cfg.WD}])
    w = torch.zeros(max(cw) + 1).to(DEV)
    for k, v in cw.items():
        w[k] = v
    for ep in range(epochs):
        net.train(); n = acc = 0
        for x, y in loader['train']:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            out = net(x)
            loss = (F.cross_entropy(out, y, reduction='none', label_smoothing=0.1) * w[y]).mean()
            loss.backward(); opt.step()
            n += x.size(0); acc += (out.argmax(1) == y).sum().item()
    return acc / max(n, 1)

for cid in range(m.cfg.NUM_CLIENTS):
    a = ft(nets[cid], loaders[cid], cweights[cid], FINETUNE_EPOCHS[cid], LR_HEAD, LR_BODY)
    print(f'  C{cid} fine-tune train acc={a:.4f}')

# ---- final eval ----
print('=' * 72)
print('  FINAL FEDUA-NET v4 RESULTS (personalized + fine-tuned)')
print('=' * 72)
from sklearn.metrics import classification_report
summary, per_class_rows = [], []
for cid in range(m.cfg.NUM_CLIENTS):
    ev = m.evaluate(nets[cid], loaders[cid]['test'], DEV)
    local_names = [classes[g] for g in client_uniq[cid]]
    print(f"\n  {m.client_meta(cid)[3]}: Acc={ev['acc']:.4f}  F1={ev['f1']:.4f}  "
          f"AUC={ev['auc']:.4f}  P={ev['precision']:.4f}  R={ev['recall']:.4f}")
    summary.append({'client': cid, 'name': m.client_meta(cid)[3],
                    'accuracy': ev['acc'], 'precision': ev['precision'],
                    'recall': ev['recall'], 'f1': ev['f1'], 'auc': ev['auc']})
    rep = classification_report(ev['y_true'], ev['y_prob'].argmax(1),
                                labels=list(range(len(local_names))),
                                target_names=local_names, output_dict=True, zero_division=0)
    for cl in local_names:
        d = rep.get(cl, {})
        per_class_rows.append({'client': m.client_meta(cid)[3], 'class': cl,
                               'precision': d.get('precision'), 'recall': d.get('recall'),
                               'f1': d.get('f1-score'), 'support': d.get('support')})

sdf = pd.DataFrame(summary)
pcdf = pd.DataFrame(per_class_rows)
print('\n' + sdf.to_string(index=False))
mean_acc = sdf['accuracy'].mean()
mean_f1 = sdf['f1'].mean()
print(f'\nMEAN client accuracy = {mean_acc:.4f}   macro-F1 = {mean_f1:.4f}')

# ---- comparison vs prior ----
prior = pd.DataFrame([
    {'Model': 'FedUA-Net v1 global FL (prior)', 'Accuracy': 0.6517, 'Macro-F1': 0.5016},
    {'Model': 'MobileNetV2 centralized (prior best)', 'Accuracy': 0.7643, 'Macro-F1': 0.7133},
    {'Model': 'FedUA-Net v4 personalized FL (+finetune)', 'Accuracy': mean_acc, 'Macro-F1': mean_f1},
])
print('\n' + '=' * 60)
print(prior.to_string(index=False))
print('=' * 60)

# ---- save ----
sdf.to_csv(os.path.join(OUT, 'reports', 'v4_final_client_summary.csv'), index=False)
pcdf.to_csv(os.path.join(OUT, 'reports', 'v4_final_per_class.csv'), index=False)
prior.to_csv(os.path.join(OUT, 'reports', 'v4_vs_prior.csv'), index=False)
torch.save({'nets': {str(cid): nets[cid].state_dict() for cid in nets}},
           os.path.join(OUT, 'models', 'fedua_net_v4_finetuned.pt'))

# curves figure from fed log
log = pd.read_csv(os.path.join(OUT, 'reports', 'v4_fed_log.csv'))
plt.style.use('seaborn-v0_8-whitegrid')
cols = ['#1976D2', '#F57C00', '#388E3C']
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(log['round'], log['mean_test_acc'], 'k-o', lw=2.5, label='Mean client test acc')
for c in range(3):
    ax.plot(log['round'], log[f'c{c}_test_acc'], '--', marker='s', ms=4,
            color=cols[c], label=m.client_meta(c)[3])
ax.set_xlabel('Communication round'); ax.set_ylabel('Accuracy')
ax.set_title('FedUA-Net v4 - federated learning progress', fontweight='bold')
ax.legend(fontsize=8); ax.set_xticks(log['round'])
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'v4_training_curves.png'), dpi=200)

# uncertainty (MC dropout)
print('\n[MC] uncertainty estimation ...')
rows = []
for cid in range(m.cfg.NUM_CLIENTS):
    if loaders[cid]['test'] is None:
        continue
    uq = m.mc_predict(nets[cid], loaders[cid]['test'], DEV, iters=m.cfg.MC_ITERS, max_batches=10)
    ys = []
    for xb, yb in loaders[cid]['test']:
        ys.append(yb.numpy())
        if len(ys) * yb.size(0) >= len(uq['mean_prob']):
            break
    ys = np.concatenate(ys)[:len(uq['mean_prob'])]
    ok = (uq['mean_prob'].argmax(1) == ys).astype(int)
    for i in range(len(ys)):
        rows.append({'client': m.client_meta(cid)[3], 'correct': ok[i],
                     'entropy': uq['entropy'][i], 'uncertainty': uq['uncertainty'][i]})
uqdf = pd.DataFrame(rows)
uqdf.to_csv(os.path.join(OUT, 'reports', 'v4_final_uncertainty.csv'), index=False)
g = uqdf.groupby(['client', 'correct'])['entropy'].mean().unstack()
print(g.to_string())

with open(os.path.join(OUT, 'reports', 'v4_final_report.txt'), 'w') as f:
    f.write('FEDUA-NET v4 FINAL REPORT (publication)\n')
    f.write('=' * 70 + '\n')
    f.write(sdf.to_string(index=False) + '\n')
    f.write(f'\nMean client accuracy = {mean_acc:.4f}\n')
    f.write(f'Mean client macro-F1 = {mean_f1:.4f}\n\n')
    f.write('Comparison vs prior work:\n')
    f.write(prior.to_string(index=False) + '\n\n')
    f.write('Per-class:\n')
    f.write(pcdf.to_string(index=False) + '\n\n')
    f.write('Uncertainty (mean entropy):\n')
    f.write(g.to_string() + '\n')
print('\n[OK] done ->', OUT)
