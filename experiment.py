# ============================================================
# FedUA-Net Tier-1 experiment harness (publication rigor)
#
# Implements, under ONE fixed architecture (SharedBody+LocalHead):
#   - Multi-seed training (splits + model init vary with seed)
#   - Baseline ladder: FedAvg / FedBN / FedProx / FedBABU / Ditto /
#     Local-only / Centralized  (+ 'fedua' = proposed: FedBN-style
#     body + local heads, post personalization fine-tune,
#     temperature scaling, conformal calibration)
#   - Full metrics: acc, macro P/R/F1, MCC, AUROC, ECE, Brier
#   - Leave-one-client-out (LOCO) unseen-site generalization
#   - Per-client conformal prediction (APS) + risk-coverage curves
#
# Raw per-(seed,strategy) results are saved to outputs_experiments/raw/.
# Cross-seed stats / Wilcoxon are computed by analyze.py.
# ============================================================
import os, sys, time, json, copy, random, argparse, contextlib
from collections import OrderedDict, Counter
from pathlib import Path

if sys.stdout.encoding != 'utf-8' or not getattr(sys.stdout, 'line_buffering', False):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fedua_net as m

os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent / '.torch_cache'))

MTHIS_NUM_CLIENTS = 3   # 3 clients: Brain MRI / Breast US / COVID X-ray
m.cfg.NUM_CLIENTS = MTHIS_NUM_CLIENTS
m.cfg.LOCAL_EPOCHS = {0: 4, 1: 10, 2: 2}

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='./Dataset', help='Path to Dataset root directory')
    ap.add_argument('--strategies', nargs='*', default=[
        'fedavg', 'fedbn', 'fedprox', 'fedbabu', 'ditto', 'local_only',
        'centralized', 'fedua'])
    ap.add_argument('--seeds', type=int, default=[0, 1, 2], nargs='*')
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--attention', default='cbam', choices=['cbam', 'channel', 'spatial', 'none'])
    ap.add_argument('--agg_weight_type', type=str, default='uniform', choices=['uniform', 'sample'],
                    help="Server aggregation weighting: 'uniform' (1/K) or 'sample' (N_k/N_total)")
    ap.add_argument('--personalize_deep', action='store_true',
                    help='CKA-guided: keep CBAM attention + projection layer local per client, never aggregated')
    ap.add_argument('--ultrasound_aug', action='store_true', default=False,
                    help='Apply ultrasound-specific data augmentations (Speckle noise + Elastic transform) for Hospital B')
    ap.add_argument('--ultrasound_aug_mild', action='store_true', default=False,
                    help='Apply milder ultrasound-specific data augmentations (SpeckleNoise(0.04, 0.3) + ElasticTransform(15.0, 4.0)) for Hospital B')
    ap.add_argument('--save_final_models', action='store_true',
                    help='Save each client final trained model state_dict for post-hoc analysis (CKA, qualitative gallery)')
    ap.add_argument('--hospital_b_subset_size', type=int, default=0,
                    help="Subsample size for Hospital B training dataset (0 = use all data)")
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--loco', action='store_true', help='run LOCO generalization')
    ap.add_argument('--out', default='./outputs_experiments')
    ap.add_argument('--resume', action='store_true',
                    help='skip strategies whose raw CSV already exists for a given seed')
    if cmd_args is not None:
        return ap.parse_args(cmd_args)
    if __name__ == '__main__':
        return ap.parse_args()
    return ap.parse_known_args([])[0]

ARGS = parse_args()
OUT = Path(ARGS.out)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RAW = OUT / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
(OUT / 'reports').mkdir(parents=True, exist_ok=True)

m.cfg.BATCH_SIZE = ARGS.batch
m.cfg.COMM_ROUNDS = ARGS.rounds
m.cfg.AGG_WEIGHT_TYPE = ARGS.agg_weight_type
m.cfg.PERSONALIZE_DEEP = ARGS.personalize_deep
m.cfg.ULTRASOUND_AUG = ARGS.ultrasound_aug
m.cfg.ULTRASOUND_AUG_MILD = ARGS.ultrasound_aug_mild
m.cfg.HOSPITAL_B_SUBSET_SIZE = ARGS.hospital_b_subset_size if ARGS.hospital_b_subset_size > 0 else None
m.cfg.DATA_ROOT = ARGS.data_root
m.ROOT = Path(ARGS.data_root)
m.DATASET_DIR = {
    'brain_tumor': m.ROOT / 'Brain Tumor MRI Dataset',
    'busi':        m.ROOT / 'Dataset_BUSI_with_GT',
    'covid':       m.ROOT / 'COVID-19_Radiography_Dataset',
}

# ------------------------------------------------------------
# Data builders (seed-aware splits via m.SEED)
# ------------------------------------------------------------
def build_data(seed):
    """Discover data with seed-dependent split; return client_dfs (with local
    labels) + global classes."""
    m.SEED = seed
    m.cfg.NUM_CLIENTS = MTHIS_NUM_CLIENTS
    all_df, classes, class_to_idx = m.discover_all(smoke=ARGS.smoke)
    all_df['gid'] = all_df['label'].map(class_to_idx)
    client_dfs = {}
    for cid in range(MTHIS_NUM_CLIENTS):
        sub = all_df[all_df['client'] == cid].copy()
        uniq = sorted(sub['gid'].unique())
        g2l = {g: i for i, g in enumerate(uniq)}
        sub['local'] = sub['gid'].map(g2l)
        client_dfs[cid] = sub
    return all_df, classes, class_to_idx, client_dfs

def loaders_for(df, client_id=None):
    return m.build_loaders(df, ARGS.batch, 0, hospital_b_subset_size=ARGS.hospital_b_subset_size, client_id=client_id)

def train_size(df):
    return int((df['split'] == 'train').sum())

def agg_weights(client_dfs, weight_type=None):
    if weight_type is None:
        weight_type = getattr(ARGS, 'agg_weight_type', 'uniform')
    if weight_type == 'uniform':
        K = len(client_dfs)
        return {c: 1.0 / K for c in client_dfs}
    else:
        sizes = {c: train_size(client_dfs[c]) for c in client_dfs}
        tot = sum(sizes.values())
        return {c: s / tot for c, s in sizes.items()}

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
def new_client_net(ncls, seed, attention=None):
    if attention is None:
        attention = ARGS.attention
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return m.ClientNet(ncls, m.cfg.BACKBONE, m.cfg.EMB, m.cfg.DROPOUT, attention=attention).to(DEV)

def new_shared_body(seed, attention=None):
    if attention is None:
        attention = ARGS.attention
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return m.SharedBody(m.cfg.BACKBONE, m.cfg.EMB, m.cfg.DROPOUT, attention=attention).to(DEV)

def sync_body(net, ref_body, include_bn=False):
    """Copy ref_body params into net.body (optionally excluding BN)."""
    m.copy_shared(ref_body, net.body, keep_bn=not include_bn)

def body_state(net, include_bn=False):
    if include_bn:
        return {k: v.detach().cpu() for k, v in net.body.state_dict().items()}
    return m.state_dict_excluding_bn(net.body)

def set_body(net_or_body, sd, include_bn=False):
    m.set_state_dict(net_or_body, sd, exclude_bn=not include_bn)

def average_bodies(states, weights, include_bn=False):
    """states: list of state-dicts; weights: parallel list."""
    _ = include_bn
    keys = list(states[0].keys())
    W = sum(weights)
    avg = {}
    for k in keys:
        acc = None
        for w, st in zip(weights, states):
            v = st[k].float()
            acc = v * w if acc is None else acc + v * w
        avg[k] = acc / W
    return avg

# ------------------------------------------------------------
# Local training (supports FedProx penalty + FedBABU head freeze)
# ------------------------------------------------------------
def build_wvec(cw, device):
    w = torch.zeros(max(cw) + 1, dtype=torch.float32)
    for k, v in cw.items():
        w[k] = v
    return w.to(device)

def local_train(net, train_loader, val_loader, cw, epochs, lr_b, lr_h, device,
                prox_ref=None, prox_mu=0.0, freeze_head=False, tag=''):
    """Train one client. Returns (tr_acc, val_acc)."""
    if freeze_head:
        for p in net.head.parameters():
            p.requires_grad = False
        head_params = []
    else:
        head_params = [p for p in net.head.parameters() if p.requires_grad]
    body_params = [p for p in net.body.parameters() if p.requires_grad]

    groups = [{'params': body_params, 'lr': lr_b, 'weight_decay': m.cfg.WD}]
    if head_params:
        groups.append({'params': head_params, 'lr': lr_h, 'weight_decay': m.cfg.WD})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    w = build_wvec(cw, device)

    bn = m.bn_param_names(net.body)
    net.train()
    n_tr = acc_tr = 0
    for ep in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = net(x, mc=False)
            loss = F.cross_entropy(out, y, reduction='none', label_smoothing=m.cfg.LABEL_SMOOTHING)
            loss = (loss * w[y]).mean()
            if prox_ref is not None and prox_mu > 0:
                pen = 0.0; cnt = 0
                for n_, p in net.body.named_parameters():
                    if n_ in bn or n_ not in prox_ref:
                        continue
                    pen = pen + (p - prox_ref[n_].to(device)).pow(2).sum()
                    cnt += 1
                loss = loss + prox_mu * pen / max(cnt, 1)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            n_tr += x.size(0)
            acc_tr += (out.argmax(1) == y).sum().item()
        sched.step()

    if freeze_head:
        for p in net.head.parameters():
            p.requires_grad = True

    net.eval(); n_v = acc_v = 0
    if val_loader is not None:
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = net(x, mc=False)
                n_v += x.size(0)
                acc_v += (out.argmax(1) == y).sum().item()
    return acc_tr / max(n_tr, 1), (acc_v / max(n_v, 1) if n_v else float('nan'))

def lr_schedule(rnd, rounds, warmup=2):
    if rnd <= warmup:
        frac = rnd / max(warmup, 1)
    else:
        prog = (rnd - warmup) / max(1, rounds - warmup)
        frac = 0.5 * (1 + np.cos(np.pi * prog))
    return (m.cfg.MIN_LR + (m.cfg.BACKBONE_LR - m.cfg.MIN_LR) * frac,
            m.cfg.MIN_LR + (m.cfg.BASE_LR - m.cfg.MIN_LR) * frac)

# ------------------------------------------------------------
# Strategies
# ------------------------------------------------------------
def run_strategy(name, client_dfs, client_ids, ncls_local, cw, loaders, seed,
                 rounds, agg_w, device, attention=None):
    """Return {cid: net}. client_ids: subset participating in federation."""
    if attention is None:
        attention = ARGS.attention
    ids = list(client_ids)
    nets = {c: new_client_net(ncls_local[c], seed, attention=attention) for c in ids}
    # shared init from client 0's body
    ref_body = new_shared_body(seed, attention=attention)
    if ids:
        m.copy_shared(nets[ids[0]].body, ref_body, keep_bn=False)
        for c in ids:
            m.copy_shared(ref_body, nets[c].body, keep_bn=False)

    if name == 'local_only':
        TOTAL_LOCAL_EPOCHS = {c: m.cfg.LOCAL_EPOCHS.get(c, 2) * rounds for c in ids}
        for c in ids:
            local_train(nets[c], loaders[c]['train'], loaders[c]['val'],
                        cw[c], TOTAL_LOCAL_EPOCHS[c], m.cfg.BACKBONE_LR, m.cfg.BASE_LR, device)
        return nets

    if name == 'fedavg':
        include_bn = True
        prox_ref = None
    elif name == 'fedbn':
        include_bn = False
        prox_ref = None
    elif name == 'fedprox':
        include_bn = False
        prox_ref = 'live'
    elif name == 'fedbabu':
        include_bn = False
        prox_ref = None
    elif name == 'ditto':
        include_bn = False
        prox_ref = None
    elif name == 'fedua':
        include_bn = False
        prox_ref = None
    elif name == 'ensemble':
        include_bn = False
        prox_ref = None
    else:
        raise ValueError(name)

    prox_mu = 0.01 if name == 'fedprox' else (1.0 if name == 'ditto' else 0.0)

    for rnd in range(1, rounds + 1):
        lr_b, lr_h = lr_schedule(rnd, rounds)
        weights, states = [], []
        ref_sd = m.state_dict_excluding_bn(ref_body)
        for c in ids:
            if name != 'ditto':
                sync_body(nets[c], ref_body, include_bn=include_bn)
            freeze = (name == 'fedbabu')
            prox_ref_body = ref_sd if (name in ('fedprox', 'ditto')) else None
            local_train(nets[c], loaders[c]['train'], loaders[c]['val'], cw[c],
                        m.cfg.LOCAL_EPOCHS[c], lr_b, lr_h, device,
                        prox_ref=prox_ref_body, prox_mu=prox_mu,
                        freeze_head=freeze, tag=f'{name}/c{c}/r{rnd}')
            if name == 'ditto':
                weights.append(agg_w[c]); states.append(body_state(nets[c], include_bn=False))
            else:
                weights.append(agg_w[c]); states.append(body_state(nets[c], include_bn=include_bn))
        avg = average_bodies(states, weights)
        set_body(ref_body, avg, include_bn=include_bn)

    # FedBABU: final head fine-tune
    if name == 'fedbabu':
        for c in ids:
            local_train(nets[c], loaders[c]['train'], loaders[c]['val'], cw[c],
                        3, m.cfg.BACKBONE_LR * 0.5, m.cfg.BASE_LR, device)

    # Proposed FedUA-Net: per-client personalization fine-tune with validation checkpointing
    if name == 'fedua':
        FINETUNE_EPOCHS = {0: 8, 1: 15, 2: 5}
        LR_HEAD, LR_BODY = 4e-4, 4e-5
        for c in ids:
            best_net_sd = copy.deepcopy(nets[c].state_dict())
            best_val_acc = -1.0
            for ep in range(FINETUNE_EPOCHS.get(c, 5)):
                tra, va = local_train(nets[c], loaders[c]['train'], loaders[c]['val'], cw[c],
                                      1, LR_BODY, LR_HEAD, device)
                if va >= best_val_acc:
                    best_val_acc = va
                    best_net_sd = copy.deepcopy(nets[c].state_dict())
            nets[c].load_state_dict(best_net_sd)
    return nets

def run_centralized(client_dfs, classes, seed, device, attention=None):
    """Pool all train+val (gid labels), train one 11-class net."""
    if attention is None:
        attention = ARGS.attention
    frames = []
    for c in client_dfs:
        sub = client_dfs[c][client_dfs[c]['split'].isin(['train', 'val'])].copy()
        frames.append(sub)
    tr = pd.concat(frames, ignore_index=True)
    te = pd.concat([client_dfs[c][client_dfs[c]['split'] == 'test']
                    for c in client_dfs], ignore_index=True)

    class GidDS(Dataset):
        def __init__(self, df, tf):
            self.paths = df['path'].tolist()
            self.y = df['gid'].tolist()
            self.c = df['client'].tolist()
            self.tf = tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = torchvision.io.read_image(self.paths[i])
            if img.shape[0] == 1: img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3: img = img[:3]
            return self.tf(img), self.y[i], self.c[i]

    trds = GidDS(tr, m.train_transforms())
    teds = GidDS(te, m.eval_transforms())
    tr_loader = DataLoader(trds, batch_size=ARGS.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(teds, batch_size=ARGS.batch, shuffle=False, num_workers=0)

    net = m.ClientNet(len(classes), m.cfg.BACKBONE, m.cfg.EMB, m.cfg.DROPOUT, attention=attention).to(device)
    epochs = 8
    body = [p for p in net.body.parameters() if p.requires_grad]
    head = [p for p in net.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': body, 'lr': m.cfg.BACKBONE_LR, 'weight_decay': m.cfg.WD},
                             {'params': head, 'lr': m.cfg.BASE_LR, 'weight_decay': m.cfg.WD}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        net.train(); n = a = 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x, mc=False)
            loss = F.cross_entropy(out, y, label_smoothing=m.cfg.LABEL_SMOOTHING)
            loss.backward(); opt.step()
            n += x.size(0); a += (out.argmax(1) == y).sum().item()
        sched.step()

    # eval
    y_true, y_prob, cli = [], [], []
    net.eval()
    with torch.no_grad():
        for x, y, c in te_loader:
            x = x.to(device)
            y_true.append(y.numpy()); cli.append(c.numpy())
            y_prob.append(net(x, mc=False).cpu().numpy())
    return {'y_true': np.concatenate(y_true), 'y_prob': _to_probs(np.concatenate(y_prob)),
            'cli': np.concatenate(cli), 'net': net}

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def full_metrics(y_true, y_prob):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 matthews_corrcoef, roc_curve, auc)
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(1)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    aucs = []
    for c in range(y_prob.shape[1]):
        b = (y_true == c).astype(int)
        if len(np.unique(b)) < 2:
            continue
        fpr, tpr, _ = roc_curve(b, y_prob[:, c])
        aucs.append(auc(fpr, tpr))
    auc_mean = float(np.nanmean(aucs)) if aucs else float('nan')

    # ECE — 15 equal-width bins
    conf = y_prob.max(1); correct = (y_pred == y_true).astype(float)
    n_bins = 15
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc  = float(correct[mask].mean())
        ece += (mask.sum() / len(conf)) * abs(avg_conf - avg_acc)

    # Brier score
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    brier = float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))

    return {'acc': acc, 'precision': float(p), 'recall': float(r), 'f1': float(f1), 'mcc': mcc,
            'auc': auc_mean, 'ece': float(ece), 'brier': brier,
            'y_true': y_true, 'y_prob': y_prob}

# ------------------------------------------------------------
# Calibration & conformal
# ------------------------------------------------------------
def collect_logits(net, loader, device):
    logits, labels = [], []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits.append(net(x, mc=False).float().cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)

def calibrate_temp(net, cal_loader, device):
    logits, labels = collect_logits(net, cal_loader, device)
    T = nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=30)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    for _ in range(15):
        opt.step(closure)
    return max(float(T.item()), 0.05)

def conformal_aps(cal_prob, cal_label, test_prob, alpha=0.1):
    """Adaptive Prediction Sets (per client) using exact conformal classification.
    Guarantees marginal coverage >= 1 - alpha on exchangeable test samples."""
    n = len(cal_label)
    scores = []
    for p, y in zip(cal_prob, cal_label):
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        idx = int(np.where(order == y)[0][0])
        scores.append(cum[idx])
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(scores, q_level, method='higher'))
    sets = []
    for p in test_prob:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        k = int(np.searchsorted(cum, qhat, side='left') + 1)
        k = min(max(k, 1), len(order))
        sets.append(set(order[:k].tolist()))
    return sets, qhat

def risk_coverage_curve(y_true, y_prob):
    """Sort by entropy ascending; accumulate accuracy vs coverage."""
    p = np.clip(y_prob, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    ent = -np.sum(p * np.log(p), axis=1)
    order = np.argsort(ent)
    y_true = y_true[order]; y_prob = y_prob[order]
    pred = y_prob.argmax(1)
    cum_correct = np.cumsum(pred == y_true)
    cov = (np.arange(1, len(order) + 1) / len(order))
    acc_at_cov = cum_correct / np.arange(1, len(order) + 1)
    out = {}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        k = int(np.searchsorted(cov, t, side='left')) - 1
        if k >= 0:
            out[f'acc_at_cov_{t:.2f}'] = float(acc_at_cov[k])
    auc_cov = float(np.trapezoid(acc_at_cov, cov))
    return {'acc_at_cov': {f'{t:.2f}': out.get(f'acc_at_cov_{t:.2f}', float('nan'))
                           for t in (0.5, 0.7, 0.8, 0.9, 0.95)},
            'acc_cov_auc': auc_cov, 'cov': cov.tolist(), 'acc_cov': acc_at_cov.tolist()}

# ------------------------------------------------------------
# LOCO (unseen-site generalization)
# ------------------------------------------------------------
def extract_embeddings(net, df, device):
    tf = m.eval_transforms()
    class DS(Dataset):
        def __init__(s):
            s.paths = df['path'].tolist(); s.y = df['gid'].tolist()
        def __len__(s): return len(s.paths)
        def __getitem__(s, i):
            img = torchvision.io.read_image(s.paths[i])
            if img.shape[0] == 1: img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3: img = img[:3]
            return tf(img), s.y[i]
    loader = DataLoader(DS(), batch_size=64, shuffle=False, num_workers=0)
    emb, ys = [], []
    net.eval()
    body = getattr(net, 'body', net)
    with torch.no_grad():
        for x, y in loader:
            v = body(x.to(device), mc=False).cpu().numpy()
            emb.append(v); ys.append(y.numpy())
    return np.concatenate(emb), np.concatenate(ys)

def run_loco(client_dfs, ncls_local, cw, seed, device, rounds=8):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef)
    rows = []
    for H in range(m.cfg.NUM_CLIENTS):
        src = [c for c in range(m.cfg.NUM_CLIENTS) if c != H]
        agg = agg_weights(client_dfs)
        src_loaders = {c: loaders_for(client_dfs[c]) for c in src}
        nets = run_strategy('fedbn', client_dfs, src, ncls_local, cw, src_loaders,
                            seed, rounds, agg, device)
        body = new_shared_body(seed)
        m.copy_shared(nets[src[0]].body, body, keep_bn=False)
        # linear probe on target site
        tr_df = client_dfs[H][client_dfs[H]['split'] == 'train']
        te_df = client_dfs[H][client_dfs[H]['split'] == 'test']
        emb, y = extract_embeddings(body, tr_df, device)
        clf = LogisticRegression(max_iter=800, C=1.0)
        clf.fit(emb, y)
        emb_te, y_te = extract_embeddings(body, te_df, device)
        pred = clf.predict(emb_te)
        rows.append({'seed': seed, 'held_out': H,
                     'held_out_name': m.client_meta(H)[3],
                     'acc': float(accuracy_score(y_te, pred)),
                     'f1': float(f1_score(y_te, pred, average='macro', zero_division=0)),
                     'mcc': float(matthews_corrcoef(y_te, pred))})
    return rows

def _softmax_from_logits(logits):
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)

def _to_probs(out):
    out = out.astype(np.float64, copy=False)
    return _softmax_from_logits(out)

def _eval_arrays(net, loader):
    net.eval(); y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            y_true.append(y.numpy())
            y_prob.append(net(x.to(DEV), mc=False).float().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    return y_true, _to_probs(y_prob)

# ------------------------------------------------------------
# Calibration + conformal + risk-coverage per client
# ------------------------------------------------------------
# METHODOLOGICAL LIMITATION NOTE:
# In the FedUA-Net training pipeline, the validation split (loaders[c]['val']) is used for
# early-stopping checkpoint selection during personalization fine-tuning, and then reused here
# as the calibration set for temperature scaling (calibrate_temp) and conformal APS (conformal_aps).
# Reusing data that guided model selection mildly violates the exchangeability assumption
# required for formal conformal coverage guarantees. A fully conservative setup would partition
# a distinct calibration fold separate from the model-selection validation set.
def run_calibration(name, nets, client_dfs, loaders, seed):
    """Per-client temperature scaling + APS conformal + risk-coverage.
    ECE is computed on both raw and temperature-scaled probabilities.
    Writes cal_{name}_seed{seed}.csv to RAW."""
    rows = []
    for c in client_dfs:
        net = nets[c]
        cal_loader = loaders[c]['val']; te_loader = loaders[c]['test']
        if cal_loader is None or te_loader is None:
            continue
        T = calibrate_temp(net, cal_loader, DEV)
        cal_logits, cal_y = collect_logits(net, cal_loader, DEV)
        te_logits,  te_y  = collect_logits(net, te_loader,  DEV)

        cal_p_raw = torch.softmax(cal_logits, dim=-1).numpy()
        te_p_raw  = torch.softmax(te_logits,  dim=-1).numpy()
        cal_p     = torch.softmax(cal_logits / T, dim=-1).numpy()
        te_p      = torch.softmax(te_logits  / T, dim=-1).numpy()

        cal_y = cal_y.numpy(); te_y = te_y.numpy()
        met_raw = full_metrics(te_y, te_p_raw)
        met     = full_metrics(te_y, te_p)
        rows.append({'strategy': name, 'seed': seed, 'client': c,
                     'client_name': m.client_meta(c)[3],
                     'metric': 'calib', 'temp': T,
                     'ece_raw':   met_raw['ece'],   'brier_raw': met_raw['brier'],
                     'ece_cal':   met['ece'],        'brier_cal': met['brier'],
                     'acc_uncal': met_raw['acc'],    'acc_cal':   met['acc']})
        # conformal APS
        for alpha in (0.05, 0.10, 0.20):
            sets, qhat = conformal_aps(cal_p, cal_y, te_p, alpha=alpha)
            cover = float(np.mean([te_y[i] in sets[i] for i in range(len(te_y))]))
            size  = float(np.mean([len(s) for s in sets]))
            rows.append({'strategy': name, 'seed': seed, 'client': c,
                         'client_name': m.client_meta(c)[3],
                         'metric': 'conformal', 'temp': T,
                         'alpha': alpha, 'coverage': cover,
                         'mean_set_size': size, 'qhat': qhat})
        # risk-coverage curve
        rc = risk_coverage_curve(te_y, te_p)
        rc_row = {'strategy': name, 'seed': seed, 'client': c,
                  'client_name': m.client_meta(c)[3],
                  'metric': 'risk_cov', 'acc_cov_auc': rc['acc_cov_auc']}
        for k, v in rc['acc_at_cov'].items():
            rc_row[f'acc_at_cov_{k}'] = v
        rows.append(rc_row)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / f'cal_{name}_seed{seed}.csv', index=False)
    return df

# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------
def main():
    print(f'PyTorch {torch.__version__} | device={DEV} | Attention={ARGS.attention}')
    if DEV == 'cuda':
        print(f'GPU={torch.cuda.get_device_name(0)}  free VRAM {torch.cuda.mem_get_info()[0] / 2**30:.1f}GB')

    all_rows = []
    for seed in ARGS.seeds:
        print(f'\n========== SEED {seed} ==========')
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        all_df, classes, class_to_idx, client_dfs = build_data(seed)
        cids = list(range(m.cfg.NUM_CLIENTS))
        loaders = {c: loaders_for(client_dfs[c], client_id=c) for c in cids}
        cw = {c: m.class_weights(client_dfs[c], hospital_b_subset_size=ARGS.hospital_b_subset_size) for c in cids}
        ncls_local = {c: int(client_dfs[c]['local'].max()) + 1 for c in cids}
        agg = agg_weights(client_dfs, weight_type=ARGS.agg_weight_type)
        print(f'agg_weights ({ARGS.agg_weight_type}):', {k: round(v, 3) for k, v in agg.items()})

        strategies = [s for s in ARGS.strategies if s != 'centralized']
        for name in strategies:
            out_csv = RAW / f'raw_{name}_seed{seed}.csv'
            if ARGS.resume and out_csv.exists():
                print(f'[{name}] seed={seed} already done -- skipping (--resume)')
                existing = pd.read_csv(out_csv)
                all_rows.extend(existing.to_dict('records'))
                continue
            t0 = time.time()
            print(f'[{name}] training ...')
            nets = run_strategy(name, client_dfs, cids, ncls_local, cw, loaders,
                                seed, ARGS.rounds, agg, DEV, attention=ARGS.attention)
            strat_rows = []
            for c in cids:
                ev = full_metrics(*_eval_arrays(nets[c], loaders[c]['test']))
                hosp_name = m.client_meta(c)[3]
                row = {'seed': seed, 'strategy': name, 'client': c,
                       'client_name': hosp_name,
                       'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'],
                       'auc': ev['auc'], 'ece': ev['ece'], 'brier': ev['brier'],
                       'precision': ev['precision'], 'recall': ev['recall']}
                all_rows.append(row)
                strat_rows.append(row)
                print(f'  [{hosp_name}] Acc={ev["acc"]*100:.2f}% | F1={ev["f1"]*100:.2f}% | '
                      f'MCC={ev["mcc"]:.3f} | ECE={ev["ece"]:.4f} | Brier={ev["brier"]:.4f}')
            print(f'  [{name}] done in {time.time() - t0:.1f}s')
            pd.DataFrame(strat_rows).to_csv(out_csv, index=False)
            if ARGS.save_final_models:
                ckpt_dir = OUT / 'final_models'
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                for c in cids:
                    torch.save(nets[c].state_dict(), ckpt_dir / f'{name}_seed{seed}_client{c}.pt')
            run_calibration(name, nets, client_dfs, loaders, seed)
            print(f'  [{name}] calibration/conformal done')

        # centralized (pooled, once per seed)
        if 'centralized' in ARGS.strategies:
            cen_csv = RAW / f'raw_centralized_seed{seed}.csv'
            if ARGS.resume and cen_csv.exists():
                print('[centralized] already done -- skipping (--resume)')
                existing = pd.read_csv(cen_csv)
                all_rows.extend(existing.to_dict('records'))
            else:
                t0 = time.time()
                print('[centralized] training on pooled data ...')
                res = run_centralized(client_dfs, classes, seed, DEV, attention=ARGS.attention)
                cen_rows = []
                for c in cids:
                    mask = res['cli'] == c
                    if mask.sum() == 0:
                        continue
                    ev = full_metrics(res['y_true'][mask], res['y_prob'][mask])
                    row = {'seed': seed, 'strategy': 'centralized', 'client': c,
                           'client_name': m.client_meta(c)[3],
                           'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'],
                           'auc': ev['auc'], 'ece': ev['ece'],
                           'brier': ev['brier'], 'precision': ev['precision'],
                           'recall': ev['recall']}
                    all_rows.append(row)
                    cen_rows.append(row)
                ev = full_metrics(res['y_true'], res['y_prob'])
                pooled_row = {'seed': seed, 'strategy': 'centralized',
                              'client': 'pooled', 'client_name': 'Pooled (11 cls)',
                              'acc': ev['acc'], 'f1': ev['f1'], 'mcc': ev['mcc'],
                              'auc': ev['auc'], 'ece': ev['ece'], 'brier': ev['brier'],
                              'precision': ev['precision'], 'recall': ev['recall']}
                all_rows.append(pooled_row)
                cen_rows.append(pooled_row)
                print(f'  [centralized] done in {time.time() - t0:.1f}s')
                pd.DataFrame(cen_rows).to_csv(cen_csv, index=False)

        # LOCO
        if ARGS.loco:
            t0 = time.time()
            print('[loco] leave-one-client-out ...')
            loco_rows = run_loco(client_dfs, ncls_local, cw, seed, DEV)
            pd.DataFrame(loco_rows).to_csv(RAW / f'loco_seed{seed}.csv', index=False)
            print(f'  [loco] done in {time.time() - t0:.1f}s')

    pd.DataFrame(all_rows).to_csv(RAW / 'raw_all.csv', index=False)
    print('\n[OK] raw results ->', RAW)

if __name__ == '__main__':
    main()
