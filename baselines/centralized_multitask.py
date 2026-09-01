# ==============================================================================
# FedUA-Net: Fair Centralized Multi-Head Baseline
# ==============================================================================
# Multi-Task Centralized training:
# - Single shared backbone (EfficientNetV2-S + CBAM)
# - 3 separate task-specific classification heads (4-class MRI, 3-class US, 4-class X-Ray)
# - Eliminates cross-task label interference while pooling all clinical training data.
# ==============================================================================

import os
import sys
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fedua_net as m

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class CentralizedMultiTaskNet(nn.Module):
    def __init__(self, ncls_dict, backbone='efficientnet_v2_s', emb=512, dropout=0.3, attention='cbam'):
        super().__init__()
        self.body = m.SharedBody(backbone=backbone, emb=emb, dropout=dropout, attention=attention)
        self.heads = nn.ModuleDict({
            str(cid): m.LocalHead(emb, ncls) for cid, ncls in ncls_dict.items()
        })
        
    def forward(self, x, cid, mc=False):
        feat = self.body(x, mc=mc)
        return self.heads[str(cid)](feat)

# ------------------------------------------------------------------------------
# Metrics Helpers
# ------------------------------------------------------------------------------
def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return ece

def fit_temperature(model, val_loader, cid):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEV)
            logits = model(x, cid, mc=False)
            all_logits.append(logits.cpu())
            all_labels.append(y)
            
    if not all_logits:
        return 1.0
        
    logits = torch.cat(all_logits, dim=0).to(DEV)
    labels = torch.cat(all_labels, dim=0).to(DEV)
    
    T = nn.Parameter(torch.tensor(1.0, device=DEV))
    optimizer = torch.optim.LBFGS([T], lr=0.05, max_iter=30)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
        
    for _ in range(15):
        optimizer.step(eval_loss)
        
    return max(float(T.item()), 0.05)

# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
def evaluate_client(model, test_loader, val_loader, cid):
    T_opt = fit_temperature(model, val_loader, cid)
    model.eval()
    all_raw_probs, all_cal_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEV)
            logits = model(x, cid, mc=False)
            raw_p = F.softmax(logits, dim=1).cpu().numpy()
            cal_p = F.softmax(logits / T_opt, dim=1).cpu().numpy()
            
            all_raw_probs.append(raw_p)
            all_cal_probs.append(cal_p)
            all_labels.append(y.numpy())
            
    all_raw_probs = np.concatenate(all_raw_probs, axis=0)
    all_cal_probs = np.concatenate(all_cal_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    preds = np.argmax(all_raw_probs, axis=1)
    acc = accuracy_score(all_labels, preds) * 100.0
    f1 = f1_score(all_labels, preds, average='macro') * 100.0
    mcc = matthews_corrcoef(all_labels, preds)
    raw_ece = compute_ece(all_raw_probs, all_labels)
    cal_ece = compute_ece(all_cal_probs, all_labels)
    
    return {
        'acc': acc,
        'f1': f1,
        'mcc': mcc,
        'raw_ece': raw_ece,
        'cal_ece': cal_ece,
        'temp': T_opt,
        'n_samples': len(all_labels)
    }

# ------------------------------------------------------------------------------
# Multi-Task Centralized Training
# ------------------------------------------------------------------------------
def train_centralized_multitask(seed, data_root, epochs=12, batch_size=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    m.SEED = seed
    m.cfg.DATA_ROOT = data_root
    m.ROOT = Path(data_root)
    m.DATASET_DIR = {
        'brain_tumor': m.ROOT / 'Brain Tumor MRI Dataset',
        'busi':        m.ROOT / 'Dataset_BUSI_with_GT',
        'covid':       m.ROOT / 'COVID-19_Radiography_Dataset',
    }
    
    all_df, classes, class_to_idx = m.discover_all(smoke=False)
    client_dfs = {}
    ncls_dict = {}
    loaders = {}
    
    for cid in range(3):
        sub = all_df[all_df['client'] == cid].copy()
        uniq = sorted(sub['label'].unique())
        sub['local'] = sub['label'].map({c: i for i, c in enumerate(uniq)})
        client_dfs[cid] = sub
        ncls_dict[cid] = len(uniq)
        loaders[cid] = m.build_loaders(sub, batch_size, 0)
        
    model = CentralizedMultiTaskNet(ncls_dict, backbone=m.cfg.BACKBONE, emb=m.cfg.EMB,
                                    dropout=m.cfg.DROPOUT, attention='cbam').to(DEV)
    
    # Optimizer with differential learning rate
    params = [
        {'params': model.body.features.parameters(), 'lr': m.cfg.BACKBONE_LR},
        {'params': model.body.attention.parameters(), 'lr': m.cfg.BASE_LR},
        {'params': model.body.fc.parameters(), 'lr': m.cfg.BASE_LR},
        {'params': model.body.prelu.parameters(), 'lr': m.cfg.BASE_LR},
        {'params': model.heads.parameters(), 'lr': m.cfg.BASE_LR},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=m.cfg.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=m.cfg.MIN_LR)
    
    # Class weights per client
    cw = {c: m.class_weights(client_dfs[c]) for c in range(3)}
    cw_tensors = {c: torch.tensor([cw[c][i] for i in range(len(cw[c]))], dtype=torch.float, device=DEV) for c in range(3)}
    
    print(f"\n--- Training Fair Centralized Multi-Head Model (Seed {seed}) for {epochs} Epochs ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_iterators = {c: iter(loaders[c]['train']) for c in range(3)}
        steps_per_epoch = max(len(loaders[c]['train']) for c in range(3))
        
        running_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            step_loss = 0.0
            
            for cid in range(3):
                try:
                    imgs, labels = next(train_iterators[cid])
                except StopIteration:
                    train_iterators[cid] = iter(loaders[cid]['train'])
                    imgs, labels = next(train_iterators[cid])
                    
                imgs = imgs.to(DEV)
                labels = labels.to(DEV)
                
                logits = model(imgs, cid, mc=False)
                loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=m.cfg.LABEL_SMOOTHING)
                w = cw_tensors[cid]
                loss = (loss * w[labels]).mean()
                
                step_loss = step_loss + loss / 3.0
                
            step_loss.backward()
            optimizer.step()
            running_loss += step_loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs:02d} | Multi-Task Train Loss: {running_loss / steps_per_epoch:.4f}")
        
    # Evaluate across the 3 client test sets
    results = []
    print("\n--- Evaluation on Test Sets ---")
    for cid in range(3):
        site_name = m.client_meta(cid)[3]
        metrics = evaluate_client(model, loaders[cid]['test'], loaders[cid]['val'], cid)
        metrics['client'] = cid
        metrics['site_name'] = site_name
        metrics['seed'] = seed
        results.append(metrics)
        print(f"[{site_name}] Acc: {metrics['acc']:.2f}% | Macro F1: {metrics['f1']:.2f}% | MCC: {metrics['mcc']:.3f} | ECE: {metrics['raw_ece']:.4f} -> {metrics['cal_ece']:.4f}")
        
    return results

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fair Multi-Head Centralized Baseline")
    parser.add_argument('--data_root', default='./Dataset', help='Path to dataset root')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Random seeds')
    parser.add_argument('--epochs', type=int, default=12, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--out_dir', default='./outputs_experiments/reports', help='Output directory for metrics')
    args = parser.parse_args()
    
    all_runs = []
    for seed in args.seeds:
        seed_results = train_centralized_multitask(seed, args.data_root, epochs=args.epochs, batch_size=args.batch_size)
        all_runs.extend(seed_results)
        
    df_results = pd.DataFrame(all_runs)
    
    print("\n" + "=" * 70)
    print("FAIR CENTRALIZED (MULTI-HEAD) 3-SEED BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Per-client cross-seed aggregation
    for cid in range(3):
        sub = df_results[df_results['client'] == cid]
        site_name = sub['site_name'].iloc[0]
        print(f"\n{site_name}:")
        print(f"  Accuracy:       {sub['acc'].mean():.2f}% ± {sub['acc'].std():.2f}%")
        print(f"  Macro F1:       {sub['f1'].mean():.2f}% ± {sub['f1'].std():.2f}%")
        print(f"  MCC:            {sub['mcc'].mean():.3f} ± {sub['mcc'].std():.3f}")
        print(f"  Raw ECE:        {sub['raw_ece'].mean():.4f} ± {sub['raw_ece'].std():.4f}")
        print(f"  Calibrated ECE: {sub['cal_ece'].mean():.4f} ± {sub['cal_ece'].std():.4f}")
        
    # Pooled cross-seed aggregation
    pooled_acc = df_results.groupby('seed').apply(lambda g: (g['acc'] * g['n_samples']).sum() / g['n_samples'].sum())
    pooled_f1 = df_results.groupby('seed')['f1'].mean()
    pooled_mcc = df_results.groupby('seed')['mcc'].mean()
    pooled_raw_ece = df_results.groupby('seed')['raw_ece'].mean()
    pooled_cal_ece = df_results.groupby('seed')['cal_ece'].mean()
    
    print("\n" + "-" * 70)
    print("POOLED MULTI-TASK BENCHMARK (Mean ± Std across 3 seeds):")
    print(f"  Multi-Task Accuracy: {pooled_acc.mean():.2f}% ± {pooled_acc.std():.2f}%")
    print(f"  Macro F1:            {pooled_f1.mean():.2f}% ± {pooled_f1.std():.2f}%")
    print(f"  MCC:                 {pooled_mcc.mean():.3f} ± {pooled_mcc.std():.3f}")
    print(f"  Raw ECE:             {pooled_raw_ece.mean():.4f} ± {pooled_raw_ece.std():.4f}")
    print(f"  Calibrated ECE:      {pooled_cal_ece.mean():.4f} ± {pooled_cal_ece.std():.4f}")
    print("=" * 70)
    
    out_path = Path(args.out_dir) / 'centralized_multihead_benchmark.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved raw multi-head results to: {out_path}")

if __name__ == '__main__':
    main()
