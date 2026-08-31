import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fedua_net as m
import experiment as exp

def setup_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11.5,
        'axes.titleweight': 'bold',
        'legend.fontsize': 9.5,
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./outputs_checkpoints_personalized/final_models/fedua_seed0_client1.pt')
    parser.add_argument('--output_dir', default='./results/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[GALLERY] Loading data and model for Hospital B (Client 1)...")
    all_df, classes, class_to_idx, client_dfs = exp.build_data(seed=0)
    b_df = client_dfs[1]
    
    # Client 1 labels
    ncls = int(b_df['local'].max()) + 1
    loaders = exp.loaders_for(b_df, client_id=1)
    
    # Determine class names for Hospital B
    local_to_global = {b_df[b_df['local'] == i]['gid'].iloc[0]: i for i in range(ncls)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[g_idx].replace('busi_', '') for g_idx in sorted(local_to_global.keys())]
    print(f"[OK] Hospital B local classes ({ncls}): {class_names}")
    
    # Load model
    model_file = Path(args.model_path)
    if not model_file.exists():
        # Fallback to alternate checkpoint location if needed
        alt_file = Path('./outputs_checkpoints_uniform_baseline/final_models/fedua_seed0_client1.pt')
        if alt_file.exists():
            model_file = alt_file
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {model_file}")
            
    print(f"[GALLERY] Using checkpoint: {model_file}")
    net = m.ClientNet(num_classes=ncls, backbone=m.cfg.BACKBONE, emb=m.cfg.EMB, dropout=m.cfg.DROPOUT)
    net.load_state_dict(torch.load(model_file, map_location=device))
    net.to(device).eval()
    
    # Temperature scaling & Conformal evaluation
    cal_loader = loaders['val']
    te_loader = loaders['test']
    
    T = exp.calibrate_temp(net, cal_loader, device)
    cal_logits, cal_y = exp.collect_logits(net, cal_loader, device)
    te_logits, te_y = exp.collect_logits(net, te_loader, device)
    
    cal_p = torch.softmax(cal_logits / T, dim=-1).numpy()
    te_p = torch.softmax(te_logits / T, dim=-1).numpy()
    cal_y, te_y = cal_y.numpy(), te_y.numpy()
    
    # Conformal APS at alpha=0.10 (90% target coverage)
    sets, qhat = exp.conformal_aps(cal_p, cal_y, te_p, alpha=0.10)
    pred = te_p.argmax(1)
    
    # Get test dataframe paths in exact iteration order
    test_sub = b_df[b_df['split'] == 'test'].reset_index(drop=True)
    
    misclassified_idx = np.where(pred != te_y)[0]
    print(f"[GALLERY] Total test samples: {len(te_y)} | Total misclassified: {len(misclassified_idx)}")
    
    results_list = []
    for idx in misclassified_idx:
        true_c = class_names[te_y[idx]]
        pred_c = class_names[pred[idx]]
        conf_set = [class_names[j] for j in sets[idx]]
        covered = te_y[idx] in sets[idx]
        prob = te_p[idx]
        img_path = test_sub.iloc[idx]['path']
        results_list.append({
            'idx': idx,
            'path': img_path,
            'true_cls': true_c,
            'pred_cls': pred_c,
            'conf_set': conf_set,
            'covered': covered,
            'top_prob': float(prob.max()),
            'probs': prob
        })
        
    print("\n" + "=" * 80)
    print("                     HOSPITAL B FAILURE CASES & CONFORMAL SETS")
    print("=" * 80)
    for r in results_list[:10]:
        print(f"Sample {r['idx']:03d} | True: {r['true_cls']:10s} | Pred: {r['pred_cls']:10s} | TopProb: {r['top_prob']:.3f} | Set: {str(r['conf_set']):25s} | Covered: {r['covered']}")

    # Select 4 representative cases for the figure:
    # 2-3 where conformal set correctly widened to cover ground truth, 1 where set failed
    covered_cases = [r for r in results_list if r['covered']]
    uncovered_cases = [r for r in results_list if not r['covered']]
    
    chosen = []
    if covered_cases:
        chosen.extend(covered_cases[:3])
    if uncovered_cases:
        chosen.extend(uncovered_cases[:1])
    while len(chosen) < 4 and len(results_list) > len(chosen):
        for r in results_list:
            if r not in chosen:
                chosen.append(r)
                if len(chosen) == 4:
                    break
                    
    # Render Figure
    setup_academic_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    
    for i, r in enumerate(chosen[:4]):
        ax = axes[i]
        try:
            img = Image.open(r['path']).convert('RGB')
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center')
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        status_color = '#2ca02c' if r['covered'] else '#d62728'
        status_text = '✓ Certified Set Covers Ground Truth' if r['covered'] else '✗ Out-of-Coverage Uncertainty'
        
        title_text = (
            f"Case #{r['idx'] + 1}\n"
            f"Ground Truth: {r['true_cls'].capitalize()}\n"
            f"Argmax Pred: {r['pred_cls'].capitalize()} ({r['top_prob']*100:.1f}%)\n"
            f"Conformal Set (α=0.10):\n"
            f"{{{', '.join([c.capitalize() for c in r['conf_set']])}}}"
        )
        ax.set_title(title_text, fontsize=10, pad=8, fontweight='bold')
        
        # Add badge box below image
        ax.text(0.5, -0.08, status_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9.5, fontweight='bold', color=status_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor=status_color, lw=1.5))
        
    plt.suptitle("Clinical Utility of Conformal Prediction Sets on Challenging Ultrasound Cases (Hospital B)",
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    fig_path = out_dir / 'fig8_failure_gallery.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Failure case gallery figure saved to: {fig_path}")

if __name__ == '__main__':
    main()
