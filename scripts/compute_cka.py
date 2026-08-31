import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fedua_net as m
import experiment as exp

def setup_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

def linear_cka(X, Y):
    """X: (n_samples, d1), Y: (n_samples, d2), both mean-centered internally."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    return float(hsic / (norm_x * norm_y + 1e-12))

class FeatureExtractor:
    def __init__(self, net):
        self.net = net
        self.activations = {}
        self.hooks = []
        
        # Hook selected layers
        self.hooks.append(net.body.features[1].register_forward_hook(self._get_hook('features[1]')))
        self.hooks.append(net.body.features[3].register_forward_hook(self._get_hook('features[3]')))
        self.hooks.append(net.body.features[5].register_forward_hook(self._get_hook('features[5]')))
        self.hooks.append(net.body.attention.register_forward_hook(self._get_hook('attention')))
        self.hooks.append(net.body.fc.register_forward_hook(self._get_hook('fc')))

    def _get_hook(self, name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                out = output.detach().cpu()
                if out.ndim == 4:
                    out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
                elif out.ndim > 2:
                    out = out.flatten(1)
                self.activations[name] = out
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

def collect_layer_activations(net, data_loader, device, max_samples=200):
    extractor = FeatureExtractor(net)
    net.to(device).eval()
    
    layer_acts = {'features[1]': [], 'features[3]': [], 'features[5]': [], 'attention': [], 'fc': []}
    n_collected = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _ = net(x, mc=False)
            for k in layer_acts:
                layer_acts[k].append(extractor.activations[k])
            n_collected += x.size(0)
            if n_collected >= max_samples:
                break
                
    extractor.remove()
    return {k: torch.cat(v, dim=0)[:max_samples].numpy() for k, v in layer_acts.items()}

def evaluate_cka_for_dir(models_dir, probe_loader, device, seed=0, strategy='fedua'):
    models_path = Path(models_dir)
    print(f"\n[CKA] Evaluating checkpoints in: {models_path}")
    
    # Client class counts: A (Brain MRI: 4), B (US: 3), C (COVID X-Ray: 4)
    ncls_map = {0: 4, 1: 3, 2: 4}
    client_acts = {}
    
    for c in (0, 1, 2):
        ckpt_file = models_path / f'{strategy}_seed{seed}_client{c}.pt'
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        
        net = m.ClientNet(num_classes=ncls_map[c], backbone=m.cfg.BACKBONE, emb=m.cfg.EMB, dropout=m.cfg.DROPOUT)
        net.load_state_dict(torch.load(ckpt_file, map_location=device))
        acts = collect_layer_activations(net, probe_loader, device, max_samples=200)
        client_acts[c] = acts
        print(f"  Client {c} activations extracted from {ckpt_file.name}")

    layers = ['features[1]', 'features[3]', 'features[5]', 'attention', 'fc']
    layer_labels = ['Early (features[1])', 'Mid (features[3])', 'Mid-Late (features[5])', 'Dual CBAM (attention)', 'Projection (fc)']
    
    results = {}
    for l_key, l_name in zip(layers, layer_labels):
        cka_ab = linear_cka(client_acts[0][l_key], client_acts[1][l_key])
        cka_ac = linear_cka(client_acts[0][l_key], client_acts[2][l_key])
        cka_bc = linear_cka(client_acts[1][l_key], client_acts[2][l_key])
        mean_cka = (cka_ab + cka_ac + cka_bc) / 3.0
        results[l_name] = {
            'layer_key': l_key,
            'cka_A_B': cka_ab,
            'cka_A_C': cka_ac,
            'cka_B_C': cka_bc,
            'mean_cka': mean_cka
        }
        print(f"  Layer {l_name:24s} -> A-B: {cka_ab:.4f} | A-C: {cka_ac:.4f} | B-C: {cka_bc:.4f} | Mean: {mean_cka:.4f}")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_models', default='./outputs_checkpoints_uniform_baseline/final_models')
    parser.add_argument('--personalized_models', default='./outputs_checkpoints_personalized/final_models')
    parser.add_argument('--output_dir', default='./results')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build validation probe data loader across clients
    print("[CKA] Building validation probe dataset...")
    all_df, classes, class_to_idx, client_dfs = exp.build_data(seed=0)
    val_subsets = []
    for c in (0, 1, 2):
        v = client_dfs[c][client_dfs[c]['split'] == 'val'].head(70)
        val_subsets.append(v)
    probe_df = pd.concat(val_subsets, ignore_index=True)
    probe_ds = m.MedImgDataset(probe_df, m.eval_transforms())
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=32, shuffle=False)
    print(f"[OK] Probe dataset: {len(probe_ds)} validation images.")
    
    # Compute CKA for baseline & personalized
    baseline_res = evaluate_cka_for_dir(args.baseline_models, probe_loader, device)
    personalized_res = evaluate_cka_for_dir(args.personalized_models, probe_loader, device)
    
    # Build comparison dataframe
    rows = []
    for k in baseline_res:
        b_mean = baseline_res[k]['mean_cka']
        p_mean = personalized_res[k]['mean_cka']
        delta = p_mean - b_mean
        rows.append({
            'layer': k,
            'cka_uniform_baseline': b_mean,
            'cka_personalize_deep': p_mean,
            'delta': delta,
            'baseline_A_B': baseline_res[k]['cka_A_B'],
            'baseline_A_C': baseline_res[k]['cka_A_C'],
            'baseline_B_C': baseline_res[k]['cka_B_C'],
            'pers_A_B': personalized_res[k]['cka_A_B'],
            'pers_A_C': personalized_res[k]['cka_A_C'],
            'pers_B_C': personalized_res[k]['cka_B_C'],
        })
        
    df_res = pd.DataFrame(rows)
    csv_path = out_dir / 'cka_before_after.csv'
    df_res.to_csv(csv_path, index=False)
    print(f"\n[OK] CKA comparison saved to: {csv_path}")
    print("\n" + "=" * 75)
    print("                     CKA REPRESENTATION SIMILARITY COMPARISON")
    print("=" * 75)
    print(df_res[['layer', 'cka_uniform_baseline', 'cka_personalize_deep', 'delta']].to_string(index=False))
    
    # Render Two-Panel Figure (Fig 7b)
    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    
    layers = [r['layer'] for r in rows]
    pairs = ['Hospital A - B', 'Hospital A - C', 'Hospital B - C']
    
    # Matrix for baseline
    b_mat = np.array([
        [r['baseline_A_B'], r['baseline_A_C'], r['baseline_B_C']] for r in rows
    ])
    # Matrix for personalized
    p_mat = np.array([
        [r['pers_A_B'], r['pers_A_C'], r['pers_B_C']] for r in rows
    ])
    
    vmin = min(b_mat.min(), p_mat.min(), 0.3)
    vmax = 1.0
    cmap = 'magma'
    
    im0 = axes[0].imshow(b_mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('(a) Uniform Aggregation (Baseline)', fontweight='bold', pad=10)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(pairs, rotation=15, ha='right')
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    axes[0].grid(False)
    
    # Annotate values
    for i in range(len(layers)):
        for j in range(3):
            val = b_mat[i, j]
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center',
                         color='white' if val < 0.75 else 'black', fontweight='bold', fontsize=9.5)
            
    im1 = axes[1].imshow(p_mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('(b) CKA-Guided Depth-Adaptive (Ours)', fontweight='bold', pad=10)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(pairs, rotation=15, ha='right')
    axes[1].grid(False)
    
    for i in range(len(layers)):
        for j in range(3):
            val = p_mat[i, j]
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center',
                         color='white' if val < 0.75 else 'black', fontweight='bold', fontsize=9.5)
            
    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Linear CKA Similarity Index', fontweight='bold')
    
    fig_path = fig_dir / 'fig7b_cka_before_after.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure saved to: {fig_path}")

if __name__ == '__main__':
    main()
