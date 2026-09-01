# ==============================================================================
# FedUA-Net: Mechanistic Representation Analysis (t-SNE, UMAP & CKA)
# ==============================================================================
# Computes:
# 1. 512-D Latent Embeddings extraction across the 3 medical imaging modalities
# 2. 2D t-SNE and UMAP projections colored by Modality and Diagnostic Class
# 3. Layer-wise Centered Kernel Alignment (CKA) similarity matrix across clients
# ==============================================================================

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fedua_net as m

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------------------
# 1. CKA (Centered Kernel Alignment) Formulation
# ------------------------------------------------------------------------------
def centering_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between feature matrices X and Y.
    X: (N, d1)
    Y: (N, d2)
    """
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)
    
    # Frobenius norm squared of cross-covariance matrix
    hsic_xy = np.linalg.norm(np.dot(X.T, Y), 'fro') ** 2
    hsic_xx = np.linalg.norm(np.dot(X.T, X), 'fro') ** 2
    hsic_yy = np.linalg.norm(np.dot(Y.T, Y), 'fro') ** 2
    
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)

# ------------------------------------------------------------------------------
# 2. Checkpoint & Data Loader Helpers
# ------------------------------------------------------------------------------
def load_client_models(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEV)
    nets = {}
    for cid in range(3):
        ncls = len(m.client_meta(cid)[2])
        net = m.ClientNet(ncls, backbone=m.cfg.BACKBONE, emb=m.cfg.EMB, dropout=m.cfg.DROPOUT, attention='cbam')
        sd = ckpt['nets'][str(cid)]
        remapped_sd = {}
        for k, v in sd.items():
            k_new = k.replace('body.cbam.', 'body.attention.')
            remapped_sd[k_new] = v
        net.load_state_dict(remapped_sd)
        net.to(DEV)
        net.eval()
        nets[cid] = net
    return nets

def extract_features_and_metadata(nets, data_root, samples_per_class=100):
    m.cfg.DATA_ROOT = data_root
    m.ROOT = Path(data_root)
    m.DATASET_DIR = {
        'brain_tumor': m.ROOT / 'Brain Tumor MRI Dataset',
        'busi':        m.ROOT / 'Dataset_BUSI_with_GT',
        'covid':       m.ROOT / 'COVID-19_Radiography_Dataset',
    }
    
    all_df, classes, class_to_idx = m.discover_all(smoke=False)
    
    embeddings_list = []
    modalities_list = []
    class_labels_list = []
    modality_names = ['Brain MRI', 'Breast Ultrasound', 'Chest X-Ray']
    
    for cid in range(3):
        net = nets[cid]
        sub = all_df[(all_df['client'] == cid) & (all_df['split'] == 'test')].copy()
        
        # Subsample balanced items per class
        balanced_dfs = []
        for lbl, grp in sub.groupby('label'):
            n_take = min(len(grp), samples_per_class)
            balanced_dfs.append(grp.sample(n=n_take, random_state=42))
        sub_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        # Build local dataset and loader
        sub_balanced['local'] = sub_balanced['label'].map({c: i for i, c in enumerate(sorted(sub_balanced['label'].unique()))})
        ds = m.MedImgDataset(sub_balanced, m.eval_transforms())
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        cid_embeds = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEV)
                # Extract 512-D latent embedding from body before head
                emb = net.body(imgs, mc=False)
                cid_embeds.append(emb.cpu().numpy())
                
        cid_embeds = np.concatenate(cid_embeds, axis=0)
        embeddings_list.append(cid_embeds)
        modalities_list.extend([modality_names[cid]] * len(sub_balanced))
        class_labels_list.extend(sub_balanced['label'].tolist())
        
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings, modalities_list, class_labels_list

# ------------------------------------------------------------------------------
# 3. Layer-Wise CKA Similarity Extraction
# ------------------------------------------------------------------------------
class LayerHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        # Flatten spatial dimensions to (N, -1)
        if isinstance(output, torch.Tensor):
            out = output.detach()
            if out.dim() == 4:
                out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
            self.features = out.cpu().numpy()
    def close(self):
        self.hook.remove()

def compute_layerwise_cka(nets, data_root, num_probe_samples=64):
    m.cfg.DATA_ROOT = data_root
    all_df, _, _ = m.discover_all(smoke=False)
    probe_sub = all_df[all_df['split'] == 'test'].sample(n=num_probe_samples, random_state=42)
    probe_sub['local'] = 0
    ds = m.MedImgDataset(probe_sub, m.eval_transforms())
    loader = DataLoader(ds, batch_size=num_probe_samples, shuffle=False, num_workers=0)
    probe_batch, _ = next(iter(loader))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe_batch = probe_batch.to(device)
    
    # 6 representative layers from early to late
    layer_names = [
        'Stage 1 (Early Conv)',
        'Stage 3 (Mid-Early)',
        'Stage 5 (Mid-Deep)',
        'Stage 7 (Deep Conv)',
        'CBAM Attention',
        'Latent Proj (512-D)'
    ]
    
    client_layer_reps = {cid: [] for cid in range(3)}
    
    for cid in range(3):
        net = nets[cid]
        net.to(device)
        net.eval()
        hooks = [
            LayerHook(net.body.features[1]),
            LayerHook(net.body.features[3]),
            LayerHook(net.body.features[5]),
            LayerHook(net.body.features[7]),
            LayerHook(net.body.attention),
            LayerHook(net.body.prelu)
        ]
        
        with torch.no_grad():
            _ = net(probe_batch)
            
        for h in hooks:
            client_layer_reps[cid].append(h.features)
            h.close()
            
    num_layers = len(layer_names)
    cka_matrix = np.zeros((num_layers, num_layers))
    pairs = [(0, 1), (1, 2), (0, 2)]
    
    for l1 in range(num_layers):
        for l2 in range(num_layers):
            vals = []
            for c1, c2 in pairs:
                v = linear_cka(client_layer_reps[c1][l1], client_layer_reps[c2][l2])
                vals.append(v)
            cka_matrix[l1, l2] = np.mean(vals)
            
    return layer_names, cka_matrix

# ------------------------------------------------------------------------------
# 4. Publication Plot Generation
# ------------------------------------------------------------------------------
def plot_latent_spaces(embeddings, modalities, class_labels, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Computing t-SNE 2D projection...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
    emb_tsne = tsne.fit_transform(embeddings)
    
    print("[INFO] Computing UMAP 2D projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.3, random_state=42)
    emb_umap = reducer.fit_transform(embeddings)
    
    df_plot = pd.DataFrame({
        'tsne_1': emb_tsne[:, 0],
        'tsne_2': emb_tsne[:, 1],
        'umap_1': emb_umap[:, 0],
        'umap_2': emb_umap[:, 1],
        'Modality': modalities,
        'Class': [c.replace('bt_', '').replace('bu_', '').replace('cr_', '').capitalize() for c in class_labels]
    })
    
    # ----------------------------------------------------
    # Figure 6: Dual t-SNE & UMAP Latent Space Projections
    # ----------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    modality_colors = {'Brain MRI': '#1f77b4', 'Breast Ultrasound': '#2ca02c', 'Chest X-Ray': '#d62728'}
    
    # (a) t-SNE by Modality
    sns.scatterplot(
        data=df_plot, x='tsne_1', y='tsne_2', hue='Modality', palette=modality_colors,
        alpha=0.85, s=45, edgecolor='none', ax=axes[0, 0]
    )
    axes[0, 0].set_title('(a) t-SNE: Clustered by Imaging Modality', fontsize=13, fontweight='bold', pad=8)
    axes[0, 0].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[0, 0].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[0, 0].legend(frameon=True, loc='best', fontsize=10)
    
    # (b) UMAP by Modality
    sns.scatterplot(
        data=df_plot, x='umap_1', y='umap_2', hue='Modality', palette=modality_colors,
        alpha=0.85, s=45, edgecolor='none', ax=axes[0, 1]
    )
    axes[0, 1].set_title('(b) UMAP: Manifold Geometry across Modalities', fontsize=13, fontweight='bold', pad=8)
    axes[0, 1].set_xlabel('UMAP Dimension 1', fontsize=11)
    axes[0, 1].set_ylabel('UMAP Dimension 2', fontsize=11)
    axes[0, 1].legend(frameon=True, loc='best', fontsize=10)
    
    # (c) t-SNE by Diagnostic Class
    sns.scatterplot(
        data=df_plot, x='tsne_1', y='tsne_2', hue='Class', palette='tab20',
        alpha=0.85, s=40, edgecolor='none', ax=axes[1, 0]
    )
    axes[1, 0].set_title('(c) t-SNE: Disjoint Diagnostic Sub-Classes', fontsize=13, fontweight='bold', pad=8)
    axes[1, 0].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1, 0].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1, 0].legend(frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, borderaxespad=0.)
    
    # (d) UMAP by Diagnostic Class
    sns.scatterplot(
        data=df_plot, x='umap_1', y='umap_2', hue='Class', palette='tab20',
        alpha=0.85, s=40, edgecolor='none', ax=axes[1, 1]
    )
    axes[1, 1].set_title('(d) UMAP: Fine-Grained Pathological Clustering', fontsize=13, fontweight='bold', pad=8)
    axes[1, 1].set_xlabel('UMAP Dimension 1', fontsize=11)
    axes[1, 1].set_ylabel('UMAP Dimension 2', fontsize=11)
    axes[1, 1].legend(frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, borderaxespad=0.)
    
    plt.tight_layout()
    fig_path = out_dir / 'fig6_tsne_umap_latent_space.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Latent space figure saved to: {fig_path}")

def plot_cka_heatmap(layer_names, cka_matrix, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    
    sns.heatmap(
        cka_matrix, annot=True, fmt=".2f", cmap="magma",
        xticklabels=layer_names, yticklabels=layer_names,
        cbar_kws={'label': 'Linear CKA Similarity'},
        vmin=0.2, vmax=1.0, ax=ax, square=True, linewidths=0.8, linecolor='white'
    )
    
    ax.set_title('Cross-Modality Layer-Wise CKA Similarity Matrix\n(Early Shared Invariants vs. Deep Modality Specialization)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Client Model Depth (Layer $l_j$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Client Model Depth (Layer $l_i$)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    fig_path = out_dir / 'fig7_cka_similarity_heatmap.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] CKA heatmap saved to: {fig_path}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FedUA-Net Mechanistic Interpretability Suite")
    parser.add_argument('--data_root', default='./Dataset', help='Path to Dataset root directory')
    parser.add_argument('--checkpoint', default='./outputs_final/models/fedua_net_final_finetuned.pt',
                        help='Path to fine-tuned federated checkpoint')
    parser.add_argument('--out_dir', default='./paper_figures', help='Output directory for figures')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FedUA-Net: Mechanistic Representation & CKA Analysis")
    print("=" * 70)
    
    print(f"[1/4] Loading fine-tuned client networks from: {args.checkpoint}")
    nets = load_client_models(args.checkpoint)
    
    print(f"[2/4] Extracting 512-D latent embeddings across 3 modalities...")
    embeddings, modalities, class_labels = extract_features_and_metadata(nets, args.data_root)
    print(f"      Extracted {len(embeddings)} latent feature vectors of shape {embeddings.shape[1]}-D.")
    
    print(f"[3/4] Computing t-SNE and UMAP visualizations...")
    plot_latent_spaces(embeddings, modalities, class_labels, args.out_dir)
    
    print(f"[4/4] Computing Layer-Wise CKA Similarity across client models...")
    layer_names, cka_matrix = compute_layerwise_cka(nets, args.data_root)
    plot_cka_heatmap(layer_names, cka_matrix, args.out_dir)
    
    print("=" * 70)
    print("[SUCCESS] All mechanistic interpretability analyses and figures complete.")
    print("=" * 70)

if __name__ == '__main__':
    main()
