# ==============================================================================
# FedUA-Net: Publication-Grade Exact Corrected Architecture Figure Generator (v2)
# Fine-tuned spacing, generous margins, crisp typography, perfect alignment
# ==============================================================================

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def generate_medical_textures():
    # 1. Brain MRI Texture (axial slice synthetic pattern)
    mri = np.zeros((100, 100))
    y, x = np.mgrid[:100, :100]
    mask_skull = ((x - 50)**2 / 38**2 + (y - 50)**2 / 44**2) <= 1
    mask_brain = ((x - 50)**2 / 34**2 + (y - 50)**2 / 40**2) <= 1
    mask_ventricle1 = ((x - 44)**2 / 6**2 + (y - 48)**2 / 14**2) <= 1
    mask_ventricle2 = ((x - 56)**2 / 6**2 + (y - 48)**2 / 14**2) <= 1
    mask_tumor = ((x - 62)**2 / 10**2 + (y - 38)**2 / 10**2) <= 1
    
    mri[mask_skull] = 0.45
    mri[mask_brain] = 0.70 + 0.08 * np.sin(x[mask_brain]*0.5) * np.cos(y[mask_brain]*0.5)
    mri[mask_ventricle1] = 0.15
    mri[mask_ventricle2] = 0.15
    mri[mask_tumor] = 0.95

    # 2. Breast Ultrasound Texture (acoustic speckle + hypoechoic lesion)
    np.random.seed(42)
    us = 0.35 + 0.25 * np.random.rand(100, 100)
    sector_mask = (y >= 15) & (np.abs(x - 50) <= (y - 10) * 0.55)
    us[~sector_mask] = 0.05
    lesion_mask = ((x - 48)**2 / 16**2 + (y - 52)**2 / 11**2) <= 1
    shadow_mask = (x >= 36) & (x <= 60) & (y >= 58) & sector_mask
    us[shadow_mask] *= 0.4
    us[lesion_mask] = 0.15 + 0.08 * np.random.rand(np.sum(lesion_mask))

    # 3. Chest X-Ray Texture (bilateral lung fields + cardiac silhouette)
    cxr = np.ones((100, 100)) * 0.8
    lung_left = ((x - 32)**2 / 14**2 + (y - 48)**2 / 28**2) <= 1
    lung_right = ((x - 68)**2 / 14**2 + (y - 48)**2 / 28**2) <= 1
    heart = ((x - 53)**2 / 16**2 + (y - 60)**2 / 18**2) <= 1
    cxr[lung_left] = 0.22 + 0.05 * np.sin(y[lung_left]*0.8)
    cxr[lung_right] = 0.22 + 0.05 * np.sin(y[lung_right]*0.8)
    cxr[heart] = 0.75
    med = (x >= 45) & (x <= 55) & (y <= 75)
    cxr[med] = 0.65
    
    return mri, us, cxr

def build_figure1(output_jpg, output_png=None):
    fig = plt.figure(figsize=(19, 10.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 105)
    ax.axis('off')
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Color Palette matching the clean publication style
    c_card_border = '#0F172A'   # Deep crisp border
    c_server_bg = '#F8FAFC'     # Server box background
    c_purple_bg = '#FAF5FF'     # Calibration pipeline background
    c_purple_border = '#6B21A8' # Calibration border
    
    # Layer block colors
    c_conv_fill = '#E0F2FE'     # Soft blue
    c_conv_border = '#0284C7'
    c_cbam_fill = '#DCFCE7'     # Soft green
    c_cbam_border = '#16A34A'
    c_gap_fill = '#F1F5F9'      # Soft gray
    c_gap_border = '#64748B'
    c_prelu_fill = '#ECFDF5'    # Mint green
    c_prelu_border = '#059669'
    c_drop_fill = '#FFEDD5'     # Orange / Amber
    c_drop_border = '#EA580C'
    c_head_fill = '#FFE4E6'     # Crimson / Pink
    c_head_border = '#E11D48'

    def draw_round_box(x, y, w, h, bg, border, lw=1.5, radius=1.2, z=1):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=bg, edgecolor=border, linewidth=lw, zorder=z)
        ax.add_patch(p)
        return p

    def draw_arrow(x1, y1, x2, y2, color='#0F172A', lw=1.5, style='-|>', z=5):
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=11,
                              color=color, linewidth=lw, zorder=z)
        ax.add_patch(arr)
        return arr

    # ==========================================================================
    # 1. TOP TIER: CENTRAL SERVER
    # ==========================================================================
    draw_round_box(58, 75, 74, 25, c_server_bg, c_card_border, lw=1.8, radius=1.4, z=2)
    
    # Server Header Banner
    ax.text(95, 96.0, "Central Server", fontsize=14.0, fontweight='bold', color='#0F172A', ha='center', va='center', zorder=3)
    
    # Divider line
    ax.plot([58, 132], [92.5, 92.5], color=c_card_border, linewidth=1.5, zorder=3)
    
    # Server Contents
    ax.text(95, 89.0, r"Global Shared Backbone Weights $\theta_{\mathbf{conv}}$", fontsize=11.5, fontweight='bold', color='#1E293B', ha='center', zorder=3)
    ax.text(95, 85.2, "Uniform Server Aggregation Engine", fontsize=10.0, fontweight='bold', color='#B91C1C', ha='center', zorder=3)
    
    # Mathematical Formula
    ax.text(95, 80.2, r"$\theta_{\mathbf{conv}}^{(t+1)} = \frac{1}{K} \sum_{k=1}^K \theta_{\mathbf{conv}, k}^{(t+1)} \quad \left(w_k = \frac{1}{3}\right)$",
            fontsize=11.5, fontweight='bold', color='#0F172A', ha='center', va='center', zorder=3)
    
    ax.text(95, 76.5, "Egalitarian fairness constraint • Prevents sample-size domination across modalities",
            fontsize=8.5, color='#64748B', ha='center', zorder=3)

    # ==========================================================================
    # 2. MIDDLE TIER: 3 HOSPITAL CLIENT CARDS
    # ==========================================================================
    mri_tex, us_tex, cxr_tex = generate_medical_textures()
    
    hospitals = [
        {
            'name': 'Hospital A (Neurology)',
            'modality': 'Brain MRI',
            'classes': '4 classes',
            'x': 4, 'y': 24, 'w': 58, 'h': 43,
            'tex': mri_tex,
            'color': '#0284C7',
            'tag_bg': '#E0F2FE',
            'client_id': 'A'
        },
        {
            'name': 'Hospital B (Oncology)',
            'modality': 'Breast Ultrasound',
            'classes': '3 classes',
            'x': 66, 'y': 24, 'w': 58, 'h': 43,
            'tex': us_tex,
            'color': '#D97706',
            'tag_bg': '#FEF3C7',
            'client_id': 'B'
        },
        {
            'name': 'Hospital C (Pulmonology)',
            'modality': 'Chest X-Ray',
            'classes': '4 classes',
            'x': 128, 'y': 24, 'w': 58, 'h': 43,
            'tex': cxr_tex,
            'color': '#059669',
            'tag_bg': '#D1FAE5',
            'client_id': 'C'
        }
    ]

    for h in hospitals:
        hx, hy, hw, hh = h['x'], h['y'], h['w'], h['h']
        
        # 1. Main Hospital Card Box
        draw_round_box(hx, hy, hw, hh, '#FFFFFF', c_card_border, lw=1.8, radius=1.4, z=2)
        
        # 2. Header Section
        # Thumbnail icon box
        draw_round_box(hx + 1.8, hy + hh - 8.2, 6.8, 6.8, '#000000', c_card_border, lw=1.0, radius=0.6, z=3)
        ax.imshow(h['tex'], extent=[hx + 2.0, hx + 8.4, hy + hh - 8.0, hy + hh - 1.6], cmap='gray', zorder=4)
        
        # Hospital Title & Modality
        ax.text(hx + 10.0, hy + hh - 3.8, h['name'], fontsize=10.5, fontweight='bold', color='#0F172A', zorder=4)
        ax.text(hx + 10.0, hy + hh - 6.8, h['modality'], fontsize=9.0, fontweight='bold', color=h['color'], zorder=4)
        
        # Class Count Badge
        draw_round_box(hx + hw - 13.0, hy + hh - 6.5, 11.2, 4.2, h['tag_bg'], h['color'], lw=1.0, radius=0.6, z=3)
        ax.text(hx + hw - 7.4, hy + hh - 4.4, h['classes'], fontsize=8.2, fontweight='bold', color=h['color'], ha='center', va='center', zorder=4)

        # 3. Top Annotation: "Local BN - Private (Not Shared)"
        ax.text(hx + 38.5, hy + hh - 10.2, "Local BN — Not Shared", fontsize=8.2, fontweight='bold', color='#D97706', ha='center', zorder=4)
        draw_arrow(hx + 38.5, hy + hh - 11.2, hx + 38.5, hy + hh - 14.5, color='#D97706', lw=1.3, z=4)

        # 4. Neural Network Pipeline Layers (carefully spaced across x=hx+2.5 to hx+56)
        # (A) Input Image Stack
        ax.imshow(h['tex'], extent=[hx + 2.2, hx + 6.8, hy + 13.5, hy + 21.5], cmap='gray', zorder=3)
        ax.imshow(h['tex'], extent=[hx + 2.8, hx + 7.4, hy + 12.5, hy + 20.5], cmap='gray', zorder=4)
        draw_round_box(hx + 2.8, hy + 12.5, 4.6, 8.0, 'none', '#000000', lw=0.8, radius=0.3, z=5)
        ax.text(hx + 5.1, hy + 9.5, "Input", fontsize=8.0, fontweight='bold', color='#334155', ha='center', zorder=5)

        draw_arrow(hx + 7.8, hy + 16.5, hx + 9.8, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (B) EfficientNetV2-S Backbone Feature Extractor (4 vertical trapezoid slices)
        bx = hx + 10.2
        block_widths = [1.4, 1.4, 1.4, 1.4]
        block_heights = [14.0, 12.0, 10.0, 8.5]
        for i, (bw, bh) in enumerate(zip(block_widths, block_heights)):
            by = hy + 16.5 - bh / 2.0
            draw_round_box(bx + i * 2.0, by, bw, bh, c_conv_fill, c_conv_border, lw=1.0, radius=0.25, z=4)
        
        # Backbone label below
        ax.text(bx + 3.8, hy + 5.0, "EfficientNetV2-S\nBackbone", fontsize=7.5, fontweight='bold', color='#0284C7', ha='center', va='top', zorder=5)

        draw_arrow(bx + 8.2, hy + 16.5, bx + 10.0, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (C) CBAM Attention Module (Green Box - CLEAR HORIZONTAL TEXT)
        cbam_x = bx + 10.5
        draw_round_box(cbam_x, hy + 7.5, 5.8, 18.0, c_cbam_fill, c_cbam_border, lw=1.2, radius=0.4, z=4)
        ax.text(cbam_x + 2.9, hy + 21.0, "CBAM", fontsize=8.2, fontweight='bold', color='#15803D', ha='center', zorder=5)
        ax.text(cbam_x + 2.9, hy + 17.5, "Attention", fontsize=7.4, fontweight='bold', color='#15803D', ha='center', zorder=5)
        ax.text(cbam_x + 2.9, hy + 14.2, "Module", fontsize=7.4, fontweight='bold', color='#15803D', ha='center', zorder=5)
        ax.text(cbam_x + 2.9, hy + 10.2, r"$\mathbf{M}_c \otimes \mathbf{M}_s$", fontsize=7.0, fontweight='bold', color='#166534', ha='center', zorder=5)

        draw_arrow(cbam_x + 6.2, hy + 16.5, cbam_x + 7.8, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (D) GAP Layer (Gray Box)
        gap_x = cbam_x + 8.2
        draw_round_box(gap_x, hy + 9.5, 3.2, 14.0, c_gap_fill, c_gap_border, lw=1.0, radius=0.3, z=4)
        ax.text(gap_x + 1.6, hy + 16.5, "GAP", fontsize=7.4, fontweight='bold', color='#334155', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(gap_x + 3.6, hy + 16.5, gap_x + 5.2, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (E) PReLU Layer (Mint Green Box)
        prelu_x = gap_x + 5.6
        draw_round_box(prelu_x, hy + 10.5, 3.0, 12.0, c_prelu_fill, c_prelu_border, lw=1.0, radius=0.3, z=4)
        ax.text(prelu_x + 1.5, hy + 16.5, "PReLU", fontsize=7.4, fontweight='bold', color='#047857', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(prelu_x + 3.4, hy + 16.5, prelu_x + 5.0, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (F) MC Dropout Layer (Orange Box)
        drop_x = prelu_x + 5.4
        draw_round_box(drop_x, hy + 9.0, 3.4, 15.0, c_drop_fill, c_drop_border, lw=1.1, radius=0.3, z=4)
        ax.text(drop_x + 1.7, hy + 16.5, "MC Dropout", fontsize=7.4, fontweight='bold', color='#C2410C', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(drop_x + 3.8, hy + 16.5, drop_x + 5.4, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (G) Local Classification Head \phi_k (Pink/Crimson Box)
        head_x = drop_x + 5.8
        draw_round_box(head_x, hy + 8.0, 3.6, 17.0, c_head_fill, c_head_border, lw=1.2, radius=0.35, z=4)
        ax.text(head_x + 1.8, hy + 16.5, r"Head $\phi_k$", fontsize=7.5, fontweight='bold', color='#BE123C', ha='center', va='center', rotation=90, zorder=5)
        
        # Local Head label below
        ax.text(head_x + 1.8, hy + 5.0, r"Local Head $\phi_k$" + "\nNot Shared", fontsize=7.4, fontweight='bold', color='#BE123C', ha='center', va='top', zorder=5)
        draw_arrow(head_x + 1.8, hy + 5.0, head_x + 1.8, hy + 7.2, color='#BE123C', lw=1.0, z=5)

        draw_arrow(head_x + 4.0, hy + 16.5, head_x + 5.6, hy + 16.5, color='#334155', lw=1.1, z=5)

        # (H) Softmax Logits Output
        ax.text(head_x + 6.0, hy + 16.5, "Softmax\nLogits", fontsize=7.4, fontweight='bold', color='#0F172A', va='center', zorder=5)

    # ==========================================================================
    # 3. COMMUNICATION ARROWS BETWEEN SERVER AND HOSPITALS
    # ==========================================================================
    # Hospital A Arrows (Left)
    ax.plot([28, 28, 72], [67, 72, 72], color='#0F172A', linewidth=1.5, zorder=4)
    draw_arrow(72, 72, 72, 75, color='#0F172A', lw=1.5, z=5)
    ax.text(48, 73.6, r"Upload $\theta_{\mathbf{conv}, \mathrm{A}}$", fontsize=8.5, fontweight='bold', color='#0F172A', ha='center', zorder=6)

    ax.plot([80, 80, 36, 36], [75, 69.5, 69.5, 67], color='#B91C1C', linewidth=1.5, zorder=4)
    draw_arrow(36, 68.5, 36, 67, color='#B91C1C', lw=1.5, z=5)
    ax.text(56, 67.8, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=8.5, fontweight='bold', color='#B91C1C', ha='center', zorder=6)

    # Hospital B Arrows (Center)
    draw_arrow(90.0, 67, 90.0, 75, color='#0F172A', lw=1.5, z=5)
    ax.text(83.5, 71.0, r"Upload $\theta_{\mathbf{conv}, \mathrm{B}}$", fontsize=8.2, fontweight='bold', color='#0F172A', ha='right', va='center', zorder=6)

    draw_arrow(100.0, 75, 100.0, 67, color='#B91C1C', lw=1.5, z=5)
    ax.text(106.5, 71.0, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=8.2, fontweight='bold', color='#B91C1C', ha='left', va='center', zorder=6)

    # Hospital C Arrows (Right)
    ax.plot([162, 162, 118], [67, 72, 72], color='#0F172A', linewidth=1.5, zorder=4)
    draw_arrow(118, 72, 118, 75, color='#0F172A', lw=1.5, z=5)
    ax.text(142, 73.6, r"Upload $\theta_{\mathbf{conv}, \mathrm{C}}$", fontsize=8.5, fontweight='bold', color='#0F172A', ha='center', zorder=6)

    ax.plot([110, 110, 154, 154], [75, 69.5, 69.5, 67], color='#B91C1C', linewidth=1.5, zorder=4)
    draw_arrow(154, 68.5, 154, 67, color='#B91C1C', lw=1.5, z=5)
    ax.text(134, 67.8, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=8.5, fontweight='bold', color='#B91C1C', ha='center', zorder=6)

    # ==========================================================================
    # 4. BOTTOM TIER: POST-HOC CALIBRATION & CONFORMAL TRIAGE PIPELINE
    # ==========================================================================
    draw_round_box(4, 2.5, 182, 18.5, c_purple_bg, c_purple_border, lw=2.0, radius=1.4, z=2)
    
    # Section Header
    ax.text(95, 18.0, "Post-Hoc Calibration Pipeline", fontsize=12.5, fontweight='bold', color='#4C1D95', ha='center', va='center', zorder=3)

    # 1. Raw Logits Label
    ax.text(16, 9.8, r"Raw Logits $\mathbf{z}(x)$", fontsize=11.0, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=4)
    draw_arrow(26.0, 9.8, 33.0, 9.8, color='#4C1D95', lw=1.8, z=5)

    # 2. Temperature Scaling Box
    draw_round_box(34, 5.2, 34, 9.2, '#FFFFFF', '#7C3AED', lw=1.5, radius=0.8, z=4)
    ax.text(51, 10.8, r"Temperature Scaling ($T^*$)", fontsize=10.0, fontweight='bold', color='#5B21B6', ha='center', zorder=5)
    ax.text(51, 7.8, "Validation-Guided NLL Optimization", fontsize=7.8, color='#6D28D9', ha='center', zorder=5)

    draw_arrow(69.0, 9.8, 77.0, 9.8, color='#4C1D95', lw=1.8, z=5)

    # 3. Calibrated Probabilities Label
    ax.text(89, 11.2, "Calibrated", fontsize=10.5, fontweight='bold', color='#1E293B', ha='center', zorder=4)
    ax.text(89, 8.4, r"Probabilities $\tilde{p}(x)$", fontsize=10.5, fontweight='bold', color='#1E293B', ha='center', zorder=4)
    draw_arrow(101.0, 9.8, 108.0, 9.8, color='#4C1D95', lw=1.8, z=5)

    # 4. APS Conformal Prediction Box
    draw_round_box(109, 5.2, 34, 9.2, '#FFFFFF', '#7C3AED', lw=1.5, radius=0.8, z=4)
    ax.text(126, 10.8, "APS Conformal Prediction", fontsize=10.0, fontweight='bold', color='#5B21B6', ha='center', zorder=5)
    ax.text(126, 7.8, r"Adaptive Prediction Sets Threshold $\hat{q}$", fontsize=7.8, color='#6D28D9', ha='center', zorder=5)

    draw_arrow(144.0, 9.8, 151.0, 9.8, color='#4C1D95', lw=1.8, z=5)

    # 5. Certified Prediction Set Output Box
    draw_round_box(152, 5.2, 32, 9.2, '#FFFFFF', '#059669', lw=1.5, radius=0.8, z=4)
    ax.text(168, 10.8, r"Prediction Set $\mathcal{C}(x)$", fontsize=10.5, fontweight='bold', color='#065F46', ha='center', zorder=5)
    ax.text(168, 7.8, r"coverage guarantee $\geq 1 - \alpha$", fontsize=8.2, fontweight='bold', color='#047857', ha='center', zorder=5)

    # Save outputs
    plt.savefig(output_jpg, dpi=300, format='jpg', pil_kwargs={'quality': 98})
    if output_png:
        plt.savefig(output_png, dpi=300, format='png')
    plt.close()
    print(f"[OK] Generated exact corrected publication Figure 1 at: {output_jpg}")

if __name__ == '__main__':
    out_dir = r"d:\Research\FedUA-Net\paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    jpg_path = os.path.join(out_dir, "fig1_architecture.jpg")
    png_path = os.path.join(out_dir, "fig1_architecture.png")
    build_figure1(jpg_path, png_path)
