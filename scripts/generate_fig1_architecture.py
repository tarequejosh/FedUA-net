# ==============================================================================
# FedUA-Net Publication-Quality Figure 1 Generator
# Ultra-High Resolution (300 DPI), Crisp Vector Text, Professional IEEE TMI Aesthetics
# ==============================================================================
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, BoxStyle, ArrowStyle, FancyArrowPatch
import numpy as np

def create_publication_fig1(output_jpg_path, output_png_path=None):
    # Setup ultra high-resolution canvas
    fig = plt.figure(figsize=(16, 8.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Background color: very subtle off-white
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    
    # Color Palette (Nature / IEEE TMI professional palette)
    c_navy = '#1A365D'       # Primary header & dark elements
    c_blue = '#2B6CB0'       # Primary accent
    c_light_blue = '#EBF8FF' # Light box fills
    c_teal = '#2C7A7B'       # Attention & feature maps
    c_light_teal = '#E6FFFA'
    c_crimson = '#9B2C2C'    # Proposed FedUA-Net / Highlights
    c_light_red = '#FFF5F5'
    c_purple = '#553C9A'     # Conformal & UQ
    c_light_purple = '#FAF5FF'
    c_amber = '#C05621'      # Scarcity & BN
    c_light_amber = '#FFFAF0'
    c_green = '#276749'      # Clinical triage success
    c_light_green = '#F0FFF4'
    c_gray_dark = '#2D3748'  # Primary text
    c_gray_mid = '#718096'   # Secondary text
    c_border = '#CBD5E0'     # Subtle borders
    
    # Helper to draw rounded container boxes
    def draw_box(x, y, w, h, bg_color, border_color, lw=1.5, radius=1.2, zorder=1):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=lw, zorder=zorder)
        ax.add_patch(box)
        return box

    def draw_badge(x, y, w, h, text, bg_color, text_color='#FFFFFF', fontsize=9.5, fontweight='bold', zorder=5):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size=0.6",
                             facecolor=bg_color, edgecolor='none', zorder=zorder)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize,
                fontweight=fontweight, ha='center', va='center', zorder=zorder+1)

    def draw_arrow(x1, y1, x2, y2, color=c_blue, lw=2.0, style='-|>', zorder=4):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=style, mutation_scale=14,
                                color=color, linewidth=lw, zorder=zorder)
        ax.add_patch(arrow)
        return arrow

    # ==========================================================================
    # 0. MAIN TITLE BAR & SUBTITLE
    # ==========================================================================
    draw_box(1.5, 92.5, 97, 6.2, '#FFFFFF', c_navy, lw=1.8, radius=0.8, zorder=2)
    ax.text(50, 96.5, "FedUA-Net: Federated Multi-Task Medical Imaging Framework with Calibrated Conformal Triage",
            fontsize=13.5, fontweight='bold', color=c_navy, ha='center', va='center', zorder=3)
    ax.text(50, 94.0, "Privacy-Preserving Cross-Modality Learning • Decoupled BN/Heads • Uniform Server Aggregation • Split-Conformal APS Guarantees",
            fontsize=9.2, fontweight='normal', color=c_gray_mid, ha='center', va='center', zorder=3)

    # ==========================================================================
    # PANEL 1: MULTI-TASK HEALTHCARE CONSORTIUM (LEFT: x=1.5 to 29.5)
    # ==========================================================================
    draw_box(1.5, 2.5, 27.5, 88.5, '#FFFFFF', c_border, lw=1.2, radius=1.0, zorder=2)
    draw_badge(3.0, 87.0, 24.5, 3.2, "1. HETEROGENEOUS CONSORTIUM", c_navy, fontsize=10)
    
    ax.text(15.2, 84.5, "Multi-Center Cohorts (Disjoint Tasks)", fontsize=9.2, fontweight='bold', color=c_gray_dark, ha='center')
    ax.text(15.2, 82.5, r"$\mathcal{Y}_j \cap \mathcal{Y}_k = \emptyset \quad (11\ \text{Total Diagnostic Classes})$",
            fontsize=9.0, color=c_crimson, fontweight='bold', ha='center')

    # Hospital A: Brain MRI
    draw_box(2.8, 56.5, 24.9, 24.2, c_light_blue, c_blue, lw=1.2, radius=0.8, zorder=3)
    draw_badge(4.0, 76.5, 14.0, 2.8, "Hospital A (Neurology)", c_blue, fontsize=8.5, zorder=4)
    ax.text(4.2, 73.2, "• Modality: Brain T1-MRI (Contrast)", fontsize=8.5, fontweight='bold', color=c_gray_dark, zorder=4)
    ax.text(4.2, 70.8, "• Classes (4): Glioma, Meningioma,", fontsize=8.0, color=c_gray_dark, zorder=4)
    ax.text(5.2, 68.8, "Pituitary Tumor, No Tumor", fontsize=8.0, color=c_gray_dark, zorder=4)
    ax.text(4.2, 66.5, "• Dataset: N = 4,855 train / 857 val / 1,311 test", fontsize=7.8, color=c_navy, zorder=4)
    # MRI mini graphic placeholder box
    draw_box(4.0, 58.0, 22.5, 6.8, '#FFFFFF', c_blue, lw=0.8, radius=0.5, zorder=4)
    ax.text(15.2, 61.4, "[ Volumetric Soft-Tissue MRI ]\nHigh structural SNR & tissue contrast",
            fontsize=7.8, color=c_blue, ha='center', va='center', zorder=5)

    # Hospital B: Breast Ultrasound
    draw_box(2.8, 29.5, 24.9, 25.5, c_light_amber, c_amber, lw=1.2, radius=0.8, zorder=3)
    draw_badge(4.0, 51.0, 14.5, 2.8, "Hospital B (Oncology)", c_amber, fontsize=8.5, zorder=4)
    ax.text(4.2, 47.8, "• Modality: Breast Ultrasound (BUSI)", fontsize=8.5, fontweight='bold', color=c_gray_dark, zorder=4)
    ax.text(4.2, 45.4, "• Classes (3): Benign, Malignant, Normal", fontsize=8.0, color=c_gray_dark, zorder=4)
    ax.text(4.2, 43.0, "• Scarcity: N = 546 train (Low Sample Mass)", fontsize=7.8, color=c_crimson, fontweight='bold', zorder=4)
    draw_box(4.0, 31.0, 22.5, 10.5, '#FFFFFF', c_amber, lw=0.8, radius=0.5, zorder=4)
    ax.text(15.2, 36.2, "[ Acoustic Reflection & Speckle ]\nAcoustic shadows, wave attenuation,\noperator-dependent scanning geometry",
            fontsize=7.6, color=c_amber, ha='center', va='center', zorder=5)

    # Hospital C: COVID-19 Chest X-Ray
    draw_box(2.8, 4.2, 24.9, 23.8, c_light_teal, c_teal, lw=1.2, radius=0.8, zorder=3)
    draw_badge(4.0, 24.0, 15.0, 2.8, "Hospital C (Pulmonology)", c_teal, fontsize=8.5, zorder=4)
    ax.text(4.2, 20.8, "• Modality: Digital Chest Radiograph", fontsize=8.5, fontweight='bold', color=c_gray_dark, zorder=4)
    ax.text(4.2, 18.4, "• Classes (4): COVID-19, Lung Opacity,", fontsize=8.0, color=c_gray_dark, zorder=4)
    ax.text(5.2, 16.4, "Viral Pneumonia, Normal", fontsize=8.0, color=c_gray_dark, zorder=4)
    ax.text(4.2, 14.0, "• Dataset: N = 14,815 train (94% Mass)", fontsize=7.8, color=c_teal, zorder=4)
    draw_box(4.0, 5.5, 22.5, 7.0, '#FFFFFF', c_teal, lw=0.8, radius=0.5, zorder=4)
    ax.text(15.2, 9.0, "[ Photonic Transmission Radiography ]\nPlanar lung projection & structural ribs",
            fontsize=7.8, color=c_teal, ha='center', va='center', zorder=5)

    # ==========================================================================
    # PANEL 2: DECOUPLED ARCHITECTURE & FEDERATED AGGREGATION (CENTER: x=30.5 to 68.5)
    # ==========================================================================
    draw_box(30.5, 2.5, 37.5, 88.5, '#FFFFFF', c_border, lw=1.2, radius=1.0, zorder=2)
    draw_badge(32.0, 87.0, 34.5, 3.2, "2. FedUA-Net DECOUPLED MODEL & AGGREGATION", c_crimson, fontsize=10)

    # --- TOP SUB-PANEL: CENTRAL COORDINATOR SERVER ---
    draw_box(32.0, 64.5, 34.5, 21.0, '#F7FAFC', c_navy, lw=1.5, radius=0.8, zorder=3)
    draw_badge(33.5, 81.5, 21.0, 2.6, "Central Coordinator Server", c_navy, fontsize=8.5, zorder=4)
    
    # Server Uniform Aggregation Box
    draw_box(33.5, 66.0, 31.5, 14.5, '#FFFFFF', c_crimson, lw=1.2, radius=0.6, zorder=4)
    ax.text(49.2, 77.8, "Uniform Server Aggregation Engine", fontsize=9.2, fontweight='bold', color=c_crimson, ha='center', zorder=5)
    ax.text(49.2, 74.5, r"$\theta_{\text{conv}}^{(t+1)} = \frac{1}{K} \sum_{k=1}^K \theta_{\text{conv}, k}^{(t+1)} \quad (w_k = 1/3)$",
            fontsize=10.0, fontweight='bold', color=c_navy, ha='center', zorder=5)
    ax.text(49.2, 70.8, "• Egalitarian Fairness: Eliminates gradient starvation on small clinics", fontsize=7.8, color=c_gray_dark, ha='center', zorder=5)
    ax.text(49.2, 68.2, "• Shared Backbone Only: 21.45M parameters (85.8 MB float32 / 42.9 MB float16)", fontsize=7.6, color=c_gray_mid, ha='center', zorder=5)

    # Communication Arrows between Server and Local Clients
    draw_arrow(43.5, 64.5, 43.5, 57.5, color=c_crimson, lw=2.2, style='->', zorder=6)
    draw_arrow(55.0, 57.5, 55.0, 64.5, color=c_blue, lw=2.2, style='->', zorder=6)
    ax.text(39.0, 61.0, "Broadcast\n" + r"$\theta_{\text{conv}}^{(t)}$", fontsize=8.0, fontweight='bold', color=c_crimson, ha='center', va='center', zorder=7)
    ax.text(59.5, 61.0, "Upload\n" + r"$\theta_{\text{conv}, k}^{(t+1)}$", fontsize=8.0, fontweight='bold', color=c_blue, ha='center', va='center', zorder=7)

    # --- BOTTOM SUB-PANEL: LOCAL CLIENT ARCHITECTURE ON HOSPITAL k ---
    draw_box(32.0, 4.2, 34.5, 52.5, '#F7FAFC', c_blue, lw=1.4, radius=0.8, zorder=3)
    draw_badge(33.5, 53.0, 26.5, 2.8, "Local Decoupled Architecture (Hospital k)", c_blue, fontsize=8.5, zorder=4)

    # 1. Input Scan
    draw_box(33.5, 43.5, 31.5, 7.5, '#FFFFFF', c_gray_mid, lw=1.0, radius=0.5, zorder=4)
    ax.text(35.0, 48.0, "Input Image " + r"$x_k \in \mathcal{X}_k$", fontsize=8.5, fontweight='bold', color=c_navy, zorder=5)
    ax.text(35.0, 45.3, "224×224 RGB • Site-specific scanner preprocessing", fontsize=7.8, color=c_gray_mid, zorder=5)
    draw_arrow(49.2, 43.5, 49.2, 40.5, color=c_navy, lw=1.5, zorder=6)

    # 2. Shared Visual Backbone (EfficientNetV2-S + CBAM)
    draw_box(33.5, 23.5, 31.5, 17.0, c_light_teal, c_teal, lw=1.4, radius=0.6, zorder=4)
    draw_badge(34.5, 37.0, 24.5, 2.4, "Shared Visual Backbone (g_θ)", c_teal, fontsize=8.0, zorder=5)
    
    # Sub-blocks inside backbone: EfficientNetV2-S & CBAM
    draw_box(34.5, 29.5, 29.5, 6.5, '#FFFFFF', c_teal, lw=0.8, radius=0.4, zorder=5)
    ax.text(49.2, 33.8, "EfficientNetV2-S Feature Stages (Pre-trained)", fontsize=8.0, fontweight='bold', color=c_navy, ha='center', zorder=6)
    ax.text(49.2, 31.2, "Universal edge, curvature, and structural texture primitive filters", fontsize=7.3, color=c_gray_mid, ha='center', zorder=6)

    draw_box(34.5, 24.5, 29.5, 4.5, '#FFFFFF', c_crimson, lw=0.8, radius=0.4, zorder=5)
    ax.text(49.2, 26.8, "CBAM Attention: Channel (M_c) ⊗ Spatial (M_s) Refinement", fontsize=7.8, fontweight='bold', color=c_crimson, ha='center', zorder=6)

    draw_arrow(49.2, 23.5, 49.2, 20.8, color=c_teal, lw=1.5, zorder=6)

    # 3. Decoupled Components: Local BN & Local Head
    # Local BN Box (Left)
    draw_box(33.5, 5.5, 15.0, 15.0, c_light_amber, c_amber, lw=1.2, radius=0.6, zorder=4)
    draw_badge(34.2, 17.2, 13.6, 2.2, "Private Local BN", c_amber, fontsize=7.5, zorder=5)
    ax.text(41.0, 14.5, r"$\theta_{\text{BN}, k} = \{\gamma, \beta, \mu, \sigma^2\}$", fontsize=7.8, fontweight='bold', color=c_navy, ha='center', zorder=5)
    ax.text(41.0, 11.2, "• Retained locally\n• Absorbs scanner\n  domain shift\n• Never aggregated",
            fontsize=7.2, color=c_gray_dark, ha='center', va='center', zorder=5)

    # Local Classification Head (Right)
    draw_box(50.0, 5.5, 15.0, 15.0, c_light_red, c_crimson, lw=1.2, radius=0.6, zorder=4)
    draw_badge(50.7, 17.2, 13.6, 2.2, "Private Local Head", c_crimson, fontsize=7.5, zorder=5)
    ax.text(57.5, 14.5, r"$h_{\phi_k} \in \mathbb{R}^{512 \times C_k}$", fontsize=7.8, fontweight='bold', color=c_navy, ha='center', zorder=5)
    ax.text(57.5, 11.2, "• Task-specific $C_k$\n• Dense projection\n• PReLU + Dropout\n• Private weights",
            fontsize=7.2, color=c_gray_dark, ha='center', va='center', zorder=5)

    # Output Arrow from Local Model to UQ Pipeline
    draw_arrow(68.0, 48.0, 70.0, 48.0, color=c_purple, lw=2.5, style='->', zorder=6)
    ax.text(69.0, 51.0, "Logits\n" + r"$\mathbf{z}(x)$", fontsize=8.5, fontweight='bold', color=c_purple, ha='center', va='center', zorder=7)

    # ==========================================================================
    # PANEL 3: POST-HOC CALIBRATED CONFORMAL TRIAGE (RIGHT: x=69.5 to 98.5)
    # ==========================================================================
    draw_box(69.5, 2.5, 29.0, 88.5, '#FFFFFF', c_border, lw=1.2, radius=1.0, zorder=2)
    draw_badge(71.0, 87.0, 26.0, 3.2, "3. CALIBRATED CONFORMAL TRIAGE", c_purple, fontsize=10)

    # Stage A: Temperature Scaling
    draw_box(70.8, 65.5, 26.4, 19.5, c_light_purple, c_purple, lw=1.2, radius=0.8, zorder=3)
    draw_badge(72.0, 81.5, 17.0, 2.4, "Validation Temperature Scaling", c_purple, fontsize=8.0, zorder=4)
    ax.text(72.2, 78.5, r"• Optimizes $T_k^* > 0$ via L-BFGS on $\mathcal{D}_{val, k}$", fontsize=7.8, color=c_navy, zorder=4)
    ax.text(72.2, 75.8, r"$\tilde{p}_c(x) = \mathrm{softmax}(\mathbf{z}(x) / T_k^*)_c$", fontsize=8.5, fontweight='bold', color=c_purple, zorder=4)
    draw_box(72.0, 67.0, 24.0, 7.2, '#FFFFFF', c_purple, lw=0.8, radius=0.4, zorder=4)
    ax.text(84.0, 71.8, "ECE Reduction: 0.0504 → 0.0307 (-39%)", fontsize=8.0, fontweight='bold', color=c_crimson, ha='center', zorder=5)
    ax.text(84.0, 69.0, "Learned consortium avg temp: T = 0.787", fontsize=7.4, color=c_gray_mid, ha='center', zorder=5)

    draw_arrow(84.0, 65.5, 84.0, 61.5, color=c_purple, lw=1.8, zorder=5)

    # Stage B: Conformal APS Engine
    draw_box(70.8, 38.5, 26.4, 22.5, c_light_blue, c_blue, lw=1.2, radius=0.8, zorder=3)
    draw_badge(72.0, 57.0, 22.5, 2.4, "Adaptive Prediction Sets (APS)", c_blue, fontsize=8.0, zorder=4)
    ax.text(72.2, 54.0, r"• Sorts calibrated probs: $\pi_{(1)} \geq \dots \geq \pi_{(C_k)}$", fontsize=7.8, color=c_navy, zorder=4)
    ax.text(72.2, 51.5, r"• Non-conformity score: $s_i = \sum_{j=1}^{k_i} \pi_{(j)}(x_i)$", fontsize=7.8, color=c_navy, zorder=4)
    ax.text(72.2, 49.0, r"• Quantile Threshold $\hat{q}_k$ calibrated at $1-\alpha$", fontsize=7.8, color=c_navy, zorder=4)
    draw_box(72.0, 40.0, 24.0, 7.5, '#FFFFFF', c_blue, lw=0.8, radius=0.4, zorder=4)
    ax.text(84.0, 44.8, r"Certified Guarantee: $P(Y \in \mathcal{C}(X)) \geq 1-\alpha$", fontsize=7.8, fontweight='bold', color=c_navy, ha='center', zorder=5)
    ax.text(84.0, 42.0, "Empirical Coverage: 99.1% (Set size: 2.33 classes)", fontsize=7.4, color=c_teal, fontweight='bold', ha='center', zorder=5)

    draw_arrow(84.0, 38.5, 84.0, 34.5, color=c_purple, lw=1.8, zorder=5)

    # Stage C: Clinical Decision Support Output
    draw_box(70.8, 4.2, 26.4, 29.8, '#FFFFFF', c_border, lw=1.2, radius=0.8, zorder=3)
    draw_badge(72.0, 30.5, 21.0, 2.4, "Clinical Decision Support Output", c_navy, fontsize=8.0, zorder=4)

    # Branch 1: High Confidence
    draw_box(72.0, 18.0, 24.0, 11.2, c_light_green, c_green, lw=1.0, radius=0.5, zorder=4)
    ax.text(73.0, 26.5, "✓ High Confidence Case (|C(x)| = 1)", fontsize=8.0, fontweight='bold', color=c_green, zorder=5)
    ax.text(73.0, 24.0, "• Output: Single definitive class label", fontsize=7.4, color=c_gray_dark, zorder=5)
    ax.text(73.0, 21.5, "• Risk-Coverage: 98.12% acc at 50% cov", fontsize=7.4, color=c_navy, zorder=5)
    ax.text(73.0, 19.2, "• Autonomous rapid clinical triage", fontsize=7.4, color=c_gray_mid, zorder=5)

    # Branch 2: Ambiguous / Borderline Case
    draw_box(72.0, 5.5, 24.0, 11.2, c_light_red, c_crimson, lw=1.0, radius=0.5, zorder=4)
    ax.text(73.0, 14.0, "⚠ Ambiguous Case (|C(x)| ≥ 2)", fontsize=8.0, fontweight='bold', color=c_crimson, zorder=5)
    ax.text(73.0, 11.5, "• Output: Certified multi-label set", fontsize=7.4, color=c_gray_dark, zorder=5)
    ax.text(73.0, 9.2, "• e.g. {Benign, Malignant}", fontsize=7.4, color=c_navy, fontweight='bold', zorder=5)
    ax.text(73.0, 7.0, "• Alerts Radiologist for Secondary Review", fontsize=7.4, color=c_crimson, fontweight='bold', zorder=5)

    # Save to high-res JPG and PNG
    plt.savefig(output_jpg_path, dpi=300, format='jpg', pil_kwargs={'quality': 95})
    if output_png_path:
        plt.savefig(output_png_path, dpi=300, format='png')
    plt.close()
    print(f"[OK] Successfully generated publication Figure 1 at: {output_jpg_path}")

if __name__ == '__main__':
    out_dir = r"d:\Research\FedUA-Net\paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    jpg_path = os.path.join(out_dir, "fig1_architecture.jpg")
    png_path = os.path.join(out_dir, "fig1_architecture.png")
    create_publication_fig1(jpg_path, png_path)
