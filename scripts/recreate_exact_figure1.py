# ==============================================================================
# FedUA-Net: Compact, Large-Text Exact Architecture Diagram (v4)
# Exact proportions, large high-contrast fonts, no overlapping, 100% compact
# ==============================================================================

import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def extract_and_enhance_scans(orig_img_path):
    orig = Image.open(orig_img_path)
    mri_crop = orig.crop((44, 275, 118, 348)).resize((240, 240), Image.Resampling.LANCZOS)
    us_crop = orig.crop((485, 275, 570, 348)).resize((240, 240), Image.Resampling.LANCZOS)
    cxr_crop = orig.crop((928, 275, 1002, 348)).resize((240, 240), Image.Resampling.LANCZOS)
    
    mri_input = orig.crop((44, 400, 118, 480)).resize((240, 260), Image.Resampling.LANCZOS)
    us_input = orig.crop((485, 400, 570, 480)).resize((240, 260), Image.Resampling.LANCZOS)
    cxr_input = orig.crop((928, 400, 1002, 480)).resize((240, 260), Image.Resampling.LANCZOS)

    return (mri_crop, us_crop, cxr_crop), (mri_input, us_input, cxr_input)

def render_figure(out_jpg, out_png=None):
    orig_path = r"d:\Research\FedUA-Net\fig1_architecture_1787497217049.jpg"
    thumbnails, inputs = extract_and_enhance_scans(orig_path)
    
    # 1376 x 768 coordinate space mapped to a large 300 DPI canvas
    # Using 137.6 x 76.8 data units
    fig = plt.figure(figsize=(13.76 * 1.5, 7.68 * 1.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 137.6)
    ax.set_ylim(0, 76.8)
    ax.axis('off')
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    c_border = '#000000'
    c_server_top = '#EEEEEE'
    c_server_bot = '#FFFFFF'
    c_hosp_bg = '#FFFFFF'
    
    c_conv_fill = '#E0F2FE'
    c_conv_border = '#0284C7'
    c_cbam_fill = '#DCFCE7'
    c_cbam_border = '#16A34A'
    c_gap_fill = '#F1F5F9'
    c_gap_border = '#64748B'
    c_prelu_fill = '#ECFDF5'
    c_prelu_border = '#059669'
    c_drop_fill = '#FFEDD5'
    c_drop_border = '#EA580C'
    c_head_fill = '#FFE4E6'
    c_head_border = '#E11D48'
    c_purple_bg = '#F3E8FF'
    c_purple_border = '#000000'

    def draw_round_box(x, y, w, h, bg, border, lw=2.2, radius=1.0, z=1):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=bg, edgecolor=border, linewidth=lw, zorder=z)
        ax.add_patch(p)
        return p

    def draw_arrow(x1, y1, x2, y2, color='#000000', lw=2.0, style='-|>', z=5):
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=14,
                              color=color, linewidth=lw, zorder=z)
        ax.add_patch(arr)
        return arr

    # ==========================================================================
    # 1. TOP SERVER BOX (Exact compact placement: x=45, y=56.5, w=48, h=18)
    # ==========================================================================
    draw_round_box(45.0, 56.5, 48.0, 18.0, c_server_bot, c_border, lw=2.2, radius=1.2, z=2)
    
    # Header bar
    header_patch = FancyBboxPatch((45.1, 69.8), 47.8, 4.6,
                                  boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor=c_server_top, edgecolor='none', zorder=3)
    ax.add_patch(header_patch)
    ax.plot([45, 93], [69.8, 69.8], color=c_border, linewidth=2.0, zorder=4)
    
    ax.text(69.0, 72.2, "Central Server", fontsize=15.0, fontweight='bold', color='#000000', ha='center', va='center', zorder=5)
    ax.text(69.0, 66.5, r"Global Shared Backbone Weights $\theta_{\mathbf{conv}}$", fontsize=12.5, fontweight='bold', color='#000000', ha='center', zorder=4)
    ax.text(69.0, 63.2, "Weighted FedAvg Aggregation", fontsize=11.0, color='#000000', ha='center', zorder=4)
    ax.text(69.0, 58.5, r"$\theta(x) = \frac{1}{K} \sum_{k=1}^K \theta_{\mathbf{conv}, k} \quad \left(w_k = \frac{1}{3}\right)$",
            fontsize=12.5, fontweight='bold', color='#000000', ha='center', va='center', zorder=4)

    # ==========================================================================
    # 2. MIDDLE 3 HOSPITAL CARDS (Compact, large elements, no overlap)
    # ==========================================================================
    hospitals = [
        {
            'name': 'Hospital A (Neurology)',
            'modality': 'Brain MRI',
            'classes': '4 classes',
            'desc_label': 'Brain MRI',
            'x': 3.5, 'y': 17.5, 'w': 42.0, 'h': 34.0,
            'thumb': thumbnails[0],
            'input': inputs[0],
            'client_id': 'A'
        },
        {
            'name': 'Hospital B (Oncology)',
            'modality': 'Breast Ultrasound',
            'classes': '3 classes',
            'desc_label': 'Breast\nUltrasound',
            'x': 47.8, 'y': 17.5, 'w': 42.0, 'h': 34.0,
            'thumb': thumbnails[1],
            'input': inputs[1],
            'client_id': 'B'
        },
        {
            'name': 'Hospital C (Pulmonology)',
            'modality': 'Chest X-Ray',
            'classes': '4 classes',
            'desc_label': 'Chest X-Ray',
            'x': 92.1, 'y': 17.5, 'w': 42.0, 'h': 34.0,
            'thumb': thumbnails[2],
            'input': inputs[2],
            'client_id': 'C'
        }
    ]

    for h in hospitals:
        hx, hy, hw, hh = h['x'], h['y'], h['w'], h['h']
        
        # Hospital Box
        draw_round_box(hx, hy, hw, hh, c_hosp_bg, c_border, lw=2.2, radius=1.2, z=2)
        
        # Thumbnail with black border
        draw_round_box(hx + 1.2, hy + hh - 7.2, 6.2, 6.2, '#000000', '#000000', lw=1.5, radius=0.4, z=3)
        ax.imshow(h['thumb'], extent=[hx + 1.3, hx + 7.3, hy + hh - 7.1, hy + hh - 1.1], zorder=4)
        
        # Name & Modality
        ax.text(hx + 8.5, hy + hh - 3.4, h['name'], fontsize=12.0, fontweight='bold', color='#000000', zorder=4)
        ax.text(hx + 8.5, hy + hh - 6.0, h['modality'], fontsize=10.8, color='#000000', zorder=4)
        
        # Badge [ X classes ]
        draw_round_box(hx + hw - 10.2, hy + hh - 5.8, 9.2, 4.0, '#FFFFFF', '#000000', lw=1.3, radius=0.8, z=3)
        ax.text(hx + hw - 5.6, hy + hh - 3.8, h['classes'], fontsize=9.5, color='#000000', ha='center', va='center', zorder=4)

        # Local BN annotation
        ax.text(hx + 28.5, hy + hh - 8.5, "Local BN - Not Shared", fontsize=9.2, fontweight='bold', color='#C05621', ha='center', zorder=4)
        draw_arrow(hx + 28.5, hy + hh - 9.5, hx + 28.5, hy + hh - 12.0, color='#C05621', lw=1.5, z=4)

        # Input Image
        ax.text(hx + 4.3, hy + hh - 10.0, h['desc_label'], fontsize=8.8, color='#000000', ha='center', zorder=4)
        draw_round_box(hx + 1.4, hy + 7.5, 6.0, 7.2, 'none', '#000000', lw=1.1, radius=0.25, z=4)
        ax.imshow(h['input'], extent=[hx + 1.4, hx + 7.4, hy + 7.5, hy + 14.7], zorder=3)
        ax.text(hx + 4.4, hy + 5.5, "Input", fontsize=9.0, color='#000000', ha='center', zorder=4)

        draw_arrow(hx + 7.8, hy + 11.2, hx + 9.5, hy + 11.2, color='#000000', lw=1.3, z=5)

        # EfficientNetV2-S Backbone (4 layered blue blocks)
        bx = hx + 9.8
        block_widths = [1.1, 1.1, 1.1, 1.1]
        block_heights = [12.0, 10.5, 9.0, 7.5]
        for i, (bw, bh) in enumerate(zip(block_widths, block_heights)):
            by = hy + 11.2 - bh / 2.0
            draw_round_box(bx + i * 1.5, by, bw, bh, c_conv_fill, c_conv_border, lw=1.3, radius=0.2, z=4)
        
        # Backbone bracket / label below
        ax.text(bx + 2.8, hy + 3.0, "EfficientNetV2-S\nBackbone", fontsize=8.6, fontweight='bold', color='#1D4ED8', ha='center', va='top', zorder=5)

        draw_arrow(bx + 6.2, hy + 11.2, bx + 7.6, hy + 11.2, color='#000000', lw=1.3, z=5)

        # CBAM Attention Module (Green Box)
        cbam_x = bx + 7.8
        draw_round_box(cbam_x, hy + 4.5, 4.4, 13.5, c_cbam_fill, c_cbam_border, lw=1.4, radius=0.3, z=4)
        ax.text(cbam_x + 2.2, hy + 11.2, "CBAM Attention\nModule", fontsize=8.6, fontweight='bold', color='#15803D',
                ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(cbam_x + 4.8, hy + 11.2, cbam_x + 6.0, hy + 11.2, color='#000000', lw=1.3, z=5)

        # GAP (Grey Box)
        gap_x = cbam_x + 6.2
        draw_round_box(gap_x, hy + 6.0, 2.2, 10.5, c_gap_fill, c_gap_border, lw=1.2, radius=0.25, z=4)
        ax.text(gap_x + 1.1, hy + 11.2, "GAP", fontsize=8.6, fontweight='bold', color='#374151', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(gap_x + 2.5, hy + 11.2, gap_x + 3.6, hy + 11.2, color='#000000', lw=1.3, z=5)

        # PReLU (Mint Green Box)
        prelu_x = gap_x + 3.8
        draw_round_box(prelu_x, hy + 7.0, 2.0, 8.5, c_prelu_fill, c_prelu_border, lw=1.2, radius=0.25, z=4)
        ax.text(prelu_x + 1.0, hy + 11.2, "PReLU", fontsize=8.5, fontweight='bold', color='#15803D', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(prelu_x + 2.3, hy + 11.2, prelu_x + 3.4, hy + 11.2, color='#000000', lw=1.3, z=5)

        # MC Dropout (Orange Box)
        drop_x = prelu_x + 3.6
        draw_round_box(drop_x, hy + 5.5, 2.4, 11.5, c_drop_fill, c_drop_border, lw=1.3, radius=0.25, z=4)
        ax.text(drop_x + 1.2, hy + 11.2, "MC Dropout", fontsize=8.6, fontweight='bold', color='#C05621', ha='center', va='center', rotation=90, zorder=5)

        draw_arrow(drop_x + 2.7, hy + 11.2, drop_x + 3.8, hy + 11.2, color='#000000', lw=1.3, z=5)

        # Local Head \phi_k (Pink Box)
        head_x = drop_x + 4.0
        draw_round_box(head_x, hy + 4.5, 2.4, 13.5, c_head_fill, c_head_border, lw=1.4, radius=0.3, z=4)
        
        # Local Head label below
        ax.text(head_x + 1.2, hy + 3.0, r"Local Head $\phi_k$" + "\n" + r"$\phi_k$ - Not Shared",
                fontsize=8.4, fontweight='bold', color='#C53030', ha='center', va='top', zorder=5)
        draw_arrow(head_x + 1.2, hy + 2.8, head_x + 1.2, hy + 4.2, color='#C53030', lw=1.2, z=5)

        draw_arrow(head_x + 2.7, hy + 11.2, head_x + 3.8, hy + 11.2, color='#000000', lw=1.3, z=5)

        # Softmax Logits Output
        ax.text(head_x + 4.2, hy + 11.2, "softmax logits", fontsize=8.5, color='#000000', va='center', rotation=90, zorder=5)

    # ==========================================================================
    # 3. COMMUNICATION ARROWS (Large, clean, perfectly placed)
    # ==========================================================================
    # Hospital A (Left)
    ax.plot([20.0, 20.0, 54.0], [51.5, 54.8, 54.8], color='#000000', linewidth=1.8, zorder=4)
    draw_arrow(54.0, 54.8, 54.0, 56.5, color='#000000', lw=1.8, z=5)
    ax.text(13.0, 53.6, r"Upload $\theta_{\mathbf{conv}, \mathrm{k}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='center', zorder=6)

    ax.plot([60.0, 60.0, 26.5, 26.5], [56.5, 53.0, 53.0, 51.5], color='#000000', linewidth=1.8, zorder=4)
    draw_arrow(26.5, 52.5, 26.5, 51.5, color='#000000', lw=1.8, z=5)
    ax.text(35.5, 53.6, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='center', zorder=6)

    # Hospital B (Center)
    draw_arrow(64.5, 51.5, 64.5, 56.5, color='#000000', lw=1.8, z=5)
    ax.text(59.0, 54.0, r"Upload $\theta_{\mathbf{conv}, \mathrm{k}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='right', va='center', zorder=6)

    draw_arrow(72.5, 56.5, 72.5, 51.5, color='#000000', lw=1.8, z=5)
    ax.text(78.0, 54.0, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='left', va='center', zorder=6)

    # Hospital C (Right)
    ax.plot([118.0, 118.0, 84.0], [51.5, 54.8, 54.8], color='#000000', linewidth=1.8, zorder=4)
    draw_arrow(84.0, 54.8, 84.0, 56.5, color='#000000', lw=1.8, z=5)
    ax.text(102.0, 53.6, r"Upload $\theta_{\mathbf{conv}, \mathrm{k}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='center', zorder=6)

    ax.plot([78.0, 78.0, 112.0, 112.0], [56.5, 53.0, 53.0, 51.5], color='#000000', linewidth=1.8, zorder=4)
    draw_arrow(112.0, 52.5, 112.0, 51.5, color='#000000', lw=1.8, z=5)
    ax.text(125.0, 53.6, r"Broadcast $\theta_{\mathbf{conv}}$", fontsize=9.8, fontweight='bold', color='#000000', ha='center', zorder=6)

    # ==========================================================================
    # 4. BOTTOM TIER: POST-HOC CALIBRATION PIPELINE (Large, Prominent Box)
    # ==========================================================================
    draw_round_box(3.5, 1.5, 130.6, 14.0, c_purple_bg, c_purple_border, lw=2.2, radius=1.2, z=2)
    
    ax.text(68.8, 13.2, "Post-Hoc Calibration Pipeline", fontsize=13.5, fontweight='bold', color='#000000', ha='center', va='center', zorder=3)

    # (1) Raw Logits
    ax.text(14.0, 6.5, "Raw Logits", fontsize=12.5, fontweight='bold', color='#000000', ha='center', va='center', zorder=4)
    draw_arrow(20.5, 6.5, 26.5, 6.5, color='#000000', lw=2.0, z=5)

    # (2) Temperature Scaling Box
    draw_round_box(27.5, 3.0, 23.0, 7.2, '#FFFFFF', '#6B21A8', lw=2.0, radius=0.8, z=4)
    ax.text(39.0, 7.6, "Temperature", fontsize=11.5, fontweight='bold', color='#000000', ha='center', zorder=5)
    ax.text(39.0, 4.6, r"Scaling ($T^*$)", fontsize=11.5, fontweight='bold', color='#000000', ha='center', zorder=5)

    draw_arrow(51.0, 6.5, 57.0, 6.5, color='#000000', lw=2.0, z=5)

    # (3) Calibrated Probabilities
    ax.text(66.0, 8.0, "Calibrated", fontsize=12.0, fontweight='bold', color='#000000', ha='center', zorder=4)
    ax.text(66.0, 5.0, "Probabilities", fontsize=12.0, fontweight='bold', color='#000000', ha='center', zorder=4)

    draw_arrow(75.5, 6.5, 81.5, 6.5, color='#000000', lw=2.0, z=5)

    # (4) APS Conformal Prediction Box
    draw_round_box(82.5, 3.0, 24.5, 7.2, '#FFFFFF', '#6B21A8', lw=2.0, radius=0.8, z=4)
    ax.text(94.75, 7.6, "APS Conformal", fontsize=11.5, fontweight='bold', color='#000000', ha='center', zorder=5)
    ax.text(94.75, 4.6, "Prediction", fontsize=11.5, fontweight='bold', color='#000000', ha='center', zorder=5)

    draw_arrow(107.5, 6.5, 113.5, 6.5, color='#000000', lw=2.0, z=5)

    # (5) Prediction Set C(x)
    ax.text(124.0, 8.0, r"Prediction Set $\mathcal{C}(x)$", fontsize=12.5, fontweight='bold', color='#000000', ha='center', zorder=4)
    ax.text(124.0, 5.0, r"coverage guarantee $\geq 1 - \alpha$", fontsize=10.5, color='#000000', ha='center', zorder=4)

    # Save outputs
    plt.savefig(out_jpg, dpi=300, format='jpg', pil_kwargs={'quality': 99})
    if out_png:
        plt.savefig(out_png, dpi=300, format='png')
    plt.close()
    print(f"[OK] Generated compact, large-text publication Figure 1 at: {out_jpg}")

if __name__ == '__main__':
    out_dir = r"d:\Research\FedUA-Net\paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    jpg_path = os.path.join(out_dir, "fig1_architecture.jpg")
    png_path = os.path.join(out_dir, "fig1_architecture.png")
    render_figure(jpg_path, png_path)
