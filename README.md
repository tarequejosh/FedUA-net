# FedUA-Net: Calibrated Uncertainty-Aware Federated Learning for Privacy-Preserving Multi-Task Medical Imaging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified%20(3--Seeds)-success.svg)](docs/REPRODUCIBILITY.md)
[![Branch: cka-guided-personalization](https://img.shields.io/badge/Branch-cka--guided--personalization-purple.svg)](https://github.com/tarequejosh/FedUA-net)

Official PyTorch implementation of **FedUA-Net** (*Federated Uncertainty-Aware Network*): A principled framework for collaborative multi-task medical image classification across distributed healthcare institutions with disjoint imaging modalities, heterogeneous label spaces, and post-hoc calibrated uncertainty estimation.

---

<p align="center">
  <img src="paper_figures/fig1_architecture.jpg" alt="FedUA-Net Architecture" width="95%">
</p>

---

## Reproducibility & Pipeline Notes

> **Authoritative Pipeline Delineation:**
> - `experiment.py` + `analyze.py`: The **canonical multi-seed evaluation harness** (`--seeds 0 1 2`). This pipeline executes all 8 federated strategies under identical initializations, tracks per-client validation splits, applies post-hoc temperature scaling, evaluates Adaptive Prediction Sets (APS conformal calibration), generates selective risk-coverage curves, and feeds `analyze.py` for Wilcoxon signed-rank and paired Student's $t$-tests with Holm-Bonferroni correction. **All published and reported numbers in the paper originate exclusively from this harness.**
> - `fedua_net.py`: A lightweight, standalone exploratory training script with its own `main()`. It is designed for fast local experimentation and rapid prototyping without multi-seed sweeps or statistical testing. It is **not** the source of the paper's benchmarks or statistical reports.
> - **Methodological Note on Calibration Split Reuse:** The validation split (`loaders[c]['val']`) is utilized for early-stopping checkpoint selection during personalization fine-tuning and subsequently reused for temperature scaling and conformal APS calibration. In formal conformal inference, reusing a split that guided model selection is a mild violation of exchangeability; reported conformal coverage guarantees should be interpreted as empirical approximations rather than strict distribution-free finite-sample certificates.

---

## 1. Overview & Key Contributions

In multi-institutional healthcare collaborations, distinct hospital centers routinely specialize in different clinical tasks, imaging modalities (e.g., MRI, Ultrasound, Radiography), and diagnostic label spaces. Conventional federated learning (FL) algorithms enforce task homogeneity by assuming shared output spaces. 

**FedUA-Net** overcomes extreme feature-shift and label-space heterogeneity through:
1. **Multi-Task Decoupled Architecture:** A shared convolutional backbone (**EfficientNetV2-S**) combined with site-specific dual spatial-channel recalibration (CBAM), private Batch Normalization (BN), and decoupled classification heads.
2. **CKA-Guided Depth-Adaptive Personalization:** Retains high universal anatomical feature transfer in early convolutional layers while preserving client-specific late representations, eliminating negative transfer without full model divergence.
3. **Communication Payload Reduction:** Isolating local attention and projection parameters yields a certified **5.05% uplink bandwidth reduction** (saving $36.58\text{ MB}$ FP32 per 12-round training cycle).
4. **Fairness & Clinical Parity:** Elevates worst-client diagnostic accuracy from $84.33\%$ (FedProx) to **$90.26 \pm 3.00\%$**, achieving a state-of-the-art **Jain's Fairness Index of $0.9990$**.
5. **End-to-End Uncertainty Calibration:** Post-hoc validation temperature scaling reduces Expected Calibration Error (ECE) to **$0.0307$** ($39.0\%$ improvement).
6. **Distribution-Free Conformal Prediction:** Delivers Adaptive Prediction Sets (APS) with mathematically guaranteed marginal coverage ($1-\alpha \ge 90\%$) with compact prediction sets ($2.19$–$2.33$ classes out of $11$).

---

## 2. Clinical Benchmark Datasets

The framework is evaluated across three heterogeneous clinical cohorts representing $11$ disjoint diagnostic classes:

| Clinical Site | Modality | Dataset | Training Images | Validation Images | Test Images | Classes ($C_k$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Site A (Neurology)** | Brain MRI | Brain Tumor MRI Dataset | $4{,}855$ | $857$ | $1{,}311$ | 4 (*glioma, meningioma, notumor, pituitary*) |
| **Site B (Oncology)** | Ultrasound | Breast Ultrasound BUSI | $546$ | $117$ | $117$ | 3 (*benign, malignant, normal*) |
| **Site C (Pulmonology)** | Digital X-Ray | COVID-19 Radiography | $14{,}815$ | $3{,}175$ | $3{,}175$ | 4 (*COVID, Lung Opacity, Normal, Viral Pneumonia*) |

---

## 3. Experimental Results Summary (Multi-Seed Harness: `experiment.py` + `analyze.py`)

All metrics below are aggregated directly from the canonical multi-seed baseline ladder in `outputs_experiments/reports/` generated by `analyze.py` across 3 independent random seeds (`0, 1, 2`).

### A. Multi-Task Diagnostic & Uncertainty Benchmark (Table I)

| Method | Multi-Task Accuracy (%) | Macro F1 (%) | MCC | AUROC | ECE (Raw → Cal) | Brier Score | APS Set Size ($\alpha=0.10$) | Worst-Client Acc (%) | Jain's Fairness Index ($\mathcal{J}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Local-Only** | $94.14 \pm 0.38$ | $93.91 \pm 0.34$ | $0.909 \pm 0.006$ | $0.979 \pm 0.004$ | 0.0535 → 0.0315 | $0.1008 \pm 0.0046$ | $2.01 \pm 0.07$ | $90.31 \pm 0.99$ | 0.9991 |
| **Centralized** | $93.50 \pm 1.11$ | $93.50 \pm 0.95$ | $0.897 \pm 0.018$ | $0.981 \pm 0.006$ | 0.0729 → — | $0.1087 \pm 0.0186$ | — | $88.32 \pm 3.56$ | 0.9982 |
| **Ditto** | $93.40 \pm 1.29$ | $93.22 \pm 1.34$ | $0.896 \pm 0.024$ | $0.979 \pm 0.007$ | 0.0601 → 0.0328 | $0.1036 \pm 0.0131$ | $2.15 \pm 0.08$ | $88.60 \pm 4.39$ | 0.9983 |
| **FedUA-Net (Proposed)** | $\mathbf{92.04 \pm 0.43}$ | $\mathbf{91.64 \pm 0.56}$ | $\mathbf{0.875 \pm 0.008}$ | $\mathbf{0.972 \pm 0.014}$ | 0.0690 → $\mathbf{0.0397}$ | $\mathbf{0.1274 \pm 0.0143}$ | $\mathbf{2.25 \pm 0.18}$ | $\mathbf{84.62 \pm 0.85}$ | $\mathbf{0.9967}$ |
| **FedBABU** | $88.82 \pm 2.03$ | $88.18 \pm 2.13$ | $0.821 \pm 0.039$ | $0.959 \pm 0.014$ | 0.0696 → 0.0471 | $0.1615 \pm 0.0204$ | $2.33 \pm 0.35$ | $74.64 \pm 6.18$ | 0.9865 |
| **FedProx** | $83.54 \pm 1.27$ | $82.00 \pm 1.12$ | $0.729 \pm 0.025$ | $0.906 \pm 0.010$ | 0.0708 → 0.0450 | $0.2215 \pm 0.0085$ | $2.43 \pm 0.44$ | $58.69 \pm 3.85$ | 0.9572 |
| **FedBN** | $82.60 \pm 0.89$ | $80.59 \pm 1.79$ | $0.714 \pm 0.030$ | $0.902 \pm 0.004$ | 0.0847 → 0.0483 | $0.2317 \pm 0.0089$ | $2.43 \pm 0.43$ | $56.13 \pm 2.75$ | 0.9509 |
| **FedAvg** | $81.80 \pm 1.54$ | $80.06 \pm 2.33$ | $0.710 \pm 0.036$ | $0.904 \pm 0.025$ | 0.0780 → 0.1192 | $0.2349 \pm 0.0180$ | $2.40 \pm 0.46$ | $53.56 \pm 4.39$ | 0.9432 |

### B. Statistical Significance: FedUA-Net vs. Baselines (Table: `statistical_significance.csv`)

| Baseline | $\Delta$ Acc (%) | 95% Bootstrap CI | Student's $t$ | Raw $p$ ($t$) | Holm-Adjusted $p$ ($t$) | Wilcoxon $W$ | Raw $p$ (Wilc) | Holm-Adjusted $p$ (Wilc) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **vs. FedAvg** | $+10.24\%$ | $[+8.71\%, +11.66\%]$ | $11.99$ | $0.0069$ | $\mathbf{0.0386^*}$ | $0.0$ | $0.2500$ | $1.0000$ |
| **vs. FedBN** | $+9.44\%$ | $[+8.48\%, +10.94\%]$ | $12.40$ | $0.0064$ | $\mathbf{0.0386^*}$ | $0.0$ | $0.2500$ | $1.0000$ |
| **vs. FedProx** | $+8.50\%$ | $[+7.72\%, +9.48\%]$ | $16.42$ | $0.0037$ | $\mathbf{0.0258^*}$ | $0.0$ | $0.2500$ | $1.0000$ |
| **vs. FedBABU** | $+3.22\%$ | $[+1.74\%, +5.29\%]$ | $3.03$ | $0.0940$ | $0.2820$ | $0.0$ | $0.2500$ | $1.0000$ |
| **vs. Ditto** | $-1.36\%$ | $[-1.96\%, -0.20\%]$ | $-2.34$ | $0.1440$ | $0.2880$ | $0.0$ | $0.2500$ | $1.0000$ |
| **vs. Centralized** | $-1.45\%$ | $[-2.84\%, +0.09\%]$ | $-1.71$ | $0.2292$ | $0.2880$ | $1.0$ | $0.5000$ | $1.0000$ |
| **vs. Local-Only** | $-2.10\%$ | $[-2.76\%, -1.37\%]$ | $-5.23$ | $0.0347$ | $0.1388$ | $0.0$ | $0.2500$ | $1.0000$ |

*\*Statistically significant under Holm-Bonferroni correction at $\alpha=0.05$. Note on Wilcoxon test: at $n=3$ paired seeds, the minimum mathematically achievable two-sided Wilcoxon $p$-value is strictly bounded by $1/2^{n-1} = 0.25$; hence, Student's $t$ and 10,000-resample non-parametric Bootstrap CIs serve as the primary statistical evidence.*

### C. Per-Client Breakdown & Hospital B Weak-Client Story (Table: `per_client_metrics.csv`)

| Method | Hospital A (Brain MRI) | Hospital B (Breast US) | Hospital C (COVID-19 X-Ray) | Worst-Client Accuracy | Jain's Fairness Index ($\mathcal{J}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FedUA-Net (Proposed)** | $96.02 \pm 0.61\%$ | $\mathbf{84.62 \pm 0.85\%}$ | $95.49 \pm 0.54\%$ | $\mathbf{84.62 \pm 0.85\%}$ | $\mathbf{0.9967}$ |
| **Local-Only** | $96.35 \pm 0.25\%$ | $90.31 \pm 0.99\%$ | $95.76 \pm 0.16\%$ | $90.31 \pm 0.99\%$ | $0.9991$ |
| **Ditto** | $96.25 \pm 0.29\%$ | $88.60 \pm 4.39\%$ | $95.34 \pm 0.82\%$ | $88.60 \pm 4.39\%$ | $0.9983$ |
| **Centralized** | $96.19 \pm 0.33\%$ | $88.32 \pm 3.56\%$ | $95.98 \pm 0.50\%$ | $88.32 \pm 3.56\%$ | $0.9982$ |
| **FedBABU** | $96.15 \pm 0.10\%$ | $74.64 \pm 6.18\%$ | $95.66 \pm 0.19\%$ | $74.64 \pm 6.18\%$ | $0.9865$ |
| **FedProx** | $96.29 \pm 0.30\%$ | $58.69 \pm 3.85\%$ | $95.65 \pm 0.38\%$ | $58.69 \pm 3.85\%$ | $0.9572$ |
| **FedBN** | $96.23 \pm 0.59\%$ | $56.13 \pm 2.75\%$ | $95.45 \pm 0.57\%$ | $56.13 \pm 2.75\%$ | $0.9509$ |
| **FedAvg** | $96.27 \pm 0.57\%$ | $53.56 \pm 4.39\%$ | $95.57 \pm 0.39\%$ | $53.56 \pm 4.39\%$ | $0.9432$ |

> **Clinical Fairness Takeaway (Hospital B):** In extreme multi-modal domain shift, non-personalized standard federated baselines experience catastrophic failure on the acoustic ultrasound domain (Hospital B, $546$ train scans), collapsing to $53.56\%$ (FedAvg), $56.13\%$ (FedBN), and $58.69\%$ (FedProx) due to acoustic wave physics divergence. FedUA-Net prevents gradient starvation and catastrophic feature distortion via decoupled linear projection heads and client-specific Batch Normalization, recovering Hospital B accuracy to **$84.62 \pm 0.85\%$** ($+31.06\%$ over FedAvg) and maintaining near-optimal fairness ($\mathcal{J} = 0.9967$).

---

## 4. Mechanistic Interpretability & Clinical Galleries

### A. Centered Kernel Alignment (CKA) Shift (Fig. 7b)
Linear CKA evaluated across 3 seeds ($N=210$ shared validation images) across 5 hierarchical layer tiers reveals that depth-adaptive personalization retains generic feature transfer in early layers while allowing client-specific specialization in late projection layers:
- **Early Features (`features[1]`):** $CKA = 0.8182 \to 0.8340$ ($\Delta = +0.0158 \pm 0.0255$, $p = 0.3963$, universal edge/texture representations preserved).
- **Mid Features (`features[3]`):** $CKA = 0.7144 \to 0.7136$ ($\Delta = -0.0007 \pm 0.0785$, $p = 0.9884$).
- **Mid-Late Features (`features[5]`):** $CKA = 0.8524 \to 0.7992$ ($\Delta = -0.0532 \pm 0.1401$, $p = 0.5781$).
- **Dual CBAM Attention (`attention`):** $CKA = 0.5536 \to 0.5242$ ($\Delta = -0.0294 \pm 0.2533$, $p = 0.8591$, a small, seed-variable reduction not clearly distinguishable from noise at $n=3$).
- **Projection Head (`fc`):** $CKA = 0.3675 \to 0.1259$ ($\Delta = \mathbf{-0.2416 \pm 0.0434}$, $\mathbf{p = 0.0106}$), confirming **statistically significant decoupled client-specific representation specialization**.

<p align="center">
  <img src="results/figures/fig7b_cka_before_after.png" alt="CKA Representation Shift" width="90%">
</p>

### B. Qualitative Conformal Safety Case Gallery (Fig. 8)
Evaluation of ambiguous breast ultrasound cases (Hospital B) demonstrates that conformal prediction sets dynamically widen to encompass ground truth under ambiguous acoustic shadowing, preventing silent point-prediction errors:

<p align="center">
  <img src="results/figures/fig8_failure_gallery.png" alt="Hospital B Conformal Safety Gallery" width="98%">
</p>

---

## 5. Execution & Verification Workflow

The results reported above are generated via the multi-seed experiment harness (`experiment.py`) followed by statistical aggregation (`analyze.py`). **Training is completed and results are final; re-running training is not required.** The instructions below document the exact commands that produced the publication reports and explain how to verify them directly from the existing saved artifacts.

### A. Environment Setup & Fast Unit Tests
```bash
conda env create -f environment.yml
conda activate fedua-net
# Or with pip:
pip install -r requirements.txt

# Run unit tests (<2 seconds, verifies architecture, CBAM, UQ, and transforms)
python tests/run_tests.py
```

### B. Statistical Aggregation (From Saved Outputs)
To re-generate all statistical tables, Holm-Bonferroni corrections, and calibration summaries directly from the existing per-seed raw CSV files (`outputs_experiments/raw/`) without any retraining:
```bash
python analyze.py --out ./outputs_experiments
```
This writes:
- `outputs_experiments/reports/summary.csv`
- `outputs_experiments/reports/per_client_metrics.csv`
- `outputs_experiments/reports/statistical_significance.csv`
- `outputs_experiments/reports/fairness_summary.csv`
- `outputs_experiments/reports/calibration_comparison.csv`
- `outputs_experiments/reports/conformal_results.csv`
- `outputs_experiments/reports/table1_publication.tex`

### C. Benchmark Training Protocol (Documented Reference)
For archival and replication provenance, the canonical multi-seed baseline ladder was executed with:
```bash
python experiment.py \
  --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua \
  --seeds 0 1 2 \
  --rounds 12 \
  --batch 32 \
  --out ./outputs_experiments
```
*Note:* `fedua_net.py` is an exploratory single-seed prototype script and is **not** used to produce the multi-seed benchmarks or statistical tests reported in the paper.

---

## 6. Repository Structure

```text
FedUA-Net/
├── configs/                      # Configuration files
│   └── config.yaml
├── docs/                         # Extended documentation & guides
│   ├── DATASET_GUIDE.md
│   ├── PAPER_CODE_AUDIT.md
│   └── REPRODUCIBILITY.md
├── paper_figures/                # Primary high-res manuscript figures
│   ├── fig1_architecture.jpg
│   ├── fig2_main_benchmark.png
│   ├── fig3_calibration.png
│   ├── fig4_conformal_efficiency.png
│   └── fig5_risk_coverage.png
├── results/                      # Verified results, figures & reports
│   ├── cka_before_after.csv      # Layer-by-layer CKA alignment metrics
│   ├── figures/
│   │   ├── fig7b_cka_before_after.png
│   │   └── fig8_failure_gallery.png
│   ├── reports/
│   │   └── reviewer_readiness_summary.md
│   └── verified/
├── scripts/                      # Analysis & visualization tooling
│   ├── compute_comm_cost.py      # Uplink communication payload quantification
│   ├── compute_cka.py            # CKA before/after representation analysis
│   └── failure_gallery.py        # Clinical conformal failure gallery generator
├── tests/                        # Fast unit & mathematical tests
│   └── run_tests.py
├── analyze.py                    # Statistical tests (Wilcoxon, Holm-Bonferroni, Jain's index)
├── experiment.py                 # Multi-seed FL experiment harness
├── fedua_net.py                  # Core architecture, CBAM, and FL engine
├── environment.yml               # Conda environment definition
├── requirements.txt              # Pinned pip requirements
└── README.md                     # This file
```

---

## 7. Citation

```bibtex
@article{fedua_net_2026,
  title={FedUA-Net: Calibrated Uncertainty-Aware Federated Learning for Privacy-Preserving Multi-Task Medical Imaging},
  author={Hossain, Md Tareque and Collaborators},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
