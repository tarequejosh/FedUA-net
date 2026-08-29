# FedUA-Net: Calibrated Uncertainty in Federated Learning for Privacy-Preserving Multi-Task Medical Imaging

Official PyTorch implementation of **FedUA-Net** (*Federated Uncertainty-Aware Network*): A framework for collaborative multi-task medical image classification across distributed healthcare institutions with disjoint imaging modalities, heterogeneous label spaces, and calibrated uncertainty estimation.

---

<p align="center">
  <img src="paper_figures/fig1_architecture.jpg" alt="FedUA-Net Architecture" width="95%">
</p>

---

## 1. Overview

In multi-institutional healthcare collaborations, different hospital centers routinely specialize in distinct clinical tasks, imaging modalities (e.g., MRI, Ultrasound, Radiography), and diagnostic label spaces. Conventional federated learning (FL) algorithms enforce task homogeneity by assuming shared output spaces. 

**FedUA-Net** addresses these challenges through:
1. **Multi-Task Decoupled Architecture:** A shared visual backbone (**EfficientNetV2-S with CBAM attention**) coupled with site-specific classification heads and private Batch Normalization (BN) layers locally maintained at each clinical node.
2. **Uniform Server Aggregation ($w_k = 1/K$):** Mitigates gradient starvation and representation skew in unbalanced clinical consortia (e.g., $14{,}815$ chest X-rays vs. $546$ breast ultrasounds).
3. **End-to-End Uncertainty Calibration:** Integrates validation-guided Temperature Scaling to reduce Expected Calibration Error (ECE) from $0.0504 \to 0.0307$ ($39.0\%$ relative improvement).
4. **Distribution-Free Conformal Prediction:** Implements Adaptive Prediction Sets (APS) split-conformal classification providing mathematically guaranteed marginal coverage ($1-\alpha \ge 90\%$) with compact prediction sets ($2.33$ classes out of $11$).
5. **Robustness Under Data Scarcity:** Outperforms standard federated baselines by $+3.99\%$ ($N=200$) and $+3.42\%$ ($N=100$) on severely constrained local cohorts.

---

## 2. Clinical Benchmark Datasets

The framework is evaluated across three heterogeneous clinical cohorts representing $11$ disjoint diagnostic classes:

| Client Node | Modality | Dataset | Training Images | Validation Images | Test Images | Classes ($C_k$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Hospital A** | Brain MRI | Brain Tumor MRI Dataset | $4{,}855$ | $857$ | $1{,}311$ | 4 (*glioma, meningioma, notumor, pituitary*) |
| **Hospital B** | Ultrasound | Breast Ultrasound BUSI | $546$ | $117$ | $117$ | 3 (*benign, malignant, normal*) |
| **Hospital C** | Digital X-Ray | COVID-19 Radiography | $14{,}815$ | $3{,}175$ | $3{,}175$ | 4 (*COVID, Lung Opacity, Normal, Viral Pneumonia*) |

Detailed acquisition sources and partition instructions are provided in [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md).

---

## 3. Experimental Results Summary (3-Seed Mean ± Std)

### Multi-Task Diagnostic & Uncertainty Benchmark (Table I)

| Method | Multi-Task Accuracy (%) | Macro F1 (%) | MCC | Raw ECE | Calibrated ECE | APS Set Size ($\alpha=0.10$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local-Only** | $94.01 \pm 0.32$ | $93.80 \pm 0.48$ | $0.906 \pm 0.005$ | 0.0463 | **0.0251** | $2.01 \pm 0.07$ |
| **Ditto** | $93.93 \pm 0.51$ | $93.67 \pm 0.62$ | $0.904 \pm 0.009$ | 0.0597 | 0.0289 | $2.18 \pm 0.09$ |
| **Centralized (Pooled)** | $93.45 \pm 1.30$ | $93.36 \pm 1.38$ | $0.895 \pm 0.024$ | 0.0714 | --- | --- |
| **FedUA-Net (Proposed)** | $\mathbf{93.30 \pm 1.13}$ | $\mathbf{93.05 \pm 1.36}$ | $\mathbf{0.894 \pm 0.019}$ | **0.0504** | $\mathbf{0.0307}$ | $\mathbf{2.33 \pm 0.23}$ |
| **FedAvg** | $92.36 \pm 0.92$ | $91.88 \pm 1.23$ | $0.878 \pm 0.016$ | 0.0697 | 0.0398 | $2.22 \pm 0.11$ |
| **FedBN** | $92.34 \pm 0.99$ | $91.76 \pm 1.02$ | $0.877 \pm 0.017$ | 0.0650 | 0.0379 | $2.20 \pm 0.16$ |
| **FedBABU** | $91.99 \pm 0.35$ | $91.50 \pm 0.34$ | $0.873 \pm 0.006$ | 0.0640 | 0.1913 | $2.14 \pm 0.17$ |
| **FedProx** | $91.97 \pm 1.03$ | $91.46 \pm 1.21$ | $0.870 \pm 0.019$ | 0.0680 | 0.0888 | $2.26 \pm 0.18$ |

Full provenance records and paper-code consistency audits are documented in [`results/RESULTS_PROVENANCE.md`](results/RESULTS_PROVENANCE.md) and [`docs/PAPER_CODE_AUDIT.md`](docs/PAPER_CODE_AUDIT.md).

---

## 4. Quick Start & Execution

### Environment Setup
```bash
# Clone repository
git clone https://github.com/tarequejosh/FedUA-net.git
cd FedUA-net

# Install dependencies
pip install -r requirements.txt
```

### Fast Unit Verification (<5 seconds)
Verify all mathematical and neural modules (CBAM, decoupled heads, BN isolation, uniform aggregation Eq. 8, temperature scaling, and conformal prediction quantiles):
```bash
python tests/run_tests.py
```

### Dataset Layout Verification
```bash
python scripts/check_datasets.py
```

### Benchmark Training Execution
```bash
# Run FedUA-Net across seeds 0, 1, 2
python scripts/run_experiments.py --method fedua --seeds 0 1 2 --rounds 12 --agg_weight_type uniform

# Run complete 8-method baseline ladder (Table I)
python scripts/run_experiments.py --all --rounds 12 --agg_weight_type uniform --seeds 0 1 2
```

### Automated Table & Figure Compilation
```bash
# Compile Markdown tables (Table I, II, III, IV)
python scripts/generate_tables.py --results_dir results/verified/main_uniform --output_dir results/tables

# Generate publication-grade figures (Fig. 2, 3, 4, 5)
python scripts/generate_figures.py --results_dir results/verified/main_uniform --output_dir results/figures
```

*(See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for full execution options and parameter flags).*

---

## 5. Repository Structure

```text
FedUA-Net/
├── configs/                   # Experiment YAML configurations
│   ├── benchmark.yaml
│   ├── scarcity_200.yaml
│   ├── scarcity_100.yaml
│   └── ablation.yaml
├── docs/                      # Documentation & Execution Guides
│   ├── PAPER_CODE_AUDIT.md
│   ├── RESULTS_PROVENANCE.md
│   ├── REPRODUCIBILITY.md
│   └── DATASET_GUIDE.md
├── results/                   # Machine-readable CSVs, tables, and figures
│   ├── verified/              # 3-seed uniform benchmark runs
│   ├── existing/              # Legacy baseline runs
│   ├── tables/                # Markdown/LaTeX compiled tables
│   └── figures/               # High-resolution benchmark figures
├── scripts/                   # CLI entrypoints and utilities
│   ├── run_experiments.py     # Main CLI runner
│   ├── generate_tables.py     # Automated table compiler
│   ├── generate_figures.py    # Automated figure compiler
│   └── check_datasets.py      # Dataset verifier
├── src/                       # Modular source code
│   ├── models/                # EfficientNetV2-S, CBAM, Decoupled Heads
│   ├── federated/             # Aggregation (Eq. 8), Client, Server
│   ├── uncertainty/           # Temperature Scaling, APS Conformal, Metrics
│   ├── data/                  # Dataset loaders and transforms
│   └── utils/                 # Seed, Config, Logging
├── tests/                     # Unit test suite
│   ├── run_tests.py           # Self-contained test runner
│   ├── test_cbam.py
│   ├── test_model.py
│   ├── test_aggregation.py
│   ├── test_temperature_scaling.py
│   ├── test_conformal.py
│   └── test_metrics.py
├── experiment.py              # Main training pipeline
├── fedua_net.py               # Core model implementation
├── requirements.txt           # Python package dependencies
├── environment.yml            # Conda environment specification
└── README.md
```

---

## 6. Citation

If you find this repository or methodology useful in your research, please cite:

```bibtex
@article{feduanet2026,
  title={Calibrated Uncertainty in Federated Learning for Privacy-Preserving Multi-Task Medical Imaging},
  author={FedUA-Net Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 7. License
This project is licensed under the MIT License — see the LICENSE file for details.
