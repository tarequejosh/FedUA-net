# Reproducibility & Benchmark Execution Guide

This document provides complete instructions for executing, evaluating, and replicating benchmarks in **FedUA-Net** (*Calibrated Uncertainty in Federated Learning for Privacy-Preserving Multi-Task Medical Imaging*).

---

## 1. Environment & Hardware Prerequisites

- **Python Version**: Python 3.10+ (tested on Python 3.11 / 3.12)
- **Core Dependencies**: PyTorch $\ge 2.0.0$, Torchvision $\ge 0.15.0$, NumPy, SciPy, Scikit-Learn, Pandas, Matplotlib, Seaborn, PyYAML.
- **Hardware Recommended**: NVIDIA GPU with $\ge 8$ GB VRAM (e.g., RTX 3060/4060/5060 or V100/A100) with CUDA 11.8+.
- **CPU Mode**: CPU execution is fully supported for testing, unit tests, and smoke sweeps.

### Quick Setup:
```bash
git clone https://github.com/tarequejosh/FedUA-net.git
cd FedUA-net
pip install -r requirements.txt
```

---

## 2. Fast Unit Tests & Smoke Verification

Verify all core algorithmic modules (CBAM, Decoupled Heads, BN Isolation, Uniform Aggregation Eq. 8, Temperature Scaling, APS Conformal Quantiles, ECE, Risk-Coverage) in $<3$ seconds without downloading datasets:

```bash
python tests/run_tests.py
```

Run a fast federated training smoke test (1 round, 1 epoch):
```bash
python scripts/run_experiments.py --smoke
```

---

## 3. Dataset Preparation Verification

Verify local medical imaging dataset integrity:
```bash
python scripts/check_datasets.py
```
*(See `docs/DATASET_GUIDE.md` for download links and partition details).*

---

## 4. Running Experiments & Benchmarks

### A. Full Multi-Task Benchmark (Table I & Table II, 8 Methods x 3 Seeds)
To train all 8 evaluated strategies (FedUA-Net, Local-Only, Ditto, Centralized, FedAvg, FedBN, FedBABU, FedProx) across seeds `0, 1, 2` under uniform server weighting:
```bash
python scripts/run_experiments.py --all --rounds 12 --batch_size 32 --lr 1e-4 --agg_weight_type uniform --seeds 0 1 2 --output_dir results/reproduced_main
```

### B. Single Strategy Training (e.g., FedUA-Net, Seed 0)
```bash
python scripts/run_experiments.py --method fedua --seed 0 --rounds 12 --agg_weight_type uniform --output_dir results/raw
```

### C. Data Scarcity Matrix on Hospital B (Table IV: N=200 & N=100)
```bash
# N=200 Scarcity Evaluation
python experiment.py --run_all_baselines --rounds 12 --hospital_b_subset_size 200 --seeds 0 1 2 --agg_weight_type uniform --output_dir results/verified/scarcity_200

# N=100 Extreme Scarcity Evaluation
python experiment.py --run_all_baselines --rounds 12 --hospital_b_subset_size 100 --seeds 0 1 2 --agg_weight_type uniform --output_dir results/verified/scarcity_100
```

---

## 5. Compiling Tables & Figures

Compile all tables and generate publication-quality figures directly from stored machine-readable CSVs:

```bash
# Generate Table I, Table II, Table III, and Table IV
python scripts/generate_tables.py --results_dir results/verified/main_uniform --base_results_dir results/verified --output_dir results/tables

# Generate Figure 2, Figure 3, Figure 4, and Figure 5
python scripts/generate_figures.py --results_dir results/verified/main_uniform --output_dir results/figures
```

Generated outputs will be saved to:
- `results/tables/table1_main_benchmark.md`
- `results/tables/table2_per_client.md`
- `results/tables/table3_ablation.md`
- `results/tables/table4_scarcity.md`
- `results/figures/fig2_main_benchmark.png`
- `results/figures/fig3_calibration.png`
- `results/figures/fig4_conformal_efficiency.png`
- `results/figures/fig5_risk_coverage.png`

---

## 6. Results Provenance Matrix

| Table / Metric | Provenance Status | Stored Evidence Location |
| :--- | :---: | :--- |
| **Table I: Main Benchmark (8 Methods)** | `VERIFIED_REPRODUCED` | `results/verified/main_uniform/reports/summary.csv` |
| **Table II: Per-Client Accuracy** | `VERIFIED_REPRODUCED` | `results/verified/main_uniform/reports/per_client_metrics.csv` |
| **Table III: Factorial Ablation Matrix** | `VERIFIED_REPRODUCED` | `results/tables/table3_ablation.md` |
| **Table IV: Data Scarcity (N=546/200/100)**| `VERIFIED_REPRODUCED` | `results/verified/scarcity_200/` & `scarcity_100/` |
| **Figure 2: Benchmark Comparison** | `VERIFIED_REPRODUCED` | `results/figures/fig2_main_benchmark.png` |
| **Figure 3: ECE Calibration** | `VERIFIED_REPRODUCED` | `results/figures/fig3_calibration.png` |
| **Figure 4: Conformal Efficiency** | `VERIFIED_REPRODUCED` | `results/figures/fig4_conformal_efficiency.png` |
| **Figure 5: Risk-Coverage Curve** | `VERIFIED_REPRODUCED` | `results/figures/fig5_risk_coverage.png` |

---

## 7. Verification & Quality Checklist

- [x] All paths use relative project roots.
- [x] No private patient data or medical images tracked in Git.
- [x] All random seeds, dataset splits, and evaluation quantiles deterministically documented.
- [x] Unit test suite runs out-of-the-box with standard Python `unittest`.
