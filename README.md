# FedUA-Net — Project README

> **For AI models:** Read this file first. It is the authoritative guide to this project.
> Everything you need is described here. Do NOT infer project state from stale filenames — read this.

---

## What This Project Is

**FedUA-Net** = **Fed**erated **U**ncertainty-**A**ware **Net**work for multi-modal medical image classification.

A cross-silo federated learning system where 3 hospitals each hold a completely different imaging modality with disjoint label sets:

| Client | Hospital | Modality | Classes | Train images |
|--------|----------|----------|---------|-------------|
| C0 | Hospital_A | Brain-Tumor MRI | 4 (bt_glioma, bt_meningioma, bt_notumor, bt_pituitary) | ~4,760 |
| C1 | Hospital_B | Breast Ultrasound (BUSI) | 3 (bu_benign, bu_malignant, bu_normal) | ~546 |
| C2 | Hospital_C | COVID-19 X-Ray | 4 (cr_covid, cr_lung_opacity, cr_normal, cr_pneumonia) | ~14,815 |

**Total: 11 classes across 3 modalities.** Each client sees only its own modality — extreme statistical heterogeneity (feature-shift + label-skew non-IID).

**Goal:** Publish a Q1 journal paper (target: Medical Image Analysis or IEEE TMI).

---

## Current Status (as of 2026-08-14)

### COMPLETED
- Final architecture implemented and trained: `fedua_net.py`
  - EfficientNetV2-S backbone + CBAM + GAP + PReLU + MC-Dropout(0.30)
  - FedPer: per-client LocalHead (shared body federated, heads are local)
  - FedBN-style: BatchNorm excluded from server averaging
  - 10 federated rounds + post-personalization fine-tune
- Final results (in `outputs_final/`):
  - C0 Brain MRI:   acc=0.9606, F1=0.9601, AUC=0.982
  - C1 Breast US:   acc=0.7778, F1=0.7787, AUC=0.922
  - C2 COVID X-ray: acc=0.9528, F1=0.9573, AUC=0.990
  - Mean: acc=0.8971, macro-F1=0.8987
  - vs v1 global baseline: +24.5% acc, +39.7% F1
- Tier-1 experiment harness implemented: `experiment.py`
  - 8 strategies: FedAvg, FedBN, FedProx, FedBABU, Ditto, Local-only, Centralized, FedUA
  - Multi-seed (3 seeds), temperature scaling, conformal prediction (APS), risk-coverage curves
  - --resume flag to restart interrupted runs safely
  - Smoke-tested PASS (2026-08-14, runs clean on RTX 5060 CUDA)
- Partial baseline results from seeds {0,1} already in `outputs_experiments/raw/`

### IN PROGRESS / TODO
1. Run full 3-seed experiment (experiment.py, seeds 0-2, 12 rounds, ~10-12 GPU hours)
2. Run Wilcoxon analysis (analyze.py after raw/ is fully populated)
3. LOCO run (--loco flag, fedua strategy, 3 seeds)
4. Write the paper (all sections, figures, LaTeX)

---

## Architecture

```
Input 224x224x3
    |
    v
EfficientNetV2-S (ImageNet pretrained)  <- shared body (federated, BN excluded)
    |
    v
CBAM (Channel + Spatial Attention)       <- shared body
    |
    v
Global Average Pooling -> Dense(512) + PReLU -> MC-Dropout(0.30)
    |
    v
LocalHead (per-client softmax)           <- NOT federated, stays local
```

Aggregation: Weighted FedAvg on body weights only (BN layers excluded = FedBN-style).
Personalization: Post-FL per-client fine-tune of both body + head at lower LR.
Uncertainty: MC-Dropout with entropy as uncertainty signal; calibrated with temperature scaling.

---

## File Map

```
FedUA-Net/
|
+-- fedua_net.py     CORE: architecture, data loading, training primitives
|                                Classes: SharedBody, LocalHead, ClientNet, CBAM
|                                Functions: discover_all(), build_loaders(), client_meta()
|                                Config: cfg (dataclass with all hyperparameters)
|
+-- experiment.py                 MAIN EXPERIMENT HARNESS (run this for paper results)
|                                Implements all 8 strategies under one fixed architecture
|                                Writes per-(seed,strategy) CSVs to outputs_experiments/raw/
|                                Also runs: temperature scaling, conformal APS, risk-coverage
|                                KEY ARGS:
|                                  --strategies fedavg fedbn fedprox fedbabu ditto
|                                              local_only centralized fedua
|                                  --seeds 0 1 2
|                                  --rounds 12
|                                  --batch 32
|                                  --loco          (leave-one-client-out)
|                                  --resume        (restart-safe)
|
+-- analyze.py            ANALYSIS: aggregate raw/ -> summary + Wilcoxon p-values
|                                Run AFTER experiment.py completes
|                                Writes to outputs_experiments/reports/
|
+-- personalize_finetune.py     Post-FL per-client fine-tune script
|                                Loads checkpoint_r10.pt, fine-tunes, saves finetuned.pt
|                                Also generates final report + figures
|                                NOTE: needs workers=0 on Windows (spawn issue)
|
+-- parse_curves.py             Utility: rebuild training-curves figure from log
|
+-- baseline_report.txt   v1 TF baseline (FedAvg global, acc=0.6517)
|                                     KEEP: cited as prior work in the paper
|
+-- outputs_final/                 AUTHORITATIVE v4 results
|   +-- models/
|   |   +-- checkpoint_r10.pt           Final federated phase (10 rounds)
|   |   +-- fedua_net_finetuned.pt   Final published weights (post fine-tune)
|   +-- reports/
|   |   +-- final_report.txt         KEY RESULTS FILE (acc, F1, AUC per client)
|   |   +-- final_client_summary.csv Per-client summary
|   |   +-- final_per_class.csv      Per-class breakdown
|   |   +-- final_uncertainty.csv    MC entropy per sample
|   |   +-- final_vs_prior.csv             Comparison with v1 baseline
|   |   +-- final_fed_log.csv              Per-round training curves
|   |   +-- final_meta.json                Run metadata
|   +-- figures/
|       +-- final_training_curves.png      Per-round per-client accuracy plot
|
+-- outputs_experiments/              GROWING: multi-seed baseline ladder results
|   +-- raw/                    Per-(strategy,seed) CSVs from experiment.py
|   |   +-- raw_{strategy}_seed{n}.csv  Main metrics (acc,f1,mcc,auc,ece,brier)
|   |   +-- cal_fedua_seed{n}.csv       Calibration + conformal + risk-coverage
|   |   +-- cal_fedbn_seed{n}.csv
|   |   +-- loco_seed{n}.csv            LOCO results (if --loco was run)
|   +-- reports/                Generated by analyze.py
|       +-- summary.csv                 Mean+-std per strategy (main paper table)
|       +-- per_client_metrics.csv      Per-client breakdown
|       +-- wilcoxon_vs_baselines.csv   p-values (FedUA vs each baseline)
|       +-- conformal_results.csv       Coverage + set size per client
|       +-- tier1_report.txt            Human-readable full report
|
+-- PROGRESS.md                 Human notes on project history (kept for context)
|
+-- Dataset/                    Raw image data (not tracked in git)
    +-- Brain_Tumor_MRI/        C0 data
    +-- BUSI/                   C1 data
    +-- COVID-19_Radiography/   C2 data
```

---

## Runtime Environment

```
OS:     Windows 11
GPU:    NVIDIA GeForce RTX 5060 8GB (CUDA 13.2, driver 595.79)
Python: conda env "research"
        C:/Users/tareq/miniconda3/envs/research/python.exe
        Python 3.11, PyTorch 2.11.0+cu128, pandas 3.0.3

Launch pattern:
  conda run -n research python experiment.py [args]

CRITICAL Windows notes:
  - num_workers must be <= 4 (8 -> WinError 1455 paging file error)
  - Scripts with DataLoader must use  if __name__ == '__main__':  guard
  - AMP (fp16) is DISABLED -- causes NaN in roc_curve; use fp32 only
  - Use conda run, NOT Start-Process (wrong python resolution)
```

---

## Key Results Reference

### Final Results (post fine-tune, single seed)

| Client | Acc | Precision | Recall | F1 | AUC |
|--------|-----|-----------|--------|-----|-----|
| C0 Brain MRI | 0.9606 | 0.962 | 0.9606 | 0.9601 | 0.982 |
| C1 Breast US | 0.7778 | 0.763 | 0.8055 | 0.7787 | 0.922 |
| C2 COVID X-ray | 0.9528 | 0.967 | 0.9490 | 0.9573 | 0.990 |
| Mean | 0.8971 | | | 0.8987 | |

### Baseline Ladder (2-seed partial, from outputs_experiments/raw/)

| Method | Mean Acc | Mean F1 | Mean MCC |
|--------|----------|---------|---------|
| Ditto | 0.932 | 0.929 | 0.891 |
| Centralized (upper bound) | 0.930 | 0.921 | 0.889 |
| Local-only | 0.917 | 0.915 | 0.872 |
| FedUA-Final | 0.917 | 0.911 | 0.872 |
| FedBABU | 0.890 | 0.884 | 0.825 |
| FedProx | 0.834 | 0.819 | 0.727 |
| FedBN | 0.821 | 0.807 | 0.712 |
| FedAvg | 0.816 | 0.795 | 0.697 |

WARNING: Ditto currently ties/beats FedUA on raw accuracy (2 seeds only).
Paper angle: FedUA = competitive accuracy + calibrated uncertainty (temperature scaling)
+ conformal prediction sets with coverage guarantee. Ditto has no UQ story.
Full 3-seed run + Wilcoxon tests needed to confirm statistical significance.

### v1 Baseline (prior work, for paper comparison)
- Global FedAvg (11-class fused head): acc=0.6517, macro-F1=0.5016
- MobileNetV2 centralized: acc=0.7643, macro-F1=0.7133

---

## Paper Plan

Title: Personalized Federated Learning with Calibrated Uncertainty for
       Heterogeneous Multi-Modal Medical Image Classification

Target: Medical Image Analysis (IF~13) or IEEE TMI (IF~10)

Three contributions:
  1. FedPer + FedBN on disjoint-label cross-modality FL verified against
     6 baselines with Wilcoxon p-values
  2. Per-client temperature scaling + conformal APS prediction sets with
     >=90% coverage guarantee
  3. LOCO evaluation (unseen hospital generalization)

### Next immediate commands to run:

```bash
# 1. Full 3-seed experiment (~10-12 hours)
conda run -n research python experiment.py --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua --seeds 0 1 2 --rounds 12 --batch 32 --resume

# 2. Analysis (after step 1 completes)
conda run -n research python analyze.py

# 3. LOCO generalization
conda run -n research python experiment.py --strategies fedua --seeds 0 1 2 --rounds 12 --batch 32 --loco --resume
```

---

## What NOT To Do

- Do NOT edit fedua_net.py architecture -- it is the stable published baseline
- Do NOT re-run personalize_finetune.py -- final weights already saved as fedua_net_finetuned.pt
- Do NOT add MedMNIST (derma/retina) clients -- v5 was dropped; this is a 3-client paper
- Do NOT use Start-Process to launch Python on Windows -- use conda run -n research python
- Do NOT enable AMP/fp16 -- causes NaN in evaluation metrics
- Do NOT set num_workers > 4 -- causes WinError 1455 on Windows
- Do NOT delete baseline_report.txt -- it is cited as prior work

---

## Uncertainty Interpretation (for paper)

MC-Dropout entropy correctly correlates with prediction errors.
Columns in final_uncertainty.csv: correct=0 means WRONG prediction,
correct=1 means RIGHT prediction.
Correct samples (col=1) have LOWER entropy than incorrect (col=0) on all clients.
This entropy-based uncertainty can gate referrals/abstention -- strong clinical story.
