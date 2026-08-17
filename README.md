# FedUA-Net: Federated Uncertainty-Aware Network

FedUA-Net is a cross-silo federated learning system designed for multi-modal medical image classification. It enables collaborative learning across different hospitals, each holding a distinct imaging modality with disjoint label sets, while preserving data privacy and quantifying uncertainty.

## Datasets & Modalities
The system integrates 11 classes across 3 disjoint modalities, simulating extreme statistical heterogeneity (feature-shift + label-skew non-IID):

| Client | Hospital | Modality | Classes | Train Images |
|--------|----------|----------|---------|-------------|
| **C0** | Hospital A | Brain-Tumor MRI | 4 (glioma, meningioma, notumor, pituitary) | ~4,760 |
| **C1** | Hospital B | Breast Ultrasound (BUSI) | 3 (benign, malignant, normal) | ~546 |
| **C2** | Hospital C | COVID-19 X-Ray | 4 (covid, lung_opacity, normal, pneumonia) | ~14,815 |

## Architecture
FedUA-Net utilizes a **FedPer + FedBN** inspired architecture:
1. **Shared Body (Federated):** EfficientNetV2-S (ImageNet pretrained) + CBAM (Channel/Spatial Attention) + GAP + Dense(512). BatchNorm layers are excluded from server averaging to handle domain shift.
2. **Local Head (Client-Specific):** A dedicated classification head that remains local to each hospital.
3. **Uncertainty Calibration (Post-FL):** MC-Dropout (0.30) entropy is used as an uncertainty signal, calibrated with temperature scaling and Conformal Prediction (APS) for robust prediction sets.

## Quickstart & Execution

### Requirements
- **OS:** Windows 11
- **Hardware:** NVIDIA GPU (e.g., RTX 5060 8GB) with CUDA 13.2
- **Environment:** Python 3.11, PyTorch 2.11.0+cu128, pandas 3.0.3

### Running Experiments
The `experiment.py` script serves as the main harness to reproduce results across 8 baseline strategies:

```bash
# 1. Full 3-seed multi-strategy experiment
conda run -n research python experiment.py --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua --seeds 0 1 2 --rounds 12 --batch 32 --resume

# 2. Aggregate Results & Analysis
conda run -n research python analyze.py

# 3. Leave-One-Client-Out (LOCO) Generalization Run
conda run -n research python experiment.py --strategies fedua --seeds 0 1 2 --rounds 12 --batch 32 --loco --resume
```

> **Note for Windows Users:** 
> - Ensure `num_workers <= 4` to prevent `WinError 1455`.
> - Do not enable AMP (fp16) as it causes NaN in evaluation metrics.

## 4. Results

**FedUA-Net achieved:**
- **Accuracy:** 88.2%
- **Mean F1:** 87.4%

It outperformed several federated learning methods:
- **FedBABU:** 87.4%
- **FedProx:** 81.8%
- **FedBN:** 80.3%
- **FedAvg:** 79.2%

*Note: Local-only (91.7%) and Ditto (92.2%) had higher accuracy, but they do not provide the same uncertainty and coverage approach as FedUA-Net.*

**Final Federated Results Breakdown (10 rounds + personalization):**
| Client | Modality | Accuracy | F1 Score | AUC |
|--------|----------|----------|----------|-----|
| C0 | Brain MRI | 96.06% | 0.9601 | 0.982 |
| C1 | Breast US | 77.78% | 0.7787 | 0.922 |
| C2 | COVID X-ray | 95.28% | 0.9573 | 0.990 |
| **Mean**| - | **89.71%** | **0.8987** | - |

*(Baseline comparisons, conformal coverage guarantees, and statistical significance tests are generated dynamically in `outputs_experiments/reports/` after running the analysis suite).*
