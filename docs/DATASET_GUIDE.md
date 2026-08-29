# Dataset Acquisition & Preparation Guide

This repository evaluates **FedUA-Net** across three distinct, publicly accessible clinical medical imaging benchmarks representing heterogeneous modalities and disjoint diagnostic categories.

---

## 1. Overview of Evaluated Benchmarks

| Hospital Node | Modality | Benchmark Dataset | Total Scans | Diagnostic Classes ($C_k$) | Primary Reference |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **Hospital A** | MRI | Brain Tumor MRI Dataset | $7{,}023$ | 4 classes: *Glioma*, *Meningioma*, *No Tumor*, *Pituitary Tumor* | Cheng et al. (2015) |
| **Hospital B** | Ultrasound | Dataset of Breast Ultrasound Images (BUSI) | $780$ | 3 classes: *Benign*, *Malignant*, *Normal* | Al-Dhabyani et al. (2020) |
| **Hospital C** | Digital X-Ray | COVID-19 Radiography Database | $21{,}165$ | 4 classes: *COVID-19*, *Lung Opacity*, *Normal*, *Viral Pneumonia* | Chowdhury et al. (2020) |

---

## 2. Directory Structure Setup

Create a `Dataset/` directory in the repository root with the following organization:

```text
Dataset/
├── Brain Tumor MRI Dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── Dataset_BUSI_with_GT/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── COVID-19_Radiography_Dataset/
    ├── COVID/
    │   └── images/
    ├── Lung_Opacity/
    │   └── images/
    ├── Normal/
    │   └── images/
    └── Viral Pneumonia/
        └── images/
```

---

## 3. Dataset Download Links & Instructions

### 1. Hospital A: Brain Tumor MRI
- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Format**: Contrast-enhanced T1-weighted axial MRI slices organized into `Training/` ($5{,}712$ images) and `Testing/` ($1{,}311$ images).
- **Split Protocol**: The $5{,}712$ training pool is partitioned into an $85\%$ local training set ($4{,}855$ images) and a $15\%$ validation set ($857$ images), while the canonical $1{,}311$ test images serve as the untouched evaluation cohort.

### 2. Hospital B: Breast Ultrasound (BUSI)
- **Source**: [Kaggle Dataset of Breast Ultrasound Images (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Format**: B-mode breast ultrasound images categorized into `benign/`, `malignant/`, and `normal/` (excluding `*_mask.png` segmentation files).
- **Split Protocol**: Stratified $70\%$ local training set ($546$ images), $15\%$ validation set ($117$ images), and $15\%$ held-out test set ($117$ images).

### 3. Hospital C: COVID-19 Radiography
- **Source**: [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Format**: Posteroanterior (PA) chest radiographs organized into `COVID/`, `Lung_Opacity/`, `Normal/`, and `Viral Pneumonia/`.
- **Split Protocol**: Stratified $70\%$ local training set ($14{,}815$ images), $15\%$ validation set ($3{,}175$ images), and $15\%$ held-out test set ($3{,}175$ images).

---

## 4. Automated Dataset Verification

After downloading and extracting the datasets into `Dataset/`, run the verification script:

```bash
python scripts/check_datasets.py
```

Expected output:
```text
======================================================================
FedUA-Net Dataset Verification Suite
======================================================================

Checking: Hospital A (Brain MRI)
  Located Path: Dataset\Brain Tumor MRI Dataset
  Images Found: 7200 (Expected >= 5000)
  [OK] Dataset validated successfully.

Checking: Hospital B (Breast Ultrasound BUSI)
  Located Path: Dataset\Dataset_BUSI_with_GT
  Images Found: 780 (Expected >= 700)
  [OK] Dataset validated successfully.

Checking: Hospital C (COVID-19 Radiography)
  Located Path: Dataset\COVID-19_Radiography_Dataset
  Images Found: 42330 (Expected >= 15000)
  [OK] Dataset validated successfully.

======================================================================
[SUCCESS] All 3 clinical datasets are verified and ready for training.
======================================================================
```
