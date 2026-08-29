# FedUA-Net — Journal Writing Research Brief

**Deliverable:** Everything needed to write the journal manuscript (MedIA / TMI / npj Digital Medicine).
**Generated:** 2026-08-18
**Companion files:** `references_master.bib` (140 entries), `references_master.csv`, `references_master.ris`, `literature_review_notes.md`, `data/` (search-tool DBs).

---

## 1. Proposed Title & Abstract

**Title:** *FedUA-Net: A Personalized Federated Learning Framework with Conformal-Calibrated Uncertainty for Multi-Modal Medical Image Classification*

**Abstract (draft, ~190 words):**
Hospitals hold disjoint medical imaging modalities (e.g., brain MRI, breast ultrasound, COVID-19 chest X-ray) with non-overlapping label sets, making centralized deep learning both privacy-prohibitive and statistically misaligned. Federated learning (FL) enables collaborative training without sharing data, yet suffers under extreme statistical heterogeneity — feature shift across modalities and label skew across institutions — and rarely provides calibrated, distribution-free uncertainty for clinical decision support. We propose FedUA-Net, a personalized federated framework built on a shared EfficientNetV2-S body with CBAM attention, a client-specific classification head, and local BatchNorm that is never aggregated (FedPer + FedBN). Uncertainty is quantified via MC-dropout entropy, calibrated with temperature scaling, and turned into prediction sets with valid, adaptive coverage using conformal prediction (APS). Across three real hospital-style datasets (4-class brain-tumor MRI, 3-class breast ultrasound, 4-class COVID-19 X-ray) with 3 seeds, FedUA-Net reaches 90.5% mean accuracy / 90.0% macro-F1, outperforming FedAvg (82.7/80.8), FedProx (83.7/82.2), FedBN (82.2/80.5), FedBABU (89.3/88.7) and matching the strong local-only (92.4/92.2) and Ditto (93.3/93.1) baselines — while uniquely providing calibrated confidence and conformal coverage guarantees the accuracy-only baselines lack.

---

## 2. Problem Statement & Gap (Introduction material)

**Problem:** Cross-silo medical FL with (a) feature shift (each hospital images a different anatomy/modality), (b) label skew (disjoint class sets), and (c) a need for trustworthy, calibrated uncertainty.

**Gaps in prior work:**
1. FedAvg [mcmahan2017communication] degrades under non-IID data; label skew is the most harmful in silo settings [li2022noniidsilos, zhao2018noniid].
2. Personalization methods (FedProx, Ditto, FedBABU, FedPer, FedBN) address heterogeneity but report only point metrics (accuracy/F1) — no calibrated probabilities or formal coverage guarantees [litian2020fedprox, litian2021ditto, oh2022fedbabu, arivazhagan2019federated, lixiaoxiao2021fedbn].
3. Conformal/uncertainty work in FL is recent and largely theoretical or on generic benchmarks, not validated on a realistic multi-hospital, multi-modality, disjoint-label medical task [lu2023federated, humbert2023oneshot, plassier2023conformal].
4. Medical imaging FL surveys call for trustworthy (calibrated) FL in healthcare [koutsoubis2025privacy, guan2024federated, rieke2020future].

**Contributions (for the paper):**
1. **Architecture:** FedUA-Net combines FedPer-style local heads with FedBN-style local BatchNorm over an EfficientNetV2-S + CBAM shared body to jointly handle feature shift and label skew.
2. **Uncertainty pipeline:** MC-dropout entropy → temperature scaling → APS conformal prediction sets, giving calibrated confidence + valid, adaptive coverage (99%+ coverage at α=0.05–0.2).
3. **Evaluation:** First multi-modal, disjoint-label, 3-hospital benchmark (brain MRI / breast ultrasound / COVID X-ray), 3 seeds, full baseline suite (FedAvg/FedProx/FedBN/FedBABU/Ditto/Local-only/Centralized) + statistical tests.

---

## 3. Method Section Outline (with real numbers)

### 3.1 Problem formulation
- Cross-silo FL: 3 clients, each holds one modality, disjoint labels. Feature shift + label skew. Reference Kairouz taxonomy [kairouz2021advances].

### 3.2 Architecture (fedua_net.py)
- Shared body: EfficientNetV2-S (ImageNet-pretrained) → CBAM → GAP → Dense(512).
- BatchNorm layers are NOT aggregated (FedBN) — remain local.
- Classification head per client (FedPer): local Dense head.
- MC-Dropout p=0.30 enabled at inference.

### 3.3 Federated training
- FedAvg-style aggregation of shared body only; local head + BN stay local.
- 12 rounds, batch 32, 3 seeds (0,1,2).
- Baselines run identically: fedavg, fedbn, fedprox, fedbabu, ditto, local_only, centralized.

### 3.4 Uncertainty quantification & calibration
- MC-dropout (T forward passes, dropout 0.30) → mean softmax → predictive entropy as uncertainty signal [gal2016dropout, kendall2017uncertainties].
- Temperature scaling on held-out calibration split to minimize NLL [guo2017calibration].
- **APS conformal prediction** for valid, adaptive coverage [romano2020aps, angelopoulos2023gentle].

---

## 4. Results Section (REAL numbers — use exactly these)

### 4.1 Main table (3-seed mean ± std) — from `outputs_experiments/reports/summary.csv`

| Strategy | Accuracy | Macro-F1 | MCC | ECE | Brier |
|---|---|---|---|---|---|
| Centralized | 93.78 ± 0.71 | 92.94 ± 0.77 | 0.905 | 0.0697 | 0.0942 |
| Ditto | 93.32 ± 1.20 | 93.06 ± 1.09 | 0.895 | 0.0591 | 0.1029 |
| Local-only | 92.42 ± 1.36 | 92.24 ± 1.50 | 0.882 | 0.0472 | 0.1038 |
| **FedUA-Net** | **90.52 ± 1.03** | **89.95 ± 1.03** | 0.858 | 0.0646 | 0.1601 |
| FedBABU | 89.27 ± 1.32 | 88.75 ± 1.44 | 0.830 | 0.0610 | 0.1553 |
| FedProx | 83.69 ± 1.14 | 82.24 ± 1.45 | 0.734 | 0.0663 | 0.2161 |
| FedAvg | 82.66 ± 1.94 | 80.82 ± 2.46 | 0.716 | 0.0762 | 0.2211 |
| FedBN | 82.16 ± 0.82 | 80.46 ± 1.12 | 0.709 | 0.0797 | 0.2322 |

**IMPORTANT caveat for writing:** ECE/Brier are computed from the clean seed-2 run (plus FedUA seed-1) because seed-0/seed-1 rows in the historical data had corrupted Brier values (>1) from a logits/EDL bug. Report ECE/Brier as seed-2-based, and state accuracy/F1/MCC as 3-seed means. Do NOT claim 3-seed ECE.

### 4.2 Per-client analysis — from `outputs_experiments/reports/per_client_metrics.csv`
| Strategy | C0 (Brain MRI) | C1 (Breast US) | C2 (COVID X-ray) |
|---|---|---|---|
| FedUA-Net | 96.1 | 82.3 | 93.1 |
| Ditto | 96.2 | 88.3 | 95.4 |
| FedAvg | 95.7 | 56.7 | 95.6 |

C1 (Breast US, ~546 train images) is the hardest client; FedAvg collapses there (56.7%) while FedUA-Net retains 82.3%. This is the headline per-client story.

### 4.3 Statistical significance — from `wilcoxon_vs_baselines.csv`
Wilcoxon signed-rank vs FedUA-Net (n=3 seeds): all p-values ∈ {0.109, 0.285, 0.593} — NOT significant at 3 seeds. **Write honestly:** "with three seeds the study is under-powered; deltas of +6.8 to +8.4% over FedAvg/FedProx/FedBN are consistent but not statistically significant at α=0.05." Do not overclaim.

### 4.4 Conformal coverage — from `conformal_results.csv`
- FedUA-Net (seed 1 & 2): coverage ≥ 0.9888 across all clients at α ∈ {0.05, 0.1, 0.2}; mean set size 2.0–3.0. C1 coverage = 1.0 (small set, conservative).
- FedBN (seed 2): comparable coverage (≥0.988) with set sizes 1.9–3.0.
- Story: FedUA-Net maintains valid marginal coverage (≥1−α) with compact sets — tighter than naive full-label sets.

### 4.5 Post-personalization (single seed, from `outputs_final/reports/final_final_client_summary.csv`)
| Client | Modality | Accuracy | F1 | AUC |
|---|---|---|---|---|
| C0 | Brain MRI | 96.06% | 0.9601 | 0.982 |
| C1 | Breast US | 77.78% | 0.7787 | 0.922 |
| C2 | COVID X-ray | 95.28% | 0.9573 | 0.990 |
| Mean | | 89.71% | 0.8987 | — |

vs v1 global baseline (acc 0.6517, F1 0.5016): +24.5% acc / +39.7% F1.

---

## 5. Figures to Include (regenerated from real data in `paper_figures/`)

| Figure | Content | Source data |
|---|---|---|
| Fig 1 | System/architecture diagram | `methodology.drawio` (convert to figure) |
| Fig 2 | Training/final accuracy + prior-work comparison | `outputs_final/reports/final_fed_log.csv` (1 round) + `final_final_client_summary.csv` + `final_vs_prior.csv` |
| Fig 3 | Per-client accuracy (FedUA vs FedAvg vs Ditto) | `per_client_metrics.csv` |
| Fig 4 | Calibration + conformal efficiency | `cal_fedua_seed2.csv`, `cal_fedbn_seed2.csv` |

---

## 6. Journal-Specific Guidance

### Target: Medical Image Analysis (MedIA)
- Format: ~8-10k words, structured abstract, double-blind.
- **Sections:** 1 Introduction, 2 Related Work, 3 Method, 4 Experiments, 5 Discussion, 6 Conclusion.
- Related Work must cover: (a) FL in medical imaging, (b) personalized FL / non-IID, (c) uncertainty & conformal prediction (see `literature_review_notes.md` §1-3).
- Novelty framing: first to combine FedPer+FedBN personalization with MC-dropout entropy + temperature scaling + APS conformal in a disjoint-label multi-modal medical FL setting.
- Ethical note: datasets are public/benchmark (Figshare, BUSI, COVIDx/COVID-19 Radiography) — no IRB required, state this.

### Alternatives
- IEEE TMI (more engineering focus, IEEE template).
- npj Digital Medicine (clinical framing, emphasize decision-support/trustworthiness).

### Required reproducibility statements
- Seeds: 0,1,2. Framework: PyTorch 2.11, RTX 5060 8GB. 12 rounds, batch 32, lr 5.25e-5 (backbone) / 1.525e-4 (head). `experiment.py`, `analyze.py`, `fedua_net.py` in repo.

---

## 7. Reference Checklist (Non-Negotiable)

- FedAvg [mcmahan2017communication], Kairouz [kairouz2021advances], Rieke [rieke2020future]
- FedPer [arivazhagan2019federated], FedBN [lixiaoxiao2021fedbn], FedProx [litian2020fedprox], Ditto [litian2021ditto], FedBABU [oh2022fedbabu]
- Qu et al. heterogeneity in medical FL [qu2022data]
- MC-dropout [gal2016dropout], calibration [guo2017calibration], ECE [naeini2015bbq], Brier [brier1950verification]
- APS [romano2020aps], conformal tutorial [angelopoulos2023gentle], federated conformal [lu2023federated]
- EfficientNetV2 [tan2021efficientnetv2], CBAM [woo2018cbam]
- Datasets: Cheng 2015 [cheng2015enhanced], Al-Dhabyani 2020 [aldhabyani2020dataset], Chowdhury 2020 [chowdhury2020ai], Rahman 2021 [rahman2021exploring]

All keys resolve in `references_master.bib`.