# FedUA-Net Paper: Corrected Table I + Reference List

Source of truth: `outputs_experiments/reports/summary.csv` (cleaned 3-seed means).
All numbers below are the real cleaned values — the draft's original Table I
used the pre-cleanup polluted values and MUST be replaced.

## Table I (corrected — 8 strategies, 3 seeds)

| Method        | Accuracy (%) | Mean F1 (%) |
|---------------|--------------|-------------|
| Centralized   | 93.8         | 92.9        |
| Ditto         | 93.3         | 93.1        |
| Local-only    | 92.4         | 92.2        |
| FedUA-Net     | 90.5         | 90.0        |
| FedBABU       | 89.3         | 88.7        |
| FedProx       | 83.7         | 82.2        |
| FedAvg        | 82.7         | 80.8        |
| FedBN         | 82.2         | 80.5        |

All rows sorted by accuracy. This is the full 8-strategy set; the draft's
"seven baselines" / "eight strategies" wording now matches the table.
Do NOT report 3-seed ECE/Brier — those are seed-2-based only
(ECE 0.047–0.080, Brier 0.094–0.232; fedua ECE 0.0646, Brier 0.1601).

## Headline numbers (verified real)

- FL-phase FedUA-Net: **90.5% acc / 90.0% F1** (3-seed mean, replaces 88.2/87.4).
- Post-fine-tuning (per-client, separate single-seed pipeline): C0 96.1%, C1 77.8%,
  C2 95.3%, **mean 89.7%** — source `outputs_final/reports/final_final_client_summary.csv`.
  Present this as a separate table; do not imply fine-tuning raised the FL-phase number.
- Conformal: FedUA-Net at α = 0.1 achieved coverage 0.989–1.000 across all three
  clients (target 90% is satisfied); mean set sizes 2.0–3.0, C1 coverage = 1.0.
  Source: `outputs_experiments/reports/conformal_results.csv`.

## Reference list (IEEE format, corrected & completed)

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas,
"Communication-Efficient Learning of Deep Networks from Decentralized Data,"
in *Proc. AISTATS*, Fort Lauderdale, FL, USA, 2017, pp. 1273–1282.

[2] X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, "FedBN: Federated Learning on
Non-IID Features via Local Batch Normalization," in *Proc. ICLR*, 2021
(arXiv:2102.07623).  ← NO page numbers; ICLR has none.

[3] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, "Federated
Learning with Personalization Layers," arXiv:1912.00818, 2019.  ← title VERIFIED correct.

[4] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning," in *Proc. ICML*, New York, NY, USA, 2016,
pp. 1050–1059.

[5] A. N. Angelopoulos and S. Bates, "Conformal Prediction: A Gentle Introduction,"
*Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494–591, 2023,
doi: 10.1561/2200000101.

### Missing references — add these

[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern
Neural Networks," in *Proc. ICML*, Sydney, Australia, 2017, pp. 1321–1330
(arXiv:1706.04599).  ← for temperature scaling

[7] Y. Romano, M. Sesia, and E. Candès, "Classification with Valid and Adaptive
Coverage," in *Proc. NeurIPS*, vol. 33, 2020, pp. 3581–3591 (arXiv:2006.02544).
← the actual APS method (draft [5] is only the tutorial)

[8] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training,"
in *Proc. ICML*, 2021, pp. 10096–10106 (arXiv:2104.00298).  ← backbone

[9] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block
Attention Module," in *Proc. ECCV*, 2018, pp. 3–19,
doi: 10.1007/978-3-030-01234-2_1.  ← attention module

[10] J. Cheng, W. Huang, S. Cao, R. Yang, W. Yang, Z. Yun, Z. Wang, and Q. Feng,
"Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation
and Partition," *PLOS ONE*, vol. 10, no. 10, e0140381, 2015,
doi: 10.1371/journal.pone.0140381.  ← brain MRI dataset

[11] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, "Dataset of Breast
Ultrasound Images," *Data Brief*, vol. 28, p. 104863, 2020,
doi: 10.1016/j.dib.2019.104863.  ← breast US dataset

[12] M. E. H. Chowdhury et al., "Can AI Help in Screening Viral and COVID-19
Pneumonia?," *IEEE Access*, vol. 8, pp. 132665–132676, 2020,
doi: 10.1109/ACCESS.2020.3010287.  ← chest X-ray dataset

[13] T. Rahman et al., "Exploring the Effect of Image Enhancement Techniques on
COVID-19 Detection using Chest X-Ray Images," *Comput. Biol. Med.*, vol. 132,
p. 104319, 2021, doi: 10.1016/j.compbiomed.2021.104319.  ← chest X-ray dataset (alt.)

[14] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
"Federated Optimization in Heterogeneous Networks," in *Proc. MLSys*, 2020
(arXiv:1812.06127).  ← FedProx baseline

[15] T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and Robust Federated
Learning Through Personalization," in *Proc. ICML*, 2021, pp. 6357–6368
(arXiv:2012.04221).  ← Ditto baseline

[16] J. Oh, S. Kim, and S.-Y. Yun, "FedBABU: Toward Enhanced Representation for
Federated Image Classification," in *Proc. ICLR*, 2022 (arXiv:2106.06042).
← FedBABU baseline

Optional: B. McMahan et al. communication-efficiency companion (1602.05629) is
not needed; Kairouz et al. 2021 "Advances and Open Problems in Federated Learning"
(arXiv:1912.04977) if you want a broad FL survey citation in the intro.

## Citation-order note

Numbered IEEE references must appear in order of first citation. Current draft
first-cites [1] (intro), [3] (personalization heads), [2] (BN), [4] (MC dropout),
[5] (conformal). Renumber so that first-appearance order is [1],[2],[3],[4],[5]
— i.e. the reference list order must follow in-text appearance, not topic.