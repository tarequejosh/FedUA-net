# FedUA-Net — Current Progress & Resume Guide

**Last updated:** 2026-08-14
**Stack:** PyTorch 2.11.0+cu128 | RTX 5060 8GB | conda env `research` (Python 3.11)
**For full context read README.md first.**

---

## Status: Phase A (Experiments) — COMPLETE (data cleaned)

### Done
- [x] v4 architecture trained (10 rounds + fine-tune) → results in `outputs_v4/`
- [x] Full 3-seed experiment completed across all baselines → `outputs_experiments/raw/`
- [x] Raw data pollution (accumulation bug in `experiment.py`) fixed and files cleaned
      → only real 3-client rows retained; ensemble dropped (no real 3-client run)
- [x] Reports regenerated from cleaned data → `outputs_experiments/reports/`
- [x] Paper figures regenerated from real data → `paper_figures/`

### Re-running the (already-completed) experiment
```bash
# Full 3-seed experiment (resume-safe, skip already-done seeds)
conda run -n research python experiment.py \
  --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua \
  --seeds 0 1 2 --rounds 12 --batch 32 --resume
```
Expected runtime: ~10-12 hours on RTX 5060.

### After an experiment run
```bash
# Aggregate + Wilcoxon tests
conda run -n research python analyze.py

# LOCO (leave-one-client-out)
conda run -n research python experiment.py \
  --strategies fedua --seeds 0 1 2 --rounds 12 --batch 32 --loco --resume
```

---

## Key Numbers (v4 final, single seed)

| Client | Acc | F1 | AUC |
|--------|-----|----|-----|
| C0 Brain MRI | 0.9606 | 0.9601 | 0.982 |
| C1 Breast US | 0.7778 | 0.7787 | 0.922 |
| C2 COVID X-ray | 0.9528 | 0.9573 | 0.990 |
| **Mean** | **0.8971** | **0.8987** | |

vs v1 global baseline (acc=0.6517, F1=0.5016): **+24.5% acc / +39.7% F1**

---

## Decisions Made

1. **v5 (DINOv2 + LoRA) dropped** — poor results on MedMNIST clients, adds complexity
2. **3-client paper only** — Brain MRI + Breast US + COVID X-ray
3. **Paper angle:** Calibrated uncertainty (temperature scaling + conformal prediction)
   differentiates FedUA from Ditto even if accuracy is tied
4. **Target journal:** Medical Image Analysis (IF~13) as primary

---

## Timeline Log (condensed)

- v1 TF implementation (archived, superseded)
- v4 PyTorch rewrite: FedPer + FedBN + CBAM + MC-Dropout on EfficientNetV2-S
- 2026-08-14: Project cleaned, v5 dropped, tier1_v2.py fixed for 3-client mode
