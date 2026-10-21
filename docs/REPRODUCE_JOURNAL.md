# Reproducing the Journal Results

This guide covers the **journal extension**: the 5-client benchmark,
federated conformal calibration, the OOD brain-MRI stress test, and the
leakage and conformal-score audit. For the conference-paper pipeline, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the history of every change
and why it was made, see [JOURNAL_PROVENANCE.md](../JOURNAL_PROVENANCE.md)
and [REVIEW_PROVENANCE.md](../REVIEW_PROVENANCE.md).

There are three entry points, from cheapest to most expensive:

| Goal | Needs | Time (RTX 5060, 8 GB) |
|---|---|---|
| Inspect the reported numbers | this repo only | none |
| Re-run the analysis from the saved per-image model outputs | repo + datasets | ~5-15 min, no GPU |
| Re-score from the released checkpoints | + checkpoints, GPU | ~1-2 h |
| Retrain everything from scratch | + GPU | ~32 h (3-client) + ~69 h (5-client) |

---

## 1. What is in this repository

```
outputs_journal_3client_v2/        # 3 clients, 8 strategies x 3 seeds   <- current results
outputs_journal_5client_v2/        # 5 clients, 5 strategies x 5 seeds + OOD   <- current results
  raw/                             # per-(strategy, seed) CSVs written by experiment.py
  reports/                         # aggregated tables (analyze.py, report_federated_conformal.py)
  rescore/                         # leakage + conformal-score audit (rescore_from_checkpoints.py)
    logits/{strategy}_seed{s}_client{c}.npz   # cached temp/conf/test (+OOD) logits and labels
    repro_check.csv                # re-inference vs. original run (see section 6)
    leakage_summary.csv, leakage_audit.json
    accuracy_by_protocol*.csv      # 'standard' vs 'dedup' protocol
    significance_by_protocol.csv
    conformal_by_score*.csv        # aps / aps_rand / lac, per strategy and per client
    federated_conformal_comparison_by_score.csv
    ood_summary.csv, ood_rows.csv  # 5-client only
outputs_journal_3client/           # SUPERSEDED pre-fix run (val/calib reuse), kept for diffing
outputs_journal_5client/           # SUPERSEDED pre-fix run, kept for diffing
review_retrain_{3,5}client.log     # training logs of the _v2 runs
review_analyze_{3,5}client.log     # analysis logs of the _v2 runs
```

**Checkpoints are not in git.** There are 200 files of about 86 MB each
(~16 GB), named `final_models/{strategy}_seed{seed}_client{c}.pt`. See
[section 5](#5-checkpoints).

---

## 2. Environment

Tested with Python 3.12.10, torch 2.11.0+cu128, torchvision 0.26.0,
numpy 2.5.1, scikit-learn 1.8.0, pandas 3.0.2, scipy 1.17.1, on Windows 11.

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
python -m pytest tests/ -q             # 41 tests, ~20 s, no data or GPU needed
```

`--workers 0` (the default) is the only DataLoader setting verified stable
on the reference machine; see the `--workers` help text in `experiment.py`.

---

## 3. Data layout

All paths are relative to `./Dataset/`. Clients A-C are as in
[DATASET_GUIDE.md](DATASET_GUIDE.md).

```
Dataset/
  Brain Tumor MRI Dataset/{Training,Testing}/{glioma,meningioma,notumor,pituitary}/   # Hospital A
  Dataset_BUSI_with_GT/{benign,malignant,normal}/                                      # Hospital B
  COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/     # Hospital C
  medmnist/dermamnist_224.npz                                                          # Hospital D
  medmnist/retinamnist_224.npz                                                         # Hospital E
  brisc2025/classification_task/test/{glioma,meningioma,no_tumor,pituitary}/           # OOD (see caveat)
  PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw/{512Glioma,512Meningioma,512Normal,512Pituitary}/  # OOD
```

The MedMNIST files are the official 224x224 MedMNIST+ releases. They can be
fetched with the `medmnist` package (`pip install medmnist`):

```python
from medmnist import DermaMNIST, RetinaMNIST
for cls in (DermaMNIST, RetinaMNIST):
    cls(split='train', download=True, size=224, root='Dataset/medmnist')
```

Splits are rebuilt deterministically from the seed (`experiment.build_data`).
Hospitals A-C use `train_test_split(random_state=seed)`; D and E use the
official MedMNIST train/val/test split. Every client's validation pool is
then split 50/25/25 into `val` (checkpoint selection), `temp` (temperature
scaling) and `conf` (conformal calibration).

> **Data caveat (read before using the OOD numbers).** 857/1000 BRISC2025
> test images and 849/1505 PMRAM images duplicate Hospital A images. 27%
> of Hospital A's test set duplicates its own training set. See
> `rescore/leakage_summary.csv` and REVIEW_PROVENANCE.md (v3). Use the
> `dedup` protocol numbers.

---

## 4. Commands

### 4a. Retrain (writes `raw/`, and `final_models/` when `--save_final_models` is set)

```bash
python experiment.py --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua \
  --seeds 0 1 2 --rounds 12 --batch 32 --num_clients 3 --personalize_deep \
  --federated_conformal --save_final_models --out ./outputs_journal_3client_v2

python experiment.py --strategies fedbn fedprox ditto local_only fedua \
  --seeds 0 1 2 3 4 --rounds 12 --batch 32 --num_clients 5 --personalize_deep \
  --federated_conformal --ood_eval brisc2025 pmram --save_final_models \
  --out ./outputs_journal_5client_v2
```

`--resume` skips any (strategy, seed) whose `raw/raw_{strategy}_seed{seed}.csv`
already exists. Note that `centralized` saves no per-client checkpoint.

### 4b. Aggregate reports (`reports/`)

```bash
for d in outputs_journal_3client_v2 outputs_journal_5client_v2; do
  python scripts/report_federated_conformal.py --out ./$d
  python analyze.py --out ./$d
done
```

### 4c. Leakage and conformal-score audit (`rescore/`)

```bash
python scripts/rescore_from_checkpoints.py --out ./outputs_journal_3client_v2 --num_clients 3 \
  --strategies fedavg fedbn fedprox fedbabu ditto local_only fedua --seeds 0 1 2

python scripts/rescore_from_checkpoints.py --out ./outputs_journal_5client_v2 --num_clients 5 \
  --strategies fedbn fedprox ditto local_only fedua --seeds 0 1 2 3 4 --ood_eval brisc2025 pmram
```

If `rescore/logits/*.npz` is present (it is, in this repo), the script
**skips inference** and needs neither checkpoints nor a GPU. It still needs
the datasets, to rebuild the splits and to hash images for the
near-duplicate check. Delete a logits file to force re-inference from its
checkpoint. Image hashes are cached in `rescore/dhash_store.npz`, which is
not committed because it holds machine-specific paths.

Protocols: `standard` uses the splits exactly as trained. `dedup` drops
every `temp`/`conf`/`test` image with a near-duplicate (16x16 dHash, at
most 10/256 bits) in the same client's `train` split before fitting T,
fitting the conformal threshold and evaluating. It also drops OOD images
that duplicate any Hospital A image.

Scores: `aps` is the original deterministic APS (conservative). `aps_rand`
is randomized APS and `lac` is 1 - p_y; both are exact and give near-nominal
coverage. All three are in `src/uncertainty/federated_conformal.py`.

---

## 5. Checkpoints

Each `final_models/{strategy}_seed{seed}_client{c}.pt` is the `state_dict`
of that client's full `fedua_net.ClientNet` (shared body + local head)
after training. Client indices are 0 = A Brain MRI, 1 = B BUSI, 2 = C COVID,
3 = D DermaMNIST, 4 = E RetinaMNIST. Load one with:

```python
import torch, experiment as ex
ex.ARGS.num_clients = 5                       # 3 for the 3-client run
_, _, _, client_dfs = ex.build_data(seed)
ncls = int(client_dfs[c]['local'].max()) + 1
net = ex.new_client_net(ncls, seed)
net.load_state_dict(torch.load(path, map_location=ex.DEV)); net.eval()
```

**Download:** the checkpoints are not public yet, but are available from
the authors on request. Once you have them, unpack them into
`outputs_journal_{3,5}client_v2/final_models/`. Until then, section 4c
still runs end-to-end from the cached logits in `rescore/logits/`.

---

## 6. Expected reproduction tolerance

`rescore/repro_check.csv` compares re-inference from the checkpoints with
the original training-time numbers:
- **3-client:** `n_calib` is identical everywhere, and accuracy is within
  one image per client (cuDNN nondeterminism).
- **5-client:** 122 of 125 rows match to the same tolerance. The other 3
  are knife-edge cases:
  - two where the ECE-based temperature fallback flips between T = 1 and
    the fitted T (the two ECE values are nearly tied);
  - one RetinaMNIST case (n_conf = 30) where the deterministic-APS
    quantile sits on a near-tie.

Retraining from scratch on different hardware or drivers will not be
bit-identical. Expect seed-level accuracy to move within the reported
standard deviations, not exactly.
