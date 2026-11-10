# FedUA-Net: Uncertainty-Aware Federated Learning for Multi-Modal Medical Imaging

Md Tareque Jamil Josh, Fernanda Miyuki Yamada, Gabriel Peixoto de Carvalho, Hiroki Takahashi
*International Workshop on Advanced Imaging Technology (IWAIT) 2027, Proc. SPIE*

This repository has the code, trained models and result files for the IWAIT 2027 paper. Every number, table and figure in the paper can be regenerated from it.

- **Paper:** [`paper/camera_ready/main.pdf`](paper/camera_ready/main.pdf), 6 pages
- **One-page abstract:** [`paper/one_page_abstract/`](paper/one_page_abstract/)

## Overview

Three hospitals train together without sharing images. Each hospital uses a different imaging modality and has its own diagnostic labels:

| Site | Modality | Dataset | Classes |
|---|---|---|---|
| A | Brain MRI | Brain Tumor MRI (Nickparvar) | 4 |
| B | Breast ultrasound | BUSI (Al-Dhabyani et al.) | 3 |
| C | Chest X-ray | COVID-19 Radiography (Chowdhury et al.; Rahman et al.) | 4 |

Together the three sites have 29,145 images and 11 classes.

How FedUA-Net works:
- The sites share an EfficientNetV2-S backbone with CBAM attention.
- Each site keeps its BatchNorm layers and classifier private.
- The server weights every hospital equally.
- After federation, each site fine-tunes the model on its own data and calibrates it with temperature scaling.
- Per-site conformal prediction then either reports a diagnosis or refers the scan to a clinician.

Main results (3 seeds):

| | Mean acc. | Ultrasound site | ECE raw → TS |
|---|---|---|---|
| FedAvg | 92.36 ± 0.92 | 85.47 | |
| FedBN | 92.34 ± 0.99 | 85.47 | |
| Ditto | 93.93 ± 0.51 | 90.60 | |
| **FedUA-Net** | **93.30 ± 1.13** | **88.60 ± 3.56** | 5.04 → 3.07 % |

- **Fewer ultrasound images:** with only 200 or 100 ultrasound training images, FedUA-Net stays 3.4–4.0 points ahead of FedBN at that site.
- **Size-proportional weighting:** if the server weights sites by data size, FedAvg and FedBN drop to 53.6 % and 56.1 % on ultrasound.
- **Conformal triage at a 90 % target:** least-ambiguous-set (LAC) prediction refers 8.4 % of MRI scans, 16.4 % of ultrasound scans and 8.0 % of X-ray scans to a clinician.

## Repository layout

```
├── paper/
│   ├── camera_ready/          main.tex, abstract_body.tex, references.bib, spie.cls, spiebib.bst,
│   │   │                      main.bbl, main.pdf
│   │   └── figures/           fig2_results, fig3_conformal (.pdf/.png); Fig. 1 is TikZ inside main.tex
│   └── one_page_abstract/     IEEE one-page abstract (.docx/.pdf) and its framework figure
├── code/
│   ├── fedua_net.py           model (EfficientNetV2-S + CBAM + projection), data loading, training utilities
│   ├── experiment.py          all federated strategies, local-only, centralised, calibration (main runs)
│   ├── run_ablations.py       attention and fine-tuning ablation
│   ├── requirements.txt, environment.yml
│   └── experiments_v2/
│       ├── dump_ckpt.py       checkpoints -> validation/test logits
│       ├── run_v2.py          data loaders and logit dumping used by dump_ckpt.py
│       ├── analyze_v2.py      temperature scaling + conformal evaluation (500 calibration/test splits)
│       └── make_paper_figs.py Figs. 2 and 3
├── models/
│   └── fedua_uniform_3seed/
│       ├── final_models/      9 FedUA-Net checkpoints (3 seeds x 3 sites, Git LFS, ~82 MB each)
│       └── raw/               per-seed metrics of the run that produced them
└── results/
    ├── main_uniform/          main experiment: all methods, 3 seeds, equal client weights
    ├── scarcity_200/, scarcity_100/   ultrasound site subsampled to 200 / 100 training images
    ├── sample_weighted/       same protocol with size-proportional weights, plus the component ablation
    └── rerun_v2/              logits of the checkpoints above and the conformal reports built from them
```

## Where every number comes from

All values are means over seeds 0, 1 and 2, read from these files.

| Paper item | Source |
|---|---|
| Table 1 (accuracy, macro-F1) | `results/main_uniform/reports/summary.csv`, `per_client_metrics.csv` |
| Table 1 (ECE raw / TS) | `results/main_uniform/reports/calibration_comparison.csv` |
| Paired *t*-tests | `results/main_uniform/reports/statistical_significance.csv` |
| Fig. 2(a), scarcity | `results/{main_uniform,scarcity_200,scarcity_100}/reports/per_client_metrics.csv` |
| Fig. 2(b), weighting | `results/sample_weighted/reports/per_client_metrics.csv` vs `results/main_uniform/...` |
| Component ablation (Sec. 4) | `results/sample_weighted/raw/ablation_all.csv` |
| Fig. 2(c), reliability | `results/rerun_v2/logits/fedua_split{s}_train{s}_nb0_c{c}.npz` |
| Table 2, conformal triage (α = 0.1) | `results/rerun_v2/reports/summary_conformal.csv` (method `fed_ts`) |
| Fig. 3, coverage / referral sweep | `results/rerun_v2/reports/conformal_sweep.csv` |

## Setup

```bash
git lfs install                      # once per machine; needed for the model files
git clone https://github.com/tarequejosh/FedUA-net.git
cd FedUA-net
conda env create -f code/environment.yml && conda activate fedua-net
```

If you cloned without Git LFS, run `git lfs pull` to download the checkpoints.

### Data

The datasets are not redistributed here. Download them and arrange them under one folder:

```
<DATA_ROOT>/
├── Brain Tumor MRI Dataset/{Training,Testing}/{glioma,meningioma,notumor,pituitary}/
├── Dataset_BUSI_with_GT/{benign,malignant,normal}/
└── COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/
```

Sources:
- Brain Tumor MRI Dataset (M. Nickparvar, Kaggle);
- BUSI (W. Al-Dhabyani et al., *Data in Brief* 2020);
- COVID-19 Radiography Database (M. E. H. Chowdhury et al. 2020; T. Rahman et al. 2021).

Set `DATA_ROOT` to an **absolute** path, or pass `--data_root`.

## Reproducing the paper

There are three levels, from cheapest to most expensive.

**1. Figures, conformal tables and the paper (CPU only, no data, about 1 minute).** These use the stored logits.

```bash
python code/experiments_v2/analyze_v2.py        # -> results/rerun_v2/reports/*.csv (Table 2)
python code/experiments_v2/make_paper_figs.py   # -> paper/camera_ready/figures, conformal_sweep.csv
cd paper/camera_ready && latexmk -pdf main.tex
```

These scripts reproduce the stored reports and figures exactly.

**2. From the released checkpoints (GPU, data needed, a few minutes).**

```bash
DATA_ROOT=/abs/path/to/Dataset python code/experiments_v2/dump_ckpt.py   # -> results/rerun_v2/logits
```

This gives 96.19 / 86.89 / 95.55 % test accuracy for sites A / B / C (92.88 % mean). The MC-dropout arrays are stochastic, but the logits are not.

To use a checkpoint directly:

```python
import sys, torch; sys.path.insert(0, 'code')
import experiment as E
net = E.new_client_net(ncls=3, seed=0)          # ncls = 4, 3, 4 for sites A, B, C (client 0, 1, 2)
net.load_state_dict(torch.load('models/fedua_uniform_3seed/final_models/fedua_seed0_client1.pt',
                               map_location='cpu'))
net.eval()
```

**3. Full retraining (GPU, several hours per seed).** Run these from `code/`:

```bash
export DATA_ROOT=/abs/path/to/Dataset
python experiment.py --agg_weight_type uniform --seeds 0 1 2 --out ../results_new/main_uniform        # Table 1
python experiment.py --agg_weight_type uniform --hospital_b_subset_size 200 --seeds 0 1 2 \
       --strategies fedbn fedua local_only --out ../results_new/scarcity_200                          # Fig. 2(a); same with 100
python experiment.py --agg_weight_type sample  --seeds 0 1 2 --out ../results_new/sample_weighted     # Fig. 2(b)
python run_ablations.py --seeds 0 1 2 --out ../results_new/sample_weighted                            # ablation
python experiment.py --agg_weight_type uniform --strategies fedua --save_final_models \
       --seeds 0 1 2 --out ../results_new/fedua_ckpt                                                  # checkpoints
```

Notes:
- `run_ablations.py` was run with size-proportional weights.
- Defaults: 12 rounds, batch size 32, CBAM attention. All 8 strategies run unless `--strategies` is given.
- GPU kernels are non-deterministic, so retraining gives numbers within seed variation rather than bit-identical ones.

## Caveats (also stated in the paper)

- **Sites are public datasets.** They stand in for hospitals, so modality and site are confounded.
- **Image-level splits.** Patient-level leakage within BUSI and the MRI collection is possible.
- **Three seeds.** These give limited statistical power.
- **Released checkpoints are a second training run.** They come from a re-run of FedUA-Net with the same protocol. Table 1 reports the first run, which did not save weights. The calibration plot and the conformal analyses use the released checkpoints.

## Citation

```bibtex
@inproceedings{josh2027fedua,
  title     = {{FedUA-Net}: Uncertainty-Aware Federated Learning for Multi-Modal Medical Imaging},
  author    = {Josh, Md Tareque Jamil and Yamada, Fernanda Miyuki and de Carvalho, Gabriel Peixoto and Takahashi, Hiroki},
  booktitle = {International Workshop on Advanced Imaging Technology (IWAIT) 2027},
  series    = {Proc. SPIE},
  year      = {2027}
}
```

## License

MIT; see [LICENSE](LICENSE). The datasets keep their own licenses.
