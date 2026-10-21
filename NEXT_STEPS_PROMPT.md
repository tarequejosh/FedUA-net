# Next steps — prompt for Claude Code (VS Code) to execute

Paste everything below this line to the Claude Code extension chat, or just say
"read NEXT_STEPS_PROMPT.md and execute it" — this file is written to be handed
to a fresh session with no memory of how it was produced.

---

I have a federated learning research project at `D:\Research\FedUA-Net-Journal` —
a copy of a conference project (the untouched original is at
`D:\Research\FedUA-Net`; never read from or write to that path, it's the
archival baseline). The journal copy already has new code written and
unit-tested by another Claude session that did NOT have shell/GPU access, so it
could only run pure-Python unit tests, never the actual pipeline. Your job is to
validate, debug, and actually run it.

## Step 0 — orient yourself

Read `JOURNAL_PROVENANCE.md` in this repo in full. It documents:
- the research direction (federated, group-conditional conformal prediction for
  personalized multi-modal medical FL)
- exactly what changed vs. the conference version (`fedua_net.py`,
  `experiment.py`, the new `src/uncertainty/federated_conformal.py` module, new
  Hospital D/E clients from medmnist, the OOD domain-shift stress test)
- the exact commands intended to be run (section 6)

Also skim `README.md`, `PROGRESS.md`, and `results/RESULTS_PROVENANCE.md` in the
same folder for the conference project's own conventions and known caveats.

## Step 1 — environment sanity check

```bash
cd D:\Research\FedUA-Net-Journal
conda activate research   # or whatever env environment.yml/README implies —
                           # check environment.yml and adjust if the env name
                           # differs on this machine
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Confirm the GPU is visible before doing anything expensive.

## Step 2 — fast unit tests (no GPU needed, run first, must pass before anything else)

```bash
python -m pytest tests/ -v
```

Pay special attention to `tests/test_federated_conformal.py` (14 tests, all
passed in isolated sandbox testing — confirm they still pass in this real
environment) and the existing `tests/run_tests.py` suite
(architecture/CBAM/UQ/transforms). Fix anything that fails — these are
pure-Python/numpy/torch tests with no dataset dependency, so a failure here
means a real bug, not a data issue.

## Step 3 — validate the new data loading paths BEFORE any long training run

These were written without execution access, so they need real verification:

**a) New MedMNIST clients** (Hospital D: Dermatology, Hospital E: Retina) load
from `Dataset/medmnist/derma/dermamnist_224.npz` and
`Dataset/medmnist/retina/retinamnist_224.npz` via a new npz-array-backed
virtual-path mechanism (`fedua_net.py`: `load_image_tensor`,
`build_medmnist_client_rows`). Write and run a tiny standalone script that:
- calls `fedua_net.build_client_rows(3)` and `build_client_rows(4)`, checks the
  resulting DataFrame has all 4 splits (train/val/calib/test) with sane
  per-class counts
- instantiates `build_loaders(...)` for one of them and pulls one batch through
  the DataLoader to confirm images decode to the right shape/dtype and labels
  are in-range

Fix any bugs found. In particular, verify the actual npz key names against
what the code assumes (`train_images`/`train_labels`/`val_images`/
`val_labels`/`test_images`/`test_labels`, the standard MedMNIST convention):

```bash
python -c "import numpy as np; d=np.load('Dataset/medmnist/derma/dermamnist_224.npz'); print(list(d.keys()))"
```

Adjust `build_medmnist_client_rows` if the real keys differ.

**b) The OOD domain-shift loader** (`fedua_net.load_ood_brain_shift` for
`'brisc2025'` and `'pmram'`) — run it standalone, confirm it finds images and
the class-name mapping (brisc2025 uses `"no_tumor"`, PMRAM uses `"512Glioma"`
etc. folder names) produces the expected 4-class label distribution matching
Hospital A's classes.

**c) The new 'calib' split** (`fedua_net.split_val_calib`) — confirm it
produces non-empty, disjoint val/calib sets for all 5 clients including the
small ones (Breast Ultrasound, RetinaMNIST). A stratified split can fail on
classes with very few examples; check for warnings/exceptions.

## Step 4 — smoke-test the full pipeline end to end on tiny data before committing GPU-hours

```bash
python experiment.py --strategies fedua --seeds 0 --rounds 2 --batch 16 \
  --num_clients 5 --smoke --federated_conformal --ood_eval brisc2025 pmram \
  --out ./outputs_journal_smoke
```

Check it completes without errors, inspect `outputs_journal_smoke/raw/` for
`cal_fedua_seed0.csv`, `fedconformal_fedua_seed0.csv`, and
`ood_fedua_seed0.csv`, and sanity-check the numbers look reasonable (coverage
roughly near 1-alpha, no NaNs, set sizes not degenerate). Fix any runtime bugs
this surfaces — this is the first time this code path has actually executed.

## Step 5 — once Step 4 is clean, run the real experiments

Per `JOURNAL_PROVENANCE.md` section 6:

```bash
# 3-client reproduction (sanity vs. conference numbers)
python experiment.py --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua \
  --seeds 0 1 2 --rounds 12 --batch 32 --num_clients 3 --personalize_deep \
  --federated_conformal --out ./outputs_journal_3client

# 5-client journal scale-up, more seeds for statistical power
python experiment.py --strategies fedbn fedprox ditto local_only fedua \
  --seeds 0 1 2 3 4 --rounds 12 --batch 32 --num_clients 5 --personalize_deep \
  --federated_conformal --ood_eval brisc2025 pmram --out ./outputs_journal_5client
```

These take hours — run them with `--resume`-safe behavior in mind (the harness
already supports `--resume`), and check in periodically rather than assuming
silent success. Use background execution if useful, but report elapsed time
and watch the printed per-round accuracy/loss for anything divergent (NaN,
collapsed accuracy) rather than letting a broken run finish.

## Step 6 — aggregate and report

```bash
python scripts/report_federated_conformal.py --out ./outputs_journal_5client
python analyze.py --out ./outputs_journal_5client   # legacy stats, unchanged pipeline
```

Read the resulting `reports/federated_conformal_comparison.csv` and
`reports/ood_robustness_summary.csv`. Report back:
- whether the federated marginal/shrunk conformal modes actually improve
  coverage validity or set size vs. the legacy per-client `'group'` mode,
  especially for the small clients (Breast Ultrasound, RetinaMNIST)
- whether Hospital A's coverage guarantee holds up under the brisc2025/PMRAM
  domain shift, or degrades (either answer is a valid, reportable finding —
  do not spin a negative result as positive)
- whether accuracy/calibration numbers for the reproduced 3-client run are
  close to the conference paper's Table I (some drift is expected since the
  calib-split fix roughly halves the old val/calib pool size)

## Ground rules

- Never modify, delete, or write into `D:\Research\FedUA-Net` (only
  `D:\Research\FedUA-Net-Journal`).
- Do not fabricate, round favorably, or cherry-pick any numbers — if something
  looks wrong or a result is disappointing, say so plainly and show the actual
  CSV output.
- If you find and fix a bug in the new code (`federated_conformal.py`,
  `fedua_net.py`, `experiment.py`, `report_federated_conformal.py`), note what
  was wrong and what you changed — a clear diff-level record, not a silent
  patch.
- If GPU memory or runtime becomes a blocker, say so rather than silently
  downgrading batch size/rounds/seeds without flagging it.

Start with Steps 0-2 and report back before moving on to Step 3.
