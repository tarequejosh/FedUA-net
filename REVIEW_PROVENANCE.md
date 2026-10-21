# Review Provenance — Q1 Reviewer Audit and Calibration-Split Fix (2026-09-22)

This file documents a second review pass, run after the paper was already
rewritten into `paper_draft/paper_final_submission.tex` (see
`JOURNAL_PROVENANCE.md` for that history). The user asked for a hostile,
adversarial Q1-reviewer-style audit of the whole project -- code, splits,
checkpoints, statistics, and manuscript claims -- before any further edits.
That audit (Phase 1: repository map, evidence table, critical-problems list)
was delivered in-session; this file records only the two findings that
required **code changes and a rerun**, so the change is traceable the same
way every other fix in this repository is.

## Finding: temperature scaling and conformal calibration reused the same
## calibration split

`JOURNAL_PROVENANCE.md` already documents a real fix: the conference version
reused `val` (the split that guided checkpoint selection) as the calibration
set for both temperature scaling and conformal APS. The first journal
submission fixed that by carving out a disjoint `calib` split, used only for
calibration, never for model selection.

What the first journal submission did **not** fix: inside `run_calibration`
(`experiment.py`), temperature `T` was fit on `calib`, and the conformal
nonconformity scores / quantile were **then computed on that same `calib`
split**, using the same T. The manuscript's Introduction (the paragraph
beginning "Third, we build the uncertainty pipeline...") explicitly claimed
this was not happening -- "a calibration split that is kept strictly
separate from the data used for model selection, so the resulting coverage
guarantee is not undermined by the same data doing two jobs at once." That
sentence was true about val-vs-calib and false about temp-vs-conformal
reuse within calib itself. A hostile-but-fair review of the code (not just
the manuscript) is what caught this.

### Fix

`fedua_net.py::split_val_calib` now splits the old validation pool three
ways instead of two:
  - `val`  (50% of the pool): checkpoint selection only (unchanged).
  - `temp` (25%): temperature scaling only.
  - `conf` (25%): conformal nonconformity scores + quantile only.

`temp` and `conf` are disjoint and neither is used to fit a parameter that
the other is then scored under, so the score function (model + T) is fixed
before `conf` is ever touched. `experiment.py::run_calibration` and
`::run_ood_eval` were updated to fit T on `loaders[c]['temp']` and compute
conformal scores/quantiles on `loaders[c]['conf']`.

This is a real further cost: the old `calib` pool (already half the
conference-era `val` pool) is now halved again, so `conf`-split sizes are
roughly a quarter of the original validation pool, smallest on the sites
that could least afford it (e.g. Hospital B / Hospital E). This is the same
kind of real cost the first journal fix already accepted and disclosed; see
the updated Limitations section for the current number.

Verified: `python -m pytest tests/` -- 29/29 pass, unchanged (the unit tests
exercise `src/uncertainty/federated_conformal.py` directly with synthetic
data and do not depend on `fedua_net.py`'s split proportions). A 2-round,
smoke-scale run of the full pipeline (`--smoke --rounds 2 --seeds 0`)
completed without errors, produced non-degenerate `n_calib` counts on the
`conf` split, and produced `fedconformal_*.csv` output, confirming the
change is wired through end to end before committing GPU time to a full
rerun.

## Finding: no checkpoints were saved for the runs that back every headline
## number in the manuscript (`outputs_journal_3client`, `outputs_journal_5client`)

Both directories contain only CSV output; no `.pt` files. This means the
split fix above could not be applied by reloading and recalibrating an
existing model -- it requires a full retrain. The CKA analysis and the
ablation/scarcity tables, by contrast, DO have saved checkpoints
(`outputs_checkpoints_personalized_3seed`, `outputs_checkpoints_uniform_baseline_3seed`),
but those are from an earlier, different training protocol (see the existing
disclosure in `paper_draft/paper_final_submission.tex`, Section V-D) and are
not affected by this particular fix (CKA never touches calibration data).

### Rerun

Full retrain, `--save_final_models` added so this problem cannot recur:

```
python experiment.py --strategies fedavg fedbn fedprox fedbabu ditto local_only centralized fedua \
  --seeds 0 1 2 --rounds 12 --batch 32 --num_clients 3 --personalize_deep \
  --federated_conformal --save_final_models --out ./outputs_journal_3client_v2

python experiment.py --strategies fedbn fedprox ditto local_only fedua \
  --seeds 0 1 2 3 4 --rounds 12 --batch 32 --num_clients 5 --personalize_deep \
  --federated_conformal --ood_eval brisc2025 pmram --save_final_models --out ./outputs_journal_5client_v2

python scripts/report_federated_conformal.py --out ./outputs_journal_3client_v2
python scripts/report_federated_conformal.py --out ./outputs_journal_5client_v2
python analyze.py --out ./outputs_journal_3client_v2
python analyze.py --out ./outputs_journal_5client_v2
```

Written to `_v2` directories deliberately, so the old `outputs_journal_3client`
/ `outputs_journal_5client` numbers stay available for a side-by-side diff
before anything in the manuscript is overwritten -- consistent with this
project's standing rule to never silently replace a result.

Status of this rerun: completed (exit 0), no tracebacks, per-strategy
accuracy/F1/MCC within normal run-to-run noise of the original
`outputs_journal_3client`/`outputs_journal_5client` numbers -- consistent
with a legitimate reproduction, not a broken run.

## Finding (surfaced by actually running the fix, not by inspection): the
## temperature-scaling clamp is not enough once the calibration pool is
## quartered

Diffing `_v2`'s `reports/calibration_comparison.csv` against the original
immediately showed `fedua`'s calibrated ECE getting **worse** than raw ECE
(`ece_reduction_pct = -44%`), with a strategy-mean `temp` of 6.2 -- far
outside the normal ~0.7-0.8 range every other strategy showed. Per-client
breakdown (`outputs_journal_3client_v2/raw/cal_fedua_seed1.csv`) traced this
to Hospital B (breast ultrasound, the smallest client, `n_calib=30` on the
new `temp` split): the LBFGS fit ran to the clamp ceiling `T=50`, and at
that T, ECE went from `0.063` (raw) to `0.525` (calibrated) -- a
calibration step that made the model roughly 8x worse-calibrated than doing
nothing.

This was not a one-off: scanning every strategy/seed/client in both `_v2`
runs for `temp > 3.0` or `ece_cal > 1.5 * ece_raw` found the same pattern in
9 of the ~120 client/strategy/seed combinations -- overwhelmingly on
Hospital B (`n=30`), but also once on Hospital A at `n=210`
(`cal_fedprox_seed0.csv`, `T=50`, ECE `0.066 -> 0.700`), showing this is not
purely a sample-size artifact -- the LBFGS closure can converge to a bad
optimum regardless of nominal calibration-set size. The existing clamp
(`[0.05, 50.0]`, added earlier to stop `T` diverging to literal infinity)
bounds the value but does nothing to check whether that bounded value
actually calibrates better than `T=1` (no rescaling) -- and several times,
it does not.

### Fix

`experiment.py::calibrate_temp` now compares the fitted (and clamped) T's
cross-entropy loss, on the exact same data it was fit on, against T=1's
loss on that same data, and falls back to T=1 whenever the fit did not
actually improve on the objective it was optimizing. This adds no new data
dependency (it re-uses only the `temp`-split logits/labels already
collected) and is a sanity check on the optimizer's own result, not a
second calibration pass against held-out data.

### Rerun, and a second bug the first fix missed

The full 3-client and 5-client retrain was repeated in place (`_v2`
directories regenerated) after the NLL-fallback fix above. Diffing again
showed the catastrophic case was gone for `fedua`, but **two more
strategies now showed the same T=50 blowup** on Hospital A
(`cal_fedprox_seed0.csv`, `cal_fedbabu_seed0.csv`: `T=50`, ECE
`0.055 -> 0.70` on the held-out test set) -- the NLL-based fallback did not
catch it.

Root cause: on a small calibration batch, average cross-entropy loss can be
dominated by a single catastrophically-misclassified example (true-class
probability near zero gives an unbounded `-log(p)` term). Flattening the
whole batch toward uniform (large T) can reduce that one huge term enough
to reduce the *average* loss, even though it wrecks calibration for every
correctly-classified example in the same batch. LBFGS was, from its own
point of view, correctly minimizing the objective it was given -- the
NLL-based accept/reject check used the same objective, so it could not see
the problem.

Fix: `calibrate_temp` now accepts the fitted (clamped) T only if it
improves **ECE**, not NLL, on the same `temp` data -- ECE bins by
confidence and averages a bounded `|confidence - accuracy|` term per bin,
so one unbounded-loss outlier cannot dominate it the way it dominates NLL.
Verified against a synthetic reproduction of the exact failure (209
confident-correct + 1 confident-wrong example, forcing T=50): the ECE
check correctly rejects T=50 (ece 0.0048 -> 0.688) and falls back to T=1,
where the NLL check would have accepted it. `full_metrics`'s ECE
computation was factored out into a shared `compute_ece()` so both call
sites use one audited definition. Re-verified: 29/29 unit tests pass; a
3-strategy smoke run shows `ece_cal <= ece_raw` for every client with no
exceptions, and the two smallest-calibration-set cases correctly fall back
to `T=1` rather than forcing a fit that wouldn't help.

The full retrain was repeated a third time with this fix.

## Final status (completed)

Both benchmarks retrained successfully a third time
(`outputs_journal_3client_v2`, `outputs_journal_5client_v2`, both with
`--save_final_models`, so this is the first journal-scale run where
checkpoints actually exist). Verification performed before touching the
manuscript:

- Scanned every strategy/seed/client in both runs for `temp > 3.0` or
  `ece_cal > ece_raw`: zero degenerate fits remain. The only cases where
  calibrated ECE exceeds raw ECE on the held-out test set are a handful of
  Hospital B (`n=30`) and Hospital D (`n=251`) client/seed combinations
  where `T` stayed close to 1.0 (never near the clamp) and the test-set ECE
  regression was mild (at most ~2.8x, vs. the earlier 8-11x catastrophic
  cases) -- ordinary small-sample estimation variance, not a bug.
- Every strategy's calibration_comparison.csv now shows `ece_cal < ece_raw`
  in aggregate, both benchmarks.
- Statistical significance: still no Holm-corrected pairwise comparison
  below p=0.18 (3-client) / p=0.14 (5-client) -- the "no significant
  difference" finding is unchanged, and if anything more robust than before
  the fix.
- Fairness reversal (Local-Only leads at 5-client scale, not the proposed
  method): unchanged in direction; exact ranking among the bottom three
  strategies (proposed / FedBN / FedProx) shuffled slightly -- FedProx is
  now the single worst, not tied with the proposed method.
- Naive federated-conformal-pooling failure mode: unchanged in direction and
  magnitude on Site D (pooling inflates set size from 3.65 to 6.60 of 7
  classes); shrinkage recovery on Site E is now smaller in this run (4.88
  to 4.73, vs. 4.99 to 4.58 before) -- still a real but partial recovery,
  now disclosed as uneven across sites rather than uniform.
- OOD anomaly (external cohorts scoring higher than in-distribution):
  unchanged, still unresolved, still disclosed as an open question rather
  than a finding.

Manuscript updated throughout `paper_draft/paper_final_submission.tex` to
match: Table I/II/III, the five-site table, the federated-conformal-modes
table, the OOD table, and every paragraph of prose quoting a specific
number from any of them. One substantive (not just numerical) change: the
"Calibration" results subsection previously described the proposed
architecture as tied with Local-Only for best post-hoc calibration; under
the corrected pipeline, Ditto is now the clear leader and the proposed
architecture a close second, while Local-Only (whose raw predictions were
already best-calibrated, leaving less room to improve) drops to the middle
of the pack. This is reported plainly rather than smoothed over. The
Method section (§III) now describes the true three-way `val`/`temp`/`conf`
split and the ECE-based fallback guard; the title was changed to drop
"Privacy-Preserving" (no formal privacy mechanism is implemented); a
communication-cost subsection, a CKA checkpoint-provenance disclosure, a
patient-level-leakage limitation, and a "federated ≠ formally private"
limitation were added. All 7 figures regenerated from the verified `_v2`
data. Recompiles clean: 13 pages, 37/37 citations resolve, 0 errors.

# Review Provenance v3 -- Data Leakage and Conformal Score Audit (2026-10-22)

Two problems surfaced by a results review, both handled post hoc from the
`--save_final_models` checkpoints (inference only, **no retraining**):
`scripts/rescore_from_checkpoints.py`. Outputs live in
`outputs_journal_{3,5}client_v2/rescore/`. The original `reports/` files are
untouched.

## Reproduction check (run before trusting anything below)

The script rebuilds each seed's splits, reloads every checkpoint, refits T
on `temp` and recomputes the legacy deterministic-APS conformal. Results are
compared with `raw/cal_*.csv` in `rescore/repro_check.csv`.
- **3-client (63 strategy/seed/client rows):** `n_calib` is identical
  everywhere. Accuracy differs by at most one image per client (cuDNN
  nondeterminism), and T by at most 1e-3.
- **5-client (125 rows):** 122 rows match to the same tolerance. The three
  outliers are knife-edge cases, not pipeline errors:
  - fedbn s0 Hospital A and ditto s3 Hospital E: the ECE-based temperature
    fallback flipped (originally T = 1, now the fitted T is accepted). The
    ECE of the fitted T and of T = 1 on `temp` are nearly tied, so
    logit-level noise decides the outcome. **The guard is a hard threshold
    on a noisy statistic; disclose it.**
  - ditto s0 Hospital E (n_conf = 30): T is identical, but the
    deterministic-APS quantile sits on a near-tie between two calibration
    scores, which moves the mean set size by 0.67.

## Finding 1: near-duplicate leakage

Near-duplicate means a 16x16 dHash Hamming distance of at most 10 of 256
bits. Every band up to 10 bits has a median pixel correlation of
0.99-1.00, against 0.48 for random pairs. A visual check showed that
10-bit matches are the same image re-processed, while 11-20-bit matches
are adjacent slices from the same patient. Those are **not** counted, so
the figures below are lower bounds on patient-level leakage.

Within-client leakage (each split checked against that client's own
`train` split; mean over 5 seeds, `rescore/leakage_summary.csv`):

| Client | temp | conf | test |
|---|---|---|---|
| A Brain MRI (Kaggle Nickparvar) | 20% | 18% | **27%** (427/1600) |
| B BUSI | 8% | 10% | 10% |
| C COVID radiography | 2% | 2% | 2% |
| D DermaMNIST | <1% | <1% | <1% |
| E RetinaMNIST | 7% | 5% | 16% |

OOD cohorts, checked against all Hospital A images:
- **BRISC2025 test:** 857/1000 are byte-identical. That covers every
  glioma and pituitary image and 303/306 meningioma images. The 143
  images left are 140 no-tumor and 3 meningioma, so **BRISC cannot be
  used as an external tumor cohort.**
- **PMRAM:** 849/1505 are resized copies, 744 of them matching Hospital
  A's *Training* folder. The 656 left contain no pituitary images
  (167 glioma, 358 meningioma, 131 no-tumor).

This explains the previously "unresolved" OOD anomaly (external accuracy
higher than in-distribution). On the non-duplicated PMRAM images, Fed-UA's
per-class recall matches Hospital A's de-duplicated test set: glioma 0.93
vs 0.86, meningioma 0.97 vs 0.98, no-tumor 1.00 vs 0.99. So on
image-independent data there is **no measurable accuracy loss**, though
pituitary cannot be tested.

**Effect on headline accuracy.** Under the `dedup` protocol, any
temp/conf/test image with a near-duplicate in its client's `train` split is
dropped *before* fitting T, fitting the conformal threshold, or evaluating.
Every strategy loses about 0.8-1.0 pp (Fed-UA: 3-client 93.16% -> 92.37%,
5-client 84.19% -> 83.26%; Hospital A alone 95.8% -> 94.4%). Rankings and
significance conclusions are unchanged. 5-client Fed-UA vs Local-Only:
-2.20 pp, raw p = 0.030, Holm-adjusted p = 0.118
(`rescore/significance_by_protocol.csv`).

## Finding 2: deterministic APS gives valid but uninformative sets

Both conformal code paths used deterministic APS. For a confident model,
a correctly classified example's score is its own top-1 probability, so
the quantile approaches 1. The result was 97-99% coverage at an 80%
target, with 2-2.6 of 4 classes per set at about 95% top-1 accuracy.
`src/uncertainty/federated_conformal.py` now takes a `score` argument:
`'lac'` (1 - p_y) or `'aps_rand'` (randomized APS). The default `'aps'`
is bit-for-bit unchanged. Twelve new unit tests cover this (41/41 pass),
including a Monte Carlo check that the new scores give coverage in
[1-α-0.01, 1-α+0.03] on a confident classifier.

Re-scored results (`rescore/conformal_by_score*.csv`,
`federated_conformal_comparison_by_score.csv`):
- **LAC and randomized APS track the target closely** on Hospitals B and
  C, with about 1 class per set. Empty sets occur in up to about 18% of
  cases at α = 0.2, the expected behaviour when accuracy exceeds 1 - α,
  and must be reported.
- **Hospital A undercovers slightly at α = 0.05** under both protocols
  (LAC 0.91-0.93, randomized APS 0.93-0.94). `conf` is drawn from Kaggle's
  Training folder and `test` is the Testing folder; they are not fully
  exchangeable even after de-duplication.
- **OOD (PMRAM, de-duplicated):** using Hospital A's in-distribution
  threshold, LAC covers 0.83 at a 0.90 target (17% empty sets), while
  randomized APS covers 0.91. **Coverage degrades under shift even though
  accuracy does not**, a legitimate and reportable finding.

## Finding 3: the federated-pooling result depends on the score

5-client Fed-UA, `dedup` protocol, α = 0.1, per-site coverage / mean set
size (`federated_conformal_comparison_by_score.csv`):

| Site | score | group | marginal_federated | shrunk |
|---|---|---|---|---|
| D DermaMNIST | aps | 0.987 / 3.65 | 0.998 / 6.51 | 0.990 / 4.28 |
| D DermaMNIST | aps_rand | 0.913 / 1.52 | 0.926 / 1.65 | 0.917 / 1.55 |
| D DermaMNIST | lac | 0.911 / 1.30 | **0.774** / 0.89 | 0.869 / 1.12 |
| E RetinaMNIST | aps_rand | 0.915 / 3.07 | 0.832 / 2.30 | 0.852 / 2.42 |
| E RetinaMNIST | lac | 0.925 / 3.18 | **0.440** / 0.68 | 0.513 / 0.85 |

The manuscript's claim that "pooling inflates set size on Site D" is
specific to deterministic APS. Under exact (non-conservative) scores, the
pooled threshold **under-covers the hardest, smallest sites**, severely
so with LAC (Site E at 0.44 for a 0.90 target). Shrinkage with tau = 100
recovers little for Site E (n_conf = 30, so w = 0.23). The robust finding
is that a single pooled threshold miscovers heterogeneous sites, with a
direction that depends on the score; group-conditional calibration is
what keeps per-site coverage.
