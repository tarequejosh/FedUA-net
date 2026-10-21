"""
Genuinely federated, group-conditional conformal calibration for personalized
cross-silo federated learning.

Conference-version limitations this module fixes (see RESULTS_PROVENANCE.md /
README.md "Methodological Note on Calibration Split Reuse" in the conference
project):
  1. The conference pipeline (experiment.py::run_calibration) ran independent
     per-client split-conformal (Romano et al. 2020, APS) using each client's
     own `val` split for BOTH personalization checkpoint selection AND
     conformal calibration -- a mild exchangeability violation, acknowledged
     but not fixed in the conference version.
  2. It never actually aggregated information ACROSS clients despite citing
     federated-conformal work (Lu et al. 2023 ICML; Humbert et al. 2023 ICML;
     Plassier et al. 2023 ICML) as related work in literature_review_notes.md
     -- each client's coverage guarantee was computed in complete isolation.

This module implements a genuinely federated calibration procedure:
  - Each client computes its OWN nonconformity scores locally (raw images and
    raw predictions never leave the client) on a calibration split that is
    DISJOINT from the split used for personalization / checkpoint selection
    (fixing limitation 1; see fedua_net.py's four-way split).
  - Each client compresses its score distribution into a small mergeable
    quantile sketch ("digest") and sends ONLY the sketch to the server -- not
    raw scores, not raw probabilities, not raw labels.
  - The server merges sketches across clients and computes:
      (a) a pooled marginal quantile (federated split-conformal, matching the
          guarantee style of Lu et al. 2023 / Humbert et al. 2023), and
      (b) group-conditional quantiles (Mondrian conformal, Vovk, Gammerman &
          Shafer 2005, ch. 4) with "group" = client/hospital identity, giving
          each hospital its own valid coverage guarantee, directly comparable
          to (but not confused with) the pooled guarantee -- fixing
          limitation 2.
      (c) a shrinkage variant that blends (a) and (b) per client according to
          how much calibration data that client actually holds, so a small
          hospital (e.g. n=117) does not rely on a high-variance threshold
          estimated from very few examples. This is an empirical, declared
          heuristic (not a new theoretical guarantee) -- it is evaluated
          empirically for whether it tightens sets without breaking observed
          per-client coverage; see JOURNAL_PROVENANCE.md.

Everything here is validated purely algorithmically (tests/test_federated_conformal.py)
-- it does not require GPU training and is independent of which backbone /
architecture produced the probabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 1. Local nonconformity scores (APS -- same definition as the conference
#    version's src/uncertainty/conformal.py / experiment.py::conformal_aps)
# ----------------------------------------------------------------------------

def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Adaptive Prediction Set nonconformity score (Romano, Sesia & Candes,
    NeurIPS 2020): s_i = sum of sorted softmax probabilities down to and
    including the true class. Factored out so both the legacy per-client
    path and the new federated path use one audited implementation."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, n_classes); got shape {probs.shape}")
    if len(labels) != len(probs):
        raise ValueError("probs and labels must have the same length")
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    rank = (order == labels[:, None]).argmax(axis=1)
    return cum[np.arange(len(labels)), rank]


# REVIEW FIX v3 (see REVIEW_PROVENANCE.md): the deterministic APS score above
# is valid but very conservative for confident classifiers -- a correctly-
# classified example's score is its own top-1 probability, so the quantile is
# driven toward 1 and sets keep classes until ~all probability mass is
# covered (observed: 97-99% coverage at a 80% target, ~2 of 4 classes per
# set at ~95% top-1 accuracy). Two standard alternatives are offered:
#   'lac'      : s = 1 - p_y (Sadinle, Lei & Wasserman, JASA 2019) --
#                smallest expected set size; sets may be empty.
#   'aps_rand' : randomized APS, s = sum_{j ranked above y} p_j + U * p_y,
#                U ~ Unif(0, 1) (Romano, Sesia & Candes 2020, eq. 5) --
#                the exact (non-conservative) form of the APS score.
# 'aps' (default) keeps the original deterministic score and set rule exactly,
# so existing results reproduce bit-for-bit.
SCORES = ("aps", "aps_rand", "lac")


def class_scores(probs: np.ndarray, score: str, u: np.ndarray = None) -> np.ndarray:
    """Nonconformity score of EVERY candidate label, shape (n, n_classes),
    for score in ('lac', 'aps_rand'). The prediction set is {k : S[i,k] <= q},
    and the calibration score is S[i, y_i], so y_i is in its set exactly when
    its score is under the threshold -- the split-conformal guarantee."""
    probs = np.asarray(probs, dtype=np.float64)
    if score == "lac":
        return 1.0 - probs
    if score == "aps_rand":
        if u is None or len(u) != len(probs):
            raise ValueError("aps_rand needs one uniform draw per example")
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cum_excl = np.cumsum(sorted_probs, axis=1) - sorted_probs
        out = np.empty_like(probs)
        np.put_along_axis(out, order, cum_excl + u[:, None] * sorted_probs, axis=1)
        return out
    raise ValueError(f"class_scores: unknown score {score!r}; expected one of {SCORES[1:]}")


def calibration_scores(probs: np.ndarray, labels: np.ndarray, score: str = "aps",
                       u: np.ndarray = None) -> np.ndarray:
    if score == "aps":
        return aps_scores(probs, labels)
    labels = np.asarray(labels, dtype=np.int64)
    return class_scores(probs, score, u)[np.arange(len(labels)), labels]


# ----------------------------------------------------------------------------
# 2. Local compression: a small mergeable quantile sketch ("digest")
# ----------------------------------------------------------------------------

@dataclass
class QuantileDigest:
    """A simple mergeable quantile sketch.

    A production deployment would use a t-digest or GK-sketch so a client
    only ever transmits O(delta) summary statistics rather than its N
    calibration scores. We implement the same *contract* (bounded-size
    summary, mergeable, quantile-queryable) with an exact equal-frequency-bin
    sketch. The scientific question this project cares about is the
    CROSS-CLIENT AGGREGATION logic (marginal pooling vs. group-conditional vs.
    shrinkage), not sketch engineering -- and an exact-binning digest lets us
    report both the compressed-path answer and the exact-pooled-data answer
    to directly measure compression error at these sample sizes (n roughly
    100-3000 per client; see tests/test_federated_conformal.py for the
    compression-error bound this is held to).
    """
    delta: int = 200  # max number of (value, weight) centroids kept
    centroids: List[Tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_scores(cls, scores: np.ndarray, delta: int = 200) -> "QuantileDigest":
        scores = np.sort(np.asarray(scores, dtype=np.float64))
        n = len(scores)
        d = cls(delta=delta)
        if n == 0:
            return d
        if n <= delta:
            d.centroids = [(float(v), 1.0) for v in scores]
            return d
        edges = np.linspace(0, n, delta + 1).astype(int)
        for i in range(delta):
            chunk = scores[edges[i]:edges[i + 1]]
            if len(chunk) == 0:
                continue
            d.centroids.append((float(chunk.mean()), float(len(chunk))))
        return d

    def total_weight(self) -> float:
        return float(sum(w for _, w in self.centroids))

    @staticmethod
    def merge(digests: Sequence["QuantileDigest"]) -> "QuantileDigest":
        """Server-side merge: concatenate centroids from every client digest.
        Requires only the already-compressed centroids -- no raw client
        scores -- matching the communication contract of federated
        quantile-sketch aggregation (see JOURNAL_PROVENANCE.md)."""
        merged = QuantileDigest(delta=max((d.delta for d in digests), default=200))
        for d in digests:
            merged.centroids.extend(d.centroids)
        merged.centroids.sort(key=lambda c: c[0])
        return merged

    def quantile(self, q: float) -> float:
        """Weighted quantile using the 'higher' convention conformal
        prediction requires (rounding must never under-cover)."""
        if not self.centroids:
            raise ValueError("Empty digest has no quantile.")
        values = np.array([v for v, _ in self.centroids])
        weights = np.array([w for _, w in self.centroids])
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cum_w = np.cumsum(weights)
        target = q * cum_w[-1]
        idx = int(np.searchsorted(cum_w, target, side="left"))
        idx = min(idx, len(values) - 1)
        return float(values[idx])


# ----------------------------------------------------------------------------
# 3. Conformal quantile level (finite-sample corrected, Romano et al. 2020)
# ----------------------------------------------------------------------------

def conformal_level(n: int, alpha: float) -> float:
    """Split-conformal finite-sample-corrected quantile level:
    ceil((n+1)(1-alpha)) / n, clipped to [0, 1]."""
    if n <= 0:
        raise ValueError("conformal_level requires n > 0")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    return float(min(1.0, max(0.0, level)))


# ----------------------------------------------------------------------------
# 4. Prediction sets from a scalar threshold, and empirical coverage
# ----------------------------------------------------------------------------

def prediction_sets(probs: np.ndarray, q_hat: float) -> Tuple[List[List[int]], np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    sets: List[List[int]] = []
    sizes = np.zeros(len(probs), dtype=np.int32)
    for i in range(len(probs)):
        mask = cum[i] >= q_hat
        k = int(np.argmax(mask)) + 1 if mask.any() else probs.shape[1]
        sets.append(order[i, :k].tolist())
        sizes[i] = k
    return sets, sizes


def prediction_sets_for_score(probs: np.ndarray, q_hat: float, score: str = "aps",
                              u: np.ndarray = None) -> Tuple[List[List[int]], np.ndarray]:
    """Same return contract as prediction_sets(), for any score in SCORES."""
    if score == "aps":
        return prediction_sets(probs, q_hat)
    mask = class_scores(probs, score, u) <= q_hat
    sets = [np.flatnonzero(row).tolist() for row in mask]
    return sets, mask.sum(axis=1).astype(np.int32)


def empirical_coverage(sets: List[List[int]], labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    covered = sum(int(labels[i]) in sets[i] for i in range(len(labels)))
    return covered / max(len(labels), 1)


# ----------------------------------------------------------------------------
# 5. Federated calibration: marginal-pooled + group-conditional + shrinkage
# ----------------------------------------------------------------------------

@dataclass
class ClientCalibrationData:
    client_id: str
    probs: np.ndarray   # (n_cal, n_classes) softmax probs on this client's OWN held-out calib split
    labels: np.ndarray  # (n_cal,) true local class indices


@dataclass
class FederatedConformalResult:
    alpha: float
    q_hat_marginal_federated: float   # one pooled threshold across ALL clients
    q_hat_group: Dict[str, float]     # per-client (Mondrian/group-conditional) threshold
    q_hat_shrunk: Dict[str, float]    # per-client threshold blended with the pooled one
    n_cal: Dict[str, int]
    shrinkage_weight: Dict[str, float]  # weight given to the LOCAL group threshold (1=fully local)
    score: str = "aps"                  # nonconformity score the thresholds were fit with
    seed: int = 0                       # RNG seed for randomized scores (aps_rand)


def _shrinkage_weight(n_local: int, tau: int) -> float:
    """How much a client's group-conditional threshold should trust its OWN
    (possibly small) calibration set vs. the pooled cross-client threshold.
    w -> 1 as n_local >> tau (trust the local group fully); w -> 0 as
    n_local -> 0 (fall back to the pooled/global threshold rather than a
    threshold estimated from a handful of examples, e.g. a ~100-image split).
    This is declared as a simple empirical-Bayes-style shrinkage heuristic,
    not a new theoretical coverage guarantee -- the guarantee we can actually
    claim is exact pooled marginal coverage; shrinkage is assessed empirically
    for whether it tightens sets without degrading measured per-client
    coverage (see JOURNAL_PROVENANCE.md)."""
    return float(n_local / (n_local + tau))


def federated_conformal_calibrate(
    client_data: Sequence[ClientCalibrationData],
    alpha: float = 0.10,
    digest_delta: int = 200,
    shrinkage_tau: int = 100,
    score: str = "aps",
    seed: int = 0,
) -> FederatedConformalResult:
    """Server-side federated conformal calibration.

    Protocol:
      1. Each client computes local APS nonconformity scores on its OWN
         calibration split (never shared) and compresses them into a
         QuantileDigest (only the digest leaves the client).
      2. The server merges all digests into one pooled digest and reads off
         the pooled marginal quantile at the conformal level for the POOLED
         sample size -- valid marginal coverage across the federation as a
         whole (the guarantee style used in federated conformal prediction,
         e.g. Lu et al. 2023 ICML, Humbert et al. 2023 ICML).
      3. The server also keeps each client's own digest and reads off a
         group-conditional (Mondrian) quantile per client, at the conformal
         level for that client's own sample size -- this is the per-hospital
         guarantee the conference version already attempted, but computed in
         complete isolation; here it is one mode of a single shared protocol,
         directly comparable to the pooled mode on the same footing.
      4. A shrinkage variant blends (2) and (3) per client, controlled by how
         much calibration data that client actually has.
    """
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")
    rng = np.random.default_rng(seed)
    digests: Dict[str, QuantileDigest] = {}
    ns: Dict[str, int] = {}
    for cd in client_data:
        u = rng.uniform(size=len(cd.labels)) if score == "aps_rand" else None
        scores = calibration_scores(cd.probs, cd.labels, score, u)
        digests[cd.client_id] = QuantileDigest.from_scores(scores, delta=digest_delta)
        ns[cd.client_id] = len(scores)

    n_total = sum(ns.values())
    if n_total == 0:
        raise ValueError("federated_conformal_calibrate received no calibration data.")

    pooled = QuantileDigest.merge(list(digests.values()))
    level_pooled = conformal_level(n_total, alpha)
    q_marginal = pooled.quantile(level_pooled)

    q_group: Dict[str, float] = {}
    q_shrunk: Dict[str, float] = {}
    w_shrink: Dict[str, float] = {}
    for cid, d in digests.items():
        n_c = ns[cid]
        if n_c == 0:
            q_group[cid] = q_marginal
            q_shrunk[cid] = q_marginal
            w_shrink[cid] = 0.0
            continue
        level_c = conformal_level(n_c, alpha)
        q_group[cid] = d.quantile(level_c)
        w = _shrinkage_weight(n_c, shrinkage_tau)
        w_shrink[cid] = w
        q_shrunk[cid] = w * q_group[cid] + (1.0 - w) * q_marginal

    return FederatedConformalResult(
        alpha=alpha,
        q_hat_marginal_federated=q_marginal,
        q_hat_group=q_group,
        q_hat_shrunk=q_shrunk,
        n_cal=ns,
        shrinkage_weight=w_shrink,
        score=score,
        seed=seed,
    )


def evaluate_federated_modes(
    result: FederatedConformalResult,
    test_data: Sequence[ClientCalibrationData],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Applies each of the three thresholds (marginal_federated, group,
    shrunk) to each client's TEST split (disjoint from calibration) and
    reports empirical coverage + mean set size per client per mode. Use this
    to compare against the legacy independent-per-client baseline (which is
    reproduced by calling federated_conformal_calibrate with a single
    ClientCalibrationData -- i.e. mode='group' computed in isolation, on the
    reused val split -- exactly what the conference version did)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    modes = {
        "marginal_federated": lambda cid: result.q_hat_marginal_federated,
        "group": lambda cid: result.q_hat_group[cid],
        "shrunk": lambda cid: result.q_hat_shrunk[cid],
    }
    # test-time draws come from a stream independent of the calibration
    # draws, and are shared across modes so the modes differ only in q_hat
    rng = np.random.default_rng([result.seed, 1])
    for td in test_data:
        out[td.client_id] = {}
        u = rng.uniform(size=len(td.labels)) if result.score == "aps_rand" else None
        for mode, q_fn in modes.items():
            q_hat = q_fn(td.client_id)
            sets, sizes = prediction_sets_for_score(td.probs, q_hat, result.score, u)
            cov = empirical_coverage(sets, td.labels)
            out[td.client_id][mode] = {
                "q_hat": q_hat,
                "coverage": cov,
                "mean_set_size": float(sizes.mean()),
                "frac_empty": float((sizes == 0).mean()),
            }
    return out
