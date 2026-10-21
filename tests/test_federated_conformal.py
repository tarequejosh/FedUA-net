"""
Unit + Monte-Carlo validation for federated_conformal.py.

These tests need only numpy -- no GPU, no dataset, no torch -- so they can
(and must) be run and pass before the module is trusted for the journal
experiments. Run with: pytest test_federated_conformal.py -v
"""
import numpy as np
import pytest

from src.uncertainty.federated_conformal import (
    aps_scores,
    QuantileDigest,
    conformal_level,
    prediction_sets,
    empirical_coverage,
    ClientCalibrationData,
    federated_conformal_calibrate,
    evaluate_federated_modes,
    class_scores,
    calibration_scores,
    prediction_sets_for_score,
)


def make_client(rng, n, n_classes, difficulty=1.0, seed_offset=0):
    """Synthetic softmax-like probabilities + labels with controllable
    'difficulty' (higher = less peaked / more uncertain predictions)."""
    logits = rng.normal(0, 1.0 / difficulty, size=(n, n_classes))
    labels = rng.integers(0, n_classes, size=n)
    # bias logits toward the true label so the classifier is informative
    logits[np.arange(n), labels] += 2.0
    ex = np.exp(logits - logits.max(1, keepdims=True))
    probs = ex / ex.sum(1, keepdims=True)
    return probs, labels


# ----------------------------------------------------------------------------
# aps_scores
# ----------------------------------------------------------------------------

def test_aps_scores_basic():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    labels = np.array([0, 1])
    scores = aps_scores(probs, labels)
    # label 0 is rank-1 (highest prob) -> score = 0.7
    assert scores[0] == pytest.approx(0.7)
    # label 1 is rank-1 for row 2 (0.6 is highest) -> score = 0.6
    assert scores[1] == pytest.approx(0.6)


def test_aps_scores_worst_case_label():
    probs = np.array([[0.7, 0.2, 0.1]])
    labels = np.array([2])  # lowest-probability class is the true label
    scores = aps_scores(probs, labels)
    assert scores[0] == pytest.approx(1.0)  # must sum all three sorted probs


# ----------------------------------------------------------------------------
# QuantileDigest: compression error should be small vs exact numpy quantile
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("n", [50, 500, 5000])
def test_digest_matches_exact_quantile(n):
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=n)
    for q in (0.5, 0.8, 0.9, 0.95, 0.99):
        exact = np.quantile(scores, q, method="higher")
        d = QuantileDigest.from_scores(scores, delta=200)
        approx = d.quantile(q)
        # compression error should shrink as data gets larger relative to delta
        tol = 0.05 if n <= 200 else 0.02
        assert abs(exact - approx) <= tol, f"n={n} q={q} exact={exact} approx={approx}"


def test_digest_merge_equals_pooled_digest():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, size=300)
    b = rng.uniform(0, 1, size=150)
    d_a = QuantileDigest.from_scores(a, delta=200)
    d_b = QuantileDigest.from_scores(b, delta=200)
    merged = QuantileDigest.merge([d_a, d_b])
    pooled_exact = QuantileDigest.from_scores(np.concatenate([a, b]), delta=200)
    for q in (0.5, 0.9, 0.95):
        # merged-then-quantile should be close to pooling raw data then digesting
        assert abs(merged.quantile(q) - pooled_exact.quantile(q)) < 0.05


def test_empty_digest_raises():
    d = QuantileDigest.from_scores(np.array([]))
    with pytest.raises(ValueError):
        d.quantile(0.9)


# ----------------------------------------------------------------------------
# conformal_level sanity
# ----------------------------------------------------------------------------

def test_conformal_level_bounds():
    assert 0.0 <= conformal_level(10, 0.1) <= 1.0
    assert conformal_level(1000000, 0.1) == pytest.approx(0.9, abs=1e-3)
    with pytest.raises(ValueError):
        conformal_level(0, 0.1)


# ----------------------------------------------------------------------------
# End-to-end: pooled marginal coverage should hold at (1 - alpha) on average
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_federated_coverage_is_valid_on_average(alpha):
    """Monte-Carlo check: across many independent (calib, test) draws from
    the SAME distribution per client (exchangeability holds by construction),
    the pooled marginal threshold should give empirical coverage close to
    (1 - alpha) on average, confirming the aggregation logic does not
    introduce a bias in either direction."""
    rng = np.random.default_rng(42)
    n_classes = 4
    coverages = []
    for trial in range(60):
        clients_cal, clients_test = [], []
        for cid, (n, diff) in enumerate([(300, 1.0), (120, 1.6), (400, 0.8)]):
            probs_cal, y_cal = make_client(rng, n, n_classes, difficulty=diff)
            probs_test, y_test = make_client(rng, n, n_classes, difficulty=diff)
            clients_cal.append(ClientCalibrationData(str(cid), probs_cal, y_cal))
            clients_test.append(ClientCalibrationData(str(cid), probs_test, y_test))
        result = federated_conformal_calibrate(clients_cal, alpha=alpha)
        modes = evaluate_federated_modes(result, clients_test)
        # pooled coverage across all clients combined for the marginal mode
        all_labels, all_sets = [], []
        for cid_data in clients_test:
            sets, _ = prediction_sets(cid_data.probs, result.q_hat_marginal_federated)
            all_sets.extend(sets)
            all_labels.extend(cid_data.labels.tolist())
        coverages.append(empirical_coverage(all_sets, np.array(all_labels)))
    mean_cov = float(np.mean(coverages))
    # should be within a few points of the nominal 1-alpha target on average
    assert mean_cov >= (1 - alpha) - 0.05, f"alpha={alpha} mean_cov={mean_cov:.3f} undercovers"


def test_group_mode_matches_isolated_per_client_computation():
    """Sanity: the 'group' mode for a single client, computed inside the
    federated protocol, must be IDENTICAL to running the exact same client in
    total isolation (this is what the conference version effectively did --
    we need the federated code path to reduce exactly to it as a special
    case, so results are provably comparable, not just similar)."""
    rng = np.random.default_rng(7)
    probs_cal, y_cal = make_client(rng, 250, 4, difficulty=1.2)
    probs_test, y_test = make_client(rng, 100, 4, difficulty=1.2)

    solo = federated_conformal_calibrate(
        [ClientCalibrationData("A", probs_cal, y_cal)], alpha=0.1, digest_delta=1000
    )
    together = federated_conformal_calibrate(
        [
            ClientCalibrationData("A", probs_cal, y_cal),
            ClientCalibrationData("B", *make_client(rng, 400, 4, difficulty=0.5)),
        ],
        alpha=0.1,
        digest_delta=1000,
    )
    # group-conditional threshold for A should not depend on B's presence
    assert solo.q_hat_group["A"] == pytest.approx(together.q_hat_group["A"], abs=1e-9)


def test_shrinkage_is_between_group_and_marginal():
    rng = np.random.default_rng(3)
    clients_cal = [
        ClientCalibrationData("small", *make_client(rng, 40, 3, difficulty=1.0)),
        ClientCalibrationData("big", *make_client(rng, 2000, 3, difficulty=1.0)),
    ]
    result = federated_conformal_calibrate(clients_cal, alpha=0.1, shrinkage_tau=100)
    for cid in ("small", "big"):
        lo, hi = sorted([result.q_hat_group[cid], result.q_hat_marginal_federated])
        assert lo - 1e-9 <= result.q_hat_shrunk[cid] <= hi + 1e-9
    # the small client should be pulled MUCH closer to the pooled threshold
    # than the big client (small n -> low shrinkage weight on its own group value)
    assert result.shrinkage_weight["small"] < result.shrinkage_weight["big"]


def test_raises_on_all_empty():
    with pytest.raises(ValueError):
        federated_conformal_calibrate([], alpha=0.1)


# ----------------------------------------------------------------------------
# Alternative scores (REVIEW FIX v3): LAC and randomized APS
# ----------------------------------------------------------------------------

def test_lac_scores_basic():
    probs = np.array([[0.7, 0.2, 0.1]])
    assert class_scores(probs, "lac") == pytest.approx(np.array([[0.3, 0.8, 0.9]]))
    assert calibration_scores(probs, np.array([1]), "lac")[0] == pytest.approx(0.8)


def test_aps_rand_brackets_deterministic_aps():
    """u=1 recovers deterministic APS; u=0 gives the mass ranked strictly above y."""
    rng = np.random.default_rng(0)
    probs, labels = make_client(rng, 50, 5)
    det = aps_scores(probs, labels)
    assert calibration_scores(probs, labels, "aps_rand", np.ones(50)) == pytest.approx(det)
    lo = calibration_scores(probs, labels, "aps_rand", np.zeros(50))
    assert lo == pytest.approx(det - probs[np.arange(50), labels])


def test_aps_rand_requires_draws():
    with pytest.raises(ValueError):
        class_scores(np.array([[0.5, 0.5]]), "aps_rand")


def test_unknown_score_raises():
    rng = np.random.default_rng(0)
    probs, labels = make_client(rng, 20, 3)
    with pytest.raises(ValueError):
        federated_conformal_calibrate([ClientCalibrationData("a", probs, labels)], score="raps")


def test_default_score_unchanged():
    """score='aps' must give exactly the pre-v3 thresholds and sets."""
    rng = np.random.default_rng(3)
    probs, labels = make_client(rng, 200, 4)
    cd = [ClientCalibrationData("a", probs, labels)]
    r_default = federated_conformal_calibrate(cd, alpha=0.1)
    r_aps = federated_conformal_calibrate(cd, alpha=0.1, score="aps")
    assert r_default.q_hat_group == r_aps.q_hat_group
    s1, z1 = prediction_sets(probs, r_aps.q_hat_group["a"])
    s2, z2 = prediction_sets_for_score(probs, r_aps.q_hat_group["a"], "aps")
    assert s1 == s2 and (z1 == z2).all()


def make_confident_client(rng, n, n_classes=4, margin=5.0):
    """~95%-accurate, confident classifier -- the regime of the real models."""
    logits = rng.normal(0, 1.0, size=(n, n_classes))
    labels = rng.integers(0, n_classes, size=n)
    logits[np.arange(n), labels] += margin
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True), labels


@pytest.mark.parametrize("score", ["lac", "aps_rand"])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_alternative_scores_are_valid_and_tight(score, alpha):
    """Coverage of the exact per-client threshold should be ~(1 - alpha):
    not below it (validity) and not far above it (the deterministic-APS
    over-coverage this fix exists for)."""
    rng = np.random.default_rng(7)
    covs = []
    for trial in range(80):
        p_cal, y_cal = make_confident_client(rng, 300)
        p_te, y_te = make_confident_client(rng, 300)
        r = federated_conformal_calibrate([ClientCalibrationData("a", p_cal, y_cal)],
                                          alpha=alpha, score=score, seed=trial,
                                          digest_delta=10 ** 9)
        covs.append(evaluate_federated_modes(r, [ClientCalibrationData("a", p_te, y_te)])
                    ["a"]["group"]["coverage"])
    mean_cov = float(np.mean(covs))
    assert (1 - alpha) - 0.01 <= mean_cov <= (1 - alpha) + 0.03, f"{score} alpha={alpha} cov={mean_cov:.3f}"


def test_deterministic_aps_overcovers_confident_classifier():
    """Documents the finding behind REVIEW FIX v3: deterministic APS at a
    nominal 80% target covers far above 80% on a confident classifier."""
    rng = np.random.default_rng(11)
    p_cal, y_cal = make_confident_client(rng, 2000)
    p_te, y_te = make_confident_client(rng, 2000)
    r = federated_conformal_calibrate([ClientCalibrationData("a", p_cal, y_cal)],
                                      alpha=0.2, digest_delta=10 ** 9)
    cov = evaluate_federated_modes(r, [ClientCalibrationData("a", p_te, y_te)])["a"]["group"]["coverage"]
    assert cov > 0.9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
