import os
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax.numpy as jnp
import numpy as np

from occubed.run_reviews import (
    binary_entropy,
    select_reviewed_label_target_eig_samples,
    select_top_acquisition_samples,
    target_eig_scores_from_prob_positive_draws,
)


def _score_single_candidate(phi, q):
    candidate_mask = jnp.asarray([[[True]]])
    prob_positive_draws = jnp.asarray(q, dtype=float).reshape(len(q), 1, 1, 1)
    scores, diagnostics = target_eig_scores_from_prob_positive_draws(
        jnp.asarray(phi, dtype=float),
        prob_positive_draws,
        candidate_mask,
        acquisition_config={"chunk_size": 4},
    )
    return float(scores[0, 0, 0]), diagnostics


def test_target_eig_zero_target_variance_gives_zero_utility():
    phi = jnp.zeros((6, 3))
    q = jnp.asarray([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])

    score, diagnostics = _score_single_candidate(phi, q)

    assert np.isclose(score, 0.0, atol=1e-8)
    assert np.isclose(float(diagnostics["target_eig"][0, 0, 0]), 0.0, atol=1e-8)


def test_target_eig_independent_label_gives_zero_utility():
    phi = jnp.asarray([[-1.0], [-1.0], [1.0], [1.0]])
    q = jnp.asarray([0.2, 0.8, 0.2, 0.8])

    score, _diagnostics = _score_single_candidate(phi, q)

    assert np.isclose(score, 0.0, atol=1e-8)


def test_target_eig_correlated_label_gives_positive_utility():
    phi = jnp.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    q = jnp.asarray([0.1, 0.2, 0.8, 0.9])

    score, _diagnostics = _score_single_candidate(phi, q)

    assert score > 0.0


def test_target_eig_obeys_binary_label_entropy_bound():
    phi = jnp.asarray([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    prob_positive_draws = jnp.asarray(
        [
            [[[0.1, 0.5, 0.9]]],
            [[[0.2, 0.5, 0.8]]],
            [[[0.8, 0.5, 0.2]]],
            [[[0.9, 0.5, 0.1]]],
        ],
        dtype=float,
    )
    candidate_mask = jnp.ones((1, 1, 3), dtype=bool)

    scores, diagnostics = target_eig_scores_from_prob_positive_draws(
        phi,
        prob_positive_draws,
        candidate_mask,
    )

    label_entropy = binary_entropy(diagnostics["p_positive"])
    assert np.all(np.asarray(scores) >= -1e-10)
    assert np.all(np.asarray(scores) <= np.asarray(label_entropy) + 1e-10)


def test_target_eig_branch_reweighting_obeys_total_expectation():
    phi = jnp.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    q = jnp.asarray([0.1, 0.2, 0.8, 0.9])
    base_mean = phi.mean(axis=0)
    p_positive = q.mean()
    p_negative = 1.0 - p_positive
    positive_weights = q / q.sum()
    negative_weights = (1.0 - q) / (1.0 - q).sum()

    branch_mean = p_positive * (positive_weights @ phi) + p_negative * (
        negative_weights @ phi
    )

    assert np.allclose(np.asarray(base_mean), np.asarray(branch_mean), atol=1e-6)


def test_target_eig_reviewed_candidates_are_excluded():
    phi = jnp.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    prob_positive_draws = jnp.asarray(
        [
            [[[0.1, 0.9]]],
            [[[0.2, 0.8]]],
            [[[0.8, 0.2]]],
            [[[0.9, 0.1]]],
        ],
        dtype=float,
    )
    reviewable = jnp.asarray([[[True, False]]])

    scores, _diagnostics = target_eig_scores_from_prob_positive_draws(
        phi,
        prob_positive_draws,
        reviewable,
    )
    selected = select_top_acquisition_samples(scores, reviewable, 2)

    assert bool(selected[0, 0, 0])
    assert not bool(selected[0, 0, 1])
    assert np.isnan(float(scores[0, 0, 1]))


def test_reviewed_bernoulli_target_eig_selects_informative_label():
    detection_logit = float(np.log(0.8) - np.log(0.2))
    bernoulli_results = SimpleNamespace(
        samples={
            "psi": jnp.asarray(
                [
                    [[0.1], [0.5]],
                    [[0.2], [0.5]],
                    [[0.8], [0.5]],
                    [[0.9], [0.5]],
                ],
                dtype=float,
            ),
            "cov_state_0": jnp.asarray([-2.0, -1.0, 1.0, 2.0]),
            "cov_det_0": jnp.full(4, detection_logit),
        }
    )
    site_covs = jnp.zeros((2, 0), dtype=float)
    obs_covs = jnp.zeros((2, 1, 1, 0), dtype=float)
    reviewed_f = jnp.full((1, 2, 1, 1), jnp.nan)
    reviewable = jnp.ones((2, 1, 1), dtype=bool)

    selected, diagnostics = select_reviewed_label_target_eig_samples(
        bernoulli_results,
        site_covs,
        obs_covs,
        reviewed_f,
        reviewable,
        n_reviews=1,
        acquisition_config={"chunk_size": 4},
    )

    assert bool(selected[0, 0, 0])
    assert not bool(selected[1, 0, 0])
    assert diagnostics is not None
    assert float(diagnostics["target_eig"][0, 0, 0]) > float(
        diagnostics["target_eig"][1, 0, 0]
    )
