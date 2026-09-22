import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from occubed.models import occu_cs_calibrated, occu_cs_init_strategy
from occubed.score_calibration import ScoreMixtureCalibration


def binary_entropy(prob):
    prob = jnp.clip(prob, 1e-12, 1.0 - 1e-12)
    return -(prob * jnp.log(prob) + (1 - prob) * jnp.log1p(-prob))


DUAL_BALANCED_MAX_SCORE_FRACTION = 13.0 / 25.0
DUAL_BALANCED_MIN_POSITIVE_COUNT = 1
EPIG_TARGET_POOL_SIZE = 512
EPIG_EPS = 1e-12
TARGETED_EIG_CANDIDATE_POOL_SIZE = 512
TARGETED_EIG_EPS = 1e-8
TARGET_EIG_TARGET_CONFIG = dict(
    include_psi=True,
    include_beta=True,
    include_alpha=True,
    psi_transform="logit",
    standardize=True,
    pca=True,
    pca_variance=0.95,
    max_target_dim=30,
    include_intercepts=True,
)
TARGET_EIG_ACQUISITION_CONFIG = dict(
    chunk_size=128,
    jitter=1e-6,
    q_eps=1e-8,
    min_branch_prob=1e-8,
    violation_tol=1e-6,
)
DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS = dict(
    score_calibration_method="cut_normal",
    score_calibration_num_warmup=100,
    score_calibration_num_samples=100,
    score_calibration_num_chains=1,
    score_calibration_num_draws=8,
    score_calibration_prevalence_mean=0.01,
    score_calibration_prevalence_strength=100.0,
)


def select_top_acquisition_samples(acquisition_score, reviewable, n_reviews):
    reviewable = jnp.asarray(reviewable, dtype=bool)
    acquisition_score = jnp.asarray(acquisition_score)
    eligible = reviewable & jnp.isfinite(acquisition_score)
    n_reviews = min(n_reviews, int(eligible.sum()))
    if n_reviews == 0:
        return jnp.zeros(reviewable.shape, dtype=bool)

    acquisition_score = jnp.where(
        eligible,
        jnp.maximum(acquisition_score, 0.0),
        -jnp.inf,
    )
    selected = jnp.zeros(reviewable.shape, dtype=bool).reshape(-1)
    selected_indices = jnp.argpartition(
        jnp.where(eligible.reshape(-1), -acquisition_score.reshape(-1), jnp.inf),
        n_reviews - 1,
    )[:n_reviews]
    selected = selected.at[selected_indices].set(True)
    return selected.reshape(reviewable.shape)


def select_top_site_diverse_samples(acquisition_score, reviewable, n_reviews):
    """Select at most one review per site using each site's top candidate."""
    reviewable = np.asarray(reviewable, dtype=bool)
    acquisition_score = np.asarray(acquisition_score, dtype=float)
    eligible = reviewable & np.isfinite(acquisition_score)
    if not eligible.any():
        return jnp.zeros(reviewable.shape, dtype=bool)

    eligible_score = np.where(eligible, np.maximum(acquisition_score, 0.0), -np.inf)
    flat_by_site = eligible_score.reshape(eligible_score.shape[0], -1)
    best_flat_index = flat_by_site.argmax(axis=1)
    best_site_score = flat_by_site.max(axis=1)
    eligible_sites = np.isfinite(best_site_score)
    n_reviews = min(n_reviews, int(eligible_sites.sum()))
    if n_reviews == 0:
        return jnp.zeros(reviewable.shape, dtype=bool)

    selected_site_indices = np.argpartition(
        np.where(eligible_sites, -best_site_score, np.inf),
        n_reviews - 1,
    )[:n_reviews]
    selected = np.zeros(reviewable.shape, dtype=bool)
    for site_index in np.asarray(selected_site_indices, dtype=int):
        site_coordinates = np.unravel_index(
            int(best_flat_index[site_index]),
            eligible_score.shape[1:],
        )
        selected[(site_index, *site_coordinates)] = True
    return jnp.asarray(selected, dtype=bool)


def select_top_dual_acquisition_samples(
    primary_score,
    secondary_score,
    reviewable,
    n_reviews,
    primary_fraction,
):
    reviewable = jnp.asarray(reviewable, dtype=bool)
    primary_reviews = int(round(float(n_reviews) * float(primary_fraction)))
    primary_reviews = int(min(max(primary_reviews, 0), n_reviews))
    secondary_reviews = int(n_reviews - primary_reviews)
    selected_primary = select_top_acquisition_samples(
        primary_score,
        reviewable,
        primary_reviews,
    )
    remaining = reviewable & ~jnp.asarray(selected_primary, dtype=bool)
    selected_secondary = select_top_acquisition_samples(
        secondary_score,
        remaining,
        secondary_reviews,
    )
    return jnp.asarray(selected_primary, dtype=bool) | jnp.asarray(
        selected_secondary, dtype=bool
    )


def _epig_binary_mutual_information(p_candidate, p_target, p11, eps=EPIG_EPS):
    p_candidate = jnp.clip(p_candidate, eps, 1.0 - eps)
    p_target = jnp.clip(p_target, eps, 1.0 - eps)
    p11 = jnp.clip(p11, eps, 1.0 - eps)
    p10 = jnp.clip(p_candidate - p11, eps, 1.0)
    p01 = jnp.clip(p_target - p11, eps, 1.0)
    p00 = jnp.clip(1.0 - p_candidate - p_target + p11, eps, 1.0)
    return (
        p11 * jnp.log(p11 / jnp.clip(p_candidate * p_target, eps, 1.0))
        + p10 * jnp.log(p10 / jnp.clip(p_candidate * (1.0 - p_target), eps, 1.0))
        + p01 * jnp.log(p01 / jnp.clip((1.0 - p_candidate) * p_target, eps, 1.0))
        + p00
        * jnp.log(p00 / jnp.clip((1.0 - p_candidate) * (1.0 - p_target), eps, 1.0))
    )


def choose_epig_target_indices(
    flat_prob_positive,
    target_mask,
    target_pool_size=EPIG_TARGET_POOL_SIZE,
    score_values=None,
):
    flat_prob_positive = jnp.asarray(flat_prob_positive, dtype=float)
    target_mask = jnp.asarray(target_mask, dtype=bool).reshape(-1)
    finite = jnp.isfinite(flat_prob_positive).all(axis=0)
    target_indices = jnp.flatnonzero(target_mask & finite)
    if target_pool_size is None or int(target_pool_size) <= 0:
        return target_indices

    target_pool_size = min(int(target_pool_size), int(target_indices.size))
    if int(target_indices.size) <= target_pool_size:
        return target_indices

    mean_prob = flat_prob_positive[:, target_indices].mean(axis=0)
    uncertainty_count = target_pool_size // 2
    uncertainty_order = jnp.argsort(-binary_entropy(mean_prob))[:uncertainty_count]
    selected = target_indices[uncertainty_order]

    remaining = target_indices[~jnp.isin(target_indices, selected)]
    secondary_count = target_pool_size - int(selected.size)
    if secondary_count <= 0 or int(remaining.size) == 0:
        return jnp.sort(selected)

    if score_values is None:
        secondary_order = remaining
    else:
        flat_score = jnp.asarray(score_values, dtype=float).reshape(-1)
        secondary_values = jnp.where(
            jnp.isfinite(flat_score[remaining]),
            flat_score[remaining],
            -jnp.inf,
        )
        secondary_order = remaining[jnp.argsort(-secondary_values)]
    if int(secondary_order.size) > secondary_count:
        positions = jnp.linspace(
            0,
            int(secondary_order.size) - 1,
            secondary_count,
        ).astype(int)
        secondary_order = secondary_order[positions]
    return jnp.sort(jnp.concatenate([selected, secondary_order[:secondary_count]]))


def epig_scores_from_prob_positive_draws(
    prob_positive_draws,
    candidate_mask,
    target_mask=None,
    target_pool_size=EPIG_TARGET_POOL_SIZE,
    score_values=None,
    eps=EPIG_EPS,
):
    prob_positive_draws = jnp.asarray(prob_positive_draws, dtype=float)
    flat_prob = prob_positive_draws.reshape(prob_positive_draws.shape[0], -1)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    if target_mask is None:
        target_mask = candidate_mask

    finite = jnp.isfinite(flat_prob).all(axis=0)
    candidate_indices = jnp.flatnonzero(candidate_mask.reshape(-1) & finite)
    target_indices = choose_epig_target_indices(
        flat_prob,
        target_mask,
        target_pool_size=target_pool_size,
        score_values=score_values,
    )
    scores = jnp.full(candidate_mask.size, jnp.nan, dtype=float)
    if int(candidate_indices.size) == 0 or int(target_indices.size) == 0:
        return scores.reshape(candidate_mask.shape)

    candidate_prob = jnp.clip(flat_prob[:, candidate_indices], eps, 1.0 - eps)
    target_prob = jnp.clip(flat_prob[:, target_indices], eps, 1.0 - eps)
    p_candidate = candidate_prob.mean(axis=0)
    p_target = target_prob.mean(axis=0)
    p11 = candidate_prob.T @ target_prob / candidate_prob.shape[0]
    mutual_information = _epig_binary_mutual_information(
        p_candidate[:, None],
        p_target[None, :],
        p11,
        eps=eps,
    )
    self_target = candidate_indices[:, None] == target_indices[None, :]
    target_counts = int(target_indices.size) - self_target.sum(axis=1)
    averaged = jnp.where(
        target_counts > 0,
        jnp.where(self_target, 0.0, mutual_information).sum(axis=1)
        / jnp.maximum(target_counts, 1),
        0.0,
    )
    return scores.at[candidate_indices].set(averaged).reshape(candidate_mask.shape)


def _with_default_config(config, default_config):
    merged = dict(default_config)
    if config is not None:
        merged.update(config)
    return merged


def _logit_clip(probability, eps=1e-5):
    probability = jnp.clip(jnp.asarray(probability, dtype=float), eps, 1.0 - eps)
    return jnp.log(probability) - jnp.log1p(-probability)


def _ordered_parameter_names(samples, prefix, n_covariates, include_intercepts=True):
    start_index = 0 if include_intercepts else 1
    names = [
        f"{prefix}_{index}"
        for index in range(start_index, int(n_covariates) + 1)
        if f"{prefix}_{index}" in samples
    ]
    if names:
        return names

    matched_names = sorted(
        name
        for name in samples
        if str(name).startswith(f"{prefix}_")
        and (include_intercepts or not str(name).endswith("_0"))
    )
    return matched_names


def build_target_matrix(
    samples,
    site_covs=None,
    obs_covs=None,
    target_config=None,
):
    target_config = _with_default_config(target_config, TARGET_EIG_TARGET_CONFIG)
    pieces = []
    n_draws = None

    def _append_values(values):
        nonlocal n_draws
        values = jnp.asarray(values, dtype=float)
        if n_draws is None:
            n_draws = int(values.shape[0])
        elif int(values.shape[0]) != n_draws:
            raise ValueError(
                "Target EIG posterior quantities have inconsistent draw counts: "
                f"{values.shape[0]} vs {n_draws}"
            )
        pieces.append(values.reshape(values.shape[0], -1))

    if target_config["include_psi"]:
        psi = jnp.asarray(samples["psi"], dtype=float)
        psi = psi.reshape(psi.shape[0], -1)
        if target_config.get("psi_transform", "logit") == "logit":
            psi = _logit_clip(psi)
        _append_values(psi)

    include_intercepts = bool(target_config.get("include_intercepts", True))
    if target_config["include_beta"]:
        if site_covs is None and "beta" in samples:
            beta = jnp.asarray(samples["beta"], dtype=float)
            if not include_intercepts:
                beta = beta[..., 1:]
            _append_values(beta)
        else:
            n_site_covariates = (
                0 if site_covs is None else jnp.asarray(site_covs).shape[-1]
            )
            for parameter_name in _ordered_parameter_names(
                samples,
                "cov_state",
                n_site_covariates,
                include_intercepts=include_intercepts,
            ):
                _append_values(samples[parameter_name])

    if target_config["include_alpha"]:
        if obs_covs is None and "alpha" in samples:
            alpha = jnp.asarray(samples["alpha"], dtype=float)
            if not include_intercepts:
                alpha = alpha[..., 1:]
            _append_values(alpha)
        else:
            n_obs_covariates = (
                0 if obs_covs is None else jnp.asarray(obs_covs).shape[-1]
            )
            for parameter_name in _ordered_parameter_names(
                samples,
                "cov_det",
                n_obs_covariates,
                include_intercepts=include_intercepts,
            ):
                _append_values(samples[parameter_name])

    if not pieces:
        raise ValueError("Target EIG target matrix has no included quantities.")
    return jnp.concatenate(pieces, axis=1)


def _finite_column_mean_and_scale(values, eps=1e-12):
    values = jnp.asarray(values, dtype=float)
    finite = jnp.isfinite(values)
    counts = jnp.maximum(finite.sum(axis=0), 1)
    mean = jnp.where(finite, values, 0.0).sum(axis=0) / counts
    centered = jnp.where(finite, values - mean, 0.0)
    variance = jnp.square(centered).sum(axis=0) / counts
    scale = jnp.where(variance > eps, jnp.sqrt(variance), 1.0)
    return mean, scale


def fit_target_transform(phi_raw, target_config=None):
    target_config = _with_default_config(target_config, TARGET_EIG_TARGET_CONFIG)
    phi = jnp.asarray(phi_raw, dtype=float)

    standardize_mean = jnp.zeros(phi.shape[1], dtype=float)
    standardize_scale = jnp.ones(phi.shape[1], dtype=float)
    if target_config.get("standardize", True):
        standardize_mean, standardize_scale = _finite_column_mean_and_scale(phi)
        phi = jnp.where(
            jnp.isfinite(phi),
            (phi - standardize_mean) / standardize_scale,
            0.0,
        )
    else:
        phi = jnp.where(jnp.isfinite(phi), phi, 0.0)

    pca_mean = jnp.zeros(phi.shape[1], dtype=float)
    pca_basis = None
    if target_config.get("pca", True) and phi.shape[1] > 0:
        pca_mean = phi.mean(axis=0)
        centered = phi - pca_mean
        _u, singular_values, vh = jnp.linalg.svd(centered, full_matrices=False)
        max_target_dim = max(1, int(target_config.get("max_target_dim", 30)))
        max_target_dim = min(max_target_dim, int(vh.shape[0]), int(phi.shape[1]))
        variances = jnp.square(singular_values) / max(int(phi.shape[0]) - 1, 1)
        total_variance = float(jnp.asarray(variances.sum()))
        if total_variance <= 0.0:
            target_dim = max_target_dim
        else:
            cumulative = np.asarray(jnp.cumsum(variances) / variances.sum())
            target_dim = int(
                np.searchsorted(
                    cumulative,
                    float(target_config.get("pca_variance", 0.95)),
                    side="left",
                )
                + 1
            )
            target_dim = min(max(target_dim, 1), max_target_dim)
        pca_basis = vh[:target_dim].T

    return dict(
        standardize_mean=standardize_mean,
        standardize_scale=standardize_scale,
        pca_mean=pca_mean,
        pca_basis=pca_basis,
    )


def transform_target_matrix(phi_raw, target_transform):
    phi = jnp.asarray(phi_raw, dtype=float)
    phi = jnp.where(
        jnp.isfinite(phi),
        (phi - target_transform["standardize_mean"])
        / target_transform["standardize_scale"],
        0.0,
    )
    pca_basis = target_transform.get("pca_basis")
    if pca_basis is not None:
        phi = (phi - target_transform["pca_mean"]) @ pca_basis
    return phi


def gaussian_logdet_entropy_many(phi, weights, jitter=1e-6):
    phi = jnp.asarray(phi, dtype=float)
    weights = jnp.asarray(weights, dtype=float)
    if phi.shape[1] == 0:
        return jnp.zeros(weights.shape[0], dtype=float)

    weights = weights / jnp.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    mean = weights @ phi
    centered = phi[None, :, :] - mean[:, None, :]
    covariance = jnp.einsum("cm,cmk,cml->ckl", weights, centered, centered)
    diag = jnp.diagonal(covariance, axis1=-2, axis2=-1)
    median_diag = jnp.median(jnp.maximum(diag, 0.0), axis=1)
    jitter_value = jnp.maximum(
        float(jitter), float(jitter) * jnp.maximum(median_diag, 1.0)
    )
    eye = jnp.eye(phi.shape[1], dtype=float)
    sign, logdet = jnp.linalg.slogdet(covariance + jitter_value[:, None, None] * eye)
    return jnp.where(sign > 0, 0.5 * logdet, -jnp.inf)


def gaussian_logdet_entropy(phi, weights=None, jitter=1e-6):
    phi = jnp.asarray(phi, dtype=float)
    if weights is None:
        weights = jnp.full(phi.shape[0], 1.0 / phi.shape[0], dtype=float)
    return gaussian_logdet_entropy_many(phi, jnp.asarray(weights)[None, :], jitter)[0]


def effective_sample_size(weights):
    weights = jnp.asarray(weights, dtype=float)
    weights = weights / jnp.maximum(weights.sum(), 1e-12)
    return 1.0 / jnp.maximum(jnp.square(weights).sum(), 1e-12)


def target_eig_scores_from_prob_positive_draws(
    phi,
    prob_positive_draws,
    candidate_mask,
    acquisition_config=None,
):
    acquisition_config = _with_default_config(
        acquisition_config,
        TARGET_EIG_ACQUISITION_CONFIG,
    )
    phi = jnp.asarray(phi, dtype=float)
    prob_positive_draws = jnp.asarray(prob_positive_draws, dtype=float)
    flat_prob = prob_positive_draws.reshape(prob_positive_draws.shape[0], -1)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    finite = jnp.isfinite(flat_prob).all(axis=0)
    candidate_indices = jnp.flatnonzero(candidate_mask.reshape(-1) & finite)

    nan_scores = jnp.full(candidate_mask.size, jnp.nan, dtype=float)
    false_flags = jnp.zeros(candidate_mask.size, dtype=bool)
    diagnostics = dict(
        target_eig=nan_scores,
        target_eig_raw=nan_scores,
        p_positive=nan_scores,
        label_entropy=nan_scores,
        ess_positive=nan_scores,
        ess_negative=nan_scores,
        target_eig_lower_bound_violation=false_flags,
        target_eig_upper_bound_violation=false_flags,
    )
    if int(candidate_indices.size) == 0:
        return nan_scores.reshape(candidate_mask.shape), {
            key: value.reshape(candidate_mask.shape)
            for key, value in diagnostics.items()
        }

    q_eps = float(acquisition_config.get("q_eps", 1e-8))
    min_branch_prob = float(acquisition_config.get("min_branch_prob", 1e-8))
    violation_tol = float(acquisition_config.get("violation_tol", 1e-6))
    jitter = float(acquisition_config.get("jitter", 1e-6))
    chunk_size = max(1, int(acquisition_config.get("chunk_size", 128)))

    base_weights = jnp.full(phi.shape[0], 1.0 / phi.shape[0], dtype=float)
    base_entropy = gaussian_logdet_entropy(phi, base_weights, jitter=jitter)

    for start in range(0, int(candidate_indices.size), chunk_size):
        chunk_indices = candidate_indices[start : start + chunk_size]
        q = jnp.clip(flat_prob[:, chunk_indices].T, q_eps, 1.0 - q_eps)
        p_positive = q.mean(axis=1)
        p_negative = 1.0 - p_positive
        label_entropy = binary_entropy(p_positive)

        positive_weights = q / jnp.maximum(q.sum(axis=1, keepdims=True), q_eps)
        negative_weights = (1.0 - q) / jnp.maximum(
            (1.0 - q).sum(axis=1, keepdims=True),
            q_eps,
        )
        positive_entropy = gaussian_logdet_entropy_many(
            phi,
            positive_weights,
            jitter=jitter,
        )
        negative_entropy = gaussian_logdet_entropy_many(
            phi,
            negative_weights,
            jitter=jitter,
        )
        positive_entropy = jnp.where(
            p_positive >= min_branch_prob,
            positive_entropy,
            base_entropy,
        )
        negative_entropy = jnp.where(
            p_negative >= min_branch_prob,
            negative_entropy,
            base_entropy,
        )
        ess_positive = jnp.where(
            p_positive >= min_branch_prob,
            1.0 / jnp.maximum(jnp.square(positive_weights).sum(axis=1), 1e-12),
            0.0,
        )
        ess_negative = jnp.where(
            p_negative >= min_branch_prob,
            1.0 / jnp.maximum(jnp.square(negative_weights).sum(axis=1), 1e-12),
            0.0,
        )
        target_eig_raw = (
            base_entropy - p_positive * positive_entropy - p_negative * negative_entropy
        )
        lower_violation = target_eig_raw < -violation_tol
        upper_violation = target_eig_raw > label_entropy + violation_tol
        target_eig = jnp.minimum(jnp.maximum(target_eig_raw, 0.0), label_entropy)

        diagnostics["target_eig"] = (
            diagnostics["target_eig"].at[chunk_indices].set(target_eig)
        )
        diagnostics["target_eig_raw"] = (
            diagnostics["target_eig_raw"].at[chunk_indices].set(target_eig_raw)
        )
        diagnostics["p_positive"] = (
            diagnostics["p_positive"].at[chunk_indices].set(p_positive)
        )
        diagnostics["label_entropy"] = (
            diagnostics["label_entropy"].at[chunk_indices].set(label_entropy)
        )
        diagnostics["ess_positive"] = (
            diagnostics["ess_positive"].at[chunk_indices].set(ess_positive)
        )
        diagnostics["ess_negative"] = (
            diagnostics["ess_negative"].at[chunk_indices].set(ess_negative)
        )
        diagnostics["target_eig_lower_bound_violation"] = (
            diagnostics["target_eig_lower_bound_violation"]
            .at[chunk_indices]
            .set(lower_violation)
        )
        diagnostics["target_eig_upper_bound_violation"] = (
            diagnostics["target_eig_upper_bound_violation"]
            .at[chunk_indices]
            .set(upper_violation)
        )

    lower_count = int(
        jnp.asarray(diagnostics["target_eig_lower_bound_violation"]).sum()
    )
    upper_count = int(
        jnp.asarray(diagnostics["target_eig_upper_bound_violation"]).sum()
    )
    if lower_count or upper_count:
        print(
            "Target EIG numerical bound violations: "
            f"{lower_count} below zero, {upper_count} above label entropy."
        )

    return diagnostics["target_eig"].reshape(candidate_mask.shape), {
        key: value.reshape(candidate_mask.shape) for key, value in diagnostics.items()
    }


def score_candidates_target_eig(
    posterior,
    prob_positive_draws,
    reviewable,
    site_covs=None,
    obs_covs=None,
    target_config=None,
    acquisition_config=None,
):
    phi_raw = build_target_matrix(
        posterior,
        site_covs=site_covs,
        obs_covs=obs_covs,
        target_config=target_config,
    )
    target_transform = fit_target_transform(phi_raw, target_config=target_config)
    phi = transform_target_matrix(phi_raw, target_transform)
    prob_positive_draws = jnp.asarray(prob_positive_draws, dtype=float)
    if prob_positive_draws.shape[0] != phi.shape[0]:
        if prob_positive_draws.shape[0] % phi.shape[0] != 0:
            raise ValueError(
                "Probability-draw count is incompatible with target-draw count: "
                f"{prob_positive_draws.shape[0]} vs {phi.shape[0]}"
            )
        phi = jnp.tile(phi, (prob_positive_draws.shape[0] // phi.shape[0], 1))
    return target_eig_scores_from_prob_positive_draws(
        phi,
        prob_positive_draws,
        reviewable,
        acquisition_config=acquisition_config,
    )


def _normal_logpdf(scores, mean, sigma):
    sigma = float(max(sigma, 1e-6))
    centered = (np.asarray(scores, dtype=float) - float(mean)) / sigma
    return -0.5 * np.square(centered) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def _build_semisupervised_normal_initializations(scores, labels, labeled, min_sigma):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    labeled = np.asarray(labeled, dtype=bool)
    q25, q50, q75 = np.quantile(scores, [0.25, 0.5, 0.75])
    overall_mean = float(scores.mean())
    overall_sigma = float(
        max(
            np.std(scores, ddof=1) if scores.size > 1 else 0.0,
            (q75 - q25) / 1.349 if q75 > q25 else 0.0,
            min_sigma,
        )
    )
    lower_scores = scores[scores <= q50]
    upper_scores = scores[scores > q50]
    if lower_scores.size == 0:
        lower_scores = np.asarray([scores.min()], dtype=float)
    if upper_scores.size == 0:
        upper_scores = np.asarray([scores.max()], dtype=float)

    initializations = []

    def _append_quantile_split(prevalence):
        threshold = np.quantile(scores, 1.0 - prevalence)
        lower_partition = scores[scores <= threshold]
        upper_partition = scores[scores > threshold]
        if lower_partition.size == 0 or upper_partition.size == 0:
            return
        initializations.append(
            dict(
                mu0=float(lower_partition.mean()),
                sigma0=float(
                    max(
                        (
                            np.std(lower_partition, ddof=1)
                            if lower_partition.size > 1
                            else overall_sigma
                        ),
                        min_sigma,
                    )
                ),
                mu1=float(upper_partition.mean()),
                sigma1=float(
                    max(
                        (
                            np.std(upper_partition, ddof=1)
                            if upper_partition.size > 1
                            else overall_sigma
                        ),
                        min_sigma,
                    )
                ),
                prevalence=float(
                    np.clip(upper_partition.size / scores.size, 1e-6, 1.0 - 1e-6)
                ),
            )
        )

    for prevalence in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
        _append_quantile_split(prevalence)

    initializations.extend(
        [
            dict(
                mu0=float(lower_scores.mean()),
                sigma0=float(
                    max(
                        (
                            np.std(lower_scores, ddof=1)
                            if lower_scores.size > 1
                            else overall_sigma
                        ),
                        min_sigma,
                    )
                ),
                mu1=float(upper_scores.mean()),
                sigma1=float(
                    max(
                        (
                            np.std(upper_scores, ddof=1)
                            if upper_scores.size > 1
                            else overall_sigma
                        ),
                        min_sigma,
                    )
                ),
                prevalence=0.5,
            ),
            dict(
                mu0=float(q25),
                sigma0=overall_sigma,
                mu1=float(q75),
                sigma1=overall_sigma,
                prevalence=0.5,
            ),
            dict(
                mu0=float(overall_mean - 0.5 * overall_sigma),
                sigma0=overall_sigma,
                mu1=float(overall_mean + 0.5 * overall_sigma),
                sigma1=overall_sigma,
                prevalence=0.5,
            ),
        ]
    )

    labeled_positive = labeled & (labels > 0.5)
    labeled_negative = labeled & (labels <= 0.5)
    if labeled_positive.any() and labeled_negative.any():
        negative_scores = scores[labeled_negative]
        positive_scores = scores[labeled_positive]
        initializations.append(
            dict(
                mu0=float(negative_scores.mean()),
                sigma0=float(
                    max(
                        (
                            np.std(negative_scores, ddof=1)
                            if negative_scores.size > 1
                            else 0.0
                        ),
                        min_sigma,
                    )
                ),
                mu1=float(positive_scores.mean()),
                sigma1=float(
                    max(
                        (
                            np.std(positive_scores, ddof=1)
                            if positive_scores.size > 1
                            else 0.0
                        ),
                        min_sigma,
                    )
                ),
                prevalence=float(np.clip(labeled_positive.mean(), 1e-6, 1.0 - 1e-6)),
            )
        )

    deduplicated = []
    seen = set()
    for init in initializations:
        key = tuple(round(float(init[name]), 6) for name in init)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(init)
    return deduplicated


def _semisupervised_normal_mixture_log_likelihood(
    scores,
    labels,
    labeled,
    prevalence,
    mu0,
    sigma0,
    mu1,
    sigma1,
):
    log_weight0 = np.log1p(-prevalence) + _normal_logpdf(scores, mu0, sigma0)
    log_weight1 = np.log(prevalence) + _normal_logpdf(scores, mu1, sigma1)
    positive = labels > 0.5
    log_observed = np.where(
        labeled,
        np.where(positive, log_weight1, log_weight0),
        np.logaddexp(log_weight0, log_weight1),
    )
    return float(log_observed.sum())


def _reviewed_score_log_likelihood(scores, labels, labeled, calibration):
    labeled_scores = np.asarray(scores, dtype=float)[np.asarray(labeled, dtype=bool)]
    if labeled_scores.size == 0:
        return 0.0
    labeled_positive = (
        np.asarray(labels, dtype=float)[np.asarray(labeled, dtype=bool)] > 0.5
    )
    log_negative = _normal_logpdf(
        labeled_scores,
        calibration.mu0,
        calibration.sigma0,
    )
    log_positive = _normal_logpdf(
        labeled_scores,
        calibration.mu1,
        calibration.sigma1,
    )
    return float(np.where(labeled_positive, log_positive, log_negative).sum())


def _effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    total_weight = float(weights.sum())
    squared_weight_sum = float(np.square(weights).sum())
    if total_weight <= 0.0 or squared_weight_sum <= 0.0:
        return 1.0
    return max(total_weight**2 / squared_weight_sum, 1.0)


def _summarize_weighted_score_class(scores, weights, min_sigma):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        overall_mean = float(scores.mean()) if scores.size else 0.0
        return dict(
            mean=overall_mean,
            variance=float(max(min_sigma**2, 1.0)),
            effective_count=1.0,
        )
    mean = float(np.sum(weights * scores) / total_weight)
    variance = float(
        max(
            np.sum(weights * np.square(scores - mean)) / total_weight,
            min_sigma**2,
        )
    )
    return dict(
        mean=mean,
        variance=variance,
        effective_count=_effective_sample_size(weights),
    )


def _fit_semisupervised_score_mixture_candidates(
    obs,
    verified_f,
    *,
    max_iter=200,
    tol=1e-6,
    min_sigma=1e-6,
    min_mean_separation=1e-3,
):
    """Fit semi-supervised Normal score-mixture candidates from scores and labels."""

    obs_arr = np.asarray(obs, dtype=float)
    verified_arr = np.asarray(verified_f, dtype=float)
    valid = np.isfinite(obs_arr)
    if not np.any(valid):
        raise ValueError("No finite score observations available for calibration.")

    scores = obs_arr[valid]
    labels = verified_arr[valid]
    labeled = np.isfinite(labels)
    candidates = []
    candidate_index_by_key = {}

    for init in _build_semisupervised_normal_initializations(
        scores, labels, labeled, min_sigma
    ):
        mu0 = float(init["mu0"])
        sigma0 = float(max(init["sigma0"], min_sigma))
        mu1 = float(init["mu1"])
        sigma1 = float(max(init["sigma1"], min_sigma))
        prevalence = float(np.clip(init["prevalence"], 1e-6, 1.0 - 1e-6))

        for _ in range(max_iter):
            old_params = np.asarray([mu0, sigma0, mu1, sigma1, prevalence], dtype=float)
            log_weight0 = np.log1p(-prevalence) + _normal_logpdf(scores, mu0, sigma0)
            log_weight1 = np.log(prevalence) + _normal_logpdf(scores, mu1, sigma1)
            posterior_positive = np.exp(
                log_weight1 - np.logaddexp(log_weight0, log_weight1)
            )
            responsibilities = np.where(labeled, labels > 0.5, posterior_positive)
            resp1 = np.clip(responsibilities, 1e-6, 1.0 - 1e-6)
            resp0 = 1.0 - resp1

            weight0 = float(resp0.sum())
            weight1 = float(resp1.sum())
            if weight0 <= 0.0 or weight1 <= 0.0:
                break

            prevalence = float(np.clip(weight1 / (weight0 + weight1), 1e-6, 1.0 - 1e-6))
            mu0 = float(np.sum(resp0 * scores) / weight0)
            mu1 = float(np.sum(resp1 * scores) / weight1)
            sigma0 = float(
                np.sqrt(
                    max(np.sum(resp0 * np.square(scores - mu0)) / weight0, min_sigma**2)
                )
            )
            sigma1 = float(
                np.sqrt(
                    max(np.sum(resp1 * np.square(scores - mu1)) / weight1, min_sigma**2)
                )
            )

            if mu1 <= mu0:
                midpoint = 0.5 * (mu0 + mu1)
                mu0 = midpoint - 0.5 * min_mean_separation
                mu1 = midpoint + 0.5 * min_mean_separation

            new_params = np.asarray([mu0, sigma0, mu1, sigma1, prevalence], dtype=float)
            if np.max(np.abs(new_params - old_params)) < tol:
                break

        log_weight0 = np.log1p(-prevalence) + _normal_logpdf(scores, mu0, sigma0)
        log_weight1 = np.log(prevalence) + _normal_logpdf(scores, mu1, sigma1)
        posterior_positive = np.exp(
            log_weight1 - np.logaddexp(log_weight0, log_weight1)
        )
        resp1 = np.where(labeled, labels > 0.5, posterior_positive)
        resp1 = np.clip(resp1, 1e-6, 1.0 - 1e-6)
        resp0 = 1.0 - resp1
        log_likelihood = _semisupervised_normal_mixture_log_likelihood(
            scores, labels, labeled, prevalence, mu0, sigma0, mu1, sigma1
        )
        calibration = ScoreMixtureCalibration(
            mu0=mu0,
            sigma0=sigma0,
            mu1=mu1,
            sigma1=sigma1,
            n_negative=int((labels[labeled] <= 0.5).sum()),
            n_positive=int((labels[labeled] > 0.5).sum()),
        )
        candidate = dict(
            calibration=calibration,
            log_likelihood=float(log_likelihood),
            reviewed_log_likelihood=_reviewed_score_log_likelihood(
                scores,
                labels,
                labeled,
                calibration,
            ),
            negative_class_stats=_summarize_weighted_score_class(
                scores,
                resp0,
                min_sigma,
            ),
            positive_class_stats=_summarize_weighted_score_class(
                scores,
                resp1,
                min_sigma,
            ),
        )
        key = tuple(
            round(
                float(value),
                3,
            )
            for value in (
                calibration.mu0,
                calibration.sigma0,
                calibration.mu1,
                calibration.sigma1,
            )
        )
        existing_index = candidate_index_by_key.get(key)
        if existing_index is None:
            candidate_index_by_key[key] = len(candidates)
            candidates.append(candidate)
        elif candidate["log_likelihood"] > candidates[existing_index]["log_likelihood"]:
            candidates[existing_index] = candidate

    candidates.sort(key=lambda candidate: candidate["log_likelihood"], reverse=True)
    if not candidates:
        raise ValueError("Semi-supervised score calibration failed to converge.")
    return candidates


def fit_semisupervised_score_mixture(
    obs,
    verified_f,
    *,
    max_iter=200,
    tol=1e-6,
    min_sigma=1e-6,
    min_mean_separation=1e-3,
):
    """Fit a semi-supervised Normal score mixture from scores and reviewed labels."""

    return _fit_semisupervised_score_mixture_candidates(
        obs,
        verified_f,
        max_iter=max_iter,
        tol=tol,
        min_sigma=min_sigma,
        min_mean_separation=min_mean_separation,
    )[0]["calibration"]


def _score_only_normal_mixture_model(
    scores_std,
    labels,
    prevalence_prior_mean,
    prevalence_prior_strength,
    score_loc,
    score_scale,
):
    mu0_std = numpyro.sample("mu0_std", dist.Normal(0.0, 2.5))
    mu_gap_std = numpyro.sample("mu_gap_std", dist.HalfNormal(3.0))
    mu1_std = numpyro.deterministic("mu1_std", mu0_std + mu_gap_std)
    sigma0_std = numpyro.sample("sigma0_std", dist.HalfNormal(2.0))
    sigma1_std = numpyro.sample("sigma1_std", dist.HalfNormal(2.0))

    prevalence_mean = jnp.clip(jnp.asarray(prevalence_prior_mean), 1e-4, 0.5)
    prevalence_strength = jnp.maximum(jnp.asarray(prevalence_prior_strength), 2.0)
    prevalence = numpyro.sample(
        "prevalence",
        dist.Beta(
            prevalence_mean * prevalence_strength,
            (1.0 - prevalence_mean) * prevalence_strength,
        ),
    )

    log_s0 = dist.Normal(mu0_std, jnp.maximum(sigma0_std, 1e-8)).log_prob(scores_std)
    log_s1 = dist.Normal(mu1_std, jnp.maximum(sigma1_std, 1e-8)).log_prob(scores_std)
    log_mix = jnp.logaddexp(
        jnp.log1p(-jnp.clip(prevalence, 1e-8, 1.0 - 1e-8)) + log_s0,
        jnp.log(jnp.clip(prevalence, 1e-8, 1.0 - 1e-8)) + log_s1,
    )
    labeled = jnp.isfinite(labels)
    positive = labels > 0.5
    log_like = jnp.where(
        labeled,
        jnp.where(positive, log_s1, log_s0),
        log_mix,
    )
    numpyro.factor("score_likelihood", jnp.sum(log_like))

    numpyro.deterministic("mu0", score_loc + score_scale * mu0_std)
    numpyro.deterministic("mu1", score_loc + score_scale * mu1_std)
    numpyro.deterministic("sigma0", score_scale * sigma0_std)
    numpyro.deterministic("sigma1", score_scale * sigma1_std)


def _initial_cut_normal_values(obs, verified_f, score_loc, score_scale):
    try:
        calibration = fit_semisupervised_score_mixture(
            obs,
            verified_f,
            min_sigma=1e-6,
        )
        mu0 = float(calibration.mu0)
        mu1 = float(calibration.mu1)
        sigma0 = float(calibration.sigma0)
        sigma1 = float(calibration.sigma1)
    except Exception:
        obs_arr = np.asarray(obs, dtype=float)
        scores = obs_arr[np.isfinite(obs_arr)]
        q10, q90 = np.quantile(scores, [0.1, 0.9])
        mu0 = float(q10)
        mu1 = float(q90)
        sigma0 = float(max(scores.std(ddof=0), 1e-6))
        sigma1 = sigma0

    gap = max(mu1 - mu0, 1e-3)
    return {
        "mu0_std": jnp.asarray((mu0 - score_loc) / score_scale),
        "mu_gap_std": jnp.asarray(gap / score_scale),
        "sigma0_std": jnp.asarray(max(sigma0 / score_scale, 1e-3)),
        "sigma1_std": jnp.asarray(max(sigma1 / score_scale, 1e-3)),
        "prevalence": jnp.asarray(0.01),
    }


def _selected_draw_indices(n_samples, n_draws):
    n_draws = min(max(int(n_draws), 1), int(n_samples))
    if n_draws == int(n_samples):
        return np.arange(int(n_samples))
    return np.linspace(0, int(n_samples) - 1, n_draws).round().astype(int)


def _logsumexp_np(values, axis=0):
    values = np.asarray(values, dtype=float)
    max_value = np.nanmax(values, axis=axis, keepdims=True)
    return np.squeeze(
        max_value
        + np.log(np.nansum(np.exp(values - max_value), axis=axis, keepdims=True)),
        axis=axis,
    )


def _build_cut_normal_score_log_bf_ensemble(obs, samples, n_draws):
    obs_arr = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs_arr)
    if not np.any(valid):
        raise ValueError("No finite score observations available for calibration.")

    sample_count = len(samples["mu0"])
    draw_indices = _selected_draw_indices(sample_count, n_draws)
    valid_scores = obs_arr[valid]
    log_bf_draw_values = []
    log_density0_draws = []
    log_density1_draws = []
    for draw_index in draw_indices:
        mu0 = float(samples["mu0"][draw_index])
        sigma0 = float(samples["sigma0"][draw_index])
        mu1 = float(samples["mu1"][draw_index])
        sigma1 = float(samples["sigma1"][draw_index])
        log_density0 = _normal_logpdf(valid_scores, mu0, sigma0)
        log_density1 = _normal_logpdf(valid_scores, mu1, sigma1)
        log_density0_draws.append(log_density0)
        log_density1_draws.append(log_density1)
        log_bf_draw_values.append(log_density1 - log_density0)

    aggregated_log_bf_values = _logsumexp_np(
        np.stack(log_density1_draws),
        axis=0,
    ) - _logsumexp_np(
        np.stack(log_density0_draws),
        axis=0,
    )
    aggregated_score_log_bf = np.full(obs_arr.shape, np.nan, dtype=float)
    aggregated_score_log_bf[valid] = aggregated_log_bf_values
    score_log_bf_draws = np.full(
        (len(log_bf_draw_values),) + obs_arr.shape,
        np.nan,
        dtype=float,
    )
    for draw_position, draw_log_bf in enumerate(log_bf_draw_values):
        score_log_bf_draws[draw_position][valid] = draw_log_bf
    return aggregated_score_log_bf, score_log_bf_draws


def fit_cut_normal_score_calibration(obs, verified_f, model_kwargs):
    """Fit score calibration separately so occupancy cannot update score parameters."""

    obs_arr = np.asarray(obs, dtype=float)
    verified_arr = np.asarray(verified_f, dtype=float)
    valid = np.isfinite(obs_arr)
    if not np.any(valid):
        raise ValueError("No finite score observations available for calibration.")

    scores = obs_arr[valid]
    labels = verified_arr[valid]
    score_loc = float(scores.mean())
    score_scale = float(max(scores.std(ddof=0), 1e-6))
    scores_std = (scores - score_loc) / score_scale
    n_positive = int(np.nansum((labels > 0.5) & np.isfinite(labels)))
    n_negative = int(np.nansum((labels <= 0.5) & np.isfinite(labels)))

    calibration_mcmc_kwargs = dict(
        num_warmup=int(
            model_kwargs.get(
                "score_calibration_num_warmup",
                model_kwargs.get("score_calibration_max_iter", 100),
            )
        ),
        num_samples=int(
            model_kwargs.get(
                "score_calibration_num_samples",
                model_kwargs.get("score_calibration_max_candidates", 100),
            )
        ),
        num_chains=int(model_kwargs.get("score_calibration_num_chains", 1)),
    )
    kernel = NUTS(
        _score_only_normal_mixture_model,
        init_strategy=init_to_value(
            values=_initial_cut_normal_values(
                obs_arr,
                verified_arr,
                score_loc,
                score_scale,
            )
        ),
    )
    mcmc = MCMC(
        kernel,
        chain_method="sequential",
        **calibration_mcmc_kwargs,
    )
    mcmc.run(
        jr.PRNGKey(int(model_kwargs.get("score_calibration_random_seed", 0))),
        scores_std=jnp.asarray(scores_std),
        labels=jnp.asarray(labels),
        prevalence_prior_mean=float(
            model_kwargs.get("score_calibration_prevalence_mean", 0.01)
        ),
        prevalence_prior_strength=float(
            model_kwargs.get("score_calibration_prevalence_strength", 100.0)
        ),
        score_loc=float(score_loc),
        score_scale=float(score_scale),
    )
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples().items()}
    score_log_bf, score_log_bf_draws = _build_cut_normal_score_log_bf_ensemble(
        obs_arr,
        samples,
        int(model_kwargs.get("score_calibration_num_draws", 8)),
    )
    calibration = ScoreMixtureCalibration(
        mu0=float(np.mean(samples["mu0"])),
        sigma0=float(np.mean(samples["sigma0"])),
        mu1=float(np.mean(samples["mu1"])),
        sigma1=float(np.mean(samples["sigma1"])),
        n_negative=n_negative,
        n_positive=n_positive,
    )
    return calibration, score_log_bf, score_log_bf_draws


def _allocate_candidate_draw_counts(candidate_weights, n_draws):
    candidate_weights = np.asarray(candidate_weights, dtype=float)
    n_candidates = len(candidate_weights)
    if n_draws <= 0 or n_candidates == 0:
        return np.zeros(n_candidates, dtype=int)

    coverage_count = min(n_draws, n_candidates)
    counts = np.zeros(n_candidates, dtype=int)
    counts[:coverage_count] = 1
    remaining_draws = n_draws - coverage_count
    if remaining_draws <= 0:
        return counts

    normalized_weights = candidate_weights / candidate_weights.sum()
    raw_extra = normalized_weights * remaining_draws
    extra = np.floor(raw_extra).astype(int)
    counts += extra
    leftover = remaining_draws - int(extra.sum())
    if leftover > 0:
        residual_order = np.argsort(-(raw_extra - extra))
        counts[residual_order[:leftover]] += 1
    return counts


def _sample_calibration_from_candidate(candidate, rng, min_sigma):
    negative_stats = candidate["negative_class_stats"]
    positive_stats = candidate["positive_class_stats"]

    def _sample_class(stats):
        effective_count = max(float(stats["effective_count"]), 1.0)
        variance = max(float(stats["variance"]), min_sigma**2)
        degrees_of_freedom = max(effective_count - 1.0, 1.0)
        sampled_variance = (
            variance
            * degrees_of_freedom
            / max(
                rng.chisquare(degrees_of_freedom),
                1e-12,
            )
        )
        sampled_variance = max(float(sampled_variance), min_sigma**2)
        sampled_sigma = float(np.sqrt(sampled_variance))
        sampled_mean = float(
            rng.normal(
                loc=float(stats["mean"]),
                scale=sampled_sigma / np.sqrt(max(effective_count, 1.0)),
            )
        )
        return sampled_mean, sampled_sigma

    mu0, sigma0 = _sample_class(negative_stats)
    mu1, sigma1 = _sample_class(positive_stats)
    if mu1 <= mu0:
        midpoint = 0.5 * (mu0 + mu1)
        min_mean_separation = max(
            1e-3,
            0.1 * max(sigma0, sigma1),
        )
        mu0 = midpoint - 0.5 * min_mean_separation
        mu1 = midpoint + 0.5 * min_mean_separation
    return ScoreMixtureCalibration(
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
        n_negative=candidate["calibration"].n_negative,
        n_positive=candidate["calibration"].n_positive,
    )


def _build_score_calibration_draws(
    candidates,
    *,
    n_draws,
    max_candidates,
    reviewed_weight_temperature,
    min_sigma,
    random_seed,
):
    if n_draws <= 1:
        return [candidates[0]["calibration"]]

    selected_candidates = list(candidates[: max(1, int(max_candidates))])
    reviewed_log_likelihoods = np.asarray(
        [candidate["reviewed_log_likelihood"] for candidate in selected_candidates],
        dtype=float,
    )
    if np.allclose(reviewed_log_likelihoods, reviewed_log_likelihoods[0]):
        candidate_weights = np.full(
            len(selected_candidates), 1.0 / len(selected_candidates)
        )
    else:
        temperature = max(float(reviewed_weight_temperature), 1e-6)
        centered_scores = (
            reviewed_log_likelihoods - reviewed_log_likelihoods.max()
        ) / temperature
        centered_scores = np.clip(centered_scores, -700.0, 0.0)
        candidate_weights = np.exp(centered_scores)
        candidate_weights /= candidate_weights.sum()

    draw_counts = _allocate_candidate_draw_counts(candidate_weights, int(n_draws))
    rng = np.random.default_rng(random_seed)
    draw_calibrations = []
    for candidate, candidate_draw_count in zip(selected_candidates, draw_counts):
        if candidate_draw_count <= 0:
            continue
        draw_calibrations.append(candidate["calibration"])
        for _ in range(candidate_draw_count - 1):
            draw_calibrations.append(
                _sample_calibration_from_candidate(candidate, rng, min_sigma)
            )
    return draw_calibrations[: max(1, int(n_draws))]


def _build_score_log_bf_ensemble(obs, calibration_draws):
    obs_arr = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs_arr)
    if not np.any(valid):
        raise ValueError("No finite score observations available for calibration.")

    draw_log_bf_values = []
    log_density0_sum = None
    log_density1_sum = None
    valid_scores = obs_arr[valid]
    for calibration in calibration_draws:
        log_density0 = _normal_logpdf(valid_scores, calibration.mu0, calibration.sigma0)
        log_density1 = _normal_logpdf(valid_scores, calibration.mu1, calibration.sigma1)
        draw_log_bf_values.append(log_density1 - log_density0)
        log_density0_sum = (
            log_density0
            if log_density0_sum is None
            else np.logaddexp(log_density0_sum, log_density0)
        )
        log_density1_sum = (
            log_density1
            if log_density1_sum is None
            else np.logaddexp(log_density1_sum, log_density1)
        )

    aggregated_score_log_bf = np.full(obs_arr.shape, np.nan, dtype=float)
    aggregated_score_log_bf[valid] = log_density1_sum - log_density0_sum
    score_log_bf_draws = np.full(
        (len(draw_log_bf_values),) + obs_arr.shape,
        np.nan,
        dtype=float,
    )
    for draw_index, draw_log_bf in enumerate(draw_log_bf_values):
        score_log_bf_draws[draw_index][valid] = draw_log_bf
    return aggregated_score_log_bf, score_log_bf_draws


def resolve_calibrated_score_log_bf(obs, verified_f, model_kwargs):
    calibration_method = model_kwargs.get("score_calibration_method", "em_normal")
    if calibration_method == "cut_normal":
        return fit_cut_normal_score_calibration(obs, verified_f, model_kwargs)
    if calibration_method != "em_normal":
        raise ValueError(
            f"Unsupported score calibration method: {calibration_method!r}"
        )

    min_sigma = model_kwargs.get("score_calibration_min_sigma", 1e-6)
    candidates = _fit_semisupervised_score_mixture_candidates(
        obs,
        verified_f,
        max_iter=model_kwargs.get("score_calibration_max_iter", 200),
        tol=model_kwargs.get("score_calibration_tol", 1e-6),
        min_sigma=min_sigma,
        min_mean_separation=model_kwargs.get(
            "score_calibration_min_mean_separation", 1e-3
        ),
    )
    score_calibration = candidates[0]["calibration"]
    score_calibration_draws = _build_score_calibration_draws(
        candidates,
        n_draws=model_kwargs.get("score_calibration_num_draws", 8),
        max_candidates=model_kwargs.get("score_calibration_max_candidates", 4),
        reviewed_weight_temperature=model_kwargs.get(
            "score_calibration_reviewed_weight_temperature",
            1.0,
        ),
        min_sigma=min_sigma,
        random_seed=model_kwargs.get("score_calibration_random_seed", 0),
    )
    aggregated_score_log_bf, score_log_bf_draws = _build_score_log_bf_ensemble(
        obs,
        score_calibration_draws,
    )
    return score_calibration, aggregated_score_log_bf, score_log_bf_draws


def _build_reviewable_score_mask(score_source, site_covs, obs_covs, verified_f):
    score_values = jnp.asarray(score_source[0])
    valid_score_observation = (
        jnp.isfinite(score_values)
        & jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
    )
    verified = jnp.isfinite(verified_f[0])
    verified_value = jnp.where(verified, verified_f[0], 0.0)
    reviewable = valid_score_observation & ~verified
    return score_values, valid_score_observation, reviewable, verified, verified_value


def uncalibrated_prob_positive_draws(
    cs_results,
    obs,
    site_covs,
    obs_covs,
    verified_f,
):
    (
        score_values,
        valid_score_observation,
        reviewable,
        verified,
        verified_value,
    ) = _build_reviewable_score_mask(obs, site_covs, obs_covs, verified_f)
    if not bool(reviewable.any()):
        return None, reviewable

    n_draws = len(cs_results.samples["mu0"])
    psi = jnp.asarray(cs_results.samples["psi"]).reshape(
        n_draws, *score_values.shape[:2]
    )
    alpha = jnp.stack(
        [
            jnp.asarray(cs_results.samples[f"cov_det_{i}"]).reshape(n_draws, -1)[:, 0]
            for i in range(obs_covs.shape[-1] + 1)
        ],
        axis=-1,
    )
    design_matrix = jnp.concatenate(
        [jnp.ones(obs_covs.shape[:-1] + (1,)), jnp.asarray(obs_covs)], axis=-1
    )
    prob_detection = 1.0 / (
        1.0 + jnp.exp(-jnp.einsum("sprc,kc->kspr", design_matrix, alpha))
    )
    mu0 = jnp.asarray(cs_results.samples["mu0"])[:, None, None, None]
    mu1 = jnp.asarray(cs_results.samples["mu1"])[:, None, None, None]
    sigma0 = jnp.maximum(
        jnp.asarray(cs_results.samples["sigma0"])[:, None, None, None], 1e-6
    )
    sigma1 = jnp.maximum(
        jnp.asarray(cs_results.samples["sigma1"])[:, None, None, None], 1e-6
    )
    log_n0 = -0.5 * ((score_values[None, ...] - mu0) / sigma0) ** 2 - jnp.log(sigma0)
    log_n1 = -0.5 * ((score_values[None, ...] - mu1) / sigma1) ** 2 - jnp.log(sigma1)
    log_p = jnp.log(jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    log_1mp = jnp.log1p(-jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    log_mix1 = jnp.logaddexp(log_1mp + log_n0, log_p + log_n1)
    log_like1 = jnp.where(
        verified[None, ...],
        jnp.where(verified_value[None, ...] > 0.5, log_p + log_n1, log_1mp + log_n0),
        jnp.where(valid_score_observation[None, ...], log_mix1, 0.0),
    )
    log_like0 = jnp.where(
        verified[None, ...],
        jnp.where(verified_value[None, ...] > 0.5, -jnp.inf, log_n0),
        jnp.where(valid_score_observation[None, ...], log_n0, 0.0),
    )
    logit_psi = jnp.log(jnp.clip(psi, 1e-12, 1.0 - 1e-12)) - jnp.log1p(
        -jnp.clip(psi, 1e-12, 1.0 - 1e-12)
    )
    posterior_z = 1.0 / (
        1.0 + jnp.exp(-(logit_psi + (log_like1 - log_like0).sum(axis=-1)))
    )
    prob_positive_given_z1 = jnp.clip(
        jnp.exp(log_p + log_n1 - log_mix1), 1e-12, 1.0 - 1e-12
    )
    prob_positive = posterior_z[:, :, :, None] * prob_positive_given_z1
    return prob_positive, reviewable


def _summarize_uncalibrated_prob_positive(
    cs_results,
    obs,
    site_covs,
    obs_covs,
    verified_f,
):
    prob_positive, reviewable = uncalibrated_prob_positive_draws(
        cs_results,
        obs,
        site_covs,
        obs_covs,
        verified_f,
    )
    if prob_positive is None:
        return None, None, reviewable
    return (
        prob_positive.mean(axis=0),
        binary_entropy(prob_positive).mean(axis=0),
        reviewable,
    )


def _summarize_calibrated_prob_positive(
    cs_results,
    score_log_bf,
    site_covs,
    obs_covs,
    verified_f,
    score_log_bf_draws=None,
):
    (
        score_log_bf_values,
        valid_score_observation,
        reviewable,
        verified,
        verified_value,
    ) = _build_reviewable_score_mask(score_log_bf, site_covs, obs_covs, verified_f)
    if not bool(reviewable.any()):
        return None, None, reviewable

    n_draws = len(cs_results.samples["cov_det_0"])
    psi = jnp.asarray(cs_results.samples["psi"]).reshape(
        n_draws, *score_log_bf_values.shape[:2]
    )
    alpha = jnp.stack(
        [
            jnp.asarray(cs_results.samples[f"cov_det_{i}"]).reshape(n_draws, -1)[:, 0]
            for i in range(obs_covs.shape[-1] + 1)
        ],
        axis=-1,
    )
    design_matrix = jnp.concatenate(
        [jnp.ones(obs_covs.shape[:-1] + (1,)), jnp.asarray(obs_covs)], axis=-1
    )
    prob_detection = 1.0 / (
        1.0 + jnp.exp(-jnp.einsum("sprc,kc->kspr", design_matrix, alpha))
    )
    log_p = jnp.log(jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    log_1mp = jnp.log1p(-jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    logit_psi = jnp.log(jnp.clip(psi, 1e-12, 1.0 - 1e-12)) - jnp.log1p(
        -jnp.clip(psi, 1e-12, 1.0 - 1e-12)
    )
    if score_log_bf_draws is None:
        score_log_bf_draws_values = score_log_bf_values[None, ...]
    else:
        score_log_bf_draws_values = jnp.asarray(score_log_bf_draws)[:, 0]

    mean_prob_positive = None
    mean_prob_positive_entropy = None
    for draw_log_bf in score_log_bf_draws_values:
        log_bf = draw_log_bf[None, ...]
        log_mix1 = jnp.logaddexp(log_1mp, log_p + log_bf)
        log_like1 = jnp.where(
            verified[None, ...],
            jnp.where(verified_value[None, ...] > 0.5, log_p, log_1mp),
            jnp.where(valid_score_observation[None, ...], log_mix1, 0.0),
        )
        log_like0 = jnp.where(
            verified[None, ...],
            jnp.where(verified_value[None, ...] > 0.5, -jnp.inf, 0.0),
            0.0,
        )
        posterior_z = 1.0 / (
            1.0 + jnp.exp(-(logit_psi + (log_like1 - log_like0).sum(axis=-1)))
        )
        prob_positive_given_z1 = jnp.clip(
            jnp.exp(log_p + log_bf - log_mix1),
            1e-12,
            1.0 - 1e-12,
        )
        prob_positive = posterior_z[:, :, :, None] * prob_positive_given_z1
        draw_mean_prob_positive = prob_positive.mean(axis=0)
        draw_mean_prob_positive_entropy = binary_entropy(prob_positive).mean(axis=0)
        if mean_prob_positive is None:
            mean_prob_positive = draw_mean_prob_positive
            mean_prob_positive_entropy = draw_mean_prob_positive_entropy
        else:
            mean_prob_positive = mean_prob_positive + draw_mean_prob_positive
            mean_prob_positive_entropy = (
                mean_prob_positive_entropy + draw_mean_prob_positive_entropy
            )

    n_calibration_draws = score_log_bf_draws_values.shape[0]
    mean_prob_positive = mean_prob_positive / n_calibration_draws
    mean_prob_positive_entropy = mean_prob_positive_entropy / n_calibration_draws
    return mean_prob_positive, mean_prob_positive_entropy, reviewable


def parameter_draw_matrix(samples, parameter_names):
    pieces = []
    for parameter_name in parameter_names:
        values = jnp.asarray(samples[parameter_name], dtype=float)
        pieces.append(values.reshape(values.shape[0], -1))
    if not pieces:
        raise ValueError("Provide at least one target parameter name.")
    return jnp.concatenate(pieces, axis=1)


def calibrated_prob_positive_draws(
    samples,
    score_log_bf,
    score_log_bf_draws,
    site_covs,
    obs_covs,
    verified_f,
):
    score_log_bf_values = jnp.asarray(score_log_bf[0], dtype=float)
    site_covs = jnp.asarray(site_covs, dtype=float)
    obs_covs = jnp.asarray(obs_covs, dtype=float)
    verified = jnp.isfinite(verified_f[0])
    verified_value = jnp.where(verified, verified_f[0], 0.0)
    valid_score_observation = (
        jnp.isfinite(score_log_bf_values)
        & jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
    )

    n_draws = len(samples["cov_det_0"])
    psi = jnp.asarray(samples["psi"]).reshape(n_draws, *score_log_bf_values.shape[:2])
    alpha = jnp.stack(
        [
            jnp.asarray(samples[f"cov_det_{index}"]).reshape(n_draws, -1)[:, 0]
            for index in range(obs_covs.shape[-1] + 1)
        ],
        axis=-1,
    )
    design_matrix = jnp.concatenate(
        [jnp.ones(obs_covs.shape[:-1] + (1,)), obs_covs],
        axis=-1,
    )
    design_matrix = jnp.where(jnp.isfinite(design_matrix), design_matrix, 0.0)
    prob_detection = 1.0 / (
        1.0 + jnp.exp(-jnp.einsum("sprc,kc->kspr", design_matrix, alpha))
    )
    prob_detection = jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12)
    log_p = jnp.log(prob_detection)
    log_1mp = jnp.log1p(-prob_detection)
    clipped_psi = jnp.clip(psi, 1e-12, 1.0 - 1e-12)
    logit_psi = jnp.log(clipped_psi) - jnp.log1p(-clipped_psi)
    if score_log_bf_draws is None:
        score_log_bf_draws_values = score_log_bf_values[None, ...]
    else:
        score_log_bf_draws_values = jnp.asarray(score_log_bf_draws)[:, 0]

    prob_positive_draws = []
    for draw_log_bf in score_log_bf_draws_values:
        draw_log_bf = jnp.where(jnp.isfinite(draw_log_bf), draw_log_bf, 0.0)
        log_bf = draw_log_bf[None, ...]
        log_mix1 = jnp.logaddexp(log_1mp, log_p + log_bf)
        log_like1 = jnp.where(
            verified[None, ...],
            jnp.where(verified_value[None, ...] > 0.5, log_p, log_1mp),
            jnp.where(valid_score_observation[None, ...], log_mix1, 0.0),
        )
        log_like0 = jnp.where(
            verified[None, ...],
            jnp.where(verified_value[None, ...] > 0.5, -jnp.inf, 0.0),
            0.0,
        )
        posterior_z = 1.0 / (
            1.0 + jnp.exp(-(logit_psi + (log_like1 - log_like0).sum(axis=-1)))
        )
        prob_positive_given_z1 = jnp.clip(
            jnp.exp(log_p + log_bf - log_mix1),
            1e-12,
            1.0 - 1e-12,
        )
        prob_positive_draws.append(posterior_z[:, :, :, None] * prob_positive_given_z1)
    return jnp.concatenate(prob_positive_draws, axis=0)


def _weighted_meanfield_gaussian_entropy(parameter_matrix, weights, eps):
    weight_sums = jnp.maximum(weights.sum(axis=0), eps)
    mean = weights.T @ parameter_matrix / weight_sums[:, None]
    second_moment = weights.T @ jnp.square(parameter_matrix) / weight_sums[:, None]
    variance = jnp.maximum(second_moment - jnp.square(mean), eps)
    return 0.5 * jnp.sum(jnp.log(variance), axis=1)


def _targeted_parameter_eig_scores_from_arrays(theta, candidate_prob, eps):
    base_variance = jnp.maximum(theta.var(axis=0), eps)
    base_entropy = 0.5 * jnp.sum(jnp.log(base_variance))
    candidate_prob = jnp.clip(candidate_prob, eps, 1.0 - eps)
    entropy_positive = _weighted_meanfield_gaussian_entropy(
        theta,
        candidate_prob,
        eps,
    )
    entropy_negative = _weighted_meanfield_gaussian_entropy(
        theta,
        1.0 - candidate_prob,
        eps,
    )
    mean_prob_positive = candidate_prob.mean(axis=0)
    return base_entropy - (
        mean_prob_positive * entropy_positive
        + (1.0 - mean_prob_positive) * entropy_negative
    )


def _conditional_targeted_eig_scores(theta, candidate_prob, outcome_weights, eps):
    outcome_probability = outcome_weights.mean(axis=0)
    current_entropy = jnp.sum(
        outcome_probability
        * _weighted_meanfield_gaussian_entropy(theta, outcome_weights, eps)
    )
    candidate_prob = jnp.clip(candidate_prob, eps, 1.0 - eps)
    expected_entropy_after = jnp.zeros(candidate_prob.shape[1], dtype=float)
    for outcome_index in range(outcome_weights.shape[1]):
        outcome_weight = outcome_weights[:, outcome_index : outcome_index + 1]
        positive_weights = outcome_weight * candidate_prob
        negative_weights = outcome_weight * (1.0 - candidate_prob)
        expected_entropy_after = expected_entropy_after + positive_weights.mean(
            axis=0
        ) * _weighted_meanfield_gaussian_entropy(theta, positive_weights, eps)
        expected_entropy_after = expected_entropy_after + negative_weights.mean(
            axis=0
        ) * _weighted_meanfield_gaussian_entropy(theta, negative_weights, eps)
    return current_entropy - expected_entropy_after


def select_revealed_targeted_eig_samples(
    cs_results,
    target_parameter_names,
    score_log_bf,
    score_log_bf_draws,
    site_covs,
    obs_covs,
    verified_f,
    reviewable,
    label_values,
    n_reviews,
    candidate_pool_size=TARGETED_EIG_CANDIDATE_POOL_SIZE,
    eps=TARGETED_EIG_EPS,
):
    theta = parameter_draw_matrix(cs_results.samples, target_parameter_names)
    prob_positive = calibrated_prob_positive_draws(
        cs_results.samples,
        score_log_bf,
        score_log_bf_draws,
        site_covs,
        obs_covs,
        verified_f,
    )
    if prob_positive.shape[0] != theta.shape[0]:
        if prob_positive.shape[0] % theta.shape[0] != 0:
            raise ValueError(
                "Probability-draw count is incompatible with parameter-draw count: "
                f"{prob_positive.shape[0]} vs {theta.shape[0]}"
            )
        theta = jnp.tile(theta, (prob_positive.shape[0] // theta.shape[0], 1))

    reviewable = jnp.asarray(reviewable, dtype=bool)
    flat_prob_positive = prob_positive.reshape(prob_positive.shape[0], -1)
    eligible_indices = jnp.flatnonzero(
        reviewable.reshape(-1) & jnp.isfinite(flat_prob_positive).all(axis=0)
    )
    selected = jnp.zeros(reviewable.size, dtype=bool)
    if n_reviews <= 0 or eligible_indices.size == 0:
        return selected.reshape(reviewable.shape)

    eligible_prob = flat_prob_positive[:, eligible_indices]
    eligible_marginal_scores = _targeted_parameter_eig_scores_from_arrays(
        theta,
        eligible_prob,
        eps,
    )
    if not bool(jnp.isfinite(eligible_marginal_scores).any()):
        return selected.reshape(reviewable.shape)

    finite_order = jnp.argsort(
        jnp.where(
            jnp.isfinite(eligible_marginal_scores), -eligible_marginal_scores, jnp.inf
        )
    )
    pool_size = min(
        int(eligible_indices.size),
        max(int(n_reviews), int(candidate_pool_size)),
    )
    pool_positions = finite_order[:pool_size]
    pool_indices = eligible_indices[pool_positions]
    pool_prob = jnp.clip(eligible_prob[:, pool_positions], eps, 1.0 - eps)
    available = jnp.ones(pool_indices.shape[0], dtype=bool)

    flat_labels = jnp.asarray(label_values, dtype=float).reshape(-1)
    outcome_weights = jnp.ones((theta.shape[0], 1), dtype=float)
    for _ in range(min(int(n_reviews), int(pool_indices.size))):
        conditional_scores = _conditional_targeted_eig_scores(
            theta,
            pool_prob,
            outcome_weights,
            eps,
        )
        conditional_scores = jnp.where(available, conditional_scores, -jnp.inf)
        if not bool(jnp.isfinite(conditional_scores).any()):
            break
        selected_position = int(jnp.argmax(conditional_scores))
        selected_index = int(pool_indices[selected_position])
        selected = selected.at[selected_index].set(True)
        available = available.at[selected_position].set(False)
        selected_prob = flat_prob_positive[:, selected_index : selected_index + 1]
        if float(flat_labels[selected_index]) > 0.5:
            outcome_weights = outcome_weights * selected_prob
        else:
            outcome_weights = outcome_weights * (1.0 - selected_prob)

    return selected.reshape(reviewable.shape)


def select_parameter_bald_samples(
    cs_results, obs, site_covs, obs_covs, verified_f, n_reviews
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_uncalibrated_prob_positive(
            cs_results,
            obs,
            site_covs,
            obs_covs,
            verified_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


def build_cs_selection_state(
    model_fn,
    cs_results,
    obs,
    site_covs,
    obs_covs,
    verified_f,
    score_log_bf=None,
    score_log_bf_draws=None,
):
    if model_fn is occu_cs_calibrated:
        mean_prob_positive, mean_prob_positive_entropy, reviewable = (
            _summarize_calibrated_prob_positive(
                cs_results,
                score_log_bf,
                site_covs,
                obs_covs,
                verified_f,
                score_log_bf_draws=score_log_bf_draws,
            )
        )
    else:
        mean_prob_positive, mean_prob_positive_entropy, reviewable = (
            _summarize_uncalibrated_prob_positive(
                cs_results,
                obs,
                site_covs,
                obs_covs,
                verified_f,
            )
        )

    max_score = np.asarray(obs[0], dtype=float)
    reviewable = np.asarray(reviewable, dtype=bool)
    if mean_prob_positive is None:
        nan_scores = np.full(reviewable.shape, np.nan, dtype=float)
        return dict(
            reviewable=reviewable,
            mean_prob_positive=None,
            expected_gain_score=nan_scores,
            posterior_predictive_uncertainty=nan_scores,
            max_score=max_score,
        )

    expected_gain_score = (
        binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    )
    posterior_predictive_uncertainty = binary_entropy(mean_prob_positive)
    return dict(
        reviewable=reviewable,
        mean_prob_positive=np.asarray(mean_prob_positive, dtype=float),
        expected_gain_score=np.asarray(expected_gain_score, dtype=float),
        posterior_predictive_uncertainty=np.asarray(
            posterior_predictive_uncertainty, dtype=float
        ),
        max_score=max_score,
    )


def restrict_selection_state_to_labelable(selection_state, label_values):
    reviewable = np.asarray(selection_state["reviewable"], dtype=bool)
    labelable = np.isfinite(np.asarray(label_values, dtype=float))
    if labelable.shape != reviewable.shape:
        raise ValueError(
            "Labelable mask shape does not match selection reviewable mask shape: "
            f"{labelable.shape} != {reviewable.shape}"
        )
    selection_state = dict(selection_state)
    selection_state["reviewable"] = reviewable & labelable
    return selection_state


def select_target_eig_samples(
    cs_results,
    obs,
    site_covs,
    obs_covs,
    verified_f,
    reviewable,
    n_reviews,
    target_config=None,
    acquisition_config=None,
):
    prob_positive, _prob_reviewable = uncalibrated_prob_positive_draws(
        cs_results,
        obs,
        site_covs,
        obs_covs,
        verified_f,
    )
    if prob_positive is None:
        empty = jnp.zeros(jnp.asarray(reviewable).shape, dtype=bool)
        return empty, None
    target_eig_score, diagnostics = score_candidates_target_eig(
        cs_results.samples,
        prob_positive,
        reviewable,
        site_covs=site_covs,
        obs_covs=obs_covs,
        target_config=target_config,
        acquisition_config=acquisition_config,
    )
    selected = select_top_acquisition_samples(
        target_eig_score,
        reviewable,
        n_reviews,
    )
    return selected, diagnostics


def select_calibrated_target_eig_samples(
    cs_results,
    score_log_bf,
    score_log_bf_draws,
    site_covs,
    obs_covs,
    verified_f,
    reviewable,
    n_reviews,
    target_config=None,
    acquisition_config=None,
):
    prob_positive = calibrated_prob_positive_draws(
        cs_results.samples,
        score_log_bf,
        score_log_bf_draws,
        site_covs,
        obs_covs,
        verified_f,
    )
    target_eig_score, diagnostics = score_candidates_target_eig(
        cs_results.samples,
        prob_positive,
        reviewable,
        site_covs=site_covs,
        obs_covs=obs_covs,
        target_config=target_config,
        acquisition_config=acquisition_config,
    )
    selected = select_top_acquisition_samples(
        target_eig_score,
        reviewable,
        n_reviews,
    )
    return selected, diagnostics


def select_reviewed_label_target_eig_samples(
    bernoulli_results,
    site_covs,
    obs_covs,
    reviewed_f,
    reviewable,
    n_reviews,
    target_config=None,
    acquisition_config=None,
):
    prob_positive, _prob_reviewable = reviewed_label_prob_positive_draws(
        bernoulli_results,
        site_covs,
        obs_covs,
        reviewed_f,
    )
    if prob_positive is None:
        empty = jnp.zeros(jnp.asarray(reviewable).shape, dtype=bool)
        return empty, None
    target_eig_score, diagnostics = score_candidates_target_eig(
        bernoulli_results.samples,
        prob_positive,
        reviewable,
        site_covs=site_covs,
        obs_covs=obs_covs,
        target_config=target_config,
        acquisition_config=acquisition_config,
    )
    selected = select_top_acquisition_samples(
        target_eig_score,
        reviewable,
        n_reviews,
    )
    return selected, diagnostics


def count_reviewed_positives(reviewed_values):
    reviewed_values = jnp.asarray(reviewed_values, dtype=float)
    return int(
        jnp.nansum(
            jnp.where(
                jnp.isfinite(reviewed_values),
                reviewed_values > 0.5,
                0.0,
            )
        )
    )


def _mean_for_mask(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    selected_values = values[mask & np.isfinite(values)]
    if selected_values.size == 0:
        return float("nan")
    return float(selected_values.mean())


def _sum_for_mask(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    selected_values = values[mask & np.isfinite(values)]
    if selected_values.size == 0:
        return float("nan")
    return float(selected_values.sum())


def _summarize_target_eig_mask(prefix, diagnostics, selection_state, mask):
    return {
        f"mean_target_eig_{prefix}": _mean_for_mask(
            diagnostics["target_eig"],
            mask,
        ),
        f"mean_bald_{prefix}": _mean_for_mask(
            selection_state["expected_gain_score"],
            mask,
        ),
        f"mean_score_{prefix}": _mean_for_mask(selection_state["max_score"], mask),
        f"mean_p_positive_{prefix}": _mean_for_mask(
            diagnostics["p_positive"],
            mask,
        ),
        f"expected_num_positives_{prefix}": _sum_for_mask(
            diagnostics["p_positive"],
            mask,
        ),
        f"mean_label_entropy_{prefix}": _mean_for_mask(
            diagnostics["label_entropy"],
            mask,
        ),
        f"mean_ess_positive_{prefix}": _mean_for_mask(
            diagnostics["ess_positive"],
            mask,
        ),
        f"mean_ess_negative_{prefix}": _mean_for_mask(
            diagnostics["ess_negative"],
            mask,
        ),
    }


def summarize_target_eig_selection(
    diagnostics,
    selection_state,
    selected,
    n_reviews,
):
    if diagnostics is None:
        return {}

    top_bald = select_top_acquisition_samples(
        selection_state["expected_gain_score"],
        selection_state["reviewable"],
        n_reviews,
    )
    top_max_score = select_top_acquisition_samples(
        selection_state["max_score"],
        selection_state["reviewable"],
        n_reviews,
    )
    metrics = {}
    metrics.update(
        _summarize_target_eig_mask(
            "selected",
            diagnostics,
            selection_state,
            selected,
        )
    )
    metrics.update(
        _summarize_target_eig_mask(
            "top_bald",
            diagnostics,
            selection_state,
            top_bald,
        )
    )
    metrics.update(
        _summarize_target_eig_mask(
            "top_max_score",
            diagnostics,
            selection_state,
            top_max_score,
        )
    )
    metrics["target_eig_lower_bound_violations"] = float(
        np.asarray(diagnostics["target_eig_lower_bound_violation"], dtype=bool).sum()
    )
    metrics["target_eig_upper_bound_violations"] = float(
        np.asarray(diagnostics["target_eig_upper_bound_violation"], dtype=bool).sum()
    )
    return metrics


def build_target_eig_candidate_rows(
    dataset_metadata,
    diagnostics,
    selection_state,
    selected=None,
):
    if diagnostics is None:
        return []

    reviewable = np.asarray(selection_state["reviewable"], dtype=bool)
    selected = (
        np.zeros(reviewable.shape, dtype=bool)
        if selected is None
        else np.asarray(selected, dtype=bool)
    )
    site_ids = np.asarray(dataset_metadata["site_ids"], dtype=object)
    replicate_ids = np.asarray(dataset_metadata["replicate_ids"], dtype=object)
    replicate_times = np.asarray(dataset_metadata["replicate_times"], dtype=object)
    rows = []
    for site_index, period_index, replicate_index in np.argwhere(reviewable):
        candidate_id = int(
            np.ravel_multi_index(
                (site_index, period_index, replicate_index),
                reviewable.shape,
            )
        )
        rows.append(
            dict(
                candidate_id=candidate_id,
                site_index=int(site_index),
                site_id=str(site_ids[site_index]),
                period_index=int(period_index),
                replicate_index=int(replicate_index),
                replicate_id=str(replicate_ids[site_index, replicate_index]),
                replicate_time=str(replicate_times[site_index, replicate_index]),
                selected=bool(selected[site_index, period_index, replicate_index]),
                score=float(
                    selection_state["max_score"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                p_positive=float(
                    diagnostics["p_positive"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                target_eig_raw=float(
                    diagnostics["target_eig_raw"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                target_eig_bounded=float(
                    diagnostics["target_eig"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                target_eig=float(
                    diagnostics["target_eig"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                label_entropy=float(
                    diagnostics["label_entropy"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                ess_positive=float(
                    diagnostics["ess_positive"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                ess_negative=float(
                    diagnostics["ess_negative"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                bald_score=float(
                    selection_state["expected_gain_score"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
                ppu_score=float(
                    selection_state["posterior_predictive_uncertainty"][
                        site_index,
                        period_index,
                        replicate_index,
                    ]
                ),
            )
        )
    return rows


def select_parameter_bald_samples_site_diverse(
    cs_results, obs, site_covs, obs_covs, verified_f, n_reviews
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_uncalibrated_prob_positive(
            cs_results,
            obs,
            site_covs,
            obs_covs,
            verified_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_site_diverse_samples(acquisition_score, reviewable, n_reviews)


def select_parameter_bald_samples_calibrated(
    cs_results,
    score_log_bf,
    site_covs,
    obs_covs,
    verified_f,
    n_reviews,
    score_log_bf_draws=None,
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_calibrated_prob_positive(
            cs_results,
            score_log_bf,
            site_covs,
            obs_covs,
            verified_f,
            score_log_bf_draws=score_log_bf_draws,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


def select_parameter_bald_samples_calibrated_site_diverse(
    cs_results,
    score_log_bf,
    site_covs,
    obs_covs,
    verified_f,
    n_reviews,
    score_log_bf_draws=None,
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_calibrated_prob_positive(
            cs_results,
            score_log_bf,
            site_covs,
            obs_covs,
            verified_f,
            score_log_bf_draws=score_log_bf_draws,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_site_diverse_samples(acquisition_score, reviewable, n_reviews)


def select_posterior_predictive_uncertainty_samples(
    cs_results,
    obs,
    site_covs,
    obs_covs,
    verified_f,
    n_reviews,
):
    mean_prob_positive, _mean_prob_positive_entropy, reviewable = (
        _summarize_uncalibrated_prob_positive(
            cs_results,
            obs,
            site_covs,
            obs_covs,
            verified_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive)
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


def select_posterior_predictive_uncertainty_samples_calibrated(
    cs_results,
    score_log_bf,
    site_covs,
    obs_covs,
    verified_f,
    n_reviews,
    score_log_bf_draws=None,
):
    mean_prob_positive, _mean_prob_positive_entropy, reviewable = (
        _summarize_calibrated_prob_positive(
            cs_results,
            score_log_bf,
            site_covs,
            obs_covs,
            verified_f,
            score_log_bf_draws=score_log_bf_draws,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive)
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


def select_random_reviewable_samples(reviewable, n_reviews, random_key):
    reviewable = jnp.asarray(reviewable, dtype=bool)
    n_reviews = min(n_reviews, int(reviewable.sum()))
    if n_reviews == 0:
        return jnp.zeros(reviewable.shape, dtype=bool)

    selected = jnp.zeros(reviewable.shape, dtype=bool).reshape(-1)
    random_scores = jr.uniform(random_key, shape=selected.shape)
    selected_indices = jnp.argpartition(
        jnp.where(reviewable.reshape(-1), random_scores, jnp.inf),
        n_reviews - 1,
    )[:n_reviews]
    selected = selected.at[selected_indices].set(True)
    return selected.reshape(reviewable.shape)


def select_random_samples(
    score_source, site_covs, obs_covs, verified_f, n_reviews, random_key
):
    score_values = jnp.asarray(score_source[0])
    reviewable = (
        jnp.isfinite(score_values)
        & jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
        & ~jnp.isfinite(verified_f[0])
    )
    return select_random_reviewable_samples(reviewable, n_reviews, random_key)


def select_top_score_samples(score_source, site_covs, obs_covs, verified_f, n_reviews):
    score_values = jnp.asarray(score_source[0])
    reviewable = (
        jnp.isfinite(score_values)
        & jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
        & ~jnp.isfinite(verified_f[0])
    )
    return select_top_acquisition_samples(score_values, reviewable, n_reviews)


def select_top_score_site_diverse_samples(
    score_source, site_covs, obs_covs, verified_f, n_reviews
):
    score_values = jnp.asarray(score_source[0])
    reviewable = (
        jnp.isfinite(score_values)
        & jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
        & ~jnp.isfinite(verified_f[0])
    )
    return select_top_site_diverse_samples(score_values, reviewable, n_reviews)


def select_random_reviewed_labels(
    site_covs, obs_covs, reviewed_f, n_reviews, random_key
):
    reviewed = jnp.isfinite(reviewed_f[0])
    reviewable = (
        jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
        & ~reviewed
    )
    return select_random_reviewable_samples(reviewable, n_reviews, random_key)


def reviewed_label_prob_positive_draws(
    bernoulli_results,
    site_covs,
    obs_covs,
    reviewed_f,
):
    reviewed = jnp.isfinite(reviewed_f[0])
    reviewed_value = jnp.where(reviewed, reviewed_f[0], 0.0)
    reviewable = (
        jnp.isfinite(obs_covs).all(axis=-1)
        & jnp.isfinite(site_covs).all(axis=-1)[:, None, None]
        & ~reviewed
    )
    if not bool(reviewable.any()):
        return None, reviewable

    n_draws = len(bernoulli_results.samples["cov_det_0"])
    psi = jnp.asarray(bernoulli_results.samples["psi"]).reshape(
        n_draws, *reviewed_value.shape[:2]
    )
    alpha = jnp.stack(
        [
            jnp.asarray(bernoulli_results.samples[f"cov_det_{i}"]).reshape(n_draws, -1)[
                :, 0
            ]
            for i in range(obs_covs.shape[-1] + 1)
        ],
        axis=-1,
    )
    design_matrix = jnp.concatenate(
        [jnp.ones(obs_covs.shape[:-1] + (1,)), jnp.asarray(obs_covs)], axis=-1
    )
    design_matrix = jnp.where(jnp.isfinite(design_matrix), design_matrix, 0.0)
    prob_detection = 1.0 / (
        1.0 + jnp.exp(-jnp.einsum("sprc,kc->kspr", design_matrix, alpha))
    )
    log_p = jnp.log(jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    log_1mp = jnp.log1p(-jnp.clip(prob_detection, 1e-12, 1.0 - 1e-12))
    log_like1 = jnp.where(
        reviewed[None, ...],
        jnp.where(reviewed_value[None, ...] > 0.5, log_p, log_1mp),
        0.0,
    )
    log_like0 = jnp.where(
        reviewed[None, ...],
        jnp.where(reviewed_value[None, ...] > 0.5, -jnp.inf, 0.0),
        0.0,
    )
    logit_psi = jnp.log(jnp.clip(psi, 1e-12, 1.0 - 1e-12)) - jnp.log1p(
        -jnp.clip(psi, 1e-12, 1.0 - 1e-12)
    )
    posterior_z = 1.0 / (
        1.0 + jnp.exp(-(logit_psi + (log_like1 - log_like0).sum(axis=-1)))
    )
    return posterior_z[:, :, :, None] * prob_detection, reviewable


def _summarize_reviewed_label_prob_positive(
    bernoulli_results,
    site_covs,
    obs_covs,
    reviewed_f,
):
    prob_positive, reviewable = reviewed_label_prob_positive_draws(
        bernoulli_results,
        site_covs,
        obs_covs,
        reviewed_f,
    )
    if prob_positive is None:
        return None, None, reviewable
    return (
        prob_positive.mean(axis=0),
        binary_entropy(prob_positive).mean(axis=0),
        reviewable,
    )


def select_reviewed_label_parameter_bald_samples(
    bernoulli_results, site_covs, obs_covs, reviewed_f, n_reviews
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_reviewed_label_prob_positive(
            bernoulli_results,
            site_covs,
            obs_covs,
            reviewed_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


def build_reviewed_selection_state(
    bernoulli_results,
    score_source,
    site_covs,
    obs_covs,
    reviewed_f,
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_reviewed_label_prob_positive(
            bernoulli_results,
            site_covs,
            obs_covs,
            reviewed_f,
        )
    )
    max_score = np.asarray(score_source[0], dtype=float)
    reviewable = np.asarray(reviewable, dtype=bool)
    if mean_prob_positive is None:
        nan_scores = np.full(reviewable.shape, np.nan, dtype=float)
        return dict(
            reviewable=reviewable,
            mean_prob_positive=None,
            expected_gain_score=nan_scores,
            posterior_predictive_uncertainty=nan_scores,
            max_score=max_score,
        )

    expected_gain_score = (
        binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    )
    posterior_predictive_uncertainty = binary_entropy(mean_prob_positive)
    return dict(
        reviewable=reviewable,
        mean_prob_positive=np.asarray(mean_prob_positive, dtype=float),
        expected_gain_score=np.asarray(expected_gain_score, dtype=float),
        posterior_predictive_uncertainty=np.asarray(
            posterior_predictive_uncertainty, dtype=float
        ),
        max_score=max_score,
    )


def select_reviewed_label_parameter_bald_samples_site_diverse(
    bernoulli_results, site_covs, obs_covs, reviewed_f, n_reviews
):
    mean_prob_positive, mean_prob_positive_entropy, reviewable = (
        _summarize_reviewed_label_prob_positive(
            bernoulli_results,
            site_covs,
            obs_covs,
            reviewed_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive) - mean_prob_positive_entropy
    return select_top_site_diverse_samples(acquisition_score, reviewable, n_reviews)


def select_reviewed_label_posterior_predictive_uncertainty_samples(
    bernoulli_results,
    site_covs,
    obs_covs,
    reviewed_f,
    n_reviews,
):
    mean_prob_positive, _mean_prob_positive_entropy, reviewable = (
        _summarize_reviewed_label_prob_positive(
            bernoulli_results,
            site_covs,
            obs_covs,
            reviewed_f,
        )
    )
    if mean_prob_positive is None:
        return jnp.zeros(reviewable.shape, dtype=bool)
    acquisition_score = binary_entropy(mean_prob_positive)
    return select_top_acquisition_samples(acquisition_score, reviewable, n_reviews)


__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
    and name
    not in {
        "MCMC",
        "NUTS",
        "ScoreMixtureCalibration",
        "dist",
        "init_to_value",
        "jnp",
        "jr",
        "np",
        "numpyro",
    }
]
