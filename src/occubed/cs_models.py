from typing import Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, reshape_predictions
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects
from occubed.priors import DEFAULT_LOGIT_COEF_PRIOR


def _normal_logpdf_np(values, loc, scale):
    scale = np.maximum(float(scale), 1e-12)
    standardized = (values - float(loc)) / scale
    return -0.5 * standardized**2 - np.log(scale)


def _score_class_stats(values, fallback_scale, min_sigma):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, float(max(fallback_scale, min_sigma))
    scale = values.std(ddof=1) if values.size > 1 else fallback_scale
    return float(values.mean()), float(max(scale, min_sigma))


def _score_mixture_initialization(
    obs,
    verified_f=None,
    *,
    min_sigma=1e-3,
    min_mean_separation=1e-3,
    max_iter=100,
    tol=1e-6,
):
    obs_arr = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs_arr)
    if not np.any(valid):
        return 0.0, 1.0, min_mean_separation, 1.0

    scores = obs_arr[valid]
    fallback_scale = float(scores.std(ddof=1)) if scores.size > 1 else 1.0
    fallback_scale = max(fallback_scale, min_sigma)
    if scores.size == 1:
        mu0 = float(scores[0])
        return mu0, fallback_scale, mu0 + min_mean_separation, fallback_scale

    if verified_f is None:
        labels = np.full(scores.shape, np.nan, dtype=float)
    else:
        labels = np.asarray(verified_f, dtype=float)[valid]
    labeled = np.isfinite(labels)
    labeled_positive = labeled & (labels > 0.5)
    labeled_negative = labeled & ~labeled_positive

    quantiles = np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9])
    lower_scores = scores[scores <= quantiles[2]]
    upper_scores = scores[scores > quantiles[2]]
    lower_mu, lower_sigma = _score_class_stats(
        lower_scores,
        fallback_scale,
        min_sigma,
    )
    upper_mu, upper_sigma = _score_class_stats(
        upper_scores,
        fallback_scale,
        min_sigma,
    )

    initializations = [
        (lower_mu, lower_sigma, upper_mu, upper_sigma, 0.25),
        (
            float(quantiles[0]),
            fallback_scale,
            float(quantiles[-1]),
            fallback_scale,
            0.1,
        ),
        (
            float(quantiles[1]),
            fallback_scale,
            float(quantiles[-1]),
            fallback_scale,
            0.2,
        ),
    ]
    if labeled_negative.any():
        neg_mu, neg_sigma = _score_class_stats(
            scores[labeled_negative],
            fallback_scale,
            min_sigma,
        )
        initializations.extend(
            [
                (neg_mu, neg_sigma, upper_mu, upper_sigma, 0.1),
                (neg_mu, neg_sigma, float(quantiles[-1]), fallback_scale, 0.1),
            ]
        )
    if labeled_positive.any():
        pos_mu, pos_sigma = _score_class_stats(
            scores[labeled_positive],
            fallback_scale,
            min_sigma,
        )
        initializations.extend(
            [
                (lower_mu, lower_sigma, pos_mu, pos_sigma, 0.1),
                (float(quantiles[0]), fallback_scale, pos_mu, pos_sigma, 0.1),
            ]
        )
    if labeled_negative.any() and labeled_positive.any():
        neg_mu, neg_sigma = _score_class_stats(
            scores[labeled_negative],
            fallback_scale,
            min_sigma,
        )
        pos_mu, pos_sigma = _score_class_stats(
            scores[labeled_positive],
            fallback_scale,
            min_sigma,
        )
        prevalence = float(
            np.clip(labeled_positive.sum() / labeled.sum(), 1e-6, 1 - 1e-6)
        )
        initializations.append((neg_mu, neg_sigma, pos_mu, pos_sigma, prevalence))

    positive_label = labels > 0.5

    def run_em(initialization):
        mu0, sigma0, mu1, sigma1, prevalence = initialization
        if mu1 <= mu0:
            midpoint = 0.5 * (mu0 + mu1)
            mu0 = midpoint - 0.5 * min_mean_separation
            mu1 = midpoint + 0.5 * min_mean_separation
        sigma0 = max(float(sigma0), min_sigma)
        sigma1 = max(float(sigma1), min_sigma)
        prevalence = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))

        for _ in range(max_iter):
            old_params = np.asarray([mu0, sigma0, mu1, sigma1, prevalence])
            log_weight0 = np.log1p(-prevalence) + _normal_logpdf_np(
                scores,
                mu0,
                sigma0,
            )
            log_weight1 = np.log(prevalence) + _normal_logpdf_np(scores, mu1, sigma1)
            posterior_positive = np.exp(
                log_weight1 - np.logaddexp(log_weight0, log_weight1)
            )
            responsibilities = np.where(
                labeled,
                positive_label.astype(float),
                posterior_positive,
            )
            resp1 = np.clip(responsibilities, 1e-6, 1.0 - 1e-6)
            resp0 = 1.0 - resp1
            weight0 = float(resp0.sum())
            weight1 = float(resp1.sum())
            if weight0 <= 0.0 or weight1 <= 0.0:
                break

            prevalence = float(np.clip(weight1 / (weight0 + weight1), 1e-6, 1 - 1e-6))
            mu0 = float(np.sum(resp0 * scores) / weight0)
            mu1 = float(np.sum(resp1 * scores) / weight1)
            sigma0 = float(
                np.sqrt(
                    max(
                        np.sum(resp0 * np.square(scores - mu0)) / weight0,
                        min_sigma**2,
                    )
                )
            )
            sigma1 = float(
                np.sqrt(
                    max(
                        np.sum(resp1 * np.square(scores - mu1)) / weight1,
                        min_sigma**2,
                    )
                )
            )
            if mu1 <= mu0:
                midpoint = 0.5 * (mu0 + mu1)
                mu0 = midpoint - 0.5 * min_mean_separation
                mu1 = midpoint + 0.5 * min_mean_separation

            new_params = np.asarray([mu0, sigma0, mu1, sigma1, prevalence])
            if np.max(np.abs(new_params - old_params)) < tol:
                break

        log_weight0 = np.log1p(-prevalence) + _normal_logpdf_np(scores, mu0, sigma0)
        log_weight1 = np.log(prevalence) + _normal_logpdf_np(scores, mu1, sigma1)
        log_likelihood = np.where(
            labeled,
            np.where(positive_label, log_weight1, log_weight0),
            np.logaddexp(log_weight0, log_weight1),
        ).sum()
        return float(log_likelihood), mu0, sigma0, mu1, sigma1

    best = max(
        (run_em(initialization) for initialization in initializations),
        key=lambda x: x[0],
    )
    _, mu0, sigma0, mu1, sigma1 = best
    if mu1 <= mu0:
        midpoint = 0.5 * (mu0 + mu1)
        mu0 = midpoint - 0.5 * min_mean_separation
        mu1 = midpoint + 0.5 * min_mean_separation
    return (
        float(mu0),
        float(max(sigma0, min_sigma)),
        float(mu1),
        float(max(sigma1, min_sigma)),
    )


def occu_cs_score_init_values(
    obs,
    verified_f=None,
    *,
    score_loc=None,
    score_scale=None,
    min_sigma=1e-3,
    min_mean_separation=1e-3,
):
    """Build constrained initial values for continuous-score parameters.

    The returned dictionary is suitable for
    :func:`numpyro.infer.initialization.init_to_value`.
    """

    obs_arr = np.asarray(obs, dtype=float)
    finite = np.isfinite(obs_arr)
    if score_loc is None or score_scale is None:
        if np.any(finite):
            observed_scores = obs_arr[finite]
            score_loc = float(observed_scores.mean())
            score_scale = float(max(observed_scores.std(ddof=0), 1e-6))
        else:
            score_loc = 0.0 if score_loc is None else float(score_loc)
            score_scale = 1.0 if score_scale is None else float(score_scale)
    score_scale = float(max(score_scale, 1e-6))

    mu0, sigma0, mu1, sigma1 = _score_mixture_initialization(
        obs_arr,
        verified_f,
        min_sigma=min_sigma,
        min_mean_separation=min_mean_separation,
    )
    gap = max(mu1 - mu0, min_mean_separation)
    return {
        "mu0_std": jnp.asarray((mu0 - score_loc) / score_scale),
        "mu_gap_std": jnp.asarray(gap / score_scale),
        "sigma0_std": jnp.asarray(max(sigma0 / score_scale, min_sigma)),
        "sigma1_std": jnp.asarray(max(sigma1 / score_scale, min_sigma)),
    }


def occu_cs_init_strategy(
    obs,
    verified_f=None,
    *,
    score_loc=None,
    score_scale=None,
    min_sigma=1e-3,
    min_mean_separation=1e-3,
):
    """Return an ``init_to_value`` strategy for the original continuous-score model."""

    from numpyro.infer.initialization import init_to_value

    return init_to_value(
        values=occu_cs_score_init_values(
            obs,
            verified_f,
            score_loc=score_loc,
            score_scale=score_scale,
            min_sigma=min_sigma,
            min_mean_separation=min_mean_separation,
        )
    )


def occu_cs(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    obs: Optional[jnp.ndarray] = None,
    verified_f: Optional[jnp.ndarray] = None,
    score_loc: Optional[float] = None,
    score_scale: Optional[float] = None,
    n_species: int = 1,
    prior_beta: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    prior_alpha: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_mu: dist.Distribution | Tuple[dist.Distribution] = dist.Normal(0, 10),
    prior_sigma: dist.Distribution | Tuple[dist.Distribution] = dist.Gamma(5, 1),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Continuous-score occupancy model inspired by Rhinehart et al. (2022), modeling
    classification scores as being drawn from true or false positive distributions.

    References
    ----------
        - Rhinehart, T. A., Turek, D., & Kitzes, J. (2022). A continuous-score occupancy model that incorporates uncertain machine learning output from autonomous biodiversity surveys. Methods in Ecology and Evolution, 13, 1778–1789.

    Parameters
    ----------
    site_covs : jnp.ndarray
        Site-level covariates (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        Observation covariates (n_sites, n_periods, n_replicates, n_obs_covs).
    coords : Optional[jnp.ndarray]
        Site coordinates for spatial effects (n_sites, 2) or None.
    ell : float
        Optional distance matrix parameter.
    obs : Optional[jnp.ndarray]
        Observations (n_species, n_sites, n_periods, n_replicates) or None.
    verified_f : Optional[jnp.ndarray]
        Human-verified latent detections of shape
        (n_species, n_sites, n_periods, n_replicates). Use ``NaN`` for
        unverified entries.
    score_loc : Optional[float]
        Optional mean used to standardize continuous scores before fitting.
        When omitted, it is estimated from ``obs`` inside the model.
    score_scale : Optional[float]
        Optional scale used to standardize continuous scores before fitting.
        When omitted, it is estimated from ``obs`` inside the model.
    n_species : int
        Number of species. Used when ``obs`` is None.
    prior_beta : dist.Distribution
        Prior for occupancy coefficients.
    prior_alpha : dist.Distribution
        Prior for detection coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.
    prior_mu : dist.Distribution or Tuple[dist.Distribution]
        Prior for the false-positive score mean ``mu0`` and, when provided as a
        tuple, the scale of the positive mean separation ``mu1 - mu0``.
    prior_sigma : dist.Distribution or Tuple[dist.Distribution]
        Prior for standard deviations of continuous scores.
    prior_gp_sd : dist.Distribution
        Prior distribution for the spatial random effect scale.
    prior_gp_length : dist.Distribution
        Prior distribution for the spatial kernel length scale.
    site_random_effects : bool
        Flag indicating whether to include site-level random effects.
    obs_random_effects : bool
        Flag indicating whether to include observation-level random effects.
    prior_site_re_sd : dist.Distribution
        Prior distribution for the site-level random effect standard deviation.
    prior_obs_re_sd : dist.Distribution
        Prior distribution for the observation-level random effect standard deviation.

    Examples
    --------
    >>> from biolith.models import occu_cs, simulate_cs
    >>> from biolith.utils import fit
    >>> data, _ = simulate_cs()
    >>> results = fit(occu_cs, **data)
    >>> print(results.samples['psi'].mean())
    """

    # Check input data
    assert (
        obs is None or obs.ndim == 4
    ), "obs must be None or of shape (n_species, n_sites, n_periods, n_replicates)"
    assert (
        verified_f is None or verified_f.ndim == 4
    ), "verified_f must be None or of shape (n_species, n_sites, n_periods, n_replicates)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 4
    ), "obs_covs must be of shape (n_sites, n_periods, n_replicates, n_obs_covs)"

    n_sites = site_covs.shape[0]
    n_periods = obs_covs.shape[1]
    n_replicates = obs_covs.shape[2]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[3]
    if obs is not None:
        n_species = obs.shape[0]
    elif verified_f is not None:
        n_species = verified_f.shape[0]

    assert (
        n_sites == site_covs.shape[0] == obs_covs.shape[0]
    ), "site_covs and obs_covs must have the same number of sites"
    assert (
        n_periods == obs_covs.shape[1]
    ), "obs_covs must have the same number of periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[1], "obs must have n_sites rows"
        assert n_periods == obs.shape[2], "obs must have n_periods columns"
        assert n_replicates == obs.shape[3], "obs must have n_replicates columns"
    if verified_f is not None:
        assert n_sites == verified_f.shape[1], "verified_f must have n_sites rows"
        assert (
            n_periods == verified_f.shape[2]
        ), "verified_f must have n_periods columns"
        assert (
            n_replicates == verified_f.shape[3]
        ), "verified_f must have n_replicates columns"
    # Mask observations where covariates are missing
    obs_mask = (
        jnp.isnan(obs_covs).any(axis=-1)
        | jnp.isnan(site_covs).any(axis=-1)[:, None, None]
    )
    obs = jnp.where(obs_mask[None, ...], jnp.nan, obs) if obs is not None else None
    verified_f = (
        jnp.where(obs_mask[None, ...], jnp.nan, verified_f)
        if verified_f is not None
        else None
    )
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Standardize scores to improve the geometry of the Normal likelihood while
    # keeping reported parameters on the original scale.
    if obs is not None:
        obs_is_finite = jnp.isfinite(obs)
        if score_loc is None or score_scale is None:
            n_obs = obs_is_finite.sum()
            obs_sum = jnp.where(obs_is_finite, obs, 0.0).sum()
            score_loc = jnp.where(n_obs > 0, obs_sum / n_obs, 0.0)
            centered_obs = jnp.where(obs_is_finite, obs - score_loc, 0.0)
            obs_var = jnp.where(n_obs > 0, jnp.square(centered_obs).sum() / n_obs, 1.0)
            score_scale = jnp.maximum(jnp.sqrt(obs_var), 1e-6)
        else:
            score_loc = jnp.asarray(score_loc)
            score_scale = jnp.maximum(jnp.asarray(score_scale), 1e-6)
        obs = jnp.where(obs_is_finite, (obs - score_loc) / score_scale, jnp.nan)
    else:
        score_loc = 0.0 if score_loc is None else jnp.asarray(score_loc)
        score_scale = (
            1.0 if score_scale is None else jnp.maximum(jnp.asarray(score_scale), 1e-6)
        )

    if coords is not None:
        w = sample_spatial_effects(
            coords,
            ell=ell,
            prior_gp_sd=prior_gp_sd,
            prior_gp_length=prior_gp_length,
        )
    else:
        w = jnp.zeros(n_sites)

    # Random effects standard deviations (sampled before plates)
    if site_random_effects:
        site_re_sd = numpyro.sample("site_re_sd", prior_site_re_sd)
    if obs_random_effects:
        obs_re_sd = numpyro.sample("obs_re_sd", prior_obs_re_sd)

    # Continuous score parameters. HMC initializes real-valued sites near zero,
    # so sample these in standardized score units and transform summaries back
    # to the original score scale.
    prior_mus = prior_mu if isinstance(prior_mu, tuple) else (prior_mu, prior_mu)
    score_to_standardized = dist.transforms.AffineTransform(
        -score_loc / score_scale,
        1.0 / score_scale,
    )
    positive_score_to_standardized = dist.transforms.AffineTransform(
        0.0,
        1.0 / score_scale,
    )
    mu0_std = numpyro.sample(
        "mu0_std",
        dist.TransformedDistribution(prior_mus[0], score_to_standardized),
    )
    mu0 = numpyro.deterministic("mu0", score_loc + score_scale * mu0_std)
    try:
        mu_gap_scale = jnp.sqrt(prior_mus[1].variance)  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError):
        mu_gap_scale = 10.0
    mu_gap_std = numpyro.sample(
        "mu_gap_std",
        dist.TransformedDistribution(
            dist.HalfNormal(jnp.maximum(mu_gap_scale, 1e-6)),
            positive_score_to_standardized,
        ),
    )
    mu_gap = numpyro.deterministic("mu_gap", score_scale * mu_gap_std)
    mu1 = numpyro.deterministic("mu1", mu0 + mu_gap)
    prior_sigmas = (
        prior_sigma if isinstance(prior_sigma, tuple) else (prior_sigma, prior_sigma)
    )
    sigma0_std = numpyro.sample(
        "sigma0_std",
        dist.TransformedDistribution(
            prior_sigmas[0],  # type: ignore[arg-type]
            positive_score_to_standardized,
        ),
    )
    sigma1_std = numpyro.sample(
        "sigma1_std",
        dist.TransformedDistribution(
            prior_sigmas[1],  # type: ignore[arg-type]
            positive_score_to_standardized,
        ),
    )
    sigma0 = numpyro.deterministic("sigma0", score_scale * sigma0_std)
    sigma1 = numpyro.deterministic("sigma1", score_scale * sigma1_std)
    mu1_std = mu0_std + mu_gap_std

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((3, 2, 1, 0))
    obs = obs.transpose((3, 2, 1, 0)) if obs is not None else None
    verified_f = verified_f.transpose((3, 2, 1, 0)) if verified_f is not None else None

    with numpyro.plate("species", n_species, dim=-1):

        # Occupancy and detection regression models
        reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
        reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

        with numpyro.plate("site", n_sites, dim=-2) as site_idx:
            site_covs_batch = jnp.take(site_covs, site_idx, axis=1)
            obs_covs_batch = jnp.take(obs_covs, site_idx, axis=-1)
            obs_batch = jnp.take(obs, site_idx, axis=-2) if obs is not None else None
            verified_f_batch = (
                jnp.take(verified_f, site_idx, axis=-2)
                if verified_f is not None
                else None
            )
            w_batch = jnp.take(w, site_idx, axis=0)
            site_covs_flat, site_shape = flatten_covariates(site_covs_batch)
            obs_covs_flat, obs_shape = flatten_covariates(obs_covs_batch)

            # Site-level random effects
            if site_random_effects:
                site_re_occ = numpyro.sample("site_re_occ", dist.Normal(0, site_re_sd))  # type: ignore
                site_re_det = numpyro.sample("site_re_det", dist.Normal(0, site_re_sd))  # type: ignore
            else:
                site_re_occ = 0.0
                site_re_det = 0.0

            occ_linear = (
                reshape_predictions(reg_occ(site_covs_flat), site_shape)
                + w_batch[:, None]
                + site_re_occ
            )

            with numpyro.plate("period", n_periods, dim=-3):

                # Occupancy process
                psi = numpyro.deterministic("psi", jax.nn.sigmoid(occ_linear))
                z = numpyro.sample(
                    "z", dist.Bernoulli(probs=psi), infer={"enumerate": "parallel"}  # type: ignore
                )

                with numpyro.plate("replicate", n_replicates, dim=-4):

                    # Observation-level random effects
                    if obs_random_effects:
                        obs_re = numpyro.sample("obs_re", dist.Normal(0, obs_re_sd))  # type: ignore
                    else:
                        obs_re = 0.0

                    # Detection process. Marginalize the latent detection indicator
                    # f analytically to avoid a second enumerated variable per score.
                    p_det = jax.nn.sigmoid(
                        reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                        + site_re_det
                        + obs_re
                    )
                    p_det = jnp.clip(p_det, 1e-8, 1.0 - 1e-8)
                    log_p = jnp.log(p_det)
                    log_1mp = jnp.log1p(-p_det)

                    score_present = (
                        jnp.isfinite(obs_batch)
                        if obs_batch is not None
                        else jnp.zeros(p_det.shape, dtype=bool)
                    )
                    s_obs = (
                        jnp.where(score_present, obs_batch, 0.0)
                        if obs_batch is not None
                        else jnp.zeros(p_det.shape)
                    )
                    label_present = (
                        jnp.isfinite(verified_f_batch)
                        if verified_f is not None
                        else jnp.zeros(p_det.shape, dtype=bool)
                    )
                    y = (
                        jnp.where(label_present, verified_f_batch, 0.0)
                        if verified_f is not None
                        else jnp.zeros(p_det.shape)
                    )
                    y_pos = y > 0.5

                    log_s0_raw = dist.Normal(mu0_std, sigma0_std).log_prob(s_obs)
                    log_s1_raw = dist.Normal(mu1_std, sigma1_std).log_prob(s_obs)
                    log_s0 = jnp.where(score_present, log_s0_raw, 0.0)
                    log_s1 = jnp.where(score_present, log_s1_raw, 0.0)

                    ll_z1_verified = jnp.where(
                        y_pos,
                        log_p + log_s1,
                        log_1mp + log_s0,
                    )
                    ll_z0_verified = jnp.where(y_pos, -jnp.inf, log_s0)

                    ll_z1_unverified = jnp.where(
                        score_present,
                        jnp.logaddexp(log_1mp + log_s0_raw, log_p + log_s1_raw),
                        0.0,
                    )
                    ll_z0_unverified = log_s0

                    ll_z1 = jnp.where(
                        label_present,
                        ll_z1_verified,
                        ll_z1_unverified,
                    )
                    ll_z0 = jnp.where(
                        label_present,
                        ll_z0_verified,
                        ll_z0_unverified,
                    )
                    numpyro.factor(
                        "score_label_likelihood",
                        jnp.where(z > 0.5, ll_z1, ll_z0),
                    )


def simulate_cs(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_periods: int = 1,
    n_species: int = 1,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    simulate_missing: bool = False,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    mu0: float = 0.0,
    sigma0: float = 10.0,
    mu1: float = 10.0,
    sigma1: float = 5.0,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
) -> tuple[dict, dict]:
    """Simulate data for :func:`occu_cs`.

    Returns ``(data, true_params)`` for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate_cs
    >>> data, params = simulate_cs()
    >>> sorted(data.keys())
    ['coords', 'ell', 'obs', 'obs_covs', 'site_covs']
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    if spatial:
        coords = rng.uniform(0, 1, size=(n_sites, 2))
    else:
        coords = None

    # Make sure occupancy and detection are not too close to 0 or 1
    z = None
    while z is None or z.mean() < min_occupancy or z.mean() > max_occupancy:

        # Generate intercept and slopes
        beta = rng.normal(
            size=(n_species, n_site_covs + 1)
        )  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(
            size=(n_species, n_obs_covs + 1)
        )  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        if spatial and coords is not None:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0
        psi = 1 / (
            1
            + np.exp(
                -(
                    beta[:, 0][:, None]
                    + np.tensordot(beta[:, 1:], site_covs, axes=([1], [1]))
                    + w[None, :]
                )
            )
        )
        z = rng.binomial(
            n=1, p=psi[:, None, :], size=(n_species, n_periods, n_sites)
        )  # matrix of latent occupancy status for each site and period

        # Generate detection data
        n_replicates = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, n_periods, n_replicates, n_obs_covs))
        p = 1 / (
            1
            + np.exp(
                -(
                    alpha[:, 0][:, None, None, None]
                    + np.tensordot(alpha[:, 1:], obs_covs, axes=([1], [3]))
                )
            )
        )

        # Create matrix of detections
        obs = np.zeros((n_species, n_sites, n_periods, n_replicates))

        z_site = z.transpose(0, 2, 1)
        f = rng.binomial(
            n=1,
            p=p * z_site[..., None],
            size=(n_species, n_sites, n_periods, n_replicates),
        )
        obs = rng.normal(
            loc=np.where(f == 1, mu1, mu0),
            scale=np.where(f == 1, sigma1, sigma0),
            size=(n_species, n_sites, n_periods, n_replicates),
        )

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = (
                np.nan
            )
            site_covs[
                rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])
            ] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        coords=coords,
        ell=ell,
    ), dict(
        z=z,
        psi=psi,
        p=p,
        f=f,
        beta=beta,
        alpha=alpha,
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )


def test_occu():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(occu_cs, **data, timeout=600)

    assert np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1)
    assert np.allclose(
        [
            results.samples[k].mean()
            for k in [f"cov_state_{i}" for i in range(true_params["beta"].shape[1])]
        ],
        true_params["beta"].mean(axis=0),
        atol=0.5,
    )
    assert np.allclose(
        [
            results.samples[k].mean()
            for k in [f"cov_det_{i}" for i in range(true_params["alpha"].shape[1])]
        ],
        true_params["alpha"].mean(axis=0),
        atol=0.5,
    )
    assert np.allclose(results.samples["mu0"].mean(), true_params["mu0"], atol=1)
    assert np.allclose(results.samples["mu1"].mean(), true_params["mu1"], atol=1)
    assert np.allclose(results.samples["sigma0"].mean(), true_params["sigma0"], atol=1)
    assert np.allclose(results.samples["sigma1"].mean(), true_params["sigma1"], atol=1)


def test_occu_partial_verification_initialization():
    import jax
    from numpyro.infer.initialization import init_to_uniform
    from numpyro.infer.util import initialize_model

    data, true_params = simulate_cs(n_sites=30)
    rng = np.random.default_rng(0)
    verified_f = true_params["f"].astype(float)
    verified_mask = rng.random(size=verified_f.shape) < 0.3
    verified_f = np.where(verified_mask, verified_f, np.nan)

    initialize_model(
        jax.random.PRNGKey(0),
        occu_cs,
        init_strategy=init_to_uniform(),
        model_kwargs={**data, "verified_f": verified_f},
        validate_grad=True,
    )


def test_occu_multi_season():
    data, true_params = simulate_cs(simulate_missing=True, n_periods=3)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        num_chains=1,
        num_samples=300,
        num_warmup=300,
        timeout=600,
    )

    assert np.allclose(
        results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
    )


def test_occu_multi_species():
    data, _ = simulate_cs(simulate_missing=True, n_species=2, n_sites=30)

    from biolith.utils import fit

    results = fit(occu_cs, **data, num_chains=1, num_samples=200, timeout=600)

    assert results.samples["psi"].shape[-1] == 2


def test_occu_spatial():
    data, true_params = simulate_cs(simulate_missing=True, spatial=True)

    from biolith.utils import fit

    results = fit(occu_cs, **data, timeout=600)

    assert np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1)
    assert np.allclose(results.samples["gp_sd"].mean(), true_params["gp_sd"], atol=1.0)
    assert np.allclose(results.samples["gp_l"].mean(), true_params["gp_l"], atol=0.5)


def test_site_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        site_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert "site_re_sd" in results.samples
    assert "site_re_occ" in results.samples
    assert "site_re_det" in results.samples
    assert results.samples["site_re_sd"].mean() > 0
    assert np.allclose(
        results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
    )


def test_obs_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        obs_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert "obs_re_sd" in results.samples
    assert "obs_re" in results.samples
    assert results.samples["obs_re_sd"].mean() > 0
    assert np.allclose(
        results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
    )


def test_combined_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        site_random_effects=True,
        obs_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert "site_re_sd" in results.samples
    assert "site_re_occ" in results.samples
    assert "site_re_det" in results.samples
    assert "obs_re_sd" in results.samples
    assert "obs_re" in results.samples
    assert np.allclose(
        results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
    )
