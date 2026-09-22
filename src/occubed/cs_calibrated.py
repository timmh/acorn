from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, reshape_predictions
from biolith.utils.spatial import sample_spatial_effects
from occubed.priors import DEFAULT_LOGIT_COEF_PRIOR


def occu_cs_calibrated(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    score_log_bf: Optional[jnp.ndarray] = None,
    verified_f: Optional[jnp.ndarray] = None,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    n_species: int = 1,
    prior_beta: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    prior_alpha: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    r"""Occupancy model for externally calibrated continuous scores.

    This model assumes the score emission distributions have been fit outside
    the occupancy model. Each score is represented by the log Bayes factor

    .. math::
        \log \text{BF}_{ij} = \log p(s_{ij} \mid f_{ij}=1) - \log p(s_{ij} \mid f_{ij}=0)

    where :math:`f_{ij}` is the latent true detection indicator. With fixed
    score calibration, the occupancy model only needs the per-observation score
    evidence, so the score-mixture parameters are not sampled inside MCMC.
    """

    if score_log_bf is None and verified_f is None:
        raise ValueError("Provide at least one of `score_log_bf` or `verified_f`.")
    if score_log_bf is not None and score_log_bf.ndim != 4:
        raise ValueError(
            "score_log_bf must have shape (n_species, n_sites, n_periods, n_replicates)."
        )
    if verified_f is not None and verified_f.ndim != 4:
        raise ValueError(
            "verified_f must have shape (n_species, n_sites, n_periods, n_replicates)."
        )
    if site_covs.ndim != 2:
        raise ValueError("site_covs must have shape (n_sites, n_site_covs).")
    if obs_covs.ndim != 4:
        raise ValueError(
            "obs_covs must have shape (n_sites, n_periods, n_replicates, n_obs_covs)."
        )

    n_sites = site_covs.shape[0]
    n_periods = obs_covs.shape[1]
    n_replicates = obs_covs.shape[2]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[3]
    if score_log_bf is not None:
        n_species = score_log_bf.shape[0]
    elif verified_f is not None:
        n_species = verified_f.shape[0]

    if score_log_bf is not None and score_log_bf.shape[1:] != (
        n_sites,
        n_periods,
        n_replicates,
    ):
        raise ValueError(
            "score_log_bf shape must match site/period/replicate dimensions."
        )
    if verified_f is not None and verified_f.shape[1:] != (
        n_sites,
        n_periods,
        n_replicates,
    ):
        raise ValueError(
            "verified_f shape must match site/period/replicate dimensions."
        )

    obs_mask = (
        jnp.isnan(obs_covs).any(axis=-1)
        | jnp.isnan(site_covs).any(axis=-1)[:, None, None]
    )
    score_log_bf = (
        jnp.where(obs_mask[None, ...], jnp.nan, score_log_bf)
        if score_log_bf is not None
        else None
    )
    verified_f = (
        jnp.where(obs_mask[None, ...], jnp.nan, verified_f)
        if verified_f is not None
        else None
    )
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    if coords is not None:
        w = sample_spatial_effects(
            coords,
            ell=ell,
            prior_gp_sd=prior_gp_sd,
            prior_gp_length=prior_gp_length,
        )
    else:
        w = jnp.zeros(n_sites)

    if site_random_effects:
        site_re_sd = numpyro.sample("site_re_sd", prior_site_re_sd)
    if obs_random_effects:
        obs_re_sd = numpyro.sample("obs_re_sd", prior_obs_re_sd)

    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((3, 2, 1, 0))
    score_log_bf = (
        score_log_bf.transpose((3, 2, 1, 0)) if score_log_bf is not None else None
    )
    verified_f = verified_f.transpose((3, 2, 1, 0)) if verified_f is not None else None
    site_covs_flat, site_shape = flatten_covariates(site_covs)
    obs_covs_flat, obs_shape = flatten_covariates(obs_covs)

    with numpyro.plate("species", n_species, dim=-1):
        reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
        reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

        with numpyro.plate("site", n_sites, dim=-2):
            if site_random_effects:
                site_re_occ = numpyro.sample("site_re_occ", dist.Normal(0, site_re_sd))  # type: ignore
                site_re_det = numpyro.sample("site_re_det", dist.Normal(0, site_re_sd))  # type: ignore
            else:
                site_re_occ = 0.0
                site_re_det = 0.0

            occ_linear = (
                reshape_predictions(reg_occ(site_covs_flat), site_shape)
                + w[:, None]
                + site_re_occ
            )

            with numpyro.plate("period", n_periods, dim=-3):
                psi = numpyro.deterministic("psi", jax.nn.sigmoid(occ_linear))

                with numpyro.plate("replicate", n_replicates, dim=-4):
                    if obs_random_effects:
                        obs_re = numpyro.sample("obs_re", dist.Normal(0, obs_re_sd))  # type: ignore
                    else:
                        obs_re = 0.0

                    prob_detection = numpyro.deterministic(
                        "prob_detection",
                        jax.nn.sigmoid(
                            reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                            + site_re_det
                            + obs_re
                        ),
                    )

                    clipped_prob_detection = jnp.clip(prob_detection, 1e-12, 1 - 1e-12)
                    log_p = jnp.log(clipped_prob_detection)
                    log_1mp = jnp.log1p(-clipped_prob_detection)
                    log_like_occ = jnp.zeros_like(prob_detection)
                    log_like_unocc = jnp.zeros_like(prob_detection)

                    if verified_f is not None:
                        verified_mask = jnp.isfinite(verified_f)
                        verified_positive = verified_mask & (verified_f > 0.5)
                        verified_negative = verified_mask & ~verified_positive
                        log_like_occ = jnp.where(
                            verified_positive,
                            log_p,
                            log_like_occ,
                        )
                        log_like_occ = jnp.where(
                            verified_negative,
                            log_1mp,
                            log_like_occ,
                        )
                        log_like_unocc = jnp.where(
                            verified_positive,
                            -jnp.inf,
                            log_like_unocc,
                        )
                    else:
                        verified_mask = jnp.zeros_like(prob_detection, dtype=bool)

                    if score_log_bf is not None:
                        score_mask = jnp.isfinite(score_log_bf) & ~verified_mask
                        score_log_bf_value = jnp.where(score_mask, score_log_bf, 0.0)
                        log_like_occ = jnp.where(
                            score_mask,
                            jnp.logaddexp(log_1mp, log_p + score_log_bf_value),
                            log_like_occ,
                        )

                log_like_occ = log_like_occ.sum(axis=0)
                log_like_unocc = log_like_unocc.sum(axis=0)
                clipped_psi = jnp.clip(psi, 1e-12, 1 - 1e-12)
                numpyro.factor(
                    "score_obs",
                    jnp.logaddexp(
                        jnp.log1p(-clipped_psi) + log_like_unocc,
                        jnp.log(clipped_psi) + log_like_occ,
                    ),
                )


def test_occu_cs_calibrated():
    from biolith.utils import fit
    from occubed.score_calibration import fit_score_mixture, score_log_bayes_factor

    data, true_params = simulate_cs_for_calibrated(n_sites=40, simulate_missing=True)
    calibration = fit_score_mixture(data["obs"], true_params["f"])
    results = fit(
        occu_cs_calibrated,
        site_covs=data["site_covs"],
        obs_covs=data["obs_covs"],
        score_log_bf=score_log_bayes_factor(data["obs"], calibration),
        num_chains=1,
        num_samples=300,
        num_warmup=300,
        timeout=600,
    )
    assert np.allclose(
        results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
    )


def test_occu_cs_calibrated_with_verified_labels():
    from biolith.utils import fit
    from occubed.score_calibration import fit_score_mixture, score_log_bayes_factor

    data, true_params = simulate_cs_for_calibrated(n_sites=30)
    rng = np.random.default_rng(0)
    verified_f = true_params["f"].astype(float)
    verified_mask = rng.random(size=verified_f.shape) < 0.3
    verified_f = np.where(verified_mask, verified_f, np.nan)
    calibration = fit_score_mixture(data["obs"], true_params["f"])
    results = fit(
        occu_cs_calibrated,
        site_covs=data["site_covs"],
        obs_covs=data["obs_covs"],
        score_log_bf=score_log_bayes_factor(data["obs"], calibration),
        verified_f=verified_f,
        num_chains=1,
        num_samples=200,
        num_warmup=200,
        timeout=600,
    )
    assert np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.2)


def simulate_cs_for_calibrated(**kwargs):
    from .cs_models import simulate_cs

    return simulate_cs(**kwargs)
