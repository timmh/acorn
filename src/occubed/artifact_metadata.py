import numpy as np


def build_real_data_dataset_metadata(data, true_params):
    """Build static dataset metadata used by saved experiment artifacts."""

    site_ids = np.asarray(true_params["site_ids"], dtype=object)
    replicate_ids = np.asarray(true_params["replicate_ids"], dtype=object)
    replicate_times = np.asarray(
        true_params.get("replicate_times", replicate_ids),
        dtype=object,
    )
    score_values = np.asarray(data["obs"][0], dtype=float)
    site_covs = np.asarray(data["site_covs"], dtype=float)
    obs_covs = np.asarray(data["obs_covs"], dtype=float)
    site_valid = np.isfinite(site_covs).all(axis=-1)[:, None, None]
    obs_valid = np.isfinite(obs_covs).all(axis=-1) if obs_covs.ndim == 4 else True
    reviewable_mask = np.isfinite(score_values) & site_valid & obs_valid

    site_covariates_raw = np.asarray(
        true_params.get("site_covariates_raw", site_covs),
        dtype=float,
    )
    site_covariate_names = [
        str(name) for name in np.asarray(true_params.get("site_covariate_names", []))
    ]
    site_covariate_means = np.asarray(
        true_params.get(
            "site_covariate_means",
            np.zeros(site_covariates_raw.shape[1], dtype=float),
        ),
        dtype=float,
    )
    site_covariate_scales = np.asarray(
        true_params.get(
            "site_covariate_scales",
            np.ones(site_covariates_raw.shape[1], dtype=float),
        ),
        dtype=float,
    )
    site_coordinates = np.asarray(
        true_params.get("site_coordinates", site_covariates_raw[:, :2]),
        dtype=float,
    )

    n_obs_covs = obs_covs.shape[-1]
    obs_covariate_names = [
        str(name)
        for name in np.asarray(
            true_params.get(
                "obs_covariate_names",
                [f"obs_cov_{index + 1}" for index in range(n_obs_covs)],
            )
        )
    ]
    if n_obs_covs > 0:
        obs_covariates_raw = np.asarray(
            true_params.get("obs_covariates_raw", obs_covs.reshape(-1, n_obs_covs)),
            dtype=float,
        )
        if obs_covariates_raw.ndim > 2:
            obs_covariates_raw = obs_covariates_raw.reshape(-1, n_obs_covs)
        obs_covariate_offsets = np.asarray(
            true_params.get(
                "obs_covariate_offsets",
                np.zeros(n_obs_covs, dtype=float),
            ),
            dtype=float,
        )
        obs_covariate_scales = np.asarray(
            true_params.get(
                "obs_covariate_scales",
                np.ones(n_obs_covs, dtype=float),
            ),
            dtype=float,
        )
    else:
        obs_covariates_raw = np.zeros((0, 0), dtype=float)
        obs_covariate_offsets = np.zeros((0,), dtype=float)
        obs_covariate_scales = np.ones((0,), dtype=float)

    return dict(
        dataset_name=str(true_params["dataset_name"]),
        target_label=str(true_params["target_label"]),
        site_ids=site_ids,
        replicate_ids=replicate_ids,
        replicate_times=replicate_times,
        score_values=score_values,
        reviewable_mask=reviewable_mask,
        site_coordinates=site_coordinates,
        site_covariates_raw=site_covariates_raw,
        site_covariate_names=site_covariate_names,
        site_covariate_means=site_covariate_means,
        site_covariate_scales=site_covariate_scales,
        obs_covariates_raw=obs_covariates_raw,
        obs_covariate_names=obs_covariate_names,
        obs_covariate_offsets=obs_covariate_offsets,
        obs_covariate_scales=obs_covariate_scales,
    )
