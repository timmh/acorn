import csv
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from operator import attrgetter

import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
from numpyro.diagnostics import summary as numpyro_summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biolith.utils import fit
from occubed.artifact_metadata import build_real_data_dataset_metadata
from occubed.datasets import (
    load_acoustic_perch_dataset,
    load_iwildcam2022_speciesnet_dataset,
)
from occubed.fit_utils import disable_numpyro_mcmc_caching
from occubed.models import occu, occu_cs, occu_cs_calibrated, occu_cs_init_strategy
from occubed.paths import ARTIFACTS_DIR
from occubed.review_policies import *  # noqa: F403
from occubed.summaries import write_parameter_summary_npz

disable_numpyro_mcmc_caching()

DEFAULT_MCMC_KWARGS = dict(
    num_warmup=500,
    num_samples=500,
    num_chains=3,
    random_seed=0,
)
DEFAULT_N_REVIEWS_PER_STEP = 5
DEFAULT_MAX_REVIEWS = 100
DEFAULT_DATASET_SPECS = [
    dict(dataset_kind="acoustic", target_species="BTBW"),
    dict(dataset_kind="iwildcam", target_species="odocoileus virginianus"),
]
SUMMARY_STAT_NAME_MAP = {
    "mean": "mean",
    "std": "std",
    "median": "median",
    "5.0%": "q05",
    "95.0%": "q95",
    "n_eff": "n_eff",
    "r_hat": "r_hat",
}


def posterior_wasserstein_distance(samples, oracle_samples):
    n_draws = min(len(samples), len(oracle_samples))
    samples = jnp.asarray(samples)[:n_draws].reshape(n_draws, -1)
    oracle_samples = jnp.asarray(oracle_samples)[:n_draws].reshape(n_draws, -1)
    return float(
        jnp.abs(jnp.sort(samples, axis=0) - jnp.sort(oracle_samples, axis=0)).mean()
    )


def slugify(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "item"


def normalize_experiment_ids(experiment_ids):
    if experiment_ids is None:
        return None
    if isinstance(experiment_ids, str):
        values = experiment_ids.split(",")
    else:
        values = experiment_ids
    normalized = []
    for value in values:
        if isinstance(value, (list, tuple)):
            child_values = normalize_experiment_ids(value)
            if child_values is not None:
                normalized.extend(child_values)
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized or None


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def write_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_single_csv_row(path):
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def load_array_or_none(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def load_saved_experiment_manifest_metadata(experiment_dir):
    experiment_dir = Path(experiment_dir)
    manifest_row = read_single_csv_row(experiment_dir / "experiment_manifest.csv")
    if not manifest_row:
        return None

    experiment_id = str(manifest_row.get("experiment_id", experiment_dir.name))
    covariate_variant = str(manifest_row.get("covariate_variant", "")).strip()
    if not covariate_variant:
        covariate_variant = "null" if experiment_id.endswith("__null") else "full"
    oracle_variant = str(manifest_row.get("oracle_variant", "")).strip()
    # Historical null-covariate traces were benchmarked against the full oracle,
    # so force a rerun when resuming those artifacts.
    if not oracle_variant and covariate_variant == "null":
        return None
    if not oracle_variant:
        oracle_variant = covariate_variant

    try:
        n_steps = int(float(manifest_row["n_steps"]))
    except (KeyError, TypeError, ValueError):
        return None

    return dict(
        manifest_row=manifest_row,
        experiment_id=experiment_id,
        covariate_variant=covariate_variant,
        oracle_variant=oracle_variant,
        n_steps=n_steps,
    )


def saved_experiment_trace_is_usable(experiment_dir):
    manifest_metadata = load_saved_experiment_manifest_metadata(experiment_dir)
    if manifest_metadata is None:
        return False
    experiment_dir = Path(experiment_dir)
    for step_index in range(manifest_metadata["n_steps"]):
        step_dir = experiment_dir / "steps" / f"step_{step_index:03d}"
        if not read_single_csv_row(step_dir / "scalar_metrics.csv"):
            return False
        if not read_csv_rows(step_dir / "grouped_oracle_distances.csv"):
            return False
        if load_array_or_none(step_dir / "reviewed_values.npy") is None:
            return False
        if load_array_or_none(step_dir / "reviewed_mask.npy") is None:
            return False
        if load_array_or_none(step_dir / "selected_mask.npy") is None:
            return False
        if load_array_or_none(step_dir / "psi_mean.npy") is None:
            return False
    return True


def load_saved_experiment_trace(experiment_dir, _dataset_metadata=None):
    experiment_dir = Path(experiment_dir)
    manifest_metadata = load_saved_experiment_manifest_metadata(experiment_dir)
    if manifest_metadata is None:
        return None

    manifest_row = manifest_metadata["manifest_row"]
    experiment_id = manifest_metadata["experiment_id"]
    covariate_variant = manifest_metadata["covariate_variant"]
    oracle_variant = manifest_metadata["oracle_variant"]
    n_steps = manifest_metadata["n_steps"]

    review_counts = []
    mean_oracle_distances = []
    grouped_oracle_distances = {}
    for step_index in range(n_steps):
        step_dir = experiment_dir / "steps" / f"step_{step_index:03d}"
        scalar_metrics = read_single_csv_row(step_dir / "scalar_metrics.csv")
        grouped_rows = read_csv_rows(step_dir / "grouped_oracle_distances.csv")
        if not scalar_metrics or not grouped_rows:
            return None
        review_counts.append(int(float(scalar_metrics["review_count"])))
        mean_oracle_distances.append(float(scalar_metrics["mean_oracle_distance"]))
        for row in grouped_rows:
            grouped_oracle_distances.setdefault(str(row["group_name"]), []).append(
                float(row["distance"])
            )

    experiment_trace = dict(
        experiment_id=experiment_id,
        display_name=str(manifest_row.get("display_name", experiment_dir.name)),
        model_family=str(manifest_row.get("model_family", "")),
        model_label=str(manifest_row.get("model_label", "")),
        selection_method=str(manifest_row.get("selection_method", "")),
        review_counts=review_counts,
        mean_oracle_distances=mean_oracle_distances,
        grouped_oracle_distances=grouped_oracle_distances,
        step_artifacts=[],
        timing_metrics={
            key: float(value)
            for key, value in manifest_row.items()
            if key.endswith("_seconds") and str(value).strip() != ""
        },
    )
    experiment_trace["covariate_variant"] = covariate_variant
    experiment_trace["oracle_variant"] = oracle_variant
    if str(manifest_row.get("n_site_covariates", "")).strip():
        experiment_trace["n_site_covariates"] = int(
            float(manifest_row["n_site_covariates"])
        )
    if str(manifest_row.get("n_obs_covariates", "")).strip():
        experiment_trace["n_obs_covariates"] = int(
            float(manifest_row["n_obs_covariates"])
        )
    return experiment_trace


def build_summary_rows(samples):
    try:
        summary_dict = numpyro_summary(samples)
    except (AssertionError, ValueError):
        return build_fallback_summary_rows(samples)
    rows = []
    for parameter_name, stats in summary_dict.items():
        shapes = [np.asarray(values).shape for values in stats.values()]
        statistic_shape = shapes[0] if shapes else ()
        indices = [()] if statistic_shape == () else list(np.ndindex(statistic_shape))
        for index in indices:
            row = dict(
                parameter=str(parameter_name),
                parameter_index="" if not index else ",".join(map(str, index)),
            )
            for source_name, target_name in SUMMARY_STAT_NAME_MAP.items():
                values = np.asarray(stats[source_name])
                row[target_name] = (
                    float(values) if values.shape == () else float(values[index])
                )
            rows.append(row)
    return rows


def build_fallback_summary_rows(samples):
    rows = []
    for parameter_name, values in samples.items():
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        statistic_shape = array.shape[1:]
        indices = [()] if statistic_shape == () else list(np.ndindex(statistic_shape))
        for index in indices:
            draws = array if statistic_shape == () else array[(slice(None),) + index]
            row = dict(
                parameter=str(parameter_name),
                parameter_index="" if not index else ",".join(map(str, index)),
                mean=float(np.nanmean(draws)),
                std=float(np.nanstd(draws, ddof=0)),
                median=float(np.nanmedian(draws)),
                q05=float(np.nanquantile(draws, 0.05)),
                q95=float(np.nanquantile(draws, 0.95)),
                n_eff=float("nan"),
                r_hat=float("nan"),
            )
            rows.append(row)
    return rows


def summarize_fit_results(fit_results):
    summary_rows = build_summary_rows(fit_results.samples)
    r_hat_values = np.asarray(
        [row["r_hat"] for row in summary_rows if np.isfinite(row["r_hat"])],
        dtype=float,
    )
    n_eff_values = np.asarray(
        [row["n_eff"] for row in summary_rows if np.isfinite(row["n_eff"])],
        dtype=float,
    )
    num_posterior_draws = fit_results.mcmc.num_samples * fit_results.mcmc.num_chains
    extra_fields = {
        key: np.asarray(value)
        for key, value in (fit_results.mcmc.get_extra_fields() or {}).items()
    }
    num_diverging = (
        int(np.asarray(extra_fields["diverging"], dtype=int).sum())
        if "diverging" in extra_fields
        else 0
    )
    scalar_metrics = dict(
        num_samples=int(fit_results.mcmc.num_samples),
        num_warmup=int(fit_results.mcmc.num_warmup),
        num_chains=int(fit_results.mcmc.num_chains),
        posterior_draws=int(num_posterior_draws),
        num_parameters=int(len(summary_rows)),
        mean_r_hat=(
            float(np.nanmean(r_hat_values)) if r_hat_values.size else float("nan")
        ),
        max_r_hat=float(np.nanmax(r_hat_values)) if r_hat_values.size else float("nan"),
        min_r_hat=float(np.nanmin(r_hat_values)) if r_hat_values.size else float("nan"),
        mean_n_eff=(
            float(np.nanmean(n_eff_values)) if n_eff_values.size else float("nan")
        ),
        min_n_eff=float(np.nanmin(n_eff_values)) if n_eff_values.size else float("nan"),
        max_n_eff=float(np.nanmax(n_eff_values)) if n_eff_values.size else float("nan"),
        mean_frac_eff=(
            float(np.nanmean(n_eff_values / num_posterior_draws))
            if n_eff_values.size
            else float("nan")
        ),
        min_frac_eff=(
            float(np.nanmin(n_eff_values / num_posterior_draws))
            if n_eff_values.size
            else float("nan")
        ),
        num_diverging=num_diverging,
        frac_diverging=(
            float(num_diverging / num_posterior_draws)
            if num_posterior_draws
            else float("nan")
        ),
    )
    try:
        scalar_metrics.update(compute_mcmc_diagnostics(fit_results.mcmc))
    except (AssertionError, ValueError, KeyError):
        scalar_metrics.update(
            dict(
                mean_r_hat=float("nan"),
                mean_frac_eff=float("nan"),
                frac_diverging=scalar_metrics["frac_diverging"],
                mean_beta_sd=float("nan"),
                mean_alpha_sd=float("nan"),
            )
        )
    return summary_rows, scalar_metrics, extra_fields


def compute_mcmc_diagnostics(mcmc):
    sites = mcmc._states[mcmc._sample_field]  # pylint: disable=protected-access
    if isinstance(sites, dict):
        state_sample_field = attrgetter(
            mcmc._sample_field
        )(  # pylint: disable=protected-access
            mcmc._last_state  # pylint: disable=protected-access
        )
        if isinstance(state_sample_field, dict):
            sites = {
                key: value
                for key, value in mcmc._states[
                    mcmc._sample_field
                ].items()  # pylint: disable=protected-access
                if key in state_sample_field
            }

    summary_dict = numpyro_summary(sites)
    mean_r_hat = sum(
        value["r_hat"].mean().item() for value in summary_dict.values()
    ) / len(summary_dict)
    mean_frac_eff = (
        sum(value["n_eff"].mean().item() for value in summary_dict.values())
        / len(summary_dict)
        / (mcmc.num_samples * mcmc.num_chains)
    )
    diagnostics = mcmc.get_extra_fields()
    if diagnostics is not None and "diverging" in diagnostics:
        frac_diverging = jnp.sum(diagnostics["diverging"]).item() / (
            mcmc.num_samples * mcmc.num_chains
        )
    else:
        frac_diverging = float("nan")

    return dict(
        mean_r_hat=mean_r_hat,
        mean_frac_eff=mean_frac_eff,
        frac_diverging=frac_diverging,
        mean_beta_sd=float("nan"),
        mean_alpha_sd=float("nan"),
    )


def compute_parameter_oracle_distances(samples, oracle_samples, parameter_names):
    return {
        str(name): posterior_wasserstein_distance(samples[name], oracle_samples[name])
        for name in parameter_names
    }


def compute_psi_mean(fit_results, shape):
    psi_samples = fit_results.samples.get("psi")
    if psi_samples is None:
        return np.full(shape, np.nan, dtype=float)
    psi_samples = np.asarray(psi_samples, dtype=float).reshape(len(psi_samples), *shape)
    return psi_samples.mean(axis=0)


def build_fit_artifact(
    fit_results,
    oracle_results,
    shared_parameter_names,
    review_count,
    selected,
    reviewed_values,
    mean_oracle_distance,
    grouped_oracle_distances,
    score_log_bf=None,
    score_calibration=None,
    timing_metrics=None,
    extra_scalar_metrics=None,
    posterior_shift_rows=None,
    acquisition_candidate_rows=None,
):
    summary_rows, diagnostic_scalars, _extra_fields = summarize_fit_results(fit_results)
    reviewed_values = np.asarray(reviewed_values, dtype=float)
    reviewed_mask = np.isfinite(reviewed_values)
    selected_mask = np.asarray(selected, dtype=bool)
    parameter_oracle_distances = compute_parameter_oracle_distances(
        fit_results.samples,
        oracle_results.samples,
        shared_parameter_names,
    )
    scalar_metrics = dict(
        review_count=int(review_count),
        selected_count=int(selected_mask.sum()),
        mean_oracle_distance=float(mean_oracle_distance),
    )
    scalar_metrics.update(diagnostic_scalars)
    if timing_metrics is not None:
        scalar_metrics.update(
            {str(name): float(value) for name, value in timing_metrics.items()}
        )
    if extra_scalar_metrics is not None:
        scalar_metrics.update(
            {str(name): float(value) for name, value in extra_scalar_metrics.items()}
        )
    return dict(
        scalar_metrics=scalar_metrics,
        grouped_oracle_distance_rows=[
            dict(group_name=str(name), distance=float(value))
            for name, value in grouped_oracle_distances.items()
        ],
        posterior_shift_rows=(
            []
            if posterior_shift_rows is None
            else [
                dict(group_name=str(row["group_name"]), distance=float(row["distance"]))
                for row in posterior_shift_rows
            ]
        ),
        parameter_oracle_distance_rows=[
            dict(parameter=str(name), distance=float(value))
            for name, value in parameter_oracle_distances.items()
        ],
        parameter_summary_rows=summary_rows,
        reviewed_values=np.where(reviewed_mask, reviewed_values, np.nan),
        reviewed_mask=reviewed_mask,
        selected_mask=selected_mask,
        psi_mean=compute_psi_mean(fit_results, reviewed_values.shape[:2]),
        score_log_bf=(
            None if score_log_bf is None else np.asarray(score_log_bf, dtype=float)
        ),
        score_calibration=(
            None if score_calibration is None else dict(asdict(score_calibration))
        ),
        acquisition_candidate_rows=(
            []
            if acquisition_candidate_rows is None
            else list(acquisition_candidate_rows)
        ),
    )


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def summarize_selection_utility(
    expected_gain_score,
    reviewable,
    selected=None,
    topk_count=None,
):
    reviewable = np.asarray(reviewable, dtype=bool)
    if expected_gain_score is None:
        candidate_scores = np.asarray([], dtype=float)
        selected_scores = np.asarray([], dtype=float)
    else:
        expected_gain_score = np.asarray(expected_gain_score, dtype=float)
        candidate_scores = expected_gain_score[
            reviewable & np.isfinite(expected_gain_score)
        ]
        if selected is None:
            selected_scores = np.asarray([], dtype=float)
        else:
            selected_mask = np.asarray(selected, dtype=bool)
            selected_scores = expected_gain_score[
                selected_mask & np.isfinite(expected_gain_score)
            ]

    if topk_count is None:
        topk_count = int(selected_scores.size)
    topk_count = int(max(topk_count, 0))
    if candidate_scores.size == 0:
        topk_scores = np.asarray([], dtype=float)
    elif topk_count <= 0:
        topk_scores = np.asarray([], dtype=float)
    else:
        topk_count = min(topk_count, int(candidate_scores.size))
        topk_scores = np.partition(candidate_scores, -topk_count)[-topk_count:]

    return dict(
        expected_gain_score_max=(
            float(candidate_scores.max()) if candidate_scores.size else float("nan")
        ),
        expected_gain_score_mean=_nanmean_or_nan(candidate_scores),
        expected_gain_score_topk_mean=_nanmean_or_nan(topk_scores),
        expected_gain_score_selected_mean=_nanmean_or_nan(selected_scores),
        expected_gain_score_selected_sum=(
            float(selected_scores.sum()) if selected_scores.size else float("nan")
        ),
    )


def summarize_prequential_selection(
    mean_prob_positive,
    selected,
    true_labels,
):
    if mean_prob_positive is None or selected is None:
        return dict(
            prequential_selected_count=0.0,
            prequential_mean_log_score=float("nan"),
            prequential_mean_nll=float("nan"),
            prequential_mean_brier=float("nan"),
            prequential_selected_positive_rate=float("nan"),
            prequential_selected_predicted_positive_rate=float("nan"),
        )

    selected_mask = np.asarray(selected, dtype=bool)
    if not selected_mask.any():
        return dict(
            prequential_selected_count=0.0,
            prequential_mean_log_score=float("nan"),
            prequential_mean_nll=float("nan"),
            prequential_mean_brier=float("nan"),
            prequential_selected_positive_rate=float("nan"),
            prequential_selected_predicted_positive_rate=float("nan"),
        )

    probabilities = np.asarray(mean_prob_positive, dtype=float)[selected_mask]
    labels = np.asarray(true_labels, dtype=float)[selected_mask]
    finite = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities = np.clip(probabilities[finite], 1e-12, 1.0 - 1e-12)
    labels = labels[finite]
    if probabilities.size == 0:
        return dict(
            prequential_selected_count=0.0,
            prequential_mean_log_score=float("nan"),
            prequential_mean_nll=float("nan"),
            prequential_mean_brier=float("nan"),
            prequential_selected_positive_rate=float("nan"),
            prequential_selected_predicted_positive_rate=float("nan"),
        )

    log_scores = labels * np.log(probabilities) + (1.0 - labels) * np.log1p(
        -probabilities
    )
    brier_scores = (probabilities - labels) ** 2
    return dict(
        prequential_selected_count=float(probabilities.size),
        prequential_mean_log_score=float(log_scores.mean()),
        prequential_mean_nll=float((-log_scores).mean()),
        prequential_mean_brier=float(brier_scores.mean()),
        prequential_selected_positive_rate=float(labels.mean()),
        prequential_selected_predicted_positive_rate=float(probabilities.mean()),
    )


def summarize_posterior_shift(
    fit_results,
    previous_fit_results,
    shared_parameter_names,
    oracle_distance_groups,
):
    if previous_fit_results is None:
        return dict(
            posterior_shift_mean_distance=float("nan"),
            posterior_shift_rows=[],
        )

    parameter_distances = compute_parameter_oracle_distances(
        fit_results.samples,
        previous_fit_results.samples,
        shared_parameter_names,
    )
    posterior_shift_rows = []
    for group_name, parameter_names in oracle_distance_groups.items():
        group_values = [
            float(parameter_distances[name])
            for name in parameter_names
            if name in parameter_distances
        ]
        posterior_shift_rows.append(
            dict(
                group_name=str(group_name),
                distance=float(np.mean(group_values)) if group_values else float("nan"),
            )
        )
    return dict(
        posterior_shift_mean_distance=float(
            np.mean(list(parameter_distances.values()))
        ),
        posterior_shift_rows=posterior_shift_rows,
    )


def build_oracle_artifact(
    oracle_results,
    true_params,
    shared_parameter_names,
    timing_metrics=None,
):
    summary_rows, diagnostic_scalars, _extra_fields = summarize_fit_results(
        oracle_results
    )
    scalar_metrics = dict(
        **diagnostic_scalars,
        num_oracle_parameters=int(len(shared_parameter_names)),
        num_positive_labels=int(np.nansum(np.asarray(true_params["f"], dtype=float))),
        num_total_labels=int(
            np.isfinite(np.asarray(true_params["f"], dtype=float)).sum()
        ),
    )
    if timing_metrics is not None:
        scalar_metrics.update(
            {str(name): float(value) for name, value in timing_metrics.items()}
        )
    return dict(
        scalar_metrics=scalar_metrics,
        parameter_summary_rows=summary_rows,
        true_labels=np.asarray(true_params["f"], dtype=float),
    )


def run_cs_experiment(
    selection_method,
    data,
    true_params,
    dataset_metadata,
    site_covs=None,
    obs_covs=None,
    model_fn=occu_cs,
    model_kwargs=None,
    oracle_results=None,
    shared_parameter_names=(),
    oracle_distance_groups=None,
    mcmc_kwargs=None,
    n_reviews_per_step=DEFAULT_N_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_MAX_REVIEWS,
    artifact_writer=None,
):
    obs = data["obs"]
    site_covs = data["site_covs"] if site_covs is None else site_covs
    obs_covs = data["obs_covs"] if obs_covs is None else obs_covs
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    mcmc_kwargs = dict(DEFAULT_MCMC_KWARGS if mcmc_kwargs is None else mcmc_kwargs)
    mean_wasserstein_distances = []
    grouped_oracle_distances = {
        group_name: [] for group_name in (oracle_distance_groups or {})
    }
    review_counts = []
    step_artifacts = []
    verified_f = jnp.full_like(true_params["f"], jnp.nan, dtype=float)
    selection_key = jr.PRNGKey(0)
    experiment_start_time = time.perf_counter()
    previous_fit_results = None
    timing_totals = dict(
        total_score_calibration_wall_time_seconds=0.0,
        total_mcmc_wall_time_seconds=0.0,
        total_selection_wall_time_seconds=0.0,
        total_step_wall_time_seconds=0.0,
    )

    while True:
        step_start_time = time.perf_counter()
        current_review_count = int(jnp.isfinite(verified_f).sum())
        fit_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key
            not in {
                "score_calibration_max_iter",
                "score_calibration_tol",
                "score_calibration_min_sigma",
                "score_calibration_min_mean_separation",
                "score_calibration_method",
                "score_calibration_num_warmup",
                "score_calibration_num_samples",
                "score_calibration_num_chains",
                "score_calibration_num_draws",
                "score_calibration_max_candidates",
                "score_calibration_reviewed_weight_temperature",
                "score_calibration_prevalence_mean",
                "score_calibration_prevalence_strength",
                "score_calibration_random_seed",
                "score_calibration_seed_n_reviews",
                "epig_target_pool_size",
                "targeted_eig_candidate_pool_size",
                "target_eig_target_config",
                "target_eig_acquisition_config",
            }
        }
        current_score_log_bf = None
        current_score_log_bf_draws = None
        current_score_calibration = None
        score_calibration_wall_time_seconds = 0.0
        step_mcmc_kwargs = dict(mcmc_kwargs)
        if model_fn is occu_cs_calibrated:
            calibration_start_time = time.perf_counter()
            (
                current_score_calibration,
                current_score_log_bf,
                current_score_log_bf_draws,
            ) = resolve_calibrated_score_log_bf(
                obs,
                verified_f,
                model_kwargs,
            )
            score_calibration_wall_time_seconds = (
                time.perf_counter() - calibration_start_time
            )
            fit_kwargs["score_log_bf"] = current_score_log_bf
        if model_fn is occu_cs:
            fit_kwargs["obs"] = obs
            if step_mcmc_kwargs.get("init_strategy") is None:
                calibration_start_time = time.perf_counter()
                step_mcmc_kwargs["init_strategy"] = occu_cs_init_strategy(
                    obs,
                    verified_f,
                    score_loc=fit_kwargs.get("score_loc"),
                    score_scale=fit_kwargs.get("score_scale"),
                    min_sigma=model_kwargs.get("score_calibration_min_sigma", 1e-3),
                    min_mean_separation=model_kwargs.get(
                        "score_calibration_min_mean_separation",
                        1e-3,
                    ),
                )
                score_calibration_wall_time_seconds += (
                    time.perf_counter() - calibration_start_time
                )
        print(f"Running CS fit with {current_review_count} reviews...")
        fit_start_time = time.perf_counter()
        cs_results = fit(
            model_fn,
            site_covs=site_covs,
            obs_covs=obs_covs,
            verified_f=verified_f,
            **fit_kwargs,
            **step_mcmc_kwargs,
        )
        mcmc_wall_time_seconds = time.perf_counter() - fit_start_time
        review_counts.append(int(jnp.isfinite(verified_f).sum()))

        mean_wasserstein_distances.append(
            float(
                jnp.mean(
                    jnp.asarray(
                        [
                            posterior_wasserstein_distance(
                                cs_results.samples[name],
                                oracle_results.samples[name],
                            )
                            for name in shared_parameter_names
                        ]
                    )
                )
            )
        )
        for group_name, parameter_names in oracle_distance_groups.items():
            grouped_oracle_distances[group_name].append(
                float(
                    jnp.mean(
                        jnp.asarray(
                            [
                                posterior_wasserstein_distance(
                                    cs_results.samples[name],
                                    oracle_results.samples[name],
                                )
                                for name in parameter_names
                            ]
                        )
                    )
                )
            )

        current_grouped_distances = {
            group_name: grouped_oracle_distances[group_name][-1]
            for group_name in grouped_oracle_distances
        }
        selection_state = build_cs_selection_state(
            model_fn,
            cs_results,
            obs,
            site_covs,
            obs_covs,
            verified_f,
            score_log_bf=current_score_log_bf,
            score_log_bf_draws=current_score_log_bf_draws,
        )
        selection_state = restrict_selection_state_to_labelable(
            selection_state,
            true_params["f"][0],
        )
        posterior_shift_summary = summarize_posterior_shift(
            cs_results,
            previous_fit_results,
            shared_parameter_names,
            oracle_distance_groups,
        )
        candidate_utility_metrics = summarize_selection_utility(
            selection_state["expected_gain_score"],
            selection_state["reviewable"],
            topk_count=min(n_reviews_per_step, max_reviews - review_counts[-1]),
        )
        target_eig_diagnostics = None

        if review_counts[-1] >= max_reviews:
            selected = jnp.zeros_like(verified_f[0], dtype=bool)
            step_timing_metrics = dict(
                score_calibration_wall_time_seconds=float(
                    score_calibration_wall_time_seconds
                ),
                mcmc_wall_time_seconds=float(mcmc_wall_time_seconds),
                selection_wall_time_seconds=0.0,
                step_wall_time_seconds=float(time.perf_counter() - step_start_time),
            )
            extra_scalar_metrics = dict(
                **candidate_utility_metrics,
                **summarize_prequential_selection(
                    selection_state["mean_prob_positive"],
                    selected,
                    true_params["f"][0],
                ),
                posterior_shift_mean_distance=float(
                    posterior_shift_summary["posterior_shift_mean_distance"]
                ),
            )
            for name, value in step_timing_metrics.items():
                timing_totals["total_" + name] += float(value)
            step_artifact = build_fit_artifact(
                cs_results,
                oracle_results,
                shared_parameter_names,
                review_counts[-1],
                selected,
                verified_f[0],
                mean_wasserstein_distances[-1],
                current_grouped_distances,
                score_log_bf=current_score_log_bf,
                score_calibration=current_score_calibration,
                timing_metrics=step_timing_metrics,
                extra_scalar_metrics=extra_scalar_metrics,
                posterior_shift_rows=posterior_shift_summary["posterior_shift_rows"],
            )
            if artifact_writer is None:
                step_artifacts.append(step_artifact)
            else:
                artifact_writer.write_step(step_artifact)
            previous_fit_results = cs_results
            break

        n_reviews = min(n_reviews_per_step, max_reviews - review_counts[-1])
        selection_start_time = time.perf_counter()
        if selection_method in {"bald", "bald_site_diverse"}:
            site_diverse = selection_method == "bald_site_diverse"
            if model_fn in {occu_cs_calibrated, occu_cs}:
                reviewed_positive_count = count_reviewed_positives(verified_f)
                seed_n_reviews = int(
                    model_kwargs.get(
                        "score_calibration_seed_n_reviews",
                        n_reviews_per_step,
                    )
                )
                if reviewed_positive_count == 0 and seed_n_reviews > 0:
                    if site_diverse:
                        selected = select_top_site_diverse_samples(
                            selection_state["max_score"],
                            selection_state["reviewable"],
                            min(n_reviews, seed_n_reviews),
                        )
                    else:
                        selected = select_top_acquisition_samples(
                            selection_state["max_score"],
                            selection_state["reviewable"],
                            min(n_reviews, seed_n_reviews),
                        )
                else:
                    if site_diverse:
                        selected = select_top_site_diverse_samples(
                            selection_state["expected_gain_score"],
                            selection_state["reviewable"],
                            n_reviews,
                        )
                    else:
                        selected = select_top_acquisition_samples(
                            selection_state["expected_gain_score"],
                            selection_state["reviewable"],
                            n_reviews,
                        )
            else:
                if site_diverse:
                    selected = select_top_site_diverse_samples(
                        selection_state["expected_gain_score"],
                        selection_state["reviewable"],
                        n_reviews,
                    )
                else:
                    selected = select_top_acquisition_samples(
                        selection_state["expected_gain_score"],
                        selection_state["reviewable"],
                        n_reviews,
                    )
        elif selection_method == "dual_balanced":
            if model_fn is occu_cs_calibrated:
                reviewed_positive_count = count_reviewed_positives(verified_f)
                seed_n_reviews = int(
                    model_kwargs.get(
                        "score_calibration_seed_n_reviews",
                        n_reviews_per_step,
                    )
                )
                if (
                    reviewed_positive_count < DUAL_BALANCED_MIN_POSITIVE_COUNT
                    and seed_n_reviews > 0
                ):
                    selected = select_top_acquisition_samples(
                        selection_state["max_score"],
                        selection_state["reviewable"],
                        min(n_reviews, seed_n_reviews),
                    )
                else:
                    selected = select_top_dual_acquisition_samples(
                        selection_state["max_score"],
                        selection_state["expected_gain_score"],
                        selection_state["reviewable"],
                        n_reviews,
                        DUAL_BALANCED_MAX_SCORE_FRACTION,
                    )
            else:
                selected = select_top_dual_acquisition_samples(
                    selection_state["max_score"],
                    selection_state["expected_gain_score"],
                    selection_state["reviewable"],
                    n_reviews,
                    DUAL_BALANCED_MAX_SCORE_FRACTION,
                )
        elif selection_method == "random":
            selection_key, random_key = jr.split(selection_key)
            selected = select_random_reviewable_samples(
                selection_state["reviewable"],
                n_reviews,
                random_key,
            )
        elif selection_method == "max_score":
            selected = select_top_acquisition_samples(
                selection_state["max_score"],
                selection_state["reviewable"],
                n_reviews,
            )
        elif selection_method == "posterior_predictive_uncertainty":
            selected = select_top_acquisition_samples(
                selection_state["posterior_predictive_uncertainty"],
                selection_state["reviewable"],
                n_reviews,
            )
        elif selection_method == "target_eig":
            reviewed_positive_count = count_reviewed_positives(verified_f)
            seed_n_reviews = int(
                model_kwargs.get(
                    "score_calibration_seed_n_reviews",
                    n_reviews_per_step,
                )
            )
            if reviewed_positive_count == 0 and seed_n_reviews > 0:
                selected = select_top_acquisition_samples(
                    selection_state["max_score"],
                    selection_state["reviewable"],
                    min(n_reviews, seed_n_reviews),
                )
            elif model_fn is occu_cs:
                selected, target_eig_diagnostics = select_target_eig_samples(
                    cs_results,
                    obs,
                    site_covs,
                    obs_covs,
                    verified_f,
                    selection_state["reviewable"],
                    n_reviews,
                    target_config=model_kwargs.get("target_eig_target_config"),
                    acquisition_config=model_kwargs.get(
                        "target_eig_acquisition_config"
                    ),
                )
            elif model_fn is occu_cs_calibrated:
                selected, target_eig_diagnostics = select_calibrated_target_eig_samples(
                    cs_results,
                    current_score_log_bf,
                    current_score_log_bf_draws,
                    site_covs,
                    obs_covs,
                    verified_f,
                    selection_state["reviewable"],
                    n_reviews,
                    target_config=model_kwargs.get("target_eig_target_config"),
                    acquisition_config=model_kwargs.get(
                        "target_eig_acquisition_config"
                    ),
                )
            else:
                raise ValueError(
                    "target_eig is only supported for continuous-score models."
                )
        elif selection_method == "epig":
            if model_fn is not occu_cs_calibrated:
                raise ValueError("epig is only supported for occu_cs_calibrated.")
            prob_positive = calibrated_prob_positive_draws(
                cs_results.samples,
                current_score_log_bf,
                current_score_log_bf_draws,
                site_covs,
                obs_covs,
                verified_f,
            )
            epig_score = epig_scores_from_prob_positive_draws(
                prob_positive,
                selection_state["reviewable"],
                target_pool_size=model_kwargs.get(
                    "epig_target_pool_size",
                    EPIG_TARGET_POOL_SIZE,
                ),
                score_values=selection_state["max_score"],
            )
            selected = select_top_acquisition_samples(
                epig_score,
                selection_state["reviewable"],
                n_reviews,
            )
        elif selection_method == "revealed_targeted_eig":
            if model_fn is not occu_cs_calibrated:
                raise ValueError(
                    "revealed_targeted_eig is only supported for occu_cs_calibrated."
                )
            selected = select_revealed_targeted_eig_samples(
                cs_results,
                shared_parameter_names,
                current_score_log_bf,
                current_score_log_bf_draws,
                site_covs,
                obs_covs,
                verified_f,
                selection_state["reviewable"],
                true_params["f"][0],
                n_reviews,
                candidate_pool_size=model_kwargs.get(
                    "targeted_eig_candidate_pool_size",
                    TARGETED_EIG_CANDIDATE_POOL_SIZE,
                ),
            )
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        selection_wall_time_seconds = time.perf_counter() - selection_start_time
        selected = (
            jnp.asarray(selected, dtype=bool)
            & ~jnp.isfinite(verified_f[0])
            & jnp.isfinite(true_params["f"][0])
        )
        extra_scalar_metrics = dict(
            **summarize_selection_utility(
                selection_state["expected_gain_score"],
                selection_state["reviewable"],
                selected=selected,
                topk_count=n_reviews,
            ),
            **summarize_target_eig_selection(
                target_eig_diagnostics,
                selection_state,
                selected,
                n_reviews,
            ),
            **summarize_prequential_selection(
                selection_state["mean_prob_positive"],
                selected,
                true_params["f"][0],
            ),
            posterior_shift_mean_distance=float(
                posterior_shift_summary["posterior_shift_mean_distance"]
            ),
        )
        step_timing_metrics = dict(
            score_calibration_wall_time_seconds=float(
                score_calibration_wall_time_seconds
            ),
            mcmc_wall_time_seconds=float(mcmc_wall_time_seconds),
            selection_wall_time_seconds=float(selection_wall_time_seconds),
            step_wall_time_seconds=float(time.perf_counter() - step_start_time),
        )
        for name, value in step_timing_metrics.items():
            timing_totals["total_" + name] += float(value)

        step_artifact = build_fit_artifact(
            cs_results,
            oracle_results,
            shared_parameter_names,
            review_counts[-1],
            selected,
            verified_f[0],
            mean_wasserstein_distances[-1],
            current_grouped_distances,
            score_log_bf=current_score_log_bf,
            score_calibration=current_score_calibration,
            timing_metrics=step_timing_metrics,
            extra_scalar_metrics=extra_scalar_metrics,
            posterior_shift_rows=posterior_shift_summary["posterior_shift_rows"],
            acquisition_candidate_rows=build_target_eig_candidate_rows(
                dataset_metadata,
                target_eig_diagnostics,
                selection_state,
                selected=selected,
            ),
        )
        if artifact_writer is None:
            step_artifacts.append(step_artifact)
        else:
            artifact_writer.write_step(step_artifact)
        if not bool(selected.any()):
            previous_fit_results = cs_results
            break
        verified_f = verified_f.at[0].set(
            jnp.where(selected, true_params["f"][0], verified_f[0])
        )
        previous_fit_results = cs_results

    return dict(
        review_counts=review_counts,
        mean_oracle_distances=mean_wasserstein_distances,
        grouped_oracle_distances=grouped_oracle_distances,
        step_artifacts=step_artifacts,
        timing_metrics=dict(
            **timing_totals,
            experiment_wall_time_seconds=float(
                time.perf_counter() - experiment_start_time
            ),
        ),
    )


def run_reviewed_bernoulli_experiment(
    selection_method,
    data,
    true_params,
    dataset_metadata,
    site_covs,
    obs_covs,
    oracle_results,
    shared_parameter_names,
    oracle_distance_groups,
    model_kwargs=None,
    mcmc_kwargs=None,
    n_reviews_per_step=DEFAULT_N_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_MAX_REVIEWS,
    artifact_writer=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    mcmc_kwargs = dict(DEFAULT_MCMC_KWARGS if mcmc_kwargs is None else mcmc_kwargs)
    mean_wasserstein_distances = []
    grouped_oracle_distances = {group_name: [] for group_name in oracle_distance_groups}
    review_counts = []
    step_artifacts = []
    reviewed_f = jnp.full_like(true_params["f"], jnp.nan, dtype=float)
    selection_key = jr.PRNGKey(0)
    experiment_start_time = time.perf_counter()
    previous_fit_results = None
    timing_totals = dict(
        total_score_calibration_wall_time_seconds=0.0,
        total_mcmc_wall_time_seconds=0.0,
        total_selection_wall_time_seconds=0.0,
        total_step_wall_time_seconds=0.0,
    )

    while True:
        step_start_time = time.perf_counter()
        current_review_count = int(jnp.isfinite(reviewed_f).sum())
        print(f"Running Bernoulli fit with {current_review_count} reviews...")
        fit_start_time = time.perf_counter()
        bernoulli_results = fit(
            occu,
            site_covs=site_covs,
            obs_covs=obs_covs,
            obs=reviewed_f,
            **mcmc_kwargs,
        )
        mcmc_wall_time_seconds = time.perf_counter() - fit_start_time
        review_counts.append(int(jnp.isfinite(reviewed_f).sum()))
        mean_wasserstein_distances.append(
            float(
                jnp.mean(
                    jnp.asarray(
                        [
                            posterior_wasserstein_distance(
                                bernoulli_results.samples[name],
                                oracle_results.samples[name],
                            )
                            for name in shared_parameter_names
                        ]
                    )
                )
            )
        )
        for group_name, parameter_names in oracle_distance_groups.items():
            grouped_oracle_distances[group_name].append(
                float(
                    jnp.mean(
                        jnp.asarray(
                            [
                                posterior_wasserstein_distance(
                                    bernoulli_results.samples[name],
                                    oracle_results.samples[name],
                                )
                                for name in parameter_names
                            ]
                        )
                    )
                )
            )

        current_grouped_distances = {
            group_name: grouped_oracle_distances[group_name][-1]
            for group_name in grouped_oracle_distances
        }
        selection_state = build_reviewed_selection_state(
            bernoulli_results,
            data["obs"],
            site_covs,
            obs_covs,
            reviewed_f,
        )
        selection_state = restrict_selection_state_to_labelable(
            selection_state,
            true_params["f"][0],
        )
        posterior_shift_summary = summarize_posterior_shift(
            bernoulli_results,
            previous_fit_results,
            shared_parameter_names,
            oracle_distance_groups,
        )
        candidate_utility_metrics = summarize_selection_utility(
            selection_state["expected_gain_score"],
            selection_state["reviewable"],
            topk_count=min(n_reviews_per_step, max_reviews - review_counts[-1]),
        )
        target_eig_diagnostics = None

        if review_counts[-1] >= max_reviews:
            selected = jnp.zeros_like(reviewed_f[0], dtype=bool)
            step_timing_metrics = dict(
                score_calibration_wall_time_seconds=0.0,
                mcmc_wall_time_seconds=float(mcmc_wall_time_seconds),
                selection_wall_time_seconds=0.0,
                step_wall_time_seconds=float(time.perf_counter() - step_start_time),
            )
            extra_scalar_metrics = dict(
                **candidate_utility_metrics,
                **summarize_target_eig_selection(
                    target_eig_diagnostics,
                    selection_state,
                    selected,
                    min(n_reviews_per_step, max_reviews - review_counts[-1]),
                ),
                **summarize_prequential_selection(
                    selection_state["mean_prob_positive"],
                    selected,
                    true_params["f"][0],
                ),
                posterior_shift_mean_distance=float(
                    posterior_shift_summary["posterior_shift_mean_distance"]
                ),
            )
            for name, value in step_timing_metrics.items():
                timing_totals["total_" + name] += float(value)
            step_artifact = build_fit_artifact(
                bernoulli_results,
                oracle_results,
                shared_parameter_names,
                review_counts[-1],
                selected,
                reviewed_f[0],
                mean_wasserstein_distances[-1],
                current_grouped_distances,
                timing_metrics=step_timing_metrics,
                extra_scalar_metrics=extra_scalar_metrics,
                posterior_shift_rows=posterior_shift_summary["posterior_shift_rows"],
            )
            if artifact_writer is None:
                step_artifacts.append(step_artifact)
            else:
                artifact_writer.write_step(step_artifact)
            previous_fit_results = bernoulli_results
            break

        n_reviews = min(n_reviews_per_step, max_reviews - review_counts[-1])
        selection_start_time = time.perf_counter()
        if selection_method in {"bald", "bald_site_diverse"}:
            if selection_method == "bald_site_diverse":
                selected = select_top_site_diverse_samples(
                    selection_state["expected_gain_score"],
                    selection_state["reviewable"],
                    n_reviews,
                )
            else:
                selected = select_top_acquisition_samples(
                    selection_state["expected_gain_score"],
                    selection_state["reviewable"],
                    n_reviews,
                )
        elif selection_method == "random":
            selection_key, random_key = jr.split(selection_key)
            selected = select_random_reviewable_samples(
                selection_state["reviewable"],
                n_reviews,
                random_key,
            )
        elif selection_method == "max_score":
            selected = select_top_acquisition_samples(
                selection_state["max_score"],
                selection_state["reviewable"],
                n_reviews,
            )
        elif selection_method == "dual_balanced":
            selected = select_top_dual_acquisition_samples(
                selection_state["max_score"],
                selection_state["expected_gain_score"],
                selection_state["reviewable"],
                n_reviews,
                DUAL_BALANCED_MAX_SCORE_FRACTION,
            )
        elif selection_method == "posterior_predictive_uncertainty":
            selected = select_top_acquisition_samples(
                selection_state["posterior_predictive_uncertainty"],
                selection_state["reviewable"],
                n_reviews,
            )
        elif selection_method == "epig":
            prob_positive, reviewable = reviewed_label_prob_positive_draws(
                bernoulli_results,
                site_covs,
                obs_covs,
                reviewed_f,
            )
            if prob_positive is None:
                selected = jnp.zeros(reviewable.shape, dtype=bool)
            else:
                epig_score = epig_scores_from_prob_positive_draws(
                    prob_positive,
                    selection_state["reviewable"],
                    score_values=selection_state["max_score"],
                )
                selected = select_top_acquisition_samples(
                    epig_score,
                    selection_state["reviewable"],
                    n_reviews,
                )
        elif selection_method == "target_eig":
            selected, target_eig_diagnostics = select_reviewed_label_target_eig_samples(
                bernoulli_results,
                site_covs,
                obs_covs,
                reviewed_f,
                selection_state["reviewable"],
                n_reviews,
                target_config=model_kwargs.get("target_eig_target_config"),
                acquisition_config=model_kwargs.get("target_eig_acquisition_config"),
            )
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        selection_wall_time_seconds = time.perf_counter() - selection_start_time
        selected = (
            jnp.asarray(selected, dtype=bool)
            & ~jnp.isfinite(reviewed_f[0])
            & jnp.isfinite(true_params["f"][0])
        )
        extra_scalar_metrics = dict(
            **summarize_selection_utility(
                selection_state["expected_gain_score"],
                selection_state["reviewable"],
                selected=selected,
                topk_count=n_reviews,
            ),
            **summarize_target_eig_selection(
                target_eig_diagnostics,
                selection_state,
                selected,
                n_reviews,
            ),
            **summarize_prequential_selection(
                selection_state["mean_prob_positive"],
                selected,
                true_params["f"][0],
            ),
            posterior_shift_mean_distance=float(
                posterior_shift_summary["posterior_shift_mean_distance"]
            ),
        )
        step_timing_metrics = dict(
            score_calibration_wall_time_seconds=0.0,
            mcmc_wall_time_seconds=float(mcmc_wall_time_seconds),
            selection_wall_time_seconds=float(selection_wall_time_seconds),
            step_wall_time_seconds=float(time.perf_counter() - step_start_time),
        )
        for name, value in step_timing_metrics.items():
            timing_totals["total_" + name] += float(value)
        step_artifact = build_fit_artifact(
            bernoulli_results,
            oracle_results,
            shared_parameter_names,
            review_counts[-1],
            selected,
            reviewed_f[0],
            mean_wasserstein_distances[-1],
            current_grouped_distances,
            timing_metrics=step_timing_metrics,
            extra_scalar_metrics=extra_scalar_metrics,
            posterior_shift_rows=posterior_shift_summary["posterior_shift_rows"],
            acquisition_candidate_rows=build_target_eig_candidate_rows(
                dataset_metadata,
                target_eig_diagnostics,
                selection_state,
                selected=selected,
            ),
        )
        if artifact_writer is None:
            step_artifacts.append(step_artifact)
        else:
            artifact_writer.write_step(step_artifact)
        if not bool(selected.any()):
            previous_fit_results = bernoulli_results
            break
        reviewed_f = reviewed_f.at[0].set(
            jnp.where(selected, true_params["f"][0], reviewed_f[0])
        )
        previous_fit_results = bernoulli_results

    return dict(
        review_counts=review_counts,
        mean_oracle_distances=mean_wasserstein_distances,
        grouped_oracle_distances=grouped_oracle_distances,
        step_artifacts=step_artifacts,
        timing_metrics=dict(
            **timing_totals,
            experiment_wall_time_seconds=float(
                time.perf_counter() - experiment_start_time
            ),
        ),
    )


experiment_plot_styles = {
    "reviewed_random": dict(marker="x", linestyle="--", color="C2"),
    "reviewed_max_score": dict(marker="D", linestyle="-.", color="C2"),
    "reviewed_posterior_predictive_uncertainty": dict(
        marker="X", linestyle="-", color="C2"
    ),
    "reviewed_bald": dict(marker="^", linestyle=":", color="C2"),
    "reviewed_target_eig": dict(marker="*", linestyle="-", color="C2"),
    "reviewed_bald_site_diverse": dict(marker="v", linestyle="-", color="C2"),
    "reviewed_dual_bald_max_balanced": dict(marker="h", linestyle="--", color="C2"),
    "reviewed_epig": dict(marker="*", linestyle="-", color="C6"),
    "adaptive_calibrated_random": dict(marker="o", linestyle="--", color="C4"),
    "adaptive_calibrated_max_score": dict(marker="D", linestyle=":", color="C4"),
    "adaptive_calibrated_posterior_predictive_uncertainty": dict(
        marker="X", linestyle="-", color="C4"
    ),
    "adaptive_calibrated_bald": dict(marker="P", linestyle="-.", color="C4"),
    "adaptive_calibrated_dual_bald_max_balanced": dict(
        marker="h", linestyle="--", color="C7"
    ),
    "adaptive_calibrated_epig": dict(marker="*", linestyle="-.", color="C6"),
    "adaptive_calibrated_bald_site_diverse": dict(
        marker="s", linestyle="-", color="C4"
    ),
    "adaptive_calibrated_revealed_targeted_eig": dict(
        marker="*", linestyle="-", color="C5"
    ),
    "cut_normal_calibrated_random": dict(marker="o", linestyle="--", color="C9"),
    "cut_normal_calibrated_max_score": dict(marker="D", linestyle="-.", color="C9"),
    "cut_normal_calibrated_posterior_predictive_uncertainty": dict(
        marker="X", linestyle="-", color="C9"
    ),
    "cut_normal_calibrated_bald": dict(marker="P", linestyle=":", color="C9"),
    "cut_normal_calibrated_target_eig": dict(marker="*", linestyle="-", color="C9"),
    "original_occu_cs_bald": dict(marker="p", linestyle="-", color="C3"),
    "original_occu_cs_target_eig": dict(marker="*", linestyle="-.", color="C8"),
    "reviewed_random__null": dict(marker="x", linestyle="-", color="C1"),
    "reviewed_max_score__null": dict(marker="D", linestyle="--", color="C1"),
    "reviewed_posterior_predictive_uncertainty__null": dict(
        marker="X", linestyle=":", color="C1"
    ),
    "reviewed_bald__null": dict(marker="^", linestyle="-.", color="C1"),
    "reviewed_target_eig__null": dict(marker="*", linestyle="-", color="C1"),
    "reviewed_bald_site_diverse__null": dict(marker="v", linestyle="--", color="C1"),
    "reviewed_dual_bald_max_balanced__null": dict(
        marker="h", linestyle=":", color="C1"
    ),
    "reviewed_epig__null": dict(marker="*", linestyle="--", color="C6"),
    "adaptive_calibrated_random__null": dict(marker="o", linestyle=":", color="C0"),
    "adaptive_calibrated_max_score__null": dict(marker="D", linestyle="--", color="C0"),
    "adaptive_calibrated_posterior_predictive_uncertainty__null": dict(
        marker="X", linestyle="-", color="C0"
    ),
    "adaptive_calibrated_bald__null": dict(marker="P", linestyle="-", color="C0"),
    "adaptive_calibrated_dual_bald_max_balanced__null": dict(
        marker="h", linestyle=":", color="C7"
    ),
    "adaptive_calibrated_epig__null": dict(marker="*", linestyle=":", color="C6"),
    "adaptive_calibrated_bald_site_diverse__null": dict(
        marker="s", linestyle="-.", color="C0"
    ),
    "adaptive_calibrated_revealed_targeted_eig__null": dict(
        marker="*", linestyle=":", color="C5"
    ),
    "cut_normal_calibrated_random__null": dict(marker="o", linestyle=":", color="C7"),
    "cut_normal_calibrated_max_score__null": dict(
        marker="D", linestyle="--", color="C7"
    ),
    "cut_normal_calibrated_posterior_predictive_uncertainty__null": dict(
        marker="X", linestyle="-", color="C7"
    ),
    "cut_normal_calibrated_bald__null": dict(marker="P", linestyle="-.", color="C7"),
    "cut_normal_calibrated_target_eig__null": dict(
        marker="*", linestyle=":", color="C7"
    ),
    "original_occu_cs_bald__null": dict(marker="p", linestyle="--", color="C3"),
    "original_occu_cs_target_eig__null": dict(marker="*", linestyle=":", color="C8"),
}
ORACLE_DISTANCE_GROUP_NAMES = (
    "Overall occupancy / psi",
    "Occupancy coefficients",
    "Detection coefficients",
)


def load_real_data_dataset(dataset_kind, target_species):
    if dataset_kind == "acoustic":
        return load_acoustic_perch_dataset(target_species=target_species)
    if dataset_kind == "iwildcam":
        return load_iwildcam2022_speciesnet_dataset(target_species=target_species)
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def build_null_covariate_arrays(site_covs, obs_covs):
    site_covs_array = np.asarray(site_covs)
    obs_covs_array = np.asarray(obs_covs)
    return (
        np.zeros(site_covs_array.shape[:-1] + (0,), dtype=site_covs_array.dtype),
        np.zeros(obs_covs_array.shape[:-1] + (0,), dtype=obs_covs_array.dtype),
    )


def build_covariate_variants(data):
    null_site_covs, null_obs_covs = build_null_covariate_arrays(
        data["site_covs"],
        data["obs_covs"],
    )
    return [
        dict(
            covariate_variant="full",
            oracle_variant="full",
            experiment_id_suffix="",
            display_name_suffix="",
            model_label_suffix="",
            site_covs=data["site_covs"],
            obs_covs=data["obs_covs"],
        ),
        dict(
            covariate_variant="null",
            oracle_variant="null",
            experiment_id_suffix="__null",
            display_name_suffix=" (null covariates)",
            model_label_suffix=" (null covariates)",
            site_covs=null_site_covs,
            obs_covs=null_obs_covs,
        ),
    ]


def build_cut_normal_model_kwargs(score_calibration_kwargs=None):
    model_kwargs = dict(DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS)
    if score_calibration_kwargs:
        model_kwargs.update(score_calibration_kwargs)
    return model_kwargs


def build_cut_normal_experiment_specs(score_calibration_kwargs=None):
    model_kwargs = build_cut_normal_model_kwargs(score_calibration_kwargs)
    return [
        dict(
            experiment_id="cut_normal_calibrated_random",
            display_name="Cut Normal calibrated CS random",
            model_family="cut_normal_calibrated_cs",
            model_label="occu_cs_calibrated (cut Normal)",
            selection_method="random",
            model_fn=occu_cs_calibrated,
            model_kwargs=dict(model_kwargs),
        ),
        dict(
            experiment_id="cut_normal_calibrated_max_score",
            display_name="Cut Normal calibrated CS max score",
            model_family="cut_normal_calibrated_cs",
            model_label="occu_cs_calibrated (cut Normal)",
            selection_method="max_score",
            model_fn=occu_cs_calibrated,
            model_kwargs=dict(model_kwargs),
        ),
        dict(
            experiment_id="cut_normal_calibrated_posterior_predictive_uncertainty",
            display_name="Cut Normal calibrated CS posterior predictive uncertainty",
            model_family="cut_normal_calibrated_cs",
            model_label="occu_cs_calibrated (cut Normal)",
            selection_method="posterior_predictive_uncertainty",
            model_fn=occu_cs_calibrated,
            model_kwargs=dict(model_kwargs),
        ),
        dict(
            experiment_id="cut_normal_calibrated_bald",
            display_name="Cut Normal calibrated CS BALD",
            model_family="cut_normal_calibrated_cs",
            model_label="occu_cs_calibrated (cut Normal)",
            selection_method="bald",
            model_fn=occu_cs_calibrated,
            model_kwargs=dict(model_kwargs),
        ),
        dict(
            experiment_id="cut_normal_calibrated_target_eig",
            display_name="Cut Normal calibrated CS target EIG",
            model_family="cut_normal_calibrated_cs",
            model_label="occu_cs_calibrated (cut Normal)",
            selection_method="target_eig",
            model_fn=occu_cs_calibrated,
            model_kwargs={
                **model_kwargs,
                "target_eig_target_config": TARGET_EIG_TARGET_CONFIG,
                "target_eig_acquisition_config": TARGET_EIG_ACQUISITION_CONFIG,
            },
        ),
    ]


def build_base_experiment_specs(score_calibration_kwargs=None):
    return [
        dict(
            experiment_id="reviewed_random",
            display_name="Reviewed-only Bernoulli random",
            model_family="reviewed_bernoulli",
            model_label="occu",
            selection_method="random",
        ),
        dict(
            experiment_id="reviewed_bald",
            display_name="Reviewed-only Bernoulli BALD",
            model_family="reviewed_bernoulli",
            model_label="occu",
            selection_method="bald",
        ),
        dict(
            experiment_id="reviewed_target_eig",
            display_name="Reviewed-only Bernoulli target EIG",
            model_family="reviewed_bernoulli",
            model_label="occu",
            selection_method="target_eig",
            model_kwargs={
                "target_eig_target_config": TARGET_EIG_TARGET_CONFIG,
                "target_eig_acquisition_config": TARGET_EIG_ACQUISITION_CONFIG,
            },
        ),
        # dict(
        #     experiment_id="reviewed_bald_site_diverse",
        #     display_name="Reviewed-only Bernoulli BALD (site diverse)",
        #     model_family="reviewed_bernoulli",
        #     model_label="occu",
        #     selection_method="bald_site_diverse",
        # ),
        dict(
            experiment_id="reviewed_max_score",
            display_name="Reviewed-only Bernoulli max score",
            model_family="reviewed_bernoulli",
            model_label="occu",
            selection_method="max_score",
        ),
        # dict(
        #     experiment_id="reviewed_dual_bald_max_balanced",
        #     display_name=(
        #         "Reviewed-only Bernoulli dual BALD/max-score "
        #         "(12 BALD / 13 max-score)"
        #     ),
        #     model_family="reviewed_bernoulli",
        #     model_label="occu",
        #     selection_method="dual_balanced",
        # ),
        # dict(
        #     experiment_id="reviewed_epig",
        #     display_name="Reviewed-only Bernoulli EPIG",
        #     model_family="reviewed_bernoulli",
        #     model_label="occu",
        #     selection_method="epig",
        # ),
        dict(
            experiment_id="reviewed_posterior_predictive_uncertainty",
            display_name="Reviewed-only Bernoulli posterior predictive uncertainty",
            model_family="reviewed_bernoulli",
            model_label="occu",
            selection_method="posterior_predictive_uncertainty",
        ),
        # dict(
        #     experiment_id="adaptive_calibrated_random",
        #     display_name="CS semi-supervised calibrated random",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="random",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_max_score",
        #     display_name="CS semi-supervised calibrated max score",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="max_score",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_posterior_predictive_uncertainty",
        #     display_name="CS semi-supervised calibrated posterior predictive uncertainty",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="posterior_predictive_uncertainty",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_bald",
        #     display_name="CS semi-supervised calibrated BALD",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="bald",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_dual_bald_max_balanced",
        #     display_name="CS semi-supervised dual BALD/max-score (12 BALD / 13 max-score)",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="dual_balanced",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_bald_site_diverse",
        #     display_name="CS semi-supervised calibrated BALD (site diverse)",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="bald_site_diverse",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_epig",
        #     display_name="CS semi-supervised calibrated EPIG",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="epig",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        # dict(
        #     experiment_id="adaptive_calibrated_revealed_targeted_eig",
        #     display_name="CS semi-supervised calibrated revealed targeted EIG",
        #     model_family="calibrated_cs",
        #     model_label="occu_cs_calibrated",
        #     selection_method="revealed_targeted_eig",
        #     model_fn=occu_cs_calibrated,
        #     model_kwargs={},
        # ),
        *build_cut_normal_experiment_specs(score_calibration_kwargs),
        dict(
            experiment_id="original_occu_cs_bald",
            display_name="Original occu_cs BALD",
            model_family="original_cs",
            model_label="occu_cs",
            selection_method="bald",
            model_fn=occu_cs,
            model_kwargs={},
        ),
        dict(
            experiment_id="original_occu_cs_target_eig",
            display_name="Original occu_cs target EIG",
            model_family="original_cs",
            model_label="occu_cs",
            selection_method="target_eig",
            model_fn=occu_cs,
            model_kwargs={},
        ),
    ]


def expected_real_data_experiment_ids():
    experiment_ids = []
    for base_spec in build_base_experiment_specs():
        experiment_ids.append(base_spec["experiment_id"])
        experiment_ids.append(base_spec["experiment_id"] + "__null")
    return experiment_ids


def real_data_output_is_complete(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir / "run_manifest.csv").exists():
        return False
    experiments_dir = output_dir / "experiments"
    return all(
        saved_experiment_trace_is_usable(experiments_dir / experiment_id)
        for experiment_id in expected_real_data_experiment_ids()
    )


def build_oracle_distance_config(site_covs, obs_covs):
    n_site_covariates = int(np.asarray(site_covs).shape[-1])
    n_obs_covariates = int(np.asarray(obs_covs).shape[-1])
    occupancy_parameter_names = [
        f"cov_state_{index}" for index in range(n_site_covariates + 1)
    ]
    detection_parameter_names = [
        f"cov_det_{index}" for index in range(n_obs_covariates + 1)
    ]
    return (
        ["psi"] + occupancy_parameter_names + detection_parameter_names,
        {
            ORACLE_DISTANCE_GROUP_NAMES[0]: ["psi"],
            ORACLE_DISTANCE_GROUP_NAMES[1]: occupancy_parameter_names,
            ORACLE_DISTANCE_GROUP_NAMES[2]: detection_parameter_names,
        },
    )


def build_experiment_specs(
    data,
    _true_params,
    experiment_ids=None,
    score_calibration_kwargs=None,
):
    spec_build_start_time = time.perf_counter()
    covariate_variants = build_covariate_variants(data)
    base_experiment_specs = build_base_experiment_specs(score_calibration_kwargs)
    experiment_specs = []
    for covariate_variant in covariate_variants:
        shared_parameter_names, oracle_distance_groups = build_oracle_distance_config(
            covariate_variant["site_covs"],
            covariate_variant["obs_covs"],
        )
        for base_spec in base_experiment_specs:
            experiment_specs.append(
                {
                    **base_spec,
                    "experiment_id": (
                        base_spec["experiment_id"]
                        + covariate_variant["experiment_id_suffix"]
                    ),
                    "display_name": (
                        base_spec["display_name"]
                        + covariate_variant["display_name_suffix"]
                    ),
                    "model_label": (
                        base_spec["model_label"]
                        + covariate_variant["model_label_suffix"]
                    ),
                    "covariate_variant": covariate_variant["covariate_variant"],
                    "oracle_variant": covariate_variant["oracle_variant"],
                    "site_covs": covariate_variant["site_covs"],
                    "obs_covs": covariate_variant["obs_covs"],
                    "shared_parameter_names": list(shared_parameter_names),
                    "oracle_distance_groups": {
                        group_name: list(parameter_names)
                        for group_name, parameter_names in oracle_distance_groups.items()
                    },
                    "n_site_covariates": int(
                        np.asarray(covariate_variant["site_covs"]).shape[-1]
                    ),
                    "n_obs_covariates": int(
                        np.asarray(covariate_variant["obs_covs"]).shape[-1]
                    ),
                }
            )
    normalized_experiment_ids = normalize_experiment_ids(experiment_ids)
    if normalized_experiment_ids is not None:
        spec_by_id = {spec["experiment_id"]: spec for spec in experiment_specs}
        missing_ids = [
            experiment_id
            for experiment_id in normalized_experiment_ids
            if experiment_id not in spec_by_id
        ]
        if missing_ids:
            raise ValueError(
                "Unknown experiment id(s): "
                + ", ".join(missing_ids)
                + ". Available ids: "
                + ", ".join(sorted(spec_by_id))
            )
        experiment_specs = [
            spec_by_id[experiment_id] for experiment_id in normalized_experiment_ids
        ]
    return experiment_specs, dict(
        base_score_calibration_wall_time_seconds=0.0,
        base_score_log_bayes_factor_wall_time_seconds=0.0,
        experiment_spec_build_wall_time_seconds=float(
            time.perf_counter() - spec_build_start_time
        ),
    )


def fit_oracle_variant(
    dataset_kind,
    target_species,
    covariate_variant,
    site_covs,
    obs_covs,
    true_params,
    mcmc_kwargs,
):
    shared_parameter_names, oracle_distance_groups = build_oracle_distance_config(
        site_covs,
        obs_covs,
    )
    oracle_label = "oracle" if covariate_variant == "full" else "null oracle"
    print(
        f"Running {oracle_label} fit for {dataset_kind} dataset, target species: {target_species}..."
    )
    oracle_fit_start_time = time.perf_counter()
    oracle_results = fit(
        occu,
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=true_params["f"],
        **mcmc_kwargs,
    )
    oracle_mcmc_wall_time_seconds = time.perf_counter() - oracle_fit_start_time
    timing_metric_name = (
        "oracle_mcmc_wall_time_seconds"
        if covariate_variant == "full"
        else "null_oracle_mcmc_wall_time_seconds"
    )
    oracle_artifact = build_oracle_artifact(
        oracle_results,
        true_params,
        shared_parameter_names,
        timing_metrics={timing_metric_name: float(oracle_mcmc_wall_time_seconds)},
    )
    return dict(
        covariate_variant=covariate_variant,
        results=oracle_results,
        artifact=oracle_artifact,
        shared_parameter_names=list(shared_parameter_names),
        oracle_distance_groups={
            group_name: list(parameter_names)
            for group_name, parameter_names in oracle_distance_groups.items()
        },
        n_site_covariates=int(np.asarray(site_covs).shape[-1]),
        n_obs_covariates=int(np.asarray(obs_covs).shape[-1]),
        timing_metrics={timing_metric_name: float(oracle_mcmc_wall_time_seconds)},
    )


def plot_results(results_by_dataset, output_path):
    fig, axes = plt.subplots(
        len(results_by_dataset),
        4,
        figsize=(14, 3.8 * len(results_by_dataset)),
        sharex=False,
    )
    if len(results_by_dataset) == 1:
        axes = axes[None, :]
    for row_axes, dataset_results in zip(axes, results_by_dataset):
        experiment_items = list(dataset_results["experiment_traces"].items())
        for experiment_id, experiment_trace in experiment_items:
            row_axes[0].plot(
                experiment_trace["review_counts"],
                experiment_trace["mean_oracle_distances"],
                label=experiment_trace["display_name"],
                **experiment_plot_styles[experiment_id],
            )
        row_axes[0].set_title(
            f"{dataset_results['dataset_name']}\nMean oracle distance"
        )
        row_axes[0].set_xlabel("Reviewed observations")
        row_axes[0].set_ylabel("Wasserstein distance to oracle posterior")
        row_axes[0].grid(alpha=0.3)

        for axis_index, group_name in enumerate(
            dataset_results["oracle_distance_groups"], start=1
        ):
            for experiment_id, experiment_trace in experiment_items:
                row_axes[axis_index].plot(
                    experiment_trace["review_counts"],
                    experiment_trace["grouped_oracle_distances"][group_name],
                    label=experiment_trace["display_name"],
                    **experiment_plot_styles[experiment_id],
                )
            row_axes[axis_index].set_title(group_name)
            row_axes[axis_index].set_xlabel("Reviewed observations")
            row_axes[axis_index].grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=min(len(handles), 3), frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_dataset_metadata_artifacts(bundle, output_dir):
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "dataset_metadata"
    metadata = bundle["dataset_metadata"]
    data = bundle["data"]
    true_params = bundle["true_params"]
    write_csv_rows(
        metadata_dir / "scalars.csv",
        [
            dict(
                dataset_kind=bundle["dataset_kind"],
                dataset_name=bundle["dataset_name"],
                target_species=bundle["target_species"],
                target_label=true_params["target_label"],
                n_sites=int(np.asarray(data["site_covs"]).shape[1]),
                n_periods=int(np.asarray(data["obs"]).shape[2]),
                n_replicates=int(np.asarray(data["obs"]).shape[3]),
                n_site_covariates=int(np.asarray(data["site_covs"]).shape[-1]),
                n_obs_covariates=int(np.asarray(data["obs_covs"]).shape[-1]),
                n_reviewable=int(
                    np.asarray(metadata["reviewable_mask"], dtype=bool).sum()
                ),
            )
        ],
    )
    for key, value in metadata.items():
        write_array(metadata_dir / f"{key}.npy", value)

    inputs_dir = output_dir / "inputs"
    write_array(inputs_dir / "obs.npy", data["obs"])
    write_array(inputs_dir / "site_covs.npy", data["site_covs"])
    write_array(inputs_dir / "obs_covs.npy", data["obs_covs"])
    write_array(inputs_dir / "true_labels.npy", true_params["f"])


def save_oracle_artifact(oracle_artifact, output_dir, oracle_dir_name="oracle"):
    oracle_dir = Path(output_dir) / oracle_dir_name
    write_csv_rows(
        oracle_dir / "scalar_metrics.csv", [oracle_artifact["scalar_metrics"]]
    )
    write_parameter_summary_npz(
        oracle_dir / "parameter_summary.npz",
        oracle_artifact["parameter_summary_rows"],
    )
    write_array(oracle_dir / "true_labels.npy", oracle_artifact["true_labels"])


def save_oracle_artifacts(bundle, output_dir):
    save_oracle_artifact(bundle["oracle_artifact"], output_dir)
    null_oracle_artifact = bundle.get("null_oracle_artifact")
    if null_oracle_artifact is not None:
        save_oracle_artifact(
            null_oracle_artifact,
            output_dir,
            oracle_dir_name="null_oracle",
        )


def build_trace_rows(step_index, step_artifact):
    trace_row = dict(step_index=int(step_index), **step_artifact["scalar_metrics"])
    trace_group_rows = []
    for group_row in step_artifact["grouped_oracle_distance_rows"]:
        column_name = "grouped_oracle_distance__" + slugify(group_row["group_name"])
        trace_row[column_name] = group_row["distance"]
        trace_group_rows.append(
            dict(
                step_index=int(step_index),
                group_name=group_row["group_name"],
                distance=group_row["distance"],
            )
        )
    for group_row in step_artifact.get("posterior_shift_rows", []):
        column_name = "posterior_shift_distance__" + slugify(group_row["group_name"])
        trace_row[column_name] = group_row["distance"]
    return trace_row, trace_group_rows


def save_step_artifact(step_artifact, step_dir):
    write_csv_rows(step_dir / "scalar_metrics.csv", [step_artifact["scalar_metrics"]])
    write_csv_rows(
        step_dir / "grouped_oracle_distances.csv",
        step_artifact["grouped_oracle_distance_rows"],
    )
    if step_artifact.get("posterior_shift_rows"):
        write_csv_rows(
            step_dir / "posterior_shift_distances.csv",
            step_artifact["posterior_shift_rows"],
        )
    write_csv_rows(
        step_dir / "parameter_oracle_distances.csv",
        step_artifact["parameter_oracle_distance_rows"],
    )
    write_parameter_summary_npz(
        step_dir / "parameter_summary.npz",
        step_artifact["parameter_summary_rows"],
    )
    write_array(step_dir / "reviewed_values.npy", step_artifact["reviewed_values"])
    write_array(step_dir / "reviewed_mask.npy", step_artifact["reviewed_mask"])
    write_array(step_dir / "selected_mask.npy", step_artifact["selected_mask"])
    write_array(step_dir / "psi_mean.npy", step_artifact["psi_mean"])
    if step_artifact["score_log_bf"] is not None:
        write_array(step_dir / "score_log_bf.npy", step_artifact["score_log_bf"])
    if step_artifact["score_calibration"] is not None:
        write_csv_rows(
            step_dir / "score_calibration.csv",
            [step_artifact["score_calibration"]],
        )
    if step_artifact.get("acquisition_candidate_rows"):
        write_csv_rows(
            step_dir / "acquisition_candidate_diagnostics.csv",
            step_artifact["acquisition_candidate_rows"],
        )


def save_experiment_trace_summary(
    experiment_trace,
    experiment_dir,
    trace_rows,
    trace_group_rows,
    n_steps,
):
    experiment_dir = Path(experiment_dir)
    manifest_row = dict(
        experiment_id=experiment_trace["experiment_id"],
        display_name=experiment_trace["display_name"],
        model_family=experiment_trace["model_family"],
        model_label=experiment_trace["model_label"],
        selection_method=experiment_trace["selection_method"],
        n_steps=int(n_steps),
        final_review_count=int(experiment_trace["review_counts"][-1]),
        final_mean_oracle_distance=float(experiment_trace["mean_oracle_distances"][-1]),
        **experiment_trace.get("timing_metrics", {}),
    )
    if "covariate_variant" in experiment_trace:
        manifest_row["covariate_variant"] = experiment_trace["covariate_variant"]
    if "oracle_variant" in experiment_trace:
        manifest_row["oracle_variant"] = experiment_trace["oracle_variant"]
    if "n_site_covariates" in experiment_trace:
        manifest_row["n_site_covariates"] = int(experiment_trace["n_site_covariates"])
    if "n_obs_covariates" in experiment_trace:
        manifest_row["n_obs_covariates"] = int(experiment_trace["n_obs_covariates"])
    write_csv_rows(
        experiment_dir / "experiment_manifest.csv",
        [manifest_row],
    )
    write_csv_rows(experiment_dir / "trace.csv", trace_rows)
    write_csv_rows(
        experiment_dir / "trace_grouped_oracle_distances.csv",
        trace_group_rows,
        fieldnames=["step_index", "group_name", "distance"],
    )


class ExperimentArtifactWriter:
    """Stream posterior-heavy step artifacts to disk during long experiments."""

    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.step_root = self.experiment_dir / "steps"
        self.trace_rows = []
        self.trace_group_rows = []
        self.step_count = 0

    def write_step(self, step_artifact):
        step_index = self.step_count
        self.step_count += 1
        trace_row, trace_group_rows = build_trace_rows(step_index, step_artifact)
        self.trace_rows.append(trace_row)
        self.trace_group_rows.extend(trace_group_rows)
        save_step_artifact(step_artifact, self.step_root / f"step_{step_index:03d}")

    def finalize(self, experiment_trace):
        save_experiment_trace_summary(
            experiment_trace,
            self.experiment_dir,
            self.trace_rows,
            self.trace_group_rows,
            n_steps=self.step_count,
        )


def save_experiment_trace_artifacts(experiment_trace, experiment_dir):
    experiment_dir = Path(experiment_dir)
    trace_rows = []
    trace_group_rows = []
    for step_index, step_artifact in enumerate(experiment_trace["step_artifacts"]):
        trace_row, current_trace_group_rows = build_trace_rows(
            step_index, step_artifact
        )
        trace_rows.append(trace_row)
        trace_group_rows.extend(current_trace_group_rows)
        save_step_artifact(
            step_artifact,
            experiment_dir / "steps" / f"step_{step_index:03d}",
        )
    save_experiment_trace_summary(
        experiment_trace,
        experiment_dir,
        trace_rows,
        trace_group_rows,
        n_steps=len(experiment_trace["step_artifacts"]),
    )


def save_bundle_artifacts(bundle, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_write_start_time = time.perf_counter()
    save_dataset_metadata_artifacts(bundle, output_dir)
    save_oracle_artifacts(bundle, output_dir)
    for experiment_id, experiment_trace in bundle["dataset_results"][
        "experiment_traces"
    ].items():
        save_experiment_trace_artifacts(
            experiment_trace,
            output_dir / "experiments" / experiment_id,
        )
    plot_results(
        [bundle["dataset_results"]], output_dir / "real_data_oracle_distance.pdf"
    )
    save_run_manifest(
        bundle,
        output_dir,
        artifact_write_wall_time_seconds=float(
            time.perf_counter() - artifact_write_start_time
        ),
    )


def save_run_manifest(bundle, output_dir, artifact_write_wall_time_seconds):
    write_csv_rows(
        output_dir / "run_manifest.csv",
        [
            dict(
                dataset_kind=bundle["dataset_kind"],
                dataset_name=bundle["dataset_name"],
                target_species=bundle["target_species"],
                target_label=bundle["true_params"]["target_label"],
                num_experiments=int(
                    len(bundle["dataset_results"]["experiment_traces"])
                ),
                num_shared_parameters=int(len(bundle["shared_parameter_names"])),
                num_null_shared_parameters=int(
                    len(bundle.get("null_shared_parameter_names", []))
                ),
                num_warmup=int(bundle["mcmc_kwargs"]["num_warmup"]),
                num_samples=int(bundle["mcmc_kwargs"]["num_samples"]),
                num_chains=int(bundle["mcmc_kwargs"]["num_chains"]),
                random_seed=int(bundle["mcmc_kwargs"]["random_seed"]),
                score_calibration_num_warmup=int(
                    bundle.get("score_calibration_kwargs", {}).get(
                        "score_calibration_num_warmup",
                        DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
                            "score_calibration_num_warmup"
                        ],
                    )
                ),
                score_calibration_num_samples=int(
                    bundle.get("score_calibration_kwargs", {}).get(
                        "score_calibration_num_samples",
                        DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
                            "score_calibration_num_samples"
                        ],
                    )
                ),
                score_calibration_num_chains=int(
                    bundle.get("score_calibration_kwargs", {}).get(
                        "score_calibration_num_chains",
                        DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
                            "score_calibration_num_chains"
                        ],
                    )
                ),
                score_calibration_num_draws=int(
                    bundle.get("score_calibration_kwargs", {}).get(
                        "score_calibration_num_draws",
                        DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
                            "score_calibration_num_draws"
                        ],
                    )
                ),
                n_reviews_per_step=int(bundle["n_reviews_per_step"]),
                max_reviews=int(bundle["max_reviews"]),
                artifact_write_wall_time_seconds=float(
                    artifact_write_wall_time_seconds
                ),
                **bundle.get("timing_metrics", {}),
            )
        ],
    )


def run_real_data_experiment(
    dataset_kind,
    target_species,
    output_dir=None,
    mcmc_kwargs=None,
    score_calibration_kwargs=None,
    n_reviews_per_step=DEFAULT_N_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_MAX_REVIEWS,
    continue_existing=False,
    experiment_ids=None,
):
    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    dataset_start_time = time.perf_counter()
    mcmc_kwargs = dict(DEFAULT_MCMC_KWARGS if mcmc_kwargs is None else mcmc_kwargs)
    dataset_load_start_time = time.perf_counter()
    data, true_params = load_real_data_dataset(dataset_kind, target_species)
    dataset_load_wall_time_seconds = time.perf_counter() - dataset_load_start_time
    dataset_metadata = build_real_data_dataset_metadata(data, true_params)
    oracle_variants = {
        spec["covariate_variant"]: spec for spec in build_covariate_variants(data)
    }
    oracle_bundles = {
        covariate_variant: fit_oracle_variant(
            dataset_kind,
            target_species,
            covariate_variant,
            covariate_spec["site_covs"],
            covariate_spec["obs_covs"],
            true_params,
            mcmc_kwargs,
        )
        for covariate_variant, covariate_spec in oracle_variants.items()
    }
    oracle_bundle = oracle_bundles["full"]
    null_oracle_bundle = oracle_bundles["null"]
    oracle_parameter_names = oracle_bundle["shared_parameter_names"]
    oracle_distance_groups = oracle_bundle["oracle_distance_groups"]
    oracle_artifact = oracle_bundle["artifact"]
    null_oracle_artifact = null_oracle_bundle["artifact"]

    experiment_specs, experiment_spec_timing_metrics = build_experiment_specs(
        data,
        true_params,
        experiment_ids=experiment_ids,
        score_calibration_kwargs=score_calibration_kwargs,
    )
    experiment_traces = {}
    experiments_start_time = time.perf_counter()
    for experiment_spec in experiment_specs:
        experiment_id = experiment_spec["experiment_id"]
        experiment_dir = (
            None if output_dir is None else output_dir / "experiments" / experiment_id
        )
        if continue_existing and experiment_dir is not None:
            saved_trace = load_saved_experiment_trace(
                experiment_dir,
                dataset_metadata,
            )
            if saved_trace is not None:
                experiment_traces[experiment_id] = saved_trace
                continue
        artifact_writer = None
        if experiment_dir is not None:
            artifact_writer = ExperimentArtifactWriter(experiment_dir)
        current_oracle_bundle = oracle_bundles[experiment_spec["oracle_variant"]]
        if experiment_spec["model_family"] == "reviewed_bernoulli":
            experiment_trace = run_reviewed_bernoulli_experiment(
                experiment_spec["selection_method"],
                data,
                true_params,
                dataset_metadata,
                experiment_spec["site_covs"],
                experiment_spec["obs_covs"],
                current_oracle_bundle["results"],
                experiment_spec["shared_parameter_names"],
                experiment_spec["oracle_distance_groups"],
                model_kwargs=experiment_spec.get("model_kwargs"),
                mcmc_kwargs=mcmc_kwargs,
                n_reviews_per_step=n_reviews_per_step,
                max_reviews=max_reviews,
                artifact_writer=artifact_writer,
            )
        else:
            experiment_trace = run_cs_experiment(
                experiment_spec["selection_method"],
                data,
                true_params,
                dataset_metadata,
                site_covs=experiment_spec["site_covs"],
                obs_covs=experiment_spec["obs_covs"],
                model_fn=experiment_spec["model_fn"],
                model_kwargs=experiment_spec["model_kwargs"],
                oracle_results=current_oracle_bundle["results"],
                shared_parameter_names=experiment_spec["shared_parameter_names"],
                oracle_distance_groups=experiment_spec["oracle_distance_groups"],
                mcmc_kwargs=mcmc_kwargs,
                n_reviews_per_step=n_reviews_per_step,
                max_reviews=max_reviews,
                artifact_writer=artifact_writer,
            )
        experiment_traces[experiment_id] = dict(
            experiment_id=experiment_id,
            display_name=experiment_spec["display_name"],
            model_family=experiment_spec["model_family"],
            model_label=experiment_spec["model_label"],
            selection_method=experiment_spec["selection_method"],
            covariate_variant=experiment_spec["covariate_variant"],
            oracle_variant=experiment_spec["oracle_variant"],
            n_site_covariates=experiment_spec["n_site_covariates"],
            n_obs_covariates=experiment_spec["n_obs_covariates"],
            **experiment_trace,
        )
        if artifact_writer is not None:
            artifact_writer.finalize(experiment_traces[experiment_id])
    experiments_wall_time_seconds = time.perf_counter() - experiments_start_time

    dataset_results = dict(
        dataset_name=f"{true_params['dataset_name']} / {true_params['target_label']}",
        dataset_metadata=dataset_metadata,
        oracle_distance_groups=oracle_distance_groups,
        experiment_traces=experiment_traces,
    )
    timing_metrics = dict(
        dataset_load_wall_time_seconds=float(dataset_load_wall_time_seconds),
        experiments_wall_time_seconds=float(experiments_wall_time_seconds),
        dataset_compute_wall_time_seconds=float(
            time.perf_counter() - dataset_start_time
        ),
        **oracle_bundle["timing_metrics"],
        **null_oracle_bundle["timing_metrics"],
        **experiment_spec_timing_metrics,
    )
    bundle = dict(
        dataset_kind=dataset_kind,
        target_species=target_species,
        dataset_name=str(true_params["dataset_name"]),
        data=data,
        true_params=true_params,
        dataset_metadata=dataset_metadata,
        oracle_artifact=oracle_artifact,
        null_oracle_artifact=null_oracle_artifact,
        dataset_results=dataset_results,
        shared_parameter_names=oracle_parameter_names,
        null_shared_parameter_names=null_oracle_bundle["shared_parameter_names"],
        mcmc_kwargs=mcmc_kwargs,
        score_calibration_kwargs=score_calibration_kwargs or {},
        n_reviews_per_step=n_reviews_per_step,
        max_reviews=max_reviews,
        timing_metrics=timing_metrics,
    )
    if output_dir is not None:
        artifact_write_start_time = time.perf_counter()
        save_dataset_metadata_artifacts(bundle, output_dir)
        save_oracle_artifacts(bundle, output_dir)
        plot_results(
            [bundle["dataset_results"]],
            output_dir / "real_data_oracle_distance.pdf",
        )
        save_run_manifest(
            bundle,
            output_dir,
            artifact_write_wall_time_seconds=float(
                time.perf_counter() - artifact_write_start_time
            ),
        )
    return bundle


def run_default_experiment_suite(
    output_dir=None,
    mcmc_kwargs=None,
    score_calibration_kwargs=None,
    n_reviews_per_step=DEFAULT_N_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_MAX_REVIEWS,
    experiment_ids=None,
):
    output_dir = ARTIFACTS_DIR / "real_data" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = []
    results_by_dataset = []
    for dataset_spec in DEFAULT_DATASET_SPECS:
        dataset_output_dir = output_dir / (
            f"{dataset_spec['dataset_kind']}__{slugify(dataset_spec['target_species'])}"
        )
        bundle = run_real_data_experiment(
            dataset_spec["dataset_kind"],
            dataset_spec["target_species"],
            output_dir=dataset_output_dir,
            mcmc_kwargs=mcmc_kwargs,
            score_calibration_kwargs=score_calibration_kwargs,
            n_reviews_per_step=n_reviews_per_step,
            max_reviews=max_reviews,
            experiment_ids=experiment_ids,
        )
        bundles.append(bundle)
        results_by_dataset.append(bundle["dataset_results"])

    plot_results(results_by_dataset, output_dir / "real_data_oracle_distance.pdf")
    return bundles


def main(
    dataset=None,
    species=None,
    output_dir=None,
    num_warmup=DEFAULT_MCMC_KWARGS["num_warmup"],
    num_samples=DEFAULT_MCMC_KWARGS["num_samples"],
    num_chains=DEFAULT_MCMC_KWARGS["num_chains"],
    random_seed=DEFAULT_MCMC_KWARGS["random_seed"],
    score_num_warmup=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_num_warmup"
    ],
    score_num_samples=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_num_samples"
    ],
    score_num_chains=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_num_chains"
    ],
    score_num_draws=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_num_draws"
    ],
    score_prevalence_mean=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_prevalence_mean"
    ],
    score_prevalence_strength=DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS[
        "score_calibration_prevalence_strength"
    ],
    reviews_per_step=DEFAULT_N_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_MAX_REVIEWS,
    continue_=False,
    experiment_ids=None,
):
    if output_dir is not None:
        output_dir = Path(output_dir)
    mcmc_kwargs = dict(
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        random_seed=int(random_seed),
    )
    score_calibration_kwargs = dict(
        score_calibration_num_warmup=int(score_num_warmup),
        score_calibration_num_samples=int(score_num_samples),
        score_calibration_num_chains=int(score_num_chains),
        score_calibration_num_draws=int(score_num_draws),
        score_calibration_prevalence_mean=float(score_prevalence_mean),
        score_calibration_prevalence_strength=float(score_prevalence_strength),
        score_calibration_random_seed=int(random_seed),
    )
    if (dataset is None) != (species is None):
        raise ValueError("`dataset` and `species` must be provided together.")

    if dataset is None:
        run_default_experiment_suite(
            output_dir=output_dir,
            mcmc_kwargs=mcmc_kwargs,
            score_calibration_kwargs=score_calibration_kwargs,
            n_reviews_per_step=int(reviews_per_step),
            max_reviews=int(max_reviews),
            experiment_ids=experiment_ids,
        )
        return

    run_real_data_experiment(
        dataset,
        species,
        output_dir=output_dir,
        mcmc_kwargs=mcmc_kwargs,
        score_calibration_kwargs=score_calibration_kwargs,
        n_reviews_per_step=int(reviews_per_step),
        max_reviews=int(max_reviews),
        continue_existing=bool(continue_),
        experiment_ids=experiment_ids,
    )


__all__ = [
    "DEFAULT_DATASET_SPECS",
    "DEFAULT_MAX_REVIEWS",
    "DEFAULT_MCMC_KWARGS",
    "DEFAULT_N_REVIEWS_PER_STEP",
    "main",
    "run_default_experiment_suite",
    "run_real_data_experiment",
    "slugify",
]
