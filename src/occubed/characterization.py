"""Task agreement and pre-decision sample profiles for the main experiment artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from occubed.experiment_config import FINAL_EXPERIMENT_IDS, TARGET_EXPERIMENT_IDS
from occubed.export_sweep import read_csv_rows, read_single_csv_row
from occubed.selection_characterization import (
    covariate_novelty, discovery_credits, masked_leverage, read_candidate_diagnostics,
)
from occubed.summaries import iter_parameter_summary_arrays, parameter_summary_exists

DEFAULT_EXPERIMENT_IDS = FINAL_EXPERIMENT_IDS
POINT_PROFILE_EXPERIMENT_IDS = FINAL_EXPERIMENT_IDS
COEFFICIENT_PREFIXES = ("cov_state_", "cov_det_")
EXPERIMENT_TARGETS = {value: key for key, value in TARGET_EXPERIMENT_IDS.items()}
TARGET_METRIC_COLUMNS = {
    "combined": "mean_ecological_agreement",
    "site_occupancy": "site_occupancy_task_agreement",
    "occupancy_coefficients": "occupancy_coefficient_agreement",
    "detection_coefficients": "detection_coefficient_agreement",
}


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    result = safe_float(value)
    return int(result) if np.isfinite(result) else default


def finite_mean(values):
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def species_spec(species_dir):
    root = Path(species_dir)
    manifest = read_single_csv_row(root / "run_manifest.csv")
    return dict(dataset_kind=manifest["dataset_kind"], species=manifest["target_species"],
                species_key=f"{manifest['dataset_kind']}::{manifest['target_species']}",
                seed=int(manifest["random_seed"]), output_dir=str(root))


def _analysis_budget(dataset_kind, suite_manifest):
    return int(suite_manifest[f"{dataset_kind}_max_reviews"])


def _squeeze_leading_species_axis(array, reference_ndim):
    array = np.asarray(array)
    if array.ndim == reference_ndim + 1 and array.shape[0] == 1:
        return array[0]
    return array


def _load_parameter_stats(path, parameters=None, prefixes=()):
    stats_by_name = {}
    for parameter_name, stats in iter_parameter_summary_arrays(
        path,
        parameters=None if parameters is None else set(parameters),
        parameter_prefixes=tuple(prefixes),
        stats=("mean", "q05", "q95"),
    ):
        stats_by_name[str(parameter_name)] = {
            name: np.asarray(value, dtype=float) for name, value in stats.items()
        }
    return stats_by_name


def _coefficient_conclusions(stats_by_name, prefix):
    conclusions = {}
    for parameter_name, stats in stats_by_name.items():
        if not parameter_name.startswith(prefix):
            continue
        q05 = np.asarray(stats.get("q05", np.nan), dtype=float)
        q95 = np.asarray(stats.get("q95", np.nan), dtype=float)
        shape = np.broadcast_shapes(q05.shape, q95.shape)
        q05 = np.broadcast_to(q05, shape).reshape(-1)
        q95 = np.broadcast_to(q95, shape).reshape(-1)
        for index, (lower, upper) in enumerate(zip(q05, q95)):
            key = (parameter_name, index)
            if not np.isfinite(lower) or not np.isfinite(upper):
                conclusions[key] = None
            elif lower > 0:
                conclusions[key] = 1
            elif upper < 0:
                conclusions[key] = -1
            else:
                conclusions[key] = 0
    return conclusions


def _conclusion_agreement(current, oracle, prefix):
    current_values = _coefficient_conclusions(current, prefix)
    oracle_values = _coefficient_conclusions(oracle, prefix)
    keys = [
        key
        for key in sorted(set(current_values) & set(oracle_values))
        if current_values[key] is not None and oracle_values[key] is not None
    ]
    if not keys:
        return float("nan")
    return float(np.mean([current_values[key] == oracle_values[key] for key in keys]))


def _rankdata_average(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_corr(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float).reshape(-1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1)
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    if finite.sum() < 2:
        return float("nan")
    ranks_a = _rankdata_average(values_a[finite])
    ranks_b = _rankdata_average(values_b[finite])
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _oracle_psi_mean(species_dir, oracle_dir="oracle"):
    stats = _load_parameter_stats(
        Path(species_dir) / oracle_dir / "parameter_summary.npz",
        parameters={"psi"},
    )
    if "psi" not in stats or "mean" not in stats["psi"]:
        return np.asarray([], dtype=float)
    return np.asarray(stats["psi"]["mean"], dtype=float)


def _experiment_target(experiment_id, manifest):
    target = str(manifest.get("ecological_target", "")).strip()
    if target:
        return target
    return EXPERIMENT_TARGETS.get(experiment_id, "baseline")


def build_task_metric_rows(output_dir, experiment_ids=DEFAULT_EXPERIMENT_IDS):
    output_dir = Path(output_dir)
    rows = []
    for spec in [species_spec(output_dir)]:
        species_dir = Path(spec["output_dir"])
        for experiment_id in experiment_ids:
            experiment_dir = species_dir / "experiments" / experiment_id
            manifest = read_single_csv_row(experiment_dir / "experiment_manifest.csv")
            trace_rows = read_csv_rows(experiment_dir / "trace.csv")
            if not manifest or not trace_rows:
                continue
            oracle_dir = "null_oracle" if manifest.get("oracle_variant") == "null" else "oracle"
            oracle_psi = _oracle_psi_mean(species_dir, oracle_dir).reshape(-1)
            oracle_coefficients = _load_parameter_stats(
                species_dir / oracle_dir / "parameter_summary.npz", prefixes=COEFFICIENT_PREFIXES)
            ecological_target = _experiment_target(experiment_id, manifest)
            for fallback_index, trace_row in enumerate(trace_rows):
                step_index = safe_int(trace_row.get("step_index"), fallback_index)
                step_dir = experiment_dir / "steps" / f"step_{step_index:03d}"
                psi_path = step_dir / "psi_mean.npy"
                summary_path = step_dir / "parameter_summary.npz"
                if not psi_path.exists() or not parameter_summary_exists(summary_path):
                    continue
                current_psi = np.asarray(
                    np.load(psi_path, allow_pickle=False), dtype=float
                ).reshape(-1)
                count = min(current_psi.size, oracle_psi.size)
                finite = (
                    np.isfinite(current_psi[:count]) & np.isfinite(oracle_psi[:count])
                    if count
                    else np.asarray([], dtype=bool)
                )
                if finite.any():
                    psi_agreement = 1.0 - float(
                        np.mean(
                            np.abs(
                                current_psi[:count][finite] - oracle_psi[:count][finite]
                            )
                        )
                    )
                    psi_rank = spearman_corr(
                        current_psi[:count][finite], oracle_psi[:count][finite]
                    )
                else:
                    psi_agreement = float("nan")
                    psi_rank = float("nan")
                current_coefficients = _load_parameter_stats(
                    summary_path, prefixes=COEFFICIENT_PREFIXES
                )
                occupancy_agreement = _conclusion_agreement(
                    current_coefficients, oracle_coefficients, "cov_state_"
                )
                detection_agreement = _conclusion_agreement(
                    current_coefficients, oracle_coefficients, "cov_det_"
                )
                site_task = finite_mean((psi_agreement, psi_rank))
                combined = finite_mean(
                    (
                        psi_agreement,
                        psi_rank,
                        occupancy_agreement,
                        detection_agreement,
                    )
                )
                rows.append(
                    dict(
                        dataset_kind=spec["dataset_kind"],
                        species=spec["species"],
                        species_key=spec["species_key"],
                        experiment_id=experiment_id,
                        selection_method=manifest.get("selection_method", ""),
                        ecological_target=ecological_target,
                        review_count=safe_int(trace_row.get("review_count")),
                        occupancy_probability_agreement=psi_agreement,
                        site_rank_spearman=psi_rank,
                        site_occupancy_task_agreement=site_task,
                        occupancy_coefficient_agreement=occupancy_agreement,
                        detection_coefficient_agreement=detection_agreement,
                        mean_ecological_agreement=combined,
                    )
                )
    return pd.DataFrame.from_records(rows)


def normalized_auc(counts, values, max_budget):
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(counts) & np.isfinite(values)
    counts = counts[finite]
    values = values[finite]
    if counts.size < 2 or max_budget <= 0:
        return float("nan")
    order = np.argsort(counts)
    counts = counts[order]
    values = values[order]
    unique_counts, unique_indices = np.unique(counts, return_index=True)
    counts = unique_counts
    values = values[unique_indices]
    if max_budget < counts.min() or max_budget > counts.max():
        return float("nan")
    interior = (counts >= 0) & (counts <= max_budget)
    clipped_counts = counts[interior]
    clipped_values = values[interior]
    if clipped_counts.size == 0 or clipped_counts[0] > 0:
        clipped_counts = np.insert(clipped_counts, 0, 0.0)
        clipped_values = np.insert(clipped_values, 0, np.interp(0, counts, values))
    if clipped_counts[-1] < max_budget:
        clipped_counts = np.append(clipped_counts, float(max_budget))
        clipped_values = np.append(
            clipped_values, np.interp(max_budget, counts, values)
        )
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(clipped_values, clipped_counts) / float(max_budget))




def binary_entropy(probability):
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(
        probability * np.log(probability) + (1.0 - probability) * np.log1p(-probability)
    )


def percentile_in_pool(value, pool):
    pool = np.asarray(pool, dtype=float)
    finite = pool[np.isfinite(pool)]
    if not np.isfinite(value) or finite.size == 0:
        return float("nan")
    lower = np.sum(finite < value)
    tied = np.sum(finite == value)
    return float((lower + 0.5 * tied) / finite.size)


def _broadcast_site_values(values, shape):
    values = np.asarray(values).reshape(shape[0], *([1] * (len(shape) - 1)))
    return np.broadcast_to(values, shape)


def _load_species_arrays(species_dir):
    species_dir = Path(species_dir)
    reviewable = np.asarray(
        np.load(species_dir / "dataset_metadata" / "reviewable_mask.npy"),
        dtype=bool,
    )
    scores = np.asarray(
        np.load(species_dir / "dataset_metadata" / "score_values.npy"), dtype=float
    )
    labels = np.asarray(
        np.load(species_dir / "inputs" / "true_labels.npy"), dtype=float
    )
    labels = _squeeze_leading_species_axis(labels, reviewable.ndim)
    site_covs = np.asarray(
        np.load(species_dir / "inputs" / "site_covs.npy"), dtype=float
    )
    if site_covs.ndim == 3 and site_covs.shape[0] == 1:
        site_covs = site_covs[0]
    obs_covs = np.asarray(np.load(species_dir / "inputs" / "obs_covs.npy"), dtype=float)
    if obs_covs.ndim == reviewable.ndim + 2 and obs_covs.shape[0] == 1:
        obs_covs = obs_covs[0]
    if labels.shape != reviewable.shape or scores.shape != reviewable.shape:
        raise ValueError(
            f"Artifact shape mismatch in {species_dir}: labels={labels.shape}, "
            f"scores={scores.shape}, reviewable={reviewable.shape}."
        )
    initial_mask = reviewable & np.isfinite(labels)
    site_valid = initial_mask.any(axis=tuple(range(1, reviewable.ndim)))
    site_leverage = masked_leverage(site_covs, site_valid)
    flat_obs_covs = obs_covs.reshape(-1, obs_covs.shape[-1])
    obs_leverage = masked_leverage(flat_obs_covs, initial_mask.reshape(-1)).reshape(reviewable.shape)
    return dict(
        reviewable=reviewable,
        scores=scores,
        labels=labels,
        site_covs=site_covs,
        obs_covs=obs_covs,
        site_leverage=site_leverage,
        obs_leverage=obs_leverage,
        initial_mask=initial_mask,
    )


def _candidate_diagnostics(step_dir):
    compact = read_candidate_diagnostics(step_dir)
    if compact is not None:
        return compact
    rows = read_csv_rows(Path(step_dir) / "acquisition_candidate_diagnostics.csv")
    return {safe_int(row.get("candidate_id"), -1): row for row in rows}


def _review_phase(review_count, max_budget):
    fraction = float(review_count) / max(float(max_budget), 1.0)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _pool_mean(values, mask):
    values = np.asarray(values, dtype=float)
    finite = mask & np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def build_selected_point_profiles(output_dir, suite_manifest):
    output_dir = Path(output_dir)
    point_rows = []
    batch_rows = []
    for spec in [species_spec(output_dir)]:
        species_dir = Path(spec["output_dir"])
        if not (species_dir / "dataset_metadata" / "reviewable_mask.npy").exists():
            continue
        arrays = _load_species_arrays(species_dir)
        shape = arrays["reviewable"].shape
        site_axes = tuple(range(1, len(shape)))
        site_leverage_grid = _broadcast_site_values(arrays["site_leverage"], shape)
        max_budget = _analysis_budget(spec["dataset_kind"], suite_manifest)
        experiment_ids = suite_manifest.get("experiment_ids", POINT_PROFILE_EXPERIMENT_IDS)
        if isinstance(experiment_ids, str):
            experiment_ids = experiment_ids.split(",")
        for experiment_id in experiment_ids:
            experiment_dir = species_dir / "experiments" / experiment_id
            experiment_manifest = read_single_csv_row(
                experiment_dir / "experiment_manifest.csv"
            )
            trace_rows = read_csv_rows(experiment_dir / "trace.csv")
            if not experiment_manifest or not trace_rows:
                continue
            ecological_target = _experiment_target(experiment_id, experiment_manifest)
            selection_method = str(experiment_manifest.get("selection_method", ""))
            for fallback_index, trace_row in enumerate(trace_rows):
                if safe_int(trace_row.get("selected_count"), 0) <= 0:
                    continue
                step_index = safe_int(trace_row.get("step_index"), fallback_index)
                step_dir = experiment_dir / "steps" / f"step_{step_index:03d}"
                selected_path = step_dir / "selected_mask.npy"
                reviewed_path = step_dir / "reviewed_values.npy"
                psi_path = step_dir / "psi_mean.npy"
                if not (
                    selected_path.exists()
                    and reviewed_path.exists()
                    and psi_path.exists()
                ):
                    continue
                selected = np.asarray(
                    np.load(selected_path, allow_pickle=False), dtype=bool
                )
                reviewed = np.asarray(
                    np.load(reviewed_path, allow_pickle=False), dtype=float
                )
                psi_mean = np.asarray(
                    np.load(psi_path, allow_pickle=False), dtype=float
                )
                candidate_mask = (
                    arrays["reviewable"]
                    & np.isfinite(arrays["labels"])
                    & ~np.isfinite(reviewed)
                )
                selected &= candidate_mask
                if not selected.any():
                    continue
                reviewed_count_by_site = np.isfinite(reviewed).sum(axis=site_axes)
                reviewed_positive_by_site = (reviewed > 0.5).sum(axis=site_axes)
                selected_count_by_site = selected.sum(axis=site_axes)
                credits = discovery_credits(selected, arrays["labels"], reviewed)
                site_novelty = covariate_novelty(arrays["site_covs"], reviewed_count_by_site > 0)
                obs_novelty = covariate_novelty(
                    arrays["obs_covs"].reshape(-1, arrays["obs_covs"].shape[-1]),
                    np.isfinite(reviewed).reshape(-1),
                ).reshape(shape)
                bootstrap = (selection_method in {"target_eig", "bald", "bald_site_diverse"}
                             and not (reviewed > 0.5).any()
                             and experiment_id.startswith("cut_normal_calibrated"))
                compact_path = step_dir / "selection_pool.npz"
                if compact_path.exists():
                    with np.load(compact_path, allow_pickle=False) as compact:
                        bootstrap = bool(compact["used_max_score_bootstrap"])
                selection_stage = ("max_score_bootstrap" if bootstrap else
                                   "adaptive" if selection_method in {"target_eig", "bald", "bald_site_diverse"}
                                   else "baseline")
                log_bf_path = step_dir / "score_log_bf.npy"
                log_bf = (np.load(log_bf_path, allow_pickle=False).reshape(shape)
                          if log_bf_path.exists() else np.full(shape, np.nan))
                site_entropy = binary_entropy(psi_mean)
                if site_entropy.ndim == len(shape) - 1:
                    site_entropy_grid = np.broadcast_to(site_entropy[..., None], shape)
                else:
                    site_entropy_grid = _broadcast_site_values(
                        np.asarray(site_entropy).reshape(shape[0], -1).mean(axis=1),
                        shape,
                    )
                diagnostics = _candidate_diagnostics(step_dir)
                diagnostic_fields = (
                    "p_positive",
                    "target_eig",
                    "label_entropy",
                    "bald_score",
                    "ppu_score",
                    "ess_positive",
                    "ess_negative",
                )
                diagnostic_pools = {
                    field: np.asarray(
                        [safe_float(row.get(field)) for row in diagnostics.values()],
                        dtype=float,
                    )
                    for field in diagnostic_fields
                }
                candidate_score_values = arrays["scores"][candidate_mask]
                candidate_site_leverage = site_leverage_grid[candidate_mask]
                candidate_obs_leverage = arrays["obs_leverage"][candidate_mask]
                candidate_site_entropy = site_entropy_grid[candidate_mask]
                selected_site_indices = np.flatnonzero(selected_count_by_site > 0)
                review_count = safe_int(trace_row.get("review_count"), 0)
                for site_index, period_index, replicate_index in np.argwhere(selected):
                    coordinate = (site_index, period_index, replicate_index)
                    candidate_id = int(np.ravel_multi_index(coordinate, shape))
                    label = float(arrays["labels"][coordinate])
                    score = float(arrays["scores"][coordinate])
                    diag = diagnostics.get(candidate_id, {})
                    p_positive = safe_float(diag.get("p_positive"))
                    score_percentile = percentile_in_pool(score, candidate_score_values)
                    classifier_surprise = (
                        1.0 - score_percentile if label > 0.5 else score_percentile
                    )
                    row = dict(
                        dataset_kind=spec["dataset_kind"],
                        species=spec["species"],
                        species_key=spec["species_key"],
                        experiment_id=experiment_id,
                        selection_method=selection_method,
                        ecological_target=ecological_target,
                        review_count=review_count,
                        review_phase=_review_phase(review_count, max_budget),
                        selection_stage=selection_stage,
                        candidate_id=candidate_id,
                        site_index=int(site_index),
                        period_index=int(period_index),
                        replicate_index=int(replicate_index),
                        true_label=label,
                        score=score,
                        score_percentile=score_percentile,
                        initial_score_percentile=percentile_in_pool(score, arrays["scores"][arrays["initial_mask"]]),
                        score_log_bayes_factor=float(log_bf[coordinate]),
                        score_evidence_magnitude=float(abs(log_bf[coordinate])),
                        classifier_rank_surprise=classifier_surprise,
                        prior_reviews_at_site=int(reviewed_count_by_site[site_index]),
                        prior_positives_at_site=int(
                            reviewed_positive_by_site[site_index]
                        ),
                        new_site_review=int(reviewed_count_by_site[site_index] == 0),
                        first_positive_at_site=int(
                            label > 0.5 and reviewed_positive_by_site[site_index] == 0
                        ),
                        first_positive_site_credit=float(credits[coordinate]),
                        prior_negatives_at_site=int(reviewed_count_by_site[site_index] - reviewed_positive_by_site[site_index]),
                        selected_from_site_in_batch=int(
                            selected_count_by_site[site_index]
                        ),
                        site_leverage=float(site_leverage_grid[coordinate]),
                        site_leverage_percentile=percentile_in_pool(
                            site_leverage_grid[coordinate], candidate_site_leverage
                        ),
                        observation_leverage=float(arrays["obs_leverage"][coordinate]),
                        observation_leverage_percentile=percentile_in_pool(
                            arrays["obs_leverage"][coordinate], candidate_obs_leverage
                        ),
                        site_covariate_max_abs_z=float(np.max(np.abs(arrays["site_covs"][site_index]), initial=0)),
                        observation_covariate_max_abs_z=float(np.max(np.abs(arrays["obs_covs"][coordinate]), initial=0)),
                        site_covariate_novelty=float(site_novelty[site_index]),
                        observation_covariate_novelty=float(obs_novelty[coordinate]),
                        site_occupancy_entropy=float(site_entropy_grid[coordinate]),
                        site_occupancy_entropy_percentile=percentile_in_pool(
                            site_entropy_grid[coordinate], candidate_site_entropy
                        ),
                        p_positive=p_positive,
                        predictive_label_confidence=(max(p_positive, 1 - p_positive) if np.isfinite(p_positive) else float("nan")),
                        posterior_label_surprise=(
                            abs(label - p_positive)
                            if np.isfinite(p_positive)
                            else float("nan")
                        ),
                    )
                    for field in diagnostic_fields:
                        value = safe_float(diag.get(field))
                        row[field] = value
                        row[f"{field}_percentile"] = percentile_in_pool(
                            value, diagnostic_pools[field]
                        )
                    for prefix, values in (("site_covariate_z", arrays["site_covs"][site_index]),
                                           ("observation_covariate_z", arrays["obs_covs"][coordinate])):
                        for index, value in enumerate(values):
                            row[f"{prefix}_{index}"] = float(value)
                    point_rows.append(row)

                selected_points = [
                    row
                    for row in point_rows[-int(selected.sum()) :]
                    if row["experiment_id"] == experiment_id
                    and row["review_count"] == review_count
                ]
                batch_rows.append(
                    dict(
                        dataset_kind=spec["dataset_kind"],
                        species=spec["species"],
                        species_key=spec["species_key"],
                        experiment_id=experiment_id,
                        selection_method=selection_method,
                        ecological_target=ecological_target,
                        review_count=review_count,
                        review_phase=_review_phase(review_count, max_budget),
                        selection_stage=selection_stage,
                        selected_count=int(selected.sum()),
                        unique_selected_sites=int(selected_site_indices.size),
                        newly_reviewed_sites=int(((selected_count_by_site > 0) & (reviewed_count_by_site == 0)).sum()),
                        previously_reviewed_sites=int((reviewed_count_by_site > 0).sum()),
                        previously_confirmed_positive_sites=int((reviewed_positive_by_site > 0).sum()),
                        within_batch_site_redundancy=(
                            1.0 - selected_site_indices.size / int(selected.sum())
                        ),
                        selected_positive_rate=finite_mean(
                            row["true_label"] for row in selected_points
                        ),
                        candidate_positive_rate=_pool_mean(
                            arrays["labels"], candidate_mask
                        ),
                        selected_first_positive_rate=float(credits.sum() / selected.sum()),
                        newly_confirmed_positive_sites=float(credits.sum()),
                        selected_positive_at_unconfirmed_site_rate=finite_mean(
                            row["first_positive_at_site"] for row in selected_points),
                        candidate_first_positive_rate=_pool_mean(
                            (arrays["labels"] > 0.5)
                            & _broadcast_site_values(
                                reviewed_positive_by_site == 0, shape
                            ),
                            candidate_mask,
                        ),
                        selected_new_site_rate=finite_mean(
                            row["new_site_review"] for row in selected_points
                        ),
                        candidate_new_site_rate=_pool_mean(
                            _broadcast_site_values(reviewed_count_by_site == 0, shape),
                            candidate_mask,
                        ),
                        mean_score_percentile=finite_mean(
                            row["score_percentile"] for row in selected_points
                        ),
                        mean_site_leverage_percentile=finite_mean(
                            row["site_leverage_percentile"] for row in selected_points
                        ),
                        mean_observation_leverage_percentile=finite_mean(
                            row["observation_leverage_percentile"]
                            for row in selected_points
                        ),
                        mean_site_occupancy_entropy_percentile=finite_mean(
                            row["site_occupancy_entropy_percentile"]
                            for row in selected_points
                        ),
                        mean_p_positive=finite_mean(
                            row["p_positive"] for row in selected_points
                        ),
                        candidate_mean_p_positive=finite_mean(diagnostic_pools["p_positive"]),
                        candidate_mean_predictive_label_confidence=finite_mean(
                            np.maximum(diagnostic_pools["p_positive"], 1 - diagnostic_pools["p_positive"])),
                        candidate_mean_score_evidence_magnitude=_pool_mean(np.abs(log_bf), candidate_mask),
                        candidate_mean_site_occupancy_entropy=_pool_mean(site_entropy_grid, candidate_mask),
                        mean_predictive_label_confidence=finite_mean(row["predictive_label_confidence"] for row in selected_points),
                        mean_site_covariate_novelty=finite_mean(row["site_covariate_novelty"] for row in selected_points),
                        candidate_mean_site_covariate_novelty=_pool_mean(_broadcast_site_values(site_novelty, shape), candidate_mask),
                        mean_observation_covariate_novelty=finite_mean(row["observation_covariate_novelty"] for row in selected_points),
                        candidate_mean_observation_covariate_novelty=_pool_mean(obs_novelty, candidate_mask),
                        mean_score_evidence_magnitude=finite_mean(row["score_evidence_magnitude"] for row in selected_points),
                        candidate_mean_site_leverage=_pool_mean(site_leverage_grid, candidate_mask),
                        candidate_mean_observation_leverage=_pool_mean(arrays["obs_leverage"], candidate_mask),
                        mean_site_leverage=finite_mean(row["site_leverage"] for row in selected_points),
                        mean_observation_leverage=finite_mean(row["observation_leverage"] for row in selected_points),
                        mean_label_entropy_percentile=finite_mean(
                            row["label_entropy_percentile"] for row in selected_points
                        ),
                        mean_target_eig_percentile=finite_mean(
                            row["target_eig_percentile"] for row in selected_points
                        ),
                        mean_posterior_label_surprise=finite_mean(
                            row["posterior_label_surprise"] for row in selected_points
                        ),
                    )
                )
    return (
        pd.DataFrame.from_records(point_rows),
        pd.DataFrame.from_records(batch_rows),
    )


def export_species(species_dir, experiment_ids=FINAL_EXPERIMENT_IDS):
    """Export the same metrics/profiles as the final runs, without another launcher."""
    root = Path(species_dir)
    spec = species_spec(root)
    run = read_single_csv_row(root / "run_manifest.csv")
    budget = int(run["max_reviews"])
    destination = root / "characterization"
    destination.mkdir(exist_ok=True)
    metrics = build_task_metric_rows(root, experiment_ids)
    cadence = int(run["n_reviews_per_step"])
    expected = np.append(np.arange(0, budget, cadence), budget)
    for experiment in experiment_ids:
        counts = metrics.loc[metrics.experiment_id.eq(experiment), "review_count"].to_numpy()
        if not np.array_equal(counts, expected):
            raise ValueError(f"Incomplete task metric trajectory: {experiment}")
    points, batches = build_selected_point_profiles(root, dict(
        experiment_ids=experiment_ids, **{f"{spec['dataset_kind']}_max_reviews": budget}))
    if len(points) != budget * len(experiment_ids):
        raise ValueError("Selected-point profiles do not cover the complete trajectories.")
    auc, gains = [], []
    for experiment, group in metrics.groupby("experiment_id"):
        group = group.sort_values("review_count")
        for target, metric in TARGET_METRIC_COLUMNS.items():
            for window, endpoint in (("100_reviews", min(100, budget)), ("full", budget)):
                auc.append(dict(experiment_id=experiment, evaluation_target=target,
                                acquisition_target=EXPERIMENT_TARGETS.get(experiment, "baseline"),
                                window=window, budget=endpoint,
                                normalized_auc=normalized_auc(group.review_count, group[metric], endpoint)))
        records = group.to_dict("records")
        for before, after in zip(records, records[1:]):
            gains.append(dict(experiment_id=experiment, review_count=before["review_count"],
                              end_review_count=after["review_count"], **{
                                  f"batch_gain_{target}": after[metric] - before[metric]
                                  for target, metric in TARGET_METRIC_COLUMNS.items()}))
    if gains:
        batches = batches.merge(pd.DataFrame(gains), on=["experiment_id", "review_count"], validate="one_to_one")
    for name, frame in (("task_metric_rows", metrics), ("task_auc", pd.DataFrame(auc)),
                        ("selected_points", points), ("selected_batch_profiles", batches)):
        for key in ("dataset_kind", "species", "species_key", "seed"):
            frame[key] = spec[key]
        frame.to_csv(destination / f"{name}.csv", index=False)
    return destination


def bootstrap_mean(values, n_bootstrap=2000):
    """Species bootstrap, with percentile bounds and a normal 95% half-width."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite species summaries")
    samples = np.random.default_rng(0).choice(values, size=(n_bootstrap, len(values))).mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(mean=float(values.mean()), ci_low=float(low), ci_high=float(high),
                ci_half_width=float(1.96 * samples.std(ddof=1)), species_count=len(values))


def summarize_sweep(species_dirs, destination):
    """Paired full-trajectory task contrasts and first-100-review sample profiles."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    auc_frames, profiles = [], []
    features = ["new_site_review", "score_percentile", "predictive_label_confidence",
                "site_leverage_percentile", "observation_leverage_percentile"]
    for species_dir in species_dirs:
        spec = species_spec(species_dir)
        local = Path(species_dir) / "characterization"
        auc_frames.append(pd.read_csv(local / "task_auc.csv"))
        points = pd.read_csv(local / "selected_points.csv",
                             usecols=["experiment_id", "review_count", "candidate_id", *features])
        for experiment, group in points.groupby("experiment_id"):
            if not experiment.startswith("cut_normal_calibrated_"):
                continue
            selected = group.sort_values(["review_count", "candidate_id"]).head(100)
            profiles.append(dict(**{key: spec[key] for key in ("dataset_kind", "species", "seed")},
                                 experiment_id=experiment, **selected[features].mean().to_dict()))
    auc = pd.concat(auc_frames, ignore_index=True)
    auc = auc[auc.window.eq("full") & auc.experiment_id.str.startswith("cut_normal_calibrated_")]
    index = ["dataset_kind", "species", "seed", "evaluation_target"]
    matrix = auc.pivot(index=index, columns="experiment_id", values="normalized_auc")
    if matrix.isna().any().any():
        raise ValueError("Task comparisons are not paired across configurations.")
    contrasts = []
    for identity, row in matrix.iterrows():
        spec = dict(zip(index, identity))
        if TARGET_EXPERIMENT_IDS["combined"] not in row:
            continue
        comparators = {name: "cut_normal_calibrated_" + name for name in (
            "random", "max_score", "bald", "posterior_predictive_uncertainty")}
        if spec["evaluation_target"] != "combined":
            comparators["task_specific_acorn"] = TARGET_EXPERIMENT_IDS[spec["evaluation_target"]]
        for label, experiment in comparators.items():
            if experiment not in row:
                continue
            contrasts.append(dict(spec, comparator=label,
                                  difference=row[TARGET_EXPERIMENT_IDS["combined"]] - row[experiment]))
    contrasts = pd.DataFrame(contrasts, columns=[*index, "comparator", "difference"])
    task_keys = ["dataset_kind", "evaluation_target", "comparator"]
    species_tasks = contrasts.groupby(task_keys + ["species"], as_index=False).difference.mean()
    task_summary = [dict(zip(task_keys, identity), **bootstrap_mean(group.difference))
                    for identity, group in species_tasks.groupby(task_keys)]
    profile_frame = pd.DataFrame(profiles)
    profile_keys = ["dataset_kind", "experiment_id"]
    species_profiles = profile_frame.groupby(profile_keys + ["species"], as_index=False)[features].mean()
    profile_summary = [dict(zip(profile_keys, identity), feature=feature, **bootstrap_mean(group[feature]))
                       for identity, group in species_profiles.groupby(profile_keys) for feature in features]
    for name, frame in (("task_contrasts_by_species_seed", contrasts),
                        ("task_contrasts_by_species", species_tasks),
                        ("task_performance_summary", pd.DataFrame(task_summary)),
                        ("selection_profiles_by_species_seed", profile_frame),
                        ("selection_profiles_by_species", species_profiles),
                        ("selection_profile_summary", pd.DataFrame(profile_summary))):
        frame.to_csv(destination / f"{name}.csv", index=False)
    return destination
