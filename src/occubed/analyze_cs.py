import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from occubed.paths import ARTIFACTS_DIR, DEFAULT_SWEEP_DIR

DEFAULT_RESULTS_CSV = "combined_species_results.csv"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "cs_analysis"

SHARED_EXPERIMENT_IDS = [
    "cut_normal_calibrated_bald",
    "cut_normal_calibrated_target_eig",
    "original_occu_cs_bald",
    "original_occu_cs_target_eig",
]
AGREEMENT_BUDGETS = [0, 100, 500, 1000]
MAX_SPECIES_BY_DATASET = {
    "acoustic": 50,
    "iwildcam": 25,
}
AGREEMENT_METRICS = [
    "psi_probability_agreement",
    "site_priority_spearman",
    "coef_ci_conclusion_agreement__occupancy",
    "coef_ci_conclusion_agreement__detection",
]
DATASET_LABELS = {
    "acoustic": "Acoustic",
    "iwildcam": "Camera trap",
}
MODEL_LABELS = {
    "cut_normal_calibrated_cs": "Decoupled CS",
    "original_cs": "Original CS",
}
MODEL_ORDER = {
    "original_cs": 0,
    "cut_normal_calibrated_cs": 1,
}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``.git``."""

    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return path


def safe_float(value: object) -> float:
    """Convert a value to float, returning NaN on non-numeric input."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def infer_model_family(experiment_id: object, value: object = "") -> str:
    """Infer a normalized model family from export metadata."""

    text_value = str(value).strip()
    if text_value and text_value.lower() != "nan":
        return text_value
    text_id = str(experiment_id)
    if text_id.startswith("original_occu_cs"):
        return "original_cs"
    if text_id.startswith("cut_normal_calibrated"):
        return "cut_normal_calibrated_cs"
    return ""


def infer_selection_method(experiment_id: object, value: object = "") -> str:
    """Infer the review policy from export metadata."""

    text_value = str(value).strip()
    if text_value and text_value.lower() != "nan":
        return text_value
    text_id = str(experiment_id)
    if text_id.endswith("target_eig"):
        return "target_eig"
    if text_id.endswith("bald"):
        return "bald"
    return ""


def parse_numeric_vector(value: object, separator: str = "|") -> np.ndarray | None:
    """Parse a separator-delimited vector written by the sweep exporter."""

    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).reshape(-1)
        return values if values.size else None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return np.asarray(
            [float(token) for token in text.split(separator)], dtype=float
        )
    except ValueError:
        return None


def rank_values(values: np.ndarray) -> np.ndarray:
    """Return average ranks for Spearman correlation."""

    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation without requiring SciPy."""

    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if finite.sum() < 2:
        return float("nan")
    left_rank = rank_values(left_values[finite])
    right_rank = rank_values(right_values[finite])
    left_sd = left_rank.std(ddof=0)
    right_sd = right_rank.std(ddof=0)
    if left_sd <= 0.0 or right_sd <= 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def site_priority_metrics(
    psi_mean: object, oracle_psi_mean: object
) -> dict[str, float]:
    """Compute site-level oracle agreement metrics."""

    psi_values = parse_numeric_vector(psi_mean)
    oracle_values = parse_numeric_vector(oracle_psi_mean)
    if psi_values is None or oracle_values is None:
        return {
            "psi_probability_agreement": float("nan"),
            "site_priority_spearman": float("nan"),
        }

    shared_size = min(len(psi_values), len(oracle_values))
    if shared_size == 0:
        return {
            "psi_probability_agreement": float("nan"),
            "site_priority_spearman": float("nan"),
        }

    psi_values = psi_values[:shared_size]
    oracle_values = oracle_values[:shared_size]
    finite = np.isfinite(psi_values) & np.isfinite(oracle_values)
    if not np.any(finite):
        return {
            "psi_probability_agreement": float("nan"),
            "site_priority_spearman": float("nan"),
        }

    psi_mae = float(np.mean(np.abs(psi_values[finite] - oracle_values[finite])))
    return {
        "psi_probability_agreement": float(np.clip(1.0 - psi_mae, 0.0, 1.0)),
        "site_priority_spearman": spearman_rank_correlation(
            psi_values[finite],
            oracle_values[finite],
        ),
    }


def credible_interval_state(q05: object, q95: object) -> str | None:
    """Return the sign conclusion implied by a 90% credible interval."""

    lower = safe_float(q05)
    upper = safe_float(q95)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower > 0.0:
        return "positive"
    if upper < 0.0:
        return "negative"
    return "uncertain"


def coefficient_agreement_metrics(row: pd.Series, group: str) -> dict[str, float]:
    """Compute coefficient-conclusion agreement for one parameter group."""

    prefix = "cov_state_" if group == "occupancy" else "cov_det_"
    parameter_names = sorted(
        {
            column.removeprefix("parameter_q05__")
            for column in row.index
            if str(column).startswith(f"parameter_q05__{prefix}")
        },
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    agreements: list[bool] = []
    for parameter_name in parameter_names:
        state = credible_interval_state(
            row.get(f"parameter_q05__{parameter_name}"),
            row.get(f"parameter_q95__{parameter_name}"),
        )
        oracle_state = credible_interval_state(
            row.get(f"oracle__parameter_q05__{parameter_name}"),
            row.get(f"oracle__parameter_q95__{parameter_name}"),
        )
        if state is None or oracle_state is None:
            continue
        agreements.append(state == oracle_state)

    return {
        f"coef_ci_conclusion_agreement__{group}": (
            float(np.mean(agreements)) if agreements else float("nan")
        )
    }


def read_comparison_rows(results_csv_path: Path) -> pd.DataFrame:
    """Read only the exported columns needed for this analysis."""

    all_columns = pd.read_csv(results_csv_path, nrows=0).columns
    parameter_columns = [
        column
        for column in all_columns
        if column.startswith(("parameter_q05__", "parameter_q95__"))
        or column.startswith(("oracle__parameter_q05__", "oracle__parameter_q95__"))
    ]
    use_columns = [
        "row_kind",
        "dataset_kind",
        "species",
        "experiment_id",
        "experiment__model_family",
        "experiment__selection_method",
        "experiment__covariate_variant",
        "trace__review_count",
        "trace__mean_r_hat",
        "trace__max_r_hat",
        "trace__num_diverging",
        "trace__frac_diverging",
        "trace__mcmc_wall_time_seconds",
        "trace__score_calibration_wall_time_seconds",
        "trace__selection_wall_time_seconds",
        "trace__step_wall_time_seconds",
        "experiment__total_mcmc_wall_time_seconds",
        "experiment__total_score_calibration_wall_time_seconds",
        "experiment__total_selection_wall_time_seconds",
        "experiment__experiment_wall_time_seconds",
        "dataset__n_reviewable",
        "trace__psi_mean_values",
        "oracle__psi_mean_values",
        *parameter_columns,
    ]
    frame = pd.read_csv(
        results_csv_path,
        usecols=[column for column in use_columns if column in all_columns],
        low_memory=False,
    )
    frame = frame.loc[
        frame["row_kind"].eq("step")
        & frame["experiment_id"].isin(SHARED_EXPERIMENT_IDS)
    ].copy()
    frame["dataset_kind"] = frame["dataset_kind"].astype(str).str.lower()
    frame["species"] = frame["species"].astype(str)
    frame["species_key"] = frame["dataset_kind"] + "::" + frame["species"]
    frame["model_family"] = [
        infer_model_family(experiment_id, model_family)
        for experiment_id, model_family in zip(
            frame["experiment_id"],
            frame.get("experiment__model_family", ""),
        )
    ]
    frame["selection_method"] = [
        infer_selection_method(experiment_id, selection_method)
        for experiment_id, selection_method in zip(
            frame["experiment_id"],
            frame.get("experiment__selection_method", ""),
        )
    ]
    frame["covariate_variant"] = (
        frame.get("experiment__covariate_variant", "full")
        .astype(str)
        .str.lower()
        .replace({"nan": "full", "": "full"})
    )
    frame = frame.loc[frame["covariate_variant"].ne("null")].copy()
    numeric_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "row_kind",
            "dataset_kind",
            "species",
            "species_key",
            "experiment_id",
            "experiment__model_family",
            "experiment__selection_method",
            "experiment__covariate_variant",
            "model_family",
            "selection_method",
            "covariate_variant",
            "trace__psi_mean_values",
            "oracle__psi_mean_values",
        }
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    complete_species = []
    required = set(SHARED_EXPERIMENT_IDS)
    for (dataset_kind, species_key), group in frame.groupby(
        ["dataset_kind", "species_key"]
    ):
        available = set(group["experiment_id"].dropna().astype(str).unique())
        if required.issubset(available):
            complete_species.append((dataset_kind, species_key))
    complete_index = pd.MultiIndex.from_tuples(
        complete_species,
        names=["dataset_kind", "species_key"],
    )
    row_index = pd.MultiIndex.from_frame(frame[["dataset_kind", "species_key"]])
    return frame.loc[row_index.isin(complete_index)].reset_index(drop=True)


def selected_species_keys_from_export(results_csv_path: Path) -> set[str]:
    """Return species keys after applying the manuscript species limits."""

    species_rows = pd.read_csv(
        results_csv_path,
        usecols=["row_kind", "species_index", "dataset_kind", "species"],
        low_memory=False,
    )
    species_rows = species_rows.loc[species_rows["row_kind"].eq("species")].copy()
    if species_rows.empty:
        return set()
    species_rows["dataset_kind"] = species_rows["dataset_kind"].astype(str).str.lower()
    species_rows["species"] = species_rows["species"].astype(str)
    species_rows["species_key"] = (
        species_rows["dataset_kind"] + "::" + species_rows["species"]
    )
    species_rows["species_index"] = pd.to_numeric(
        species_rows["species_index"],
        errors="coerce",
    )
    selected: set[str] = set()
    for dataset_kind, group in species_rows.sort_values("species_index").groupby(
        "dataset_kind",
        sort=False,
    ):
        limit = MAX_SPECIES_BY_DATASET.get(dataset_kind)
        if limit is None:
            selected.update(group["species_key"].astype(str))
        else:
            selected.update(group.head(int(limit))["species_key"].astype(str))
    return selected


def load_oracle_psi_mean(species_dir: Path) -> np.ndarray | None:
    """Load the oracle posterior mean for site-level ``psi`` from an artifact."""

    path = species_dir / "oracle" / "parameter_summary.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    parameter_names = [str(name) for name in data["parameter_names"]]
    if "psi" not in parameter_names:
        return None
    parameter_index = parameter_names.index("psi")
    key = f"parameter_{parameter_index:04d}__mean"
    if key not in data:
        return None
    return np.asarray(data[key], dtype=float).reshape(-1)


def load_oracle_psi_lookup(sweep_dir: Path) -> dict[str, np.ndarray]:
    """Load oracle site-level occupancy means for every available species."""

    lookup: dict[str, np.ndarray] = {}
    for species_dir in species_artifact_dirs(sweep_dir):
        scalars_path = species_dir / "dataset_metadata" / "scalars.csv"
        if not scalars_path.exists():
            continue
        scalars = pd.read_csv(scalars_path).iloc[0].to_dict()
        dataset_kind = str(scalars.get("dataset_kind", "")).lower()
        species = str(scalars.get("target_species", ""))
        values = load_oracle_psi_mean(species_dir)
        if values is not None:
            lookup[f"{dataset_kind}::{species}"] = values
    return lookup


def build_agreement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute row-level oracle agreement metrics."""

    rows: list[dict[str, object]] = []
    for _row_index, series in frame.iterrows():
        site_metrics = site_priority_metrics(
            series.get("trace__psi_mean_values"),
            series.get("oracle_psi_vector", series.get("oracle__psi_mean_values")),
        )
        coefficient_metrics = {
            **coefficient_agreement_metrics(series, "occupancy"),
            **coefficient_agreement_metrics(series, "detection"),
        }
        metric_values = {**site_metrics, **coefficient_metrics}
        rows.append(
            {
                "dataset_kind": series["dataset_kind"],
                "species": series["species"],
                "species_key": series["species_key"],
                "experiment_id": series["experiment_id"],
                "model_family": series["model_family"],
                "selection_method": series["selection_method"],
                "review_count": int(series["trace__review_count"]),
                **metric_values,
                "mean_agreement": float(
                    np.nanmean([metric_values[name] for name in AGREEMENT_METRICS])
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_agreement_by_budget(agreement_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean oracle agreement at fixed review budgets."""

    summary = (
        agreement_frame.loc[agreement_frame["review_count"].isin(AGREEMENT_BUDGETS)]
        .groupby(["dataset_kind", "model_family", "review_count"], as_index=False)
        .agg(
            n_species=("species_key", "nunique"),
            n_trajectories=("experiment_id", "count"),
            mean_agreement=("mean_agreement", "mean"),
            psi_probability_agreement=("psi_probability_agreement", "mean"),
            site_priority_spearman=("site_priority_spearman", "mean"),
            coef_ci_conclusion_agreement__occupancy=(
                "coef_ci_conclusion_agreement__occupancy",
                "mean",
            ),
            coef_ci_conclusion_agreement__detection=(
                "coef_ci_conclusion_agreement__detection",
                "mean",
            ),
        )
    )
    summary["dataset_label"] = summary["dataset_kind"].map(DATASET_LABELS)
    summary["model_label"] = summary["model_family"].map(MODEL_LABELS)
    return summary.sort_values(
        ["dataset_kind", "review_count", "model_family"]
    ).reset_index(drop=True)


def summarize_timing(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean trajectory wall times after de-duplicating step rows."""

    experiment_rows = frame.drop_duplicates(
        ["dataset_kind", "species_key", "experiment_id"]
    ).copy()
    summary = experiment_rows.groupby(
        ["dataset_kind", "model_family"], as_index=False
    ).agg(
        n_species=("species_key", "nunique"),
        n_trajectories=("experiment_id", "count"),
        mcmc_seconds=("experiment__total_mcmc_wall_time_seconds", "mean"),
        score_calibration_seconds=(
            "experiment__total_score_calibration_wall_time_seconds",
            "mean",
        ),
        selection_seconds=("experiment__total_selection_wall_time_seconds", "mean"),
        total_seconds=("experiment__experiment_wall_time_seconds", "mean"),
    )
    for column in [
        "mcmc_seconds",
        "score_calibration_seconds",
        "selection_seconds",
        "total_seconds",
    ]:
        summary[column.replace("_seconds", "_minutes")] = summary[column] / 60.0
    return summary


def summarize_convergence(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize sampler diagnostics for ecological MCMC fits."""

    diagnostic_frame = frame.copy()
    diagnostic_frame["problematic_fit"] = (
        diagnostic_frame["trace__max_r_hat"] > 1.05
    ) | (diagnostic_frame["trace__num_diverging"] > 0)
    summary = diagnostic_frame.groupby(
        ["dataset_kind", "model_family"], as_index=False
    ).agg(
        n_fits=("experiment_id", "count"),
        mean_r_hat=("trace__mean_r_hat", "mean"),
        median_max_r_hat=("trace__max_r_hat", "median"),
        p95_max_r_hat=(
            "trace__max_r_hat",
            lambda values: (
                float(np.nanquantile(values, 0.95))
                if np.isfinite(np.asarray(values, dtype=float)).any()
                else float("nan")
            ),
        ),
        divergence_fit_rate=(
            "trace__num_diverging",
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0)),
        ),
        mean_frac_diverging=("trace__frac_diverging", "mean"),
        problematic_fit_rate=("problematic_fit", "mean"),
    )
    return summary


def slugify(value: object) -> str:
    """Return the artifact-directory slug used by the sweep scripts."""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def species_artifact_dirs(sweep_dir: Path) -> list[Path]:
    """Return species artifact directories in stable order."""

    return sorted(
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and re.match(r"^\d{3}_.+?__.+", path.name)
    )


def squeeze_species_axis(
    values: np.ndarray, reference_ndim: int | None = None
) -> np.ndarray:
    """Remove the leading species axis saved for single-species arrays."""

    array = np.asarray(values)
    if array.ndim >= 1 and array.shape[0] == 1:
        if reference_ndim is None or array.ndim == reference_ndim + 1:
            return array[0]
    return array


def heterogeneity_rows(sweep_dir: Path) -> pd.DataFrame:
    """Measure score-distribution heterogeneity by species.

    For each species, compare scores from negative replicates at sites with at
    least one positive label against negative replicates at sites with no
    positive labels.
    """

    rows: list[dict[str, object]] = []
    for species_dir in species_artifact_dirs(sweep_dir):
        scalars_path = species_dir / "dataset_metadata" / "scalars.csv"
        obs_path = species_dir / "inputs" / "obs.npy"
        labels_path = species_dir / "inputs" / "true_labels.npy"
        reviewable_path = species_dir / "dataset_metadata" / "reviewable_mask.npy"
        if not (
            scalars_path.exists()
            and obs_path.exists()
            and labels_path.exists()
            and reviewable_path.exists()
        ):
            continue

        scalars = pd.read_csv(scalars_path).iloc[0].to_dict()
        dataset_kind = str(scalars.get("dataset_kind", "")).lower()
        species = str(scalars.get("target_species", ""))
        scores = np.asarray(np.load(obs_path), dtype=float)
        reviewable = np.asarray(np.load(reviewable_path), dtype=bool)
        labels = np.asarray(np.load(labels_path), dtype=float)
        scores = squeeze_species_axis(scores, reviewable.ndim)
        labels = squeeze_species_axis(labels, reviewable.ndim)
        if scores.shape != labels.shape or scores.shape != reviewable.shape:
            continue

        valid = reviewable & np.isfinite(scores) & np.isfinite(labels)
        positive = valid & (labels > 0.5)
        negative = valid & (labels <= 0.5)
        if valid.ndim < 1:
            continue
        site_axes = tuple(range(1, valid.ndim))
        site_occupied = positive.any(axis=site_axes)
        occupied_mask = np.broadcast_to(
            site_occupied.reshape((site_occupied.shape[0],) + (1,) * (valid.ndim - 1)),
            valid.shape,
        )
        negative_at_occupied = negative & occupied_mask
        negative_at_no_positive = negative & ~occupied_mask

        def mean_for(mask: np.ndarray) -> float:
            return float(np.nanmean(scores[mask])) if np.any(mask) else float("nan")

        true_positive_mean = mean_for(positive)
        negative_occupied_mean = mean_for(negative_at_occupied)
        negative_no_positive_mean = mean_for(negative_at_no_positive)
        rows.append(
            {
                "dataset_kind": dataset_kind,
                "dataset_label": DATASET_LABELS.get(dataset_kind, dataset_kind.title()),
                "species": species,
                "species_key": f"{dataset_kind}::{species}",
                "n_sites": int(valid.any(axis=site_axes).sum()),
                "n_positive_sites": int(site_occupied.sum()),
                "n_positive_replicates": int(positive.sum()),
                "n_negative_replicates_at_positive_sites": int(
                    negative_at_occupied.sum()
                ),
                "n_negative_replicates_at_no_positive_sites": int(
                    negative_at_no_positive.sum()
                ),
                "true_positive_score_mean": true_positive_mean,
                "negative_score_mean_at_positive_sites": negative_occupied_mean,
                "negative_score_mean_at_no_positive_sites": negative_no_positive_mean,
                "negative_score_heterogeneity_gap": (
                    negative_occupied_mean - negative_no_positive_mean
                    if np.isfinite(negative_occupied_mean)
                    and np.isfinite(negative_no_positive_mean)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_heterogeneity(heterogeneity_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize score heterogeneity by dataset."""

    frame = heterogeneity_frame.loc[
        np.isfinite(heterogeneity_frame["negative_score_heterogeneity_gap"])
    ].copy()
    summary = frame.groupby(["dataset_kind", "dataset_label"], as_index=False).agg(
        n_species=("species_key", "nunique"),
        mean_gap=("negative_score_heterogeneity_gap", "mean"),
        median_gap=("negative_score_heterogeneity_gap", "median"),
        species_with_positive_gap=(
            "negative_score_heterogeneity_gap",
            lambda values: int(np.sum(np.asarray(values, dtype=float) > 0.0)),
        ),
        positive_gap_fraction=(
            "negative_score_heterogeneity_gap",
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0)),
        ),
    )
    return summary


def heterogeneity_robustness_frames(
    agreement_frame: pd.DataFrame,
    heterogeneity_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize decoupled-minus-original agreement by heterogeneity level."""

    shared = agreement_frame.loc[
        agreement_frame["review_count"].isin(AGREEMENT_BUDGETS)
    ].copy()
    species_model = (
        shared.groupby(
            [
                "dataset_kind",
                "species_key",
                "species",
                "model_family",
                "review_count",
            ],
            as_index=False,
        )["mean_agreement"]
        .mean()
        .pivot_table(
            index=["dataset_kind", "species_key", "species", "review_count"],
            columns="model_family",
            values="mean_agreement",
        )
        .reset_index()
    )
    species_model["delta_decoupled_minus_original"] = (
        species_model["cut_normal_calibrated_cs"] - species_model["original_cs"]
    )
    species_model = species_model.merge(
        heterogeneity_frame[
            [
                "species_key",
                "negative_score_heterogeneity_gap",
                "negative_score_mean_at_positive_sites",
                "negative_score_mean_at_no_positive_sites",
                "true_positive_score_mean",
            ]
        ],
        on="species_key",
        how="left",
    )

    thresholds = (
        heterogeneity_frame.groupby("dataset_kind")["negative_score_heterogeneity_gap"]
        .quantile(0.75)
        .rename("top_quartile_threshold")
        .reset_index()
    )
    species_model = species_model.merge(thresholds, on="dataset_kind", how="left")
    species_model["heterogeneity_group"] = np.where(
        species_model["negative_score_heterogeneity_gap"]
        >= species_model["top_quartile_threshold"],
        "top quartile",
        "lower three quartiles",
    )

    grouped = (
        species_model.groupby(
            ["dataset_kind", "heterogeneity_group", "review_count"],
            as_index=False,
        )
        .agg(
            n_species=("species_key", "nunique"),
            mean_heterogeneity_gap=("negative_score_heterogeneity_gap", "mean"),
            original_agreement=("original_cs", "mean"),
            decoupled_agreement=("cut_normal_calibrated_cs", "mean"),
            delta_decoupled_minus_original=(
                "delta_decoupled_minus_original",
                "mean",
            ),
        )
        .sort_values(["dataset_kind", "heterogeneity_group", "review_count"])
    )

    examples = (
        species_model.loc[species_model["review_count"].isin([0, 100])]
        .sort_values(
            ["dataset_kind", "negative_score_heterogeneity_gap", "review_count"],
            ascending=[True, False, True],
        )
        .groupby(["dataset_kind", "review_count"], group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    return grouped, examples


def build_manuscript_table(
    timing_summary: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine key comparison values into one manuscript table."""

    table = timing_summary.merge(
        convergence_summary,
        on=["dataset_kind", "model_family"],
        how="left",
    )
    agreement_wide = agreement_summary.pivot_table(
        index=["dataset_kind", "model_family"],
        columns="review_count",
        values="mean_agreement",
        aggfunc="mean",
    ).reset_index()
    agreement_wide = agreement_wide.rename(
        columns={budget: f"agreement_{budget}" for budget in AGREEMENT_BUDGETS}
    )
    table = table.merge(agreement_wide, on=["dataset_kind", "model_family"], how="left")
    table["dataset_label"] = table["dataset_kind"].map(DATASET_LABELS)
    table["model_label"] = table["model_family"].map(MODEL_LABELS)
    table["model_order"] = table["model_family"].map(MODEL_ORDER)
    return table.sort_values(["dataset_kind", "model_order"]).reset_index(drop=True)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""

    text_frame = frame.copy()
    for column in text_frame.columns:
        if pd.api.types.is_float_dtype(text_frame[column]):
            text_frame[column] = text_frame[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
        else:
            text_frame[column] = text_frame[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    columns = [str(column) for column in text_frame.columns]
    rows = [[str(value) for value in record] for record in text_frame.to_numpy()]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]

    def render_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def write_markdown_summary(
    path: Path,
    manuscript_table: pd.DataFrame,
    heterogeneity_summary: pd.DataFrame,
    heterogeneity_examples: pd.DataFrame,
) -> None:
    """Write a human-readable analysis summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Continuous-score formulation comparison",
        "",
        "## Manuscript table",
        "",
        manuscript_table[
            [
                "dataset_label",
                "model_label",
                "n_species",
                "mcmc_minutes",
                "total_minutes",
                "mean_r_hat",
                "mean_frac_diverging",
                "agreement_0",
                "agreement_100",
                "agreement_500",
                "agreement_1000",
            ]
        ].pipe(dataframe_to_markdown),
        "",
        "## Score distribution heterogeneity",
        "",
        heterogeneity_summary.pipe(dataframe_to_markdown),
        "",
        "## Largest positive heterogeneity gaps",
        "",
        heterogeneity_examples[
            [
                "dataset_label",
                "species",
                "true_positive_score_mean",
                "negative_score_mean_at_positive_sites",
                "negative_score_mean_at_no_positive_sites",
                "negative_score_heterogeneity_gap",
            ]
        ].pipe(dataframe_to_markdown),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: SimpleNamespace) -> None:
    """Run the analysis and write outputs."""

    repo_root = find_repo_root()
    sweep_dir = (repo_root / args.sweep_dir).resolve()
    results_csv_path = sweep_dir / args.results_csv
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_species_keys = selected_species_keys_from_export(results_csv_path)
    frame = read_comparison_rows(results_csv_path)
    if selected_species_keys:
        frame = frame.loc[frame["species_key"].isin(selected_species_keys)].copy()
    oracle_psi_lookup = load_oracle_psi_lookup(sweep_dir)
    frame["oracle_psi_vector"] = frame["species_key"].map(oracle_psi_lookup)
    agreement_frame = build_agreement_frame(frame)
    agreement_summary = summarize_agreement_by_budget(agreement_frame)
    timing_summary = summarize_timing(frame)
    convergence_summary = summarize_convergence(frame)
    manuscript_table = build_manuscript_table(
        timing_summary,
        convergence_summary,
        agreement_summary,
    )
    heterogeneity_frame = heterogeneity_rows(sweep_dir)
    if selected_species_keys:
        heterogeneity_frame = heterogeneity_frame.loc[
            heterogeneity_frame["species_key"].isin(selected_species_keys)
        ].copy()
    heterogeneity_summary = summarize_heterogeneity(heterogeneity_frame)
    robustness_summary, robustness_examples = heterogeneity_robustness_frames(
        agreement_frame,
        heterogeneity_frame,
    )
    heterogeneity_examples = (
        heterogeneity_frame.sort_values(
            "negative_score_heterogeneity_gap",
            ascending=False,
            na_position="last",
        )
        .groupby("dataset_kind", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )

    agreement_frame.to_csv(output_dir / "cs_agreement_rows.csv", index=False)
    agreement_summary.to_csv(output_dir / "cs_agreement_by_budget.csv", index=False)
    timing_summary.to_csv(output_dir / "cs_timing_summary.csv", index=False)
    convergence_summary.to_csv(output_dir / "cs_convergence_summary.csv", index=False)
    manuscript_table.to_csv(output_dir / "cs_model_comparison_table.csv", index=False)
    heterogeneity_frame.to_csv(
        output_dir / "score_heterogeneity_by_species.csv", index=False
    )
    heterogeneity_summary.to_csv(
        output_dir / "score_heterogeneity_summary.csv", index=False
    )
    heterogeneity_examples.to_csv(
        output_dir / "score_heterogeneity_examples.csv", index=False
    )
    robustness_summary.to_csv(
        output_dir / "cs_heterogeneity_robustness_summary.csv",
        index=False,
    )
    robustness_examples.to_csv(
        output_dir / "cs_heterogeneity_robustness_examples.csv",
        index=False,
    )
    write_markdown_summary(
        output_dir / "cs_analysis_summary.md",
        manuscript_table,
        heterogeneity_summary,
        heterogeneity_examples,
    )

    print(f"Wrote analysis outputs to {output_dir}")
    print(
        manuscript_table[
            [
                "dataset_label",
                "model_label",
                "n_species",
                "mcmc_minutes",
                "total_minutes",
                "mean_r_hat",
                "mean_frac_diverging",
                "agreement_0",
                "agreement_100",
                "agreement_500",
                "agreement_1000",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


def main(
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
    results_csv: str = DEFAULT_RESULTS_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the continuous-score analysis and write outputs."""

    run(
        SimpleNamespace(
            sweep_dir=Path(sweep_dir),
            results_csv=results_csv,
            output_dir=Path(output_dir),
        )
    )
