import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from occubed.paths import DEFAULT_SWEEP_DIR, FIGURE_DIR

plt.style.use("seaborn-v0_8-whitegrid")
WIDTH_SCALE = 1.5
NEURIPS_TEXT_WIDTH_PT = 397.48499 * WIDTH_SCALE
TEX_POINTS_PER_INCH = 72.27
FULL_WIDTH_IN = NEURIPS_TEXT_WIDTH_PT / TEX_POINTS_PER_INCH
BASE_FONT_SIZE_PT = 10.0
PLOT_LINEWIDTH = 1.0
ASPECT_RATIO = 1 / 1.61803

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "font.size": BASE_FONT_SIZE_PT,
        "axes.labelsize": BASE_FONT_SIZE_PT,
        "axes.titlesize": BASE_FONT_SIZE_PT,
        "axes.labelpad": 2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "legend.fontsize": 8.5,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "lines.linewidth": PLOT_LINEWIDTH,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.alpha": 0.25,
    }
)


SWEEP_DIR = os.environ.get("OCCUBED_SWEEP_DIR", str(DEFAULT_SWEEP_DIR))
COMBINED_RESULTS_CSV = os.environ.get(
    "OCCUBED_STOPPING_RESULTS_CSV",
    os.environ.get("OCCUBED_ANALYSIS_RESULTS_CSV", "combined_species_results.csv"),
)
DATASET_FILTER = None
SAVE_FIGURES = os.environ.get("OCCUBED_ANALYSIS_WRITE_FIGURES", "1") != "0"
FIGURE_OUTPUT_DIR = os.environ.get(
    "OCCUBED_STOPPING_FIGURE_OUTPUT_DIR",
    str(FIGURE_DIR),
)
STOPPING_EXPERIMENT_IDS = ["cut_normal_calibrated_target_eig"]
STOPPING_MAX_WORKERS = min(4, os.cpu_count() or 1)
STOPPING_PARAMETER_SWEEP_POINTS = int(
    os.environ.get(
        "OCCUBED_STOPPING_PARAMETER_SWEEP_POINTS",
        "25",
    )
)

AGREEMENT_METRIC_COLUMNS = (
    "psi_probability_agreement",
    "site_priority_spearman",
    "coef_ci_conclusion_agreement__occupancy",
    "coef_ci_conclusion_agreement__detection",
)
DATASET_LABELS = {
    "acoustic": "Acoustic",
    "iwildcam": "Camera trap",
}
DATASET_ORDER = ("acoustic", "iwildcam")
STOPPING_TRADEOFF_WIDTH_FRACTION = 0.375
EXPECTED_GAIN_PATIENCE = 2
POSTERIOR_SHIFT_PATIENCE = 2

STOPPING_COLORS = {
    "mcmc_diagnostic": "#4C78A8",
    "expected_gain_small": "#B6992D",
    "posterior_shift_small": "#54A24B",
}


@dataclass(frozen=True)
class CriterionSpec:
    name: str
    label: str
    color: str
    parameter_name: str
    parameter_label: str
    parameter_values: tuple[float, ...]
    diagnostic_column: str


@dataclass(frozen=True)
class StoppingTrajectory:
    trace: pd.DataFrame
    step_frame: pd.DataFrame


CRITERION_SPECS = (
    CriterionSpec(
        name="mcmc_diagnostic",
        label="MCMC diagnostic",
        color=STOPPING_COLORS["mcmc_diagnostic"],
        parameter_name="mean_r_hat_max",
        parameter_label=r"Mean $\hat{R}$ max",
        parameter_values=(1.0005, 1.0010, 1.0020, 1.0050, 1.0100),
        diagnostic_column="mean_r_hat",
    ),
    CriterionSpec(
        name="expected_gain_small",
        label="Expected gain",
        color=STOPPING_COLORS["expected_gain_small"],
        parameter_name="expected_gain_ratio_max",
        parameter_label="Expected-gain ratio max",
        parameter_values=(0.01, 0.02, 0.05, 0.10, 0.20, 0.40),
        diagnostic_column="expected_gain_ratio",
    ),
    CriterionSpec(
        name="posterior_shift_small",
        label="Posterior shift",
        color=STOPPING_COLORS["posterior_shift_small"],
        parameter_name="posterior_shift_ratio_max",
        parameter_label="Posterior-shift ratio max",
        parameter_values=(0.02, 0.05, 0.10, 0.20, 0.40, 0.80),
        diagnostic_column="posterior_shift_ratio",
    ),
)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def resolve_sweep_dir(repo_root: Path, sweep_dir: str | Path) -> Path:
    candidate = Path(sweep_dir).expanduser()
    search_paths = [candidate, repo_root / candidate, Path.cwd() / candidate]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve sweep directory: {sweep_dir}")


def resolve_results_csv_path(sweep_dir: Path, results_csv: str | Path) -> Path:
    candidate = Path(results_csv).expanduser()
    search_paths = [candidate, sweep_dir / candidate, Path.cwd() / candidate]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve combined results CSV: {results_csv}")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_numeric_vector(value: object, separator: str = "|") -> np.ndarray | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    tokens = [token.strip() for token in text.split(separator)]
    try:
        return np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError:
        return None


def credible_interval_conclusion_from_bounds(q05: float, q95: float) -> str | None:
    if not np.isfinite(q05) or not np.isfinite(q95):
        return None
    if q05 > 0.0:
        return "positive"
    if q95 < 0.0:
        return "negative"
    return "uncertain"


def coefficient_group_name(parameter_name: str) -> str | None:
    if parameter_name.startswith("cov_state_"):
        return "occupancy"
    if parameter_name.startswith("cov_det_"):
        return "detection"
    return None


def coefficient_states_from_row(row: pd.Series, prefix: str) -> dict[str, str]:
    states: dict[str, str] = {}
    q05_prefix = f"{prefix}q05__"
    q95_prefix = f"{prefix}q95__"
    for key, q05 in row.items():
        key = str(key)
        if not key.startswith(q05_prefix):
            continue
        parameter_name = key[len(q05_prefix) :]
        if coefficient_group_name(parameter_name) is None:
            continue
        conclusion = credible_interval_conclusion_from_bounds(
            safe_float(q05),
            safe_float(row.get(f"{q95_prefix}{parameter_name}")),
        )
        if conclusion is not None:
            states[parameter_name] = conclusion
    return states


def compute_coefficient_conclusion_metrics_from_states(
    states: dict[str, str | None],
    oracle_states: dict[str, str | None],
) -> dict[str, float]:
    shared_parameters = sorted(
        parameter_name
        for parameter_name in set(states) & set(oracle_states)
        if states[parameter_name] is not None
        and oracle_states[parameter_name] is not None
    )

    metrics = {}
    for group_name in ["occupancy", "detection"]:
        group_parameters = [
            parameter_name
            for parameter_name in shared_parameters
            if coefficient_group_name(parameter_name) == group_name
        ]
        agreement_name = f"coef_ci_conclusion_agreement__{group_name}"
        if not group_parameters:
            metrics[agreement_name] = float("nan")
            continue
        metrics[agreement_name] = float(
            np.mean(
                [
                    states[parameter_name] == oracle_states[parameter_name]
                    for parameter_name in group_parameters
                ]
            )
        )
    return metrics


def spearman_rank_correlation(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
) -> float:
    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if finite.sum() < 2:
        return float("nan")
    return float(
        pd.Series(left_values[finite]).corr(
            pd.Series(right_values[finite]),
            method="spearman",
        )
    )


def compute_site_priority_metrics(
    psi_mean: np.ndarray,
    oracle_psi_mean: np.ndarray,
) -> dict[str, float]:
    psi_values = np.asarray(psi_mean, dtype=float).reshape(-1)
    oracle_values = np.asarray(oracle_psi_mean, dtype=float).reshape(-1)
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
    aligned_psi = np.where(finite, psi_values, np.nan)
    aligned_oracle = np.where(finite, oracle_values, np.nan)
    psi_mae = float(np.nanmean(np.abs(aligned_psi - aligned_oracle), dtype=float))
    return {
        "psi_probability_agreement": float(np.clip(1.0 - psi_mae, 0.0, 1.0)),
        "site_priority_spearman": spearman_rank_correlation(
            aligned_psi,
            aligned_oracle,
        ),
    }


def add_agreement_metrics(trace_frame: pd.DataFrame) -> pd.DataFrame:
    if trace_frame.empty:
        return trace_frame

    trace_frame = trace_frame.copy()
    rows: list[dict[str, float]] = []
    parsed_vector_cache: dict[str, np.ndarray | None] = {}

    def cached_parse_vector(value: object) -> np.ndarray | None:
        text = str(value).strip()
        if text not in parsed_vector_cache:
            parsed_vector_cache[text] = parse_numeric_vector(text)
        return parsed_vector_cache[text]

    for _, row in trace_frame.iterrows():
        metrics: dict[str, float] = {
            metric_column: float("nan") for metric_column in AGREEMENT_METRIC_COLUMNS
        }

        step_psi_mean = cached_parse_vector(row.get("psi_mean_values", ""))
        oracle_psi_mean = cached_parse_vector(row.get("oracle__psi_mean_values", ""))
        if step_psi_mean is not None and oracle_psi_mean is not None:
            metrics.update(
                compute_site_priority_metrics(step_psi_mean, oracle_psi_mean)
            )

        metrics.update(
            compute_coefficient_conclusion_metrics_from_states(
                coefficient_states_from_row(row, "parameter_"),
                coefficient_states_from_row(row, "oracle__parameter_"),
            )
        )
        rows.append(metrics)

    metric_frame = pd.DataFrame.from_records(rows, index=trace_frame.index)
    for metric_column in AGREEMENT_METRIC_COLUMNS:
        trace_frame[metric_column] = pd.to_numeric(
            metric_frame[metric_column],
            errors="coerce",
        )
    return trace_frame


def coalesce_text_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    default: str = "",
) -> pd.Series:
    result = pd.Series(default, index=frame.index, dtype=object)
    missing = pd.Series(True, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").fillna("").str.strip()
        valid = values.ne("") & values.str.lower().ne("nan")
        use_values = missing & valid
        result.loc[use_values] = values.loc[use_values].astype(object)
        missing &= ~use_values
    return result.astype(str)


def add_prefixed_aliases(
    frame: pd.DataFrame, prefixes: tuple[str, ...]
) -> pd.DataFrame:
    frame = frame.copy()
    for column in list(frame.columns):
        for prefix in prefixes:
            if not column.startswith(prefix):
                continue
            alias = column[len(prefix) :]
            if not alias:
                continue
            if alias not in frame.columns:
                frame[alias] = frame[column]
                continue
            missing = frame[alias].isna() | frame[alias].astype(str).str.strip().eq("")
            frame.loc[missing, alias] = frame.loc[missing, column]
    return frame


def infer_covariate_variant(experiment_id: object) -> str:
    text = str(experiment_id).strip().lower()
    return "null" if text.endswith("__null") else "full"


def normalize_exported_results_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_prefixed_aliases(results_frame, prefixes=("trace__",))
    frame["row_kind"] = coalesce_text_columns(frame, ["row_kind"])
    frame["dataset_kind"] = coalesce_text_columns(
        frame,
        ["dataset_kind", "run__dataset_kind", "dataset__dataset_kind"],
    ).str.lower()
    frame["species"] = coalesce_text_columns(
        frame,
        ["species", "run__target_species", "dataset__target_species"],
    )
    frame["species_key"] = frame["dataset_kind"] + "::" + frame["species"]
    frame["experiment_id"] = coalesce_text_columns(frame, ["experiment_id"])
    frame["covariate_variant"] = coalesce_text_columns(
        frame,
        ["experiment__covariate_variant", "experiment__oracle_variant"],
    )

    missing_covariates = frame["covariate_variant"].eq("") & frame["experiment_id"].ne(
        ""
    )
    frame.loc[missing_covariates, "covariate_variant"] = frame.loc[
        missing_covariates,
        "experiment_id",
    ].map(infer_covariate_variant)
    frame["covariate_variant"] = frame["covariate_variant"].map(
        lambda value: "null" if "null" in str(value).lower() else str(value).lower()
    )

    needed_numeric_columns = [
        "step_index",
        "review_count",
        "mean_oracle_distance",
        "mean_r_hat",
        "expected_gain_score_max",
        "posterior_shift_mean_distance",
    ]
    for column in needed_numeric_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "psi_mean_values" not in frame.columns:
        frame["psi_mean_values"] = ""
    return frame


def read_stopping_trace_export(
    results_csv_path: Path,
    *,
    dataset_filter: list[str] | None = None,
    experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    desired_columns = [
        "row_kind",
        "species_index",
        "dataset_kind",
        "run__dataset_kind",
        "dataset__dataset_kind",
        "species",
        "run__target_species",
        "dataset__target_species",
        "experiment_id",
        "experiment__covariate_variant",
        "experiment__oracle_variant",
        "step_index",
        "trace__step_index",
        "trace__review_count",
        "trace__mean_oracle_distance",
        "trace__mean_r_hat",
        "trace__expected_gain_score_max",
        "trace__posterior_shift_mean_distance",
        "trace__psi_mean_values",
        "oracle__psi_mean_values",
    ]
    available_columns = pd.read_csv(results_csv_path, nrows=0).columns.tolist()
    desired_prefixes = (
        "parameter_q05__cov_state_",
        "parameter_q95__cov_state_",
        "parameter_q05__cov_det_",
        "parameter_q95__cov_det_",
        "oracle__parameter_q05__cov_state_",
        "oracle__parameter_q95__cov_state_",
        "oracle__parameter_q05__cov_det_",
        "oracle__parameter_q95__cov_det_",
    )
    desired_columns.extend(
        column
        for column in available_columns
        if column.startswith(desired_prefixes) and column not in desired_columns
    )
    usecols = [column for column in desired_columns if column in available_columns]
    frame = pd.read_csv(results_csv_path, usecols=usecols, low_memory=False)
    frame = normalize_exported_results_frame(frame)

    if dataset_filter:
        frame = frame.loc[frame["dataset_kind"].isin(dataset_filter)].copy()
    if experiment_ids:
        frame = frame.loc[frame["experiment_id"].isin(experiment_ids)].copy()

    trace_frame = frame.loc[frame["row_kind"].eq("step")].copy()
    trace_frame = trace_frame.sort_values(
        ["dataset_kind", "species", "experiment_id", "step_index"],
        na_position="last",
    ).reset_index(drop=True)
    return add_agreement_metrics(trace_frame)


def build_step_diagnostics(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    initial_expected_gain = float("nan")
    initial_posterior_shift = float("nan")

    for _, row in trace.iterrows():
        expected_gain_score_max = safe_float(row.get("expected_gain_score_max"))
        expected_gain_ratio = float("nan")
        if np.isfinite(expected_gain_score_max):
            if not np.isfinite(initial_expected_gain):
                initial_expected_gain = expected_gain_score_max
            elif initial_expected_gain > 0.0:
                expected_gain_ratio = float(
                    expected_gain_score_max / initial_expected_gain
                )

        posterior_shift_mean_distance = safe_float(
            row.get("posterior_shift_mean_distance")
        )
        posterior_shift_ratio = float("nan")
        if np.isfinite(posterior_shift_mean_distance):
            if not np.isfinite(initial_posterior_shift):
                initial_posterior_shift = posterior_shift_mean_distance
            elif initial_posterior_shift > 0.0:
                posterior_shift_ratio = float(
                    posterior_shift_mean_distance / initial_posterior_shift
                )

        rows.append(
            {
                "step_index": int(row["step_index"]),
                "review_count": safe_float(row.get("review_count")),
                "mean_oracle_distance": safe_float(row.get("mean_oracle_distance")),
                "mean_r_hat": safe_float(row.get("mean_r_hat")),
                "expected_gain_ratio": expected_gain_ratio,
                "posterior_shift_ratio": posterior_shift_ratio,
            }
        )

    return pd.DataFrame.from_records(rows)


def prepare_stopping_trajectory(trace: pd.DataFrame) -> StoppingTrajectory:
    trace = trace.sort_values("step_index").reset_index(drop=True)
    return StoppingTrajectory(trace=trace, step_frame=build_step_diagnostics(trace))


def data_driven_parameter_values(
    values: np.ndarray,
    fallback_values: tuple[float, ...],
    *,
    point_count: int,
) -> tuple[float, ...]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return fallback_values

    point_count = max(3, int(point_count))
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    inner_count = max(1, point_count - 2)
    quantile_values = np.quantile(
        finite_values,
        np.linspace(0.0, 1.0, inner_count),
    )
    candidates = np.concatenate(
        [
            np.asarray([np.nextafter(min_value, -np.inf)]),
            np.asarray(quantile_values, dtype=float),
            np.asarray([np.nextafter(max_value, np.inf)]),
        ]
    )
    candidates = np.unique(candidates[np.isfinite(candidates)])
    return tuple(float(value) for value in candidates)


def build_criterion_parameter_values(
    trajectories: list[StoppingTrajectory],
) -> dict[str, tuple[float, ...]]:
    parameter_values: dict[str, tuple[float, ...]] = {}
    for spec in CRITERION_SPECS:
        diagnostic_values = []
        for trajectory in trajectories:
            if spec.diagnostic_column not in trajectory.step_frame.columns:
                continue
            values = trajectory.step_frame[spec.diagnostic_column].to_numpy(dtype=float)
            diagnostic_values.append(values)

        combined_values = (
            np.concatenate(diagnostic_values)
            if diagnostic_values
            else np.asarray([], dtype=float)
        )
        parameter_values[spec.name] = data_driven_parameter_values(
            combined_values,
            spec.parameter_values,
            point_count=STOPPING_PARAMETER_SWEEP_POINTS,
        )
    return parameter_values


def find_first_streak(
    values: np.ndarray,
    *,
    predicate,
    patience: int,
) -> int | None:
    streak = 0
    for index, value in enumerate(values):
        if predicate(value):
            streak += 1
        else:
            streak = 0
        if streak >= patience:
            return index
    return None


def evaluate_criterion(
    step_frame: pd.DataFrame,
    criterion_name: str,
    parameter_value: float,
) -> tuple[int | None, bool]:
    if criterion_name == "mcmc_diagnostic":
        values = step_frame["mean_r_hat"].to_numpy(dtype=float)
        available = bool(np.isfinite(values).any())
        if not available:
            return None, False
        indices = np.flatnonzero(np.isfinite(values) & (values <= parameter_value))
        return (int(indices[0]) if indices.size else None), True

    if criterion_name == "expected_gain_small":
        values = step_frame["expected_gain_ratio"].to_numpy(dtype=float)
        available = bool(np.isfinite(values).any())
        if not available:
            return None, False
        return (
            find_first_streak(
                values,
                predicate=lambda value: np.isfinite(value) and value <= parameter_value,
                patience=EXPECTED_GAIN_PATIENCE,
            ),
            True,
        )

    if criterion_name == "posterior_shift_small":
        values = step_frame["posterior_shift_ratio"].to_numpy(dtype=float)
        available = bool(np.isfinite(values).any())
        if not available:
            return None, False
        return (
            find_first_streak(
                values,
                predicate=lambda value: np.isfinite(value) and value <= parameter_value,
                patience=POSTERIOR_SHIFT_PATIENCE,
            ),
            True,
        )

    raise KeyError(f"Unknown stopping criterion: {criterion_name}")


def best_agreement_by_metric(trace: pd.DataFrame) -> dict[str, float]:
    best_values: dict[str, float] = {}
    for metric_column in AGREEMENT_METRIC_COLUMNS:
        if metric_column not in trace.columns:
            best_values[metric_column] = float("nan")
            continue
        values = trace[metric_column].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        best_values[metric_column] = (
            float(np.max(finite_values)) if finite_values.size else float("nan")
        )
    return best_values


def row_mean_agreement(row: pd.Series) -> float:
    values = [
        safe_float(row.get(metric_column)) for metric_column in AGREEMENT_METRIC_COLUMNS
    ]
    finite_values = np.asarray(
        [value for value in values if np.isfinite(value)],
        dtype=float,
    )
    return float(np.mean(finite_values)) if finite_values.size else float("nan")


def row_agreement_regret(
    row: pd.Series,
    best_values: dict[str, float],
) -> float:
    regrets = []
    for metric_column, best_value in best_values.items():
        current_value = safe_float(row.get(metric_column))
        if np.isfinite(best_value) and np.isfinite(current_value):
            regrets.append(max(0.0, float(best_value - current_value)))
    return float(np.mean(regrets)) if regrets else float("nan")


def mean_finite(values: list[float]) -> float:
    finite_values = np.asarray(
        [value for value in values if np.isfinite(value)],
        dtype=float,
    )
    return float(np.mean(finite_values)) if finite_values.size else float("nan")


def evaluate_single_experiment_stopping_sweeps(
    trajectory: StoppingTrajectory,
    parameter_values_by_criterion: dict[str, tuple[float, ...]],
) -> list[dict[str, object]]:
    trace = trajectory.trace
    if trace.empty:
        return []

    step_frame = trajectory.step_frame
    best_row = trace.loc[trace["mean_oracle_distance"].astype(float).idxmin()]
    final_row = trace.iloc[-1]
    first_row = trace.iloc[0]
    best_agreements = best_agreement_by_metric(trace)

    records: list[dict[str, object]] = []
    for spec in CRITERION_SPECS:
        parameter_values = parameter_values_by_criterion.get(
            spec.name,
            spec.parameter_values,
        )
        for parameter_order, parameter_value in enumerate(parameter_values):
            trigger_index, available = evaluate_criterion(
                step_frame,
                spec.name,
                parameter_value,
            )
            if available:
                if trigger_index is None:
                    stop_row = final_row
                    triggered = False
                else:
                    stop_row = trace.iloc[int(trigger_index)]
                    triggered = True

                stop_review_count = float(stop_row["review_count"])
                stop_mean_oracle_distance = float(stop_row["mean_oracle_distance"])
                final_review_count = float(final_row["review_count"])
                budget_fraction_used = (
                    float(stop_review_count / final_review_count)
                    if final_review_count
                    else float("nan")
                )
                review_savings = float(final_review_count - stop_review_count)
                oracle_regret_vs_best = float(
                    stop_mean_oracle_distance - float(best_row["mean_oracle_distance"])
                )
                agreement_regret_vs_best = row_agreement_regret(
                    stop_row,
                    best_agreements,
                )
                oracle_gap_vs_final = float(
                    stop_mean_oracle_distance - float(final_row["mean_oracle_distance"])
                )
                stop_step_index = float(stop_row["step_index"])
                stop_mean_agreement = row_mean_agreement(stop_row)
            else:
                triggered = False
                final_review_count = float(final_row["review_count"])
                stop_review_count = float("nan")
                stop_mean_oracle_distance = float("nan")
                budget_fraction_used = float("nan")
                review_savings = float("nan")
                oracle_regret_vs_best = float("nan")
                agreement_regret_vs_best = float("nan")
                oracle_gap_vs_final = float("nan")
                stop_step_index = float("nan")
                stop_mean_agreement = float("nan")

            records.append(
                {
                    "dataset_kind": first_row["dataset_kind"],
                    "species": first_row["species"],
                    "experiment_id": first_row["experiment_id"],
                    "criterion": spec.name,
                    "criterion_label": spec.label,
                    "parameter_name": spec.parameter_name,
                    "parameter_label": spec.parameter_label,
                    "parameter_value": float(parameter_value),
                    "parameter_order": int(parameter_order),
                    "available": available,
                    "triggered": triggered,
                    "stop_step_index": stop_step_index,
                    "stop_review_count": stop_review_count,
                    "final_review_count": final_review_count,
                    "budget_fraction_used": budget_fraction_used,
                    "review_savings": review_savings,
                    "stop_mean_agreement": stop_mean_agreement,
                    "final_mean_agreement": row_mean_agreement(final_row),
                    "best_mean_agreement": mean_finite(list(best_agreements.values())),
                    "stop_mean_oracle_distance": stop_mean_oracle_distance,
                    "final_mean_oracle_distance": float(
                        final_row["mean_oracle_distance"]
                    ),
                    "best_mean_oracle_distance": float(
                        best_row["mean_oracle_distance"]
                    ),
                    "oracle_regret_vs_best": oracle_regret_vs_best,
                    "agreement_regret_vs_best": agreement_regret_vs_best,
                    "oracle_gap_vs_final": oracle_gap_vs_final,
                }
            )
    return records


def evaluate_stopping_sweeps(
    trace_frame: pd.DataFrame,
    *,
    experiment_ids: list[str],
    max_workers: int,
) -> pd.DataFrame:
    subset = trace_frame.loc[
        trace_frame["covariate_variant"].eq("full")
        & trace_frame["experiment_id"].isin(experiment_ids)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    trace_groups = [
        group
        for _, group in subset.groupby(
            ["species_key", "experiment_id"],
            dropna=False,
            sort=True,
        )
    ]
    trajectories: list[StoppingTrajectory] = []
    if max_workers <= 1:
        for index, trace in enumerate(trace_groups, start=1):
            trajectories.append(prepare_stopping_trajectory(trace))
            if index % 10 == 0 or index == len(trace_groups):
                print(
                    "Prepared stopping diagnostics for "
                    f"{index} / {len(trace_groups)} trajectories"
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(prepare_stopping_trajectory, trace)
                for trace in trace_groups
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                trajectories.append(future.result())
                if index % 10 == 0 or index == len(trace_groups):
                    print(
                        "Prepared stopping diagnostics for "
                        f"{index} / {len(trace_groups)} trajectories"
                    )

    parameter_values_by_criterion = build_criterion_parameter_values(trajectories)
    print(
        "Using data-driven stopping threshold grids with up to "
        f"{STOPPING_PARAMETER_SWEEP_POINTS} values per criterion."
    )

    for index, trajectory in enumerate(trajectories, start=1):
        records.extend(
            evaluate_single_experiment_stopping_sweeps(
                trajectory,
                parameter_values_by_criterion,
            )
        )
        if index % 10 == 0 or index == len(trajectories):
            print(
                f"Evaluated stopping sweeps for {index} / {len(trajectories)} trajectories"
            )
    return pd.DataFrame.from_records(records)


def ci95(std: float, count: int) -> float:
    if count <= 1 or not np.isfinite(std):
        return 0.0
    return float(1.96 * std / np.sqrt(count))


def summarize_stopping_sweeps(stopping_frame: pd.DataFrame) -> pd.DataFrame:
    if stopping_frame.empty:
        return pd.DataFrame()

    summary = (
        stopping_frame.groupby(
            [
                "dataset_kind",
                "criterion",
                "criterion_label",
                "parameter_name",
                "parameter_label",
                "parameter_value",
                "parameter_order",
            ]
        )
        .agg(
            available_rate=("available", "mean"),
            trigger_rate=("triggered", "mean"),
            species_count=(
                "budget_fraction_used",
                lambda values: int(np.isfinite(values).sum()),
            ),
            agreement_species_count=(
                "agreement_regret_vs_best",
                lambda values: int(np.isfinite(values).sum()),
            ),
            mean_budget_fraction=("budget_fraction_used", "mean"),
            std_budget_fraction=("budget_fraction_used", "std"),
            mean_agreement_regret=("agreement_regret_vs_best", "mean"),
            std_agreement_regret=("agreement_regret_vs_best", "std"),
            mean_oracle_regret=("oracle_regret_vs_best", "mean"),
            std_oracle_regret=("oracle_regret_vs_best", "std"),
            mean_review_savings=("review_savings", "mean"),
        )
        .reset_index()
        .sort_values(["dataset_kind", "criterion", "parameter_order"])
    )
    summary["budget_fraction_ci95"] = summary.apply(
        lambda row: ci95(row["std_budget_fraction"], int(row["species_count"])),
        axis=1,
    )
    summary["agreement_regret_ci95"] = summary.apply(
        lambda row: ci95(
            row["std_agreement_regret"], int(row["agreement_species_count"])
        ),
        axis=1,
    )
    summary["oracle_regret_ci95"] = summary.apply(
        lambda row: ci95(row["std_oracle_regret"], int(row["species_count"])),
        axis=1,
    )
    return summary


def maybe_save_figure(fig, output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(filename).with_suffix(".pdf")
    fig.patch.set_facecolor("none")
    for axis in fig.axes:
        axis.set_facecolor("none")
    fig.savefig(
        path,
        bbox_inches=fig.bbox_inches,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    return path


def full_width_grid_size(
    n_cols: int,
    n_rows: int,
    *,
    row_scale: float = 1.35,
) -> tuple[float, float]:
    return FULL_WIDTH_IN, FULL_WIDTH_IN * row_scale * n_rows / max(n_cols, 1)


def apply_tight_layout(
    fig,
    *,
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    pad: float = 0.02,
    w_pad: float = 0.1,
    h_pad: float = 0.10,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    fig.tight_layout(rect=rect, pad=pad, w_pad=w_pad, h_pad=h_pad)
    if wspace is not None or hspace is not None:
        fig.subplots_adjust(
            wspace=fig.subplotpars.wspace if wspace is None else wspace,
            hspace=fig.subplotpars.hspace if hspace is None else hspace,
        )


def plot_stopping_tradeoff_sweeps(
    summary_frame: pd.DataFrame,
    *,
    output_dir: Path | None,
    filename: str,
) -> None:
    if summary_frame.empty:
        print("No stopping sweep data available.")
        return

    datasets = sorted(
        summary_frame["dataset_kind"].unique(),
        key=lambda value: (
            (
                DATASET_ORDER.index(value)
                if value in DATASET_ORDER
                else len(DATASET_ORDER)
            ),
            str(value),
        ),
    )
    figure_width = FULL_WIDTH_IN * STOPPING_TRADEOFF_WIDTH_FRACTION
    legend_height = 0.38
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(
            figure_width,
            figure_width * len(datasets) * ASPECT_RATIO + legend_height,
        ),
        # sharey=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for axis, dataset_kind in zip(axes, datasets):
        dataset_frame = summary_frame.loc[summary_frame["dataset_kind"] == dataset_kind]
        for spec in CRITERION_SPECS:
            criterion_frame = dataset_frame.loc[
                (dataset_frame["criterion"] == spec.name)
                & np.isfinite(dataset_frame["mean_budget_fraction"])
                & np.isfinite(dataset_frame["mean_agreement_regret"])
            ].sort_values(["mean_budget_fraction", "parameter_order"])
            if criterion_frame.empty:
                continue

            x_values = criterion_frame["mean_budget_fraction"].to_numpy(dtype=float)
            y_values = criterion_frame["mean_agreement_regret"].to_numpy(dtype=float)
            x_ci = criterion_frame["budget_fraction_ci95"].to_numpy(dtype=float)
            y_ci = criterion_frame["agreement_regret_ci95"].to_numpy(dtype=float)

            # axis.errorbar(
            #     x_values,
            #     y_values,
            #     xerr=x_ci,
            #     yerr=y_ci,
            #     fmt="none",
            #     color=spec.color,
            #     elinewidth=0.7,
            #     alpha=0.18,
            #     capsize=0,
            #     zorder=1,
            # )
            axis.plot(
                x_values,
                y_values,
                color=spec.color,
                linewidth=PLOT_LINEWIDTH,
                alpha=0.95,
                zorder=2,
            )
            # axis.scatter(
            #     x_values,
            #     y_values,
            #     s=12.0,
            #     color=spec.color,
            #     alpha=0.65,
            #     zorder=3,
            # )

        dataset_label = DATASET_LABELS.get(dataset_kind, dataset_kind.title())
        axis.set_box_aspect(ASPECT_RATIO)
        axis.set_xlabel("$t/t_{\\max}$", labelpad=1.0)
        axis.set_ylabel(
            f"{dataset_label} $\\bar{{A}}^{{\\star}} - \\bar{{A}}^{{\\mathrm{{stop}}}}$"
        )
        axis.set_xlim(-0.02, 1.02)
        axis.margins(x=0.01, y=0.04)

    legend_handles = [
        Line2D([0], [0], color=spec.color, linewidth=PLOT_LINEWIDTH, label=spec.label)
        for spec in CRITERION_SPECS
        if spec.name in set(summary_frame["criterion"])
    ]
    fig.legend(
        handles=legend_handles,
        ncol=min(2, len(legend_handles)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.7,
        handletextpad=0.45,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.08, 1.0, 1.0),
        pad=0.01,
        w_pad=0.02,
        h_pad=0.02,
        hspace=0.18,
    )
    saved_path = maybe_save_figure(fig, output_dir, filename)
    plt.show(block=False)
    plt.close(fig)
    if saved_path is not None:
        print(f"Saved {saved_path}")
    print(
        "\nStopping-tradeoff sweep plot:"
        " each line sweeps one criterion's main threshold;"
        " regret is averaged across the four agreement metrics;"
        " lower-left curves identify earlier stopping with less lost agreement."
    )


def main() -> None:
    repo_root = find_repo_root()
    dataset_filter = (
        None
        if DATASET_FILTER is None
        else [str(value).strip().lower() for value in np.atleast_1d(DATASET_FILTER)]
    )
    sweep_dir = resolve_sweep_dir(repo_root, SWEEP_DIR)
    combined_results_path = resolve_results_csv_path(sweep_dir, COMBINED_RESULTS_CSV)
    figure_output_dir = (repo_root / FIGURE_OUTPUT_DIR) if SAVE_FIGURES else None

    trace_frame = read_stopping_trace_export(
        combined_results_path,
        dataset_filter=dataset_filter,
        experiment_ids=STOPPING_EXPERIMENT_IDS,
    )
    print(f"Using sweep directory: {sweep_dir}")
    print(f"Using combined results CSV: {combined_results_path}")
    print(f"Saving figures: {SAVE_FIGURES}")
    print("Stopping experiments: " + ", ".join(STOPPING_EXPERIMENT_IDS))
    if figure_output_dir is not None:
        print(f"Figure output directory: {figure_output_dir}")

    stopping_frame = evaluate_stopping_sweeps(
        trace_frame,
        experiment_ids=STOPPING_EXPERIMENT_IDS,
        max_workers=STOPPING_MAX_WORKERS,
    )
    summary_frame = summarize_stopping_sweeps(stopping_frame)
    if summary_frame.empty:
        print("No stopping sweep summaries were computed.")
        return

    plot_stopping_tradeoff_sweeps(
        summary_frame,
        output_dir=figure_output_dir,
        filename="budget_regret_tradeoff_sweep",
    )
