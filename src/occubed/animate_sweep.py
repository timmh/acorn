import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

try:
    import contextily as cx
    from pyproj import Transformer
except ImportError:  # pragma: no cover - validated at map-render time.
    cx = None
    Transformer = None

from types import SimpleNamespace

from occubed.paths import DEFAULT_SWEEP_DIR
from occubed.summaries import iter_parameter_summary_arrays

Z90 = 1.6448536269514722
CURRENT_COLOR = "#1f77b4"
ORACLE_COLOR = "#d62728"
SELECTED_COLOR = "#111111"
SELECTED_BATCH_COLOR = "#ffbf47"
LEGEND_STYLE = dict(
    fontsize=7,
    frameon=True,
    framealpha=0.92,
    facecolor="white",
    edgecolor="0.65",
    fancybox=True,
    borderpad=0.4,
    labelspacing=0.35,
    handlelength=2.2,
)


@dataclass(frozen=True)
class ParameterStats:
    """Marginal posterior summaries for one scalar or array parameter."""

    mean: np.ndarray
    q05: np.ndarray | None = None
    q95: np.ndarray | None = None
    std: np.ndarray | None = None


@dataclass(frozen=True)
class ScoreCalibration:
    """Class-conditional Normal score calibration."""

    mu0: float
    sigma0: float
    mu1: float
    sigma1: float
    n_negative: int
    n_positive: int
    source: str


@dataclass(frozen=True)
class SelectedSample:
    """One selected replicate used as the primary annotation in a frame."""

    site_index: int
    period_index: int
    replicate_index: int
    score: float
    site_id: str
    replicate_id: str
    replicate_time: str
    batch_size: int


@dataclass(frozen=True)
class ArtifactContext:
    """Static data shared by all frames for one species and experiment."""

    sweep_dir: Path
    dataset_dir: Path
    experiment_dir: Path
    oracle_dir: Path
    output_dir: Path
    frame_dir: Path
    dataset_name: str
    target_label: str
    display_name: str
    model_label: str
    selection_method: str
    site_ids: np.ndarray
    replicate_ids: np.ndarray
    replicate_times: np.ndarray
    site_coordinates: np.ndarray
    site_covariate_names: list[str]
    obs_covariate_names: list[str]
    site_covariates_raw: np.ndarray
    obs_covariates_raw: np.ndarray
    site_covariate_offsets: np.ndarray
    site_covariate_scales: np.ndarray
    obs_covariate_offsets: np.ndarray
    obs_covariate_scales: np.ndarray
    score_values: np.ndarray
    true_labels: np.ndarray
    oracle_summary: dict[str, ParameterStats]
    oracle_score_calibration: ScoreCalibration
    notes: list[str]


def slugify(value: object) -> str:
    """Return a filesystem-safe slug."""

    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "item"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact into a list of dictionaries."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_single_csv_row(path: Path) -> dict[str, str]:
    """Read the first row from a CSV artifact."""

    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def load_npy(path: Path, *, allow_pickle: bool = True) -> np.ndarray:
    """Load a NumPy array and raise an artifact-specific error if absent."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def load_scalar_npy(path: Path) -> str:
    """Load a scalar/string `.npy` artifact as text."""

    value = load_npy(path)
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def load_parameter_summary(path: Path) -> dict[str, ParameterStats]:
    """Load compact or legacy parameter summaries."""

    summary: dict[str, ParameterStats] = {}
    for parameter_name, stats in iter_parameter_summary_arrays(
        path,
        stats=("mean", "q05", "q95", "std"),
    ):
        summary[str(parameter_name)] = ParameterStats(
            mean=np.asarray(stats["mean"], dtype=float),
            q05=(None if "q05" not in stats else np.asarray(stats["q05"], dtype=float)),
            q95=(None if "q95" not in stats else np.asarray(stats["q95"], dtype=float)),
            std=(None if "std" not in stats else np.asarray(stats["std"], dtype=float)),
        )
    if not summary:
        raise ValueError(f"No parameter summaries could be read from {path}")
    return summary


def finite_scalar(values: np.ndarray | None, default: float = np.nan) -> float:
    """Return the first finite scalar from an array-like value."""

    if values is None:
        return default
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite[0]) if finite.size else default


def marginal_sd(stats: ParameterStats) -> np.ndarray:
    """Approximate marginal standard deviations from 90% credible intervals."""

    if stats.q05 is not None and stats.q95 is not None:
        sd = (
            np.asarray(stats.q95, dtype=float) - np.asarray(stats.q05, dtype=float)
        ) / (2.0 * Z90)
    elif stats.std is not None:
        sd = np.asarray(stats.std, dtype=float)
    else:
        sd = np.full_like(np.asarray(stats.mean, dtype=float), np.nan, dtype=float)
    return np.where(np.isfinite(sd) & (sd > 0.0), sd, np.nan)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform for arrays."""

    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def normal_pdf(x_values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Evaluate a Normal density."""

    sigma = max(float(sigma), 1e-6)
    z_values = (x_values - float(mean)) / sigma
    return np.exp(-0.5 * z_values**2) / (sigma * math.sqrt(2.0 * math.pi))


def add_readable_legend(
    axis: plt.Axes,
    *,
    loc: str,
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    """Add a fixed-location legend with an opaque background."""

    legend_kwargs = dict(LEGEND_STYLE)
    legend_kwargs["ncol"] = ncol
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = axis.legend(loc=loc, **legend_kwargs)
    if legend is not None:
        legend.set_zorder(20)


def stable_legend_location_from_oracle(
    oracle_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> str:
    """Choose a fixed legend corner from the oracle curve shape.

    The oracle curve is constant across animation frames, so this avoids legends
    jumping as the current posterior changes.
    """

    if oracle_curve is None:
        return "upper right"
    oracle_mean = np.asarray(oracle_curve[0], dtype=float)
    if oracle_mean.size == 0 or not np.isfinite(oracle_mean).any():
        return "upper right"

    window = max(1, oracle_mean.size // 4)
    segments = {
        "upper left": oracle_mean[:window],
        "lower left": oracle_mean[:window],
        "upper right": oracle_mean[-window:],
        "lower right": oracle_mean[-window:],
    }
    costs = {}
    for location, segment in segments.items():
        finite_segment = segment[np.isfinite(segment)]
        if finite_segment.size == 0:
            costs[location] = np.inf
            continue
        segment_mean = float(finite_segment.mean())
        costs[location] = (
            segment_mean if location.startswith("upper") else 1.0 - segment_mean
        )
    return min(costs, key=costs.get)


def find_dataset_dir(
    sweep_dir: Path,
    species: str | None,
    dataset_kind: str | None,
) -> Path:
    """Find the species dataset directory in a sweep artifact."""

    candidates = []
    for dataset_dir in sorted(sweep_dir.iterdir()):
        metadata_dir = dataset_dir / "dataset_metadata"
        scalars_path = metadata_dir / "scalars.csv"
        if not dataset_dir.is_dir() or not scalars_path.exists():
            continue
        row = read_single_csv_row(scalars_path)
        row_kind = str(row.get("dataset_kind", "")).lower()
        if dataset_kind and row_kind != dataset_kind.lower():
            continue
        target_species = str(row.get("target_species", ""))
        target_label = str(row.get("target_label", ""))
        haystack = {
            dataset_dir.name.lower(),
            target_species.lower(),
            target_label.lower(),
            slugify(target_species),
            slugify(target_label),
        }
        if (
            species is None
            or species.lower() in haystack
            or slugify(species) in haystack
        ):
            candidates.append(dataset_dir)

    if not candidates:
        raise ValueError(
            "No matching dataset directory found for "
            f"species={species!r}, dataset_kind={dataset_kind!r} under {sweep_dir}"
        )
    if len(candidates) > 1:
        labels = ", ".join(path.name for path in candidates[:12])
        raise ValueError(
            "Species selection is ambiguous; pass --dataset-kind or a more exact "
            f"--species value. Matches: {labels}"
        )
    return candidates[0]


def find_experiment_dir(
    dataset_dir: Path,
    experiment_id: str | None,
    model: str | None,
    policy: str | None,
    covariate_variant: str | None,
) -> Path:
    """Find the experiment directory for a model/policy selection."""

    experiments_dir = dataset_dir / "experiments"
    if experiment_id:
        experiment_dir = experiments_dir / experiment_id
        if not (experiment_dir / "experiment_manifest.csv").exists():
            raise ValueError(f"Experiment does not exist: {experiment_dir}")
        return experiment_dir

    matches = []
    for manifest_path in sorted(experiments_dir.glob("*/experiment_manifest.csv")):
        row = read_single_csv_row(manifest_path)
        if not row:
            continue
        row_model = str(row.get("model_label", ""))
        row_family = str(row.get("model_family", ""))
        row_policy = str(row.get("selection_method", ""))
        row_covariate_variant = str(row.get("covariate_variant", ""))
        model_matches = model is None or model.lower() in {
            row_model.lower(),
            row_family.lower(),
        }
        policy_matches = policy is None or policy.lower() == row_policy.lower()
        covariate_variant_matches = (
            covariate_variant is None
            or covariate_variant.lower() == row_covariate_variant.lower()
        )
        if model_matches and policy_matches and covariate_variant_matches:
            matches.append(manifest_path.parent)

    if not matches:
        raise ValueError(
            "No experiment matched "
            f"model={model!r}, policy={policy!r} in {experiments_dir}"
        )
    if len(matches) > 1:
        labels = ", ".join(path.name for path in matches[:12])
        raise ValueError(
            "Experiment selection is ambiguous; pass --experiment-id. "
            f"Matches: {labels}"
        )
    return matches[0]


def reshape_score_array(
    array: np.ndarray, score_shape: tuple[int, int, int]
) -> np.ndarray:
    """Normalize score-like arrays to `(n_sites, n_periods, n_replicates)`."""

    array = np.asarray(array, dtype=float)
    if array.shape == score_shape:
        return array
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[1:] == score_shape:
        return array[0]
    if array.size == int(np.prod(score_shape)):
        return array.reshape(score_shape)
    raise ValueError(
        "Could not reshape array with shape "
        f"{array.shape} to score shape {score_shape}"
    )


def reshape_obs_covariates(
    obs_covariates_raw: np.ndarray,
    score_shape: tuple[int, int, int],
    n_covariates: int,
) -> np.ndarray:
    """Normalize observation covariates to score shape plus covariate axis."""

    obs_covariates_raw = np.asarray(obs_covariates_raw, dtype=float)
    target_shape = score_shape + (n_covariates,)
    if n_covariates == 0:
        return np.zeros(target_shape, dtype=float)
    if obs_covariates_raw.shape == target_shape:
        return obs_covariates_raw
    if obs_covariates_raw.ndim == 2 and obs_covariates_raw.shape[1] == n_covariates:
        if obs_covariates_raw.shape[0] == int(np.prod(score_shape)):
            return obs_covariates_raw.reshape(target_shape)
    raise ValueError(
        "Could not reshape obs_covariates_raw with shape "
        f"{obs_covariates_raw.shape} to {target_shape}"
    )


def load_score_calibration(path: Path, source: str) -> ScoreCalibration:
    """Load one score-calibration CSV artifact."""

    row = read_single_csv_row(path)
    if not row:
        raise FileNotFoundError(f"Score calibration artifact is missing: {path}")
    return ScoreCalibration(
        mu0=float(row["mu0"]),
        sigma0=max(float(row["sigma0"]), 1e-6),
        mu1=float(row["mu1"]),
        sigma1=max(float(row["sigma1"]), 1e-6),
        n_negative=int(float(row.get("n_negative", 0))),
        n_positive=int(float(row.get("n_positive", 0))),
        source=source,
    )


def fit_oracle_score_calibration(
    score_values: np.ndarray,
    true_labels: np.ndarray,
) -> ScoreCalibration:
    """Fit oracle score components from stored real scores and true labels."""

    valid = np.isfinite(score_values) & np.isfinite(true_labels)
    scores = np.asarray(score_values, dtype=float)[valid]
    labels = np.asarray(true_labels, dtype=float)[valid] > 0.5
    negative_scores = scores[~labels]
    positive_scores = scores[labels]
    if negative_scores.size < 2 or positive_scores.size < 2:
        raise ValueError(
            "Cannot derive oracle score components from stored labels: "
            f"{negative_scores.size} negatives and {positive_scores.size} positives."
        )
    return ScoreCalibration(
        mu0=float(negative_scores.mean()),
        sigma0=max(float(negative_scores.std(ddof=1)), 1e-6),
        mu1=float(positive_scores.mean()),
        sigma1=max(float(positive_scores.std(ddof=1)), 1e-6),
        n_negative=int(negative_scores.size),
        n_positive=int(positive_scores.size),
        source="derived from stored true labels",
    )


def load_context(args: SimpleNamespace) -> ArtifactContext:
    """Load static artifact data and validate the chosen trace."""

    sweep_dir = Path(args.sweep_dir)
    dataset_dir = find_dataset_dir(sweep_dir, args.species, args.dataset_kind)
    experiment_dir = find_experiment_dir(
        dataset_dir,
        args.experiment_id,
        args.model,
        args.policy,
        args.covariate_variant,
    )
    manifest = read_single_csv_row(experiment_dir / "experiment_manifest.csv")
    if not manifest:
        raise ValueError(f"Missing experiment manifest: {experiment_dir}")

    oracle_variant = str(manifest.get("oracle_variant", "full")).strip() or "full"
    oracle_dir = dataset_dir / ("null_oracle" if oracle_variant == "null" else "oracle")
    if not (oracle_dir / "parameter_summary.npz").exists():
        raise FileNotFoundError(f"Missing oracle parameter summary: {oracle_dir}")

    metadata_dir = dataset_dir / "dataset_metadata"
    score_values = np.asarray(load_npy(metadata_dir / "score_values.npy"), dtype=float)
    score_shape = tuple(int(value) for value in score_values.shape)

    true_labels = reshape_score_array(
        load_npy(dataset_dir / "inputs" / "true_labels.npy"),
        score_shape,
    )
    site_covariate_names = [
        str(value) for value in load_npy(metadata_dir / "site_covariate_names.npy")
    ]
    obs_covariate_names = [
        str(value) for value in load_npy(metadata_dir / "obs_covariate_names.npy")
    ]
    obs_covariates_raw = reshape_obs_covariates(
        load_npy(metadata_dir / "obs_covariates_raw.npy"),
        score_shape,
        len(obs_covariate_names),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            dataset_dir
            / "visualizations"
            / slugify(manifest.get("experiment_id", experiment_dir.name))
        )
    )
    frame_dir = Path(args.frame_dir) if args.frame_dir else output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    notes = [
        "Oracle score components are derived post hoc from stored real scores and true labels.",
        "Effect bands use independent Normal approximations from saved 90% marginal summaries.",
    ]
    return ArtifactContext(
        sweep_dir=sweep_dir,
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        oracle_dir=oracle_dir,
        output_dir=output_dir,
        frame_dir=frame_dir,
        dataset_name=load_scalar_npy(metadata_dir / "dataset_name.npy"),
        target_label=load_scalar_npy(metadata_dir / "target_label.npy"),
        display_name=str(manifest.get("display_name", experiment_dir.name)),
        model_label=str(manifest.get("model_label", "")),
        selection_method=str(manifest.get("selection_method", "")),
        site_ids=load_npy(metadata_dir / "site_ids.npy"),
        replicate_ids=load_npy(metadata_dir / "replicate_ids.npy"),
        replicate_times=load_npy(metadata_dir / "replicate_times.npy"),
        site_coordinates=np.asarray(
            load_npy(metadata_dir / "site_coordinates.npy"),
            dtype=float,
        ),
        site_covariate_names=site_covariate_names,
        obs_covariate_names=obs_covariate_names,
        site_covariates_raw=np.asarray(
            load_npy(metadata_dir / "site_covariates_raw.npy"),
            dtype=float,
        ),
        obs_covariates_raw=obs_covariates_raw,
        site_covariate_offsets=np.asarray(
            load_npy(metadata_dir / "site_covariate_means.npy"),
            dtype=float,
        ),
        site_covariate_scales=np.asarray(
            load_npy(metadata_dir / "site_covariate_scales.npy"),
            dtype=float,
        ),
        obs_covariate_offsets=np.asarray(
            load_npy(metadata_dir / "obs_covariate_offsets.npy"),
            dtype=float,
        ),
        obs_covariate_scales=np.asarray(
            load_npy(metadata_dir / "obs_covariate_scales.npy"),
            dtype=float,
        ),
        score_values=score_values,
        true_labels=true_labels,
        oracle_summary=load_parameter_summary(oracle_dir / "parameter_summary.npz"),
        oracle_score_calibration=fit_oracle_score_calibration(
            score_values, true_labels
        ),
        notes=notes,
    )


def step_dirs_for_context(
    context: ArtifactContext,
    *,
    max_steps: int | None,
    step_stride: int,
) -> list[Path]:
    """Return sorted step directories after stride and limit filtering."""

    step_dirs = sorted((context.experiment_dir / "steps").glob("step_*"))
    if not step_dirs:
        raise ValueError(f"No step directories found under {context.experiment_dir}")
    step_dirs = step_dirs[:: max(int(step_stride), 1)]
    if max_steps is not None:
        step_dirs = step_dirs[: max(int(max_steps), 0)]
    if not step_dirs:
        raise ValueError("Step filtering left no frames to render.")
    return step_dirs


def selected_sample_for_step(
    context: ArtifactContext, selected_mask: np.ndarray
) -> SelectedSample:
    """Choose the primary selected sample for annotations in a frame."""

    selected_indices = np.argwhere(np.asarray(selected_mask, dtype=bool))
    if selected_indices.size == 0:
        raise ValueError("Selected mask contains no selected samples.")

    selected_scores = np.asarray(
        [context.score_values[tuple(index)] for index in selected_indices],
        dtype=float,
    )
    finite_scores = np.where(np.isfinite(selected_scores), selected_scores, -np.inf)
    primary_index = selected_indices[int(np.argmax(finite_scores))]
    site_index, period_index, replicate_index = [int(value) for value in primary_index]
    return SelectedSample(
        site_index=site_index,
        period_index=period_index,
        replicate_index=replicate_index,
        score=float(context.score_values[site_index, period_index, replicate_index]),
        site_id=str(context.site_ids[site_index]),
        replicate_id=str(context.replicate_ids[site_index, replicate_index]),
        replicate_time=str(context.replicate_times[site_index, replicate_index]),
        batch_size=int(selected_indices.shape[0]),
    )


def parameter_for_index(
    summary: dict[str, ParameterStats],
    prefix: str,
    index: int,
    covariate_name: str | None = None,
) -> ParameterStats | None:
    """Find a coefficient parameter by local naming conventions."""

    candidates = [f"{prefix}{index}"]
    if covariate_name:
        candidates.insert(0, f"{prefix}{covariate_name}")
    if index == 0:
        candidates.append(f"{prefix}intercept")
    for candidate in candidates:
        if candidate in summary:
            return summary[candidate]

    matching_names = sorted(
        [name for name in summary if name.startswith(prefix)],
        key=lambda name: _parameter_sort_key(name, prefix),
    )
    if len(matching_names) > index:
        return summary[matching_names[index]]
    return None


def _parameter_sort_key(name: str, prefix: str) -> tuple[int, str]:
    suffix = name.removeprefix(prefix)
    if suffix in {"intercept", "0"}:
        return (0, suffix)
    try:
        return (int(suffix), suffix)
    except ValueError:
        return (10_000, suffix)


def covariate_grid(raw_values: np.ndarray) -> np.ndarray:
    """Build a compact raw-scale grid for an effect plot."""

    raw_values = np.asarray(raw_values, dtype=float).reshape(-1)
    finite_values = raw_values[np.isfinite(raw_values)]
    if finite_values.size == 0:
        return np.linspace(-2.0, 2.0, 80)
    unique_values = np.unique(finite_values)
    if unique_values.size <= 2 and np.isin(unique_values, [0.0, 1.0]).all():
        return unique_values.astype(float)
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if math.isclose(min_value, max_value):
        padding = 1.0 if math.isclose(min_value, 0.0) else abs(min_value) * 0.1
        return np.linspace(min_value - padding, max_value + padding, 80)
    return np.linspace(min_value, max_value, 80)


def effect_curve(
    summary: dict[str, ParameterStats],
    prefix: str,
    covariate_index: int,
    covariate_name: str,
    x_values: np.ndarray,
    offset: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Approximate a logistic partial-effect curve from saved summaries."""

    intercept_stats = parameter_for_index(summary, prefix, 0)
    slope_stats = parameter_for_index(
        summary,
        prefix,
        covariate_index + 1,
        covariate_name,
    )
    if intercept_stats is None or slope_stats is None:
        return None

    intercept_mean = finite_scalar(intercept_stats.mean)
    slope_mean = finite_scalar(slope_stats.mean)
    intercept_sd = finite_scalar(marginal_sd(intercept_stats), default=0.0)
    slope_sd = finite_scalar(marginal_sd(slope_stats), default=0.0)
    if not (np.isfinite(intercept_mean) and np.isfinite(slope_mean)):
        return None

    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    x_model = (np.asarray(x_values, dtype=float) - offset) / scale
    lp_mean = intercept_mean + slope_mean * x_model
    lp_sd = np.sqrt(intercept_sd**2 + np.square(x_model * slope_sd))
    lower = sigmoid(lp_mean - Z90 * lp_sd)
    upper = sigmoid(lp_mean + Z90 * lp_sd)
    return sigmoid(lp_mean), lower, upper


def occupancy_density(
    summary: dict[str, ParameterStats],
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the site-level occupancy posterior PDF."""

    if "psi" not in summary:
        raise ValueError("Parameter summary does not contain `psi`.")
    psi_stats = summary["psi"]
    means = np.asarray(psi_stats.mean, dtype=float).reshape(-1)
    sds = marginal_sd(psi_stats).reshape(-1)
    valid = np.isfinite(means) & np.isfinite(sds) & (sds > 0.0)
    if not np.any(valid):
        raise ValueError("No finite psi means and intervals are available.")
    means = means[valid]
    sds = np.maximum(sds[valid], 1e-3)
    x_values = np.linspace(0.0, 1.0, 500)
    density = np.zeros_like(x_values)
    for mean, sd in zip(means, sds):
        component = normal_pdf(x_values, mean, sd)
        integral = np.trapezoid(component, x_values)
        if np.isfinite(integral) and integral > 0.0:
            component = component / integral
        density += component
    density /= float(len(means))
    return x_values, density


def plot_occupancy_pdf(
    axis: plt.Axes,
    current_summary: dict[str, ParameterStats],
    oracle_summary: dict[str, ParameterStats],
    selected: SelectedSample,
) -> None:
    """Plot current and oracle occupancy posterior PDFs."""

    current_x, current_density = occupancy_density(current_summary)
    oracle_x, oracle_density = occupancy_density(oracle_summary)
    axis.plot(current_x, current_density, color=CURRENT_COLOR, lw=2.0, label="Current")
    axis.fill_between(current_x, 0.0, current_density, color=CURRENT_COLOR, alpha=0.12)
    axis.plot(oracle_x, oracle_density, color=ORACLE_COLOR, lw=2.0, label="Oracle")
    axis.fill_between(oracle_x, 0.0, oracle_density, color=ORACLE_COLOR, alpha=0.10)

    for summary, color, label in (
        (current_summary, CURRENT_COLOR, "Current selected-site mean"),
        (oracle_summary, ORACLE_COLOR, "Oracle selected-site mean"),
    ):
        psi_mean = np.asarray(summary["psi"].mean, dtype=float).reshape(-1)
        if selected.site_index < psi_mean.size and np.isfinite(
            psi_mean[selected.site_index]
        ):
            axis.axvline(
                float(psi_mean[selected.site_index]),
                color=color,
                lw=1.5,
                ls=":",
                alpha=0.9,
                label=label,
            )

    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Site occupancy probability")
    axis.set_ylabel("Density")
    axis.set_title("Posterior occupancy PDF")
    axis.grid(alpha=0.25)
    add_readable_legend(axis, loc="upper left")


def score_range(
    score_values: np.ndarray,
    calibrations: tuple[ScoreCalibration, ScoreCalibration],
) -> np.ndarray:
    """Build a score grid covering observed scores and component tails."""

    finite_scores = score_values[np.isfinite(score_values)]
    anchors = []
    for calibration in calibrations:
        anchors.extend(
            [
                calibration.mu0 - 4.0 * calibration.sigma0,
                calibration.mu0 + 4.0 * calibration.sigma0,
                calibration.mu1 - 4.0 * calibration.sigma1,
                calibration.mu1 + 4.0 * calibration.sigma1,
            ]
        )
    if finite_scores.size:
        anchors.extend([float(finite_scores.min()), float(finite_scores.max())])
    min_value = min(anchors)
    max_value = max(anchors)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0
    return np.linspace(min_value, max_value, 500)


def plot_score_mixture(
    axis: plt.Axes,
    context: ArtifactContext,
    current_calibration: ScoreCalibration,
    selected_mask: np.ndarray,
    selected: SelectedSample,
) -> None:
    """Plot current and oracle score mixture components."""

    oracle_calibration = context.oracle_score_calibration
    x_values = score_range(
        context.score_values,
        (current_calibration, oracle_calibration),
    )
    lines = [
        (current_calibration, CURRENT_COLOR, "Current"),
        (oracle_calibration, ORACLE_COLOR, "Oracle"),
    ]
    for calibration, color, label_prefix in lines:
        axis.plot(
            x_values,
            normal_pdf(x_values, calibration.mu0, calibration.sigma0),
            color=color,
            lw=1.8,
            ls="--",
            label=f"{label_prefix} negative",
        )
        axis.plot(
            x_values,
            normal_pdf(x_values, calibration.mu1, calibration.sigma1),
            color=color,
            lw=1.8,
            ls="-",
            label=f"{label_prefix} positive",
        )

    selected_scores = context.score_values[np.asarray(selected_mask, dtype=bool)]
    for score in selected_scores[np.isfinite(selected_scores)]:
        axis.axvline(float(score), color="0.55", lw=0.7, ls=":", alpha=0.35)
    if np.isfinite(selected.score):
        axis.axvline(
            selected.score,
            color=SELECTED_COLOR,
            lw=1.5,
            ls=":",
            label="Selected score",
        )

    axis.set_xlabel("Classifier score")
    axis.set_ylabel("Density")
    axis.set_title("Score mixture components")
    axis.grid(alpha=0.25)
    add_readable_legend(axis, loc="upper left", ncol=2)


def plot_map(
    axis: plt.Axes,
    context: ArtifactContext,
    selected_mask: np.ndarray,
    selected: SelectedSample,
) -> None:
    """Plot site locations with the selected site highlighted."""

    if cx is None or Transformer is None:
        raise ImportError(
            "Contextily and pyproj are required for the geographic basemap. "
            "Install `contextily` and `pyproj` or render with an environment "
            "that provides them."
        )

    coordinates = np.asarray(context.site_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        axis.text(0.5, 0.5, "No site coordinates", ha="center", va="center")
        axis.set_axis_off()
        return

    latitudes = coordinates[:, 0]
    longitudes = coordinates[:, 1]
    valid = np.isfinite(latitudes) & np.isfinite(longitudes)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_coords = np.full_like(longitudes, np.nan, dtype=float)
    y_coords = np.full_like(latitudes, np.nan, dtype=float)
    if np.any(valid):
        x_coords[valid], y_coords[valid] = transformer.transform(
            longitudes[valid],
            latitudes[valid],
        )
    valid_projected = np.isfinite(x_coords) & np.isfinite(y_coords)
    if np.any(valid):
        x_min = float(np.nanmin(x_coords[valid_projected]))
        x_max = float(np.nanmax(x_coords[valid_projected]))
        y_min = float(np.nanmin(y_coords[valid_projected]))
        y_max = float(np.nanmax(y_coords[valid_projected]))
        x_pad = max((x_max - x_min) * 0.12, 20_000.0)
        y_pad = max((y_max - y_min) * 0.12, 20_000.0)
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
    try:
        cx.add_basemap(
            axis,
            source=cx.providers.Esri.WorldTopoMap,
            crs="EPSG:3857",
            zoom="auto",
            zoom_adjust=1,
            attribution_size=4,
        )
    except (OSError, ValueError) as error:
        axis.text(
            0.5,
            0.5,
            f"Basemap unavailable\n{error}",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=8,
            color="0.25",
        )
        axis.set_facecolor("#eef2ea")
    axis.scatter(
        x_coords[valid_projected],
        y_coords[valid_projected],
        s=18,
        color="0.62",
        edgecolor="white",
        linewidth=0.35,
        label="Sites",
        alpha=0.9,
        zorder=5,
    )
    selected_sites = np.unique(np.argwhere(np.asarray(selected_mask, dtype=bool))[:, 0])
    selected_sites = selected_sites[selected_sites < len(latitudes)]
    selected_valid = selected_sites[
        np.isfinite(x_coords[selected_sites]) & np.isfinite(y_coords[selected_sites])
    ]
    if selected_valid.size:
        axis.scatter(
            x_coords[selected_valid],
            y_coords[selected_valid],
            s=80,
            facecolor="none",
            edgecolor=SELECTED_BATCH_COLOR,
            linewidth=2.2,
            label="Selected batch",
            zorder=6,
        )
    if (
        selected.site_index < len(latitudes)
        and np.isfinite(x_coords[selected.site_index])
        and np.isfinite(y_coords[selected.site_index])
    ):
        axis.scatter(
            [x_coords[selected.site_index]],
            [y_coords[selected.site_index]],
            s=105,
            marker="*",
            color=SELECTED_COLOR,
            edgecolor="white",
            linewidth=0.6,
            label="Primary selected",
            zorder=7,
        )
    axis.set_title("Review location")
    axis.set_xticks([])
    axis.set_yticks([])
    add_readable_legend(axis, loc="upper left")


def plot_effect(
    axis: plt.Axes,
    current_summary: dict[str, ParameterStats],
    oracle_summary: dict[str, ParameterStats],
    prefix: str,
    title_prefix: str,
    covariate_index: int,
    covariate_name: str,
    raw_values: np.ndarray,
    selected_value: float,
    offset: float,
    scale: float,
) -> bool:
    """Plot one current/oracle covariate effect panel."""

    x_values = covariate_grid(raw_values)
    current_curve = effect_curve(
        current_summary,
        prefix,
        covariate_index,
        covariate_name,
        x_values,
        offset,
        scale,
    )
    oracle_curve = effect_curve(
        oracle_summary,
        prefix,
        covariate_index,
        covariate_name,
        x_values,
        offset,
        scale,
    )
    if current_curve is None and oracle_curve is None:
        return False

    for curve, color, label in (
        (current_curve, CURRENT_COLOR, "Current"),
        (oracle_curve, ORACLE_COLOR, "Oracle"),
    ):
        if curve is None:
            continue
        mean, lower, upper = curve
        axis.plot(x_values, mean, color=color, lw=1.7, label=label)
        axis.fill_between(x_values, lower, upper, color=color, alpha=0.12)

    if np.isfinite(selected_value):
        axis.axvline(
            float(selected_value),
            color=SELECTED_COLOR,
            lw=1.5,
            ls=":",
            label="Selected value",
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel(pretty_label(covariate_name))
    axis.set_ylabel("Probability")
    axis.set_title(fill(f"{title_prefix}: {pretty_label(covariate_name)}", 34))
    axis.grid(alpha=0.22)
    add_readable_legend(
        axis,
        loc=stable_legend_location_from_oracle(oracle_curve),
    )
    return True


def pretty_label(value: str) -> str:
    """Format underscored artifact names for plot labels."""

    return str(value).replace("_", " ")


def plot_effects(
    fig: plt.Figure,
    grid_spec: GridSpec,
    context: ArtifactContext,
    current_summary: dict[str, ParameterStats],
    selected: SelectedSample,
) -> None:
    """Plot all available site and observation covariate effects."""

    effect_specs = []
    for covariate_index, covariate_name in enumerate(context.site_covariate_names):
        raw_values = context.site_covariates_raw[:, covariate_index]
        selected_value = (
            context.site_covariates_raw[selected.site_index, covariate_index]
            if context.site_covariates_raw.size
            else np.nan
        )
        offset = (
            float(context.site_covariate_offsets[covariate_index])
            if covariate_index < len(context.site_covariate_offsets)
            else 0.0
        )
        scale = (
            float(context.site_covariate_scales[covariate_index])
            if covariate_index < len(context.site_covariate_scales)
            else 1.0
        )
        effect_specs.append(
            (
                "cov_state_",
                "Occupancy",
                covariate_index,
                covariate_name,
                raw_values,
                selected_value,
                offset,
                scale,
            )
        )
    for covariate_index, covariate_name in enumerate(context.obs_covariate_names):
        raw_values = context.obs_covariates_raw[..., covariate_index]
        selected_value = context.obs_covariates_raw[
            selected.site_index,
            selected.period_index,
            selected.replicate_index,
            covariate_index,
        ]
        offset = (
            float(context.obs_covariate_offsets[covariate_index])
            if covariate_index < len(context.obs_covariate_offsets)
            else 0.0
        )
        scale = (
            float(context.obs_covariate_scales[covariate_index])
            if covariate_index < len(context.obs_covariate_scales)
            else 1.0
        )
        effect_specs.append(
            (
                "cov_det_",
                "Detection",
                covariate_index,
                covariate_name,
                raw_values,
                selected_value,
                offset,
                scale,
            )
        )

    rendered_count = 0
    max_columns = 4
    for effect_index, effect_spec in enumerate(effect_specs):
        row = effect_index // max_columns
        column = effect_index % max_columns
        axis = fig.add_subplot(grid_spec[row, column])
        rendered = plot_effect(
            axis,
            current_summary,
            context.oracle_summary,
            *effect_spec,
        )
        if rendered:
            rendered_count += 1
        else:
            axis.set_axis_off()

    for empty_index in range(len(effect_specs), grid_spec.nrows * grid_spec.ncols):
        row = empty_index // max_columns
        column = empty_index % max_columns
        axis = fig.add_subplot(grid_spec[row, column])
        axis.set_axis_off()

    if rendered_count == 0:
        axis = fig.add_subplot(grid_spec[0, :])
        axis.text(
            0.5,
            0.5,
            "No covariate effect summaries are available for this experiment.",
            ha="center",
            va="center",
        )
        axis.set_axis_off()


def frame_title(
    context: ArtifactContext,
    step_number: int,
    review_count: int,
    selected: SelectedSample,
) -> tuple[str, str, str]:
    """Build compact frame title lines."""

    replicate_label = selected.replicate_id
    if len(replicate_label) > 72:
        replicate_label = "..." + replicate_label[-69:]
    return (
        f"{context.dataset_name} | {context.target_label} | {context.display_name}",
        f"Step {step_number:03d}, reviewed {review_count}, "
        f"selected batch {selected.batch_size}; primary site {selected.site_id}, "
        f"score {selected.score:.3g}, time {selected.replicate_time}",
        replicate_label,
    )


def render_frame(
    fig: plt.Figure,
    context: ArtifactContext,
    step_dir: Path,
) -> None:
    """Render one animation frame into an existing figure."""

    fig.clear()
    current_summary = load_parameter_summary(step_dir / "parameter_summary.npz")
    current_calibration = load_score_calibration(
        step_dir / "score_calibration.csv",
        source="saved current step",
    )
    selected_mask = np.asarray(load_npy(step_dir / "selected_mask.npy"), dtype=bool)
    selected = selected_sample_for_step(context, selected_mask)
    scalar_metrics = read_single_csv_row(step_dir / "scalar_metrics.csv")
    review_count = int(float(scalar_metrics.get("review_count", 0)))
    step_number = int(step_dir.name.rsplit("_", maxsplit=1)[-1])

    n_effects = len(context.site_covariate_names) + len(context.obs_covariate_names)
    effect_rows = max(1, math.ceil(max(n_effects, 1) / 4))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.12 * effect_rows],
        hspace=0.36,
        left=0.055,
        right=0.99,
        bottom=0.07,
        top=0.885,
    )
    top_grid = outer[0].subgridspec(1, 3, wspace=0.28)
    plot_occupancy_pdf(
        fig.add_subplot(top_grid[0, 0]),
        current_summary,
        context.oracle_summary,
        selected,
    )
    plot_score_mixture(
        fig.add_subplot(top_grid[0, 1]),
        context,
        current_calibration,
        selected_mask,
        selected,
    )
    if cx is None or Transformer is None:
        raise ImportError(
            "Contextily and pyproj are required for the geographic basemap. "
            "Install `contextily` and `pyproj` or render with an environment "
            "that provides them."
        )
    plot_map(
        fig.add_subplot(top_grid[0, 2]),
        context,
        selected_mask,
        selected,
    )
    effect_grid = outer[1].subgridspec(effect_rows, 4, wspace=0.3, hspace=0.55)
    plot_effects(fig, effect_grid, context, current_summary, selected)

    title_line, subtitle_line, detail_line = frame_title(
        context,
        step_number,
        review_count,
        selected,
    )
    fig.text(
        0.5,
        0.985,
        title_line,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(0.5, 0.957, subtitle_line, ha="center", va="top", fontsize=10.7)
    fig.text(0.5, 0.935, detail_line, ha="center", va="top", fontsize=9.5)
    fig.text(
        0.01,
        0.004,
        "Post-hoc reconstruction from saved sweep artifacts; no model refit.",
        fontsize=7.5,
        color="0.35",
    )


def parse_step_list(value: str | None) -> list[int]:
    """Parse a comma-separated step list."""

    if not value:
        return []
    steps = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        steps.append(int(chunk))
    return steps


def configure_ffmpeg() -> None:
    """Configure Matplotlib's ffmpeg writer, including imageio-ffmpeg if present."""

    if animation.writers.is_available("ffmpeg"):
        return
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib cannot find ffmpeg. Install ffmpeg or `imageio-ffmpeg` "
            "to export MP4 files."
        ) from exc
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "imageio-ffmpeg was imported, but Matplotlib still cannot use ffmpeg."
        )


def save_png_frames(
    context: ArtifactContext,
    step_dirs: list[Path],
    requested_steps: list[int],
    *,
    dpi: int,
    figsize: tuple[float, float],
) -> list[Path]:
    """Save selected debugging PNG frames."""

    if not requested_steps:
        return []
    step_by_number = {
        int(step_dir.name.rsplit("_", maxsplit=1)[-1]): step_dir
        for step_dir in step_dirs
    }
    saved_paths = []
    for step_number in requested_steps:
        if step_number not in step_by_number:
            raise ValueError(
                f"Requested PNG step is not in selected frame set: {step_number}"
            )
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        render_frame(fig, context, step_by_number[step_number])
        png_path = context.frame_dir / f"step_{step_number:03d}.png"
        fig.savefig(png_path, dpi=dpi, facecolor="white")
        plt.close(fig)
        saved_paths.append(png_path)
    return saved_paths


def save_mp4(
    context: ArtifactContext,
    step_dirs: list[Path],
    output_path: Path,
    *,
    fps: int,
    dpi: int,
    figsize: tuple[float, float],
) -> Path:
    """Render all selected steps to an MP4 animation."""

    configure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(output_path), dpi=dpi):
        for index, step_dir in enumerate(step_dirs, start=1):
            print(
                f"Rendering frame {index}/{len(step_dirs)}: {step_dir.name}",
                flush=True,
            )
            render_frame(fig, context, step_dir)
            writer.grab_frame(facecolor="white")
    plt.close(fig)
    return output_path


def main(
    sweep_dir: str | Path = DEFAULT_SWEEP_DIR,
    species: str = "BTBW",
    dataset_kind: str = "acoustic",
    experiment_id: str | None = None,
    model: str = "occu_cs_calibrated",
    policy: str = "bald",
    covariate_variant: str = "full",
    output_dir: str | Path | None = None,
    frame_dir: str | Path | None = None,
    output: str | Path | None = None,
    png_steps: str = "0,1,10",
    skip_video: bool = False,
    max_steps: int | None = None,
    step_stride: int = 1,
    fps: int = 6,
    dpi: int = 130,
    fig_width: float = 18.0,
    fig_height: float = 11.0,
) -> int:
    """Create a species/model/policy-specific MP4 visualization."""

    args = SimpleNamespace(
        sweep_dir=Path(sweep_dir),
        species=species,
        dataset_kind=dataset_kind,
        experiment_id=experiment_id,
        model=model,
        policy=policy,
        covariate_variant=covariate_variant,
        output_dir=Path(output_dir) if output_dir is not None else None,
        frame_dir=Path(frame_dir) if frame_dir is not None else None,
        output=Path(output) if output is not None else None,
        png_steps=png_steps,
        skip_video=bool(skip_video),
        max_steps=max_steps,
        step_stride=int(step_stride),
        fps=int(fps),
        dpi=int(dpi),
        fig_width=float(fig_width),
        fig_height=float(fig_height),
    )
    context = load_context(args)
    step_dirs = step_dirs_for_context(
        context,
        max_steps=args.max_steps,
        step_stride=args.step_stride,
    )
    steps_without_selection = [
        step_dir
        for step_dir in step_dirs
        if not np.asarray(load_npy(step_dir / "selected_mask.npy"), dtype=bool).any()
    ]
    if steps_without_selection:
        skipped_names = ", ".join(step_dir.name for step_dir in steps_without_selection)
        print(
            "Skipping steps without selected samples; no sample can be highlighted "
            f"for review in these frames: {skipped_names}",
            flush=True,
        )
        step_dirs = [
            step_dir
            for step_dir in step_dirs
            if step_dir not in steps_without_selection
        ]
        if not step_dirs:
            raise ValueError("No selected-sample frames remain after filtering.")

    # Validate that current score calibrations exist for the selected trace before
    # starting a long render. Reviewed-only experiments do not currently save
    # these artifacts, so the requested score-mixture panel cannot be produced.
    missing_calibrations = [
        step_dir
        for step_dir in step_dirs
        if not (step_dir / "score_calibration.csv").exists()
    ]
    if missing_calibrations:
        example = missing_calibrations[0]
        raise FileNotFoundError(
            "The selected experiment does not contain current score-calibration "
            f"artifacts needed for the score-mixture panel. First missing step: {example}"
        )

    figsize = (float(args.fig_width), float(args.fig_height))
    png_steps = parse_step_list(args.png_steps)
    saved_pngs = save_png_frames(
        context,
        step_dirs,
        png_steps,
        dpi=int(args.dpi),
        figsize=figsize,
    )
    for png_path in saved_pngs:
        print(f"Wrote PNG frame: {png_path}")

    output_path = (
        Path(args.output)
        if args.output
        else context.output_dir
        / f"{slugify(context.target_label)}__{slugify(context.experiment_dir.name)}.mp4"
    )
    if not args.skip_video:
        save_mp4(
            context,
            step_dirs,
            output_path,
            fps=int(args.fps),
            dpi=int(args.dpi),
            figsize=figsize,
        )
        print(f"Wrote MP4 animation: {output_path}")
    for note in context.notes:
        print(f"Note: {note}")
    return 0
