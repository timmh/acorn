# %%
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from occubed.paths import (
    DATA_DIR,
    DEFAULT_SWEEP_DIR,
    FIGURE_DIR,
    PACKAGE_DIR,
)
from occubed.submit_sweep import default_species_by_dataset
from occubed.summaries import iter_parameter_summary_arrays

if "ipykernel" in sys.modules:
    from IPython.display import Markdown, display
else:  # pragma: no cover - plain-script fallback
    Markdown = None

    def display(value):
        print(value)


plt.style.use("seaborn-v0_8-whitegrid")
WIDTH_SCALE = 1.5
NEURIPS_TEXT_WIDTH_PT = 397.48499 * WIDTH_SCALE
TEX_POINTS_PER_INCH = 72.27
FULL_WIDTH_IN = NEURIPS_TEXT_WIDTH_PT / TEX_POINTS_PER_INCH
BASE_FONT_SIZE_PT = 10.0
PLOT_LINEWIDTH = 1.0
AXIS_TITLE_FONTSIZE = BASE_FONT_SIZE_PT * 0.8
AGGREGATED_SMOOTHING_WINDOW = 10
AGGREGATED_SMOOTHING_ROUGHNESS_THRESHOLD = 0.40
AGGREGATED_SMOOTHING_MAX_JUMP_THRESHOLD = 0.80
AGGREGATED_SMOOTHING_LONG_WINDOW = 10
AGGREGATED_SMOOTHING_LONG_HORIZON_MIN_SPAN = 100
AGGREGATED_SMOOTHING_LONG_ROUGHNESS_THRESHOLD = 0.20
AGGREGATED_SMOOTHING_LONG_MAX_JUMP_THRESHOLD = 0.40
AGREEMENT_TREND_WINDOW = int(
    os.environ.get("OCCUBED_ANALYSIS_AGREEMENT_TREND_WINDOW", "7")
)
AGREEMENT_RAW_TRACE_ALPHA = 0.22
AGREEMENT_RAW_TRACE_LINEWIDTH = PLOT_LINEWIDTH * 0.55
AGREEMENT_TREND_LINEWIDTH = PLOT_LINEWIDTH * 1.35

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


# %%
SWEEP_DIR = os.environ.get("OCCUBED_SWEEP_DIR", str(DEFAULT_SWEEP_DIR))
COMBINED_RESULTS_CSV = os.environ.get(
    "OCCUBED_ANALYSIS_RESULTS_CSV",
    "combined_species_results.csv",
)
DATASET_FILTER = None
SAVE_FIGURES = os.environ.get("OCCUBED_ANALYSIS_WRITE_FIGURES", "1") != "0"
FIGURE_OUTPUT_DIR = os.environ.get(
    "OCCUBED_ANALYSIS_FIGURE_OUTPUT_DIR",
    str(FIGURE_DIR),
)
SAVE_TABLES = os.environ.get("OCCUBED_ANALYSIS_WRITE_TABLES", "1") != "0"
TABLE_OUTPUT_DIR = os.environ.get(
    "OCCUBED_ANALYSIS_TABLE_OUTPUT_DIR",
    str(PACKAGE_DIR / "manuscript" / "tables"),
)
REVIEW_TARGET_FRACTION = 0.90
MAX_REVIEW_COUNT_BY_DATASET = {
    "acoustic": sys.maxsize,
    "iwildcam": sys.maxsize,
}
# DEFAULT_MAX_SPECIES_PER_DATASET = 0
MAX_SPECIES_BY_DATASET = {
    "acoustic": 50,
    "iwildcam": 25,
}
SPECIES_EXAMPLES = {
    "acoustic": None,
    "iwildcam": None,
}
PARAMETER_TRAJECTORY_EXPERIMENT_IDS = [
    # "adaptive_calibrated_bald",
    # "adaptive_calibrated_dual_bald_max_balanced",
    # "adaptive_calibrated_random",
    # "original_occu_cs_bald",
    # "original_occu_cs_target_eig",
    "cut_normal_calibrated_target_eig",
    "cut_normal_calibrated_random",
    "reviewed_target_eig",
    "reviewed_random",
]
PARAMETER_TRAJECTORY_SUBDIR = "species_parameter_trajectories"
EXPORT_PARAMETER_TRAJECTORIES = (
    os.environ.get("OCCUBED_ANALYSIS_EXPORT_PARAMETER_TRAJECTORIES", "1") != "0"
)
RUN_STOPPING_ANALYSIS = (
    os.environ.get("OCCUBED_ANALYSIS_RUN_STOPPING_ANALYSIS", "1") != "0"
)
DATASET_ONLY = os.environ.get("OCCUBED_ANALYSIS_DATASET_ONLY", "0") != "0"
STOPPING_EXPERIMENT_IDS = [
    "cut_normal_calibrated_target_eig",
]
MAX_AGREEMENT_SPECIES_PER_DATASET = int(
    os.environ.get("OCCUBED_ANALYSIS_MAX_AGREEMENT_SPECIES_PER_DATASET", "0")
)
PARAMETER_TRAJECTORY_PLOT_WORKERS = int(
    os.environ.get("OCCUBED_ANALYSIS_PARAMETER_TRAJECTORY_PLOT_WORKERS", "0")
)
FULL_EXPERIMENT_IDS = [
    "reviewed_random",
    "reviewed_bald",
    "reviewed_max_score",
    "reviewed_posterior_predictive_uncertainty",
    # "adaptive_calibrated_random",
    # "adaptive_calibrated_bald",
    # "adaptive_calibrated_dual_bald_max_balanced",
    # "adaptive_calibrated_max_score",
    # "adaptive_calibrated_posterior_predictive_uncertainty",
    # "original_occu_cs_bald",
]
NULL_EXPERIMENT_IDS = [
    f"{experiment_id}__null" for experiment_id in FULL_EXPERIMENT_IDS
]
CUT_NORMAL_EXPERIMENT_IDS = [
    "cut_normal_calibrated_random",
    "cut_normal_calibrated_bald",
    "cut_normal_calibrated_max_score",
    "cut_normal_calibrated_posterior_predictive_uncertainty",
]
INFORMATION_GAIN_EXPERIMENT_IDS = [
    # "reviewed_epig",
    # "adaptive_calibrated_epig",
    # "adaptive_calibrated_revealed_targeted_eig",
    # "original_occu_cs_target_eig",
    "reviewed_target_eig",
    "cut_normal_calibrated_target_eig",
]
ORACLE_DISTANCE_EXPERIMENT_IDS = [
    *FULL_EXPERIMENT_IDS,
    *CUT_NORMAL_EXPERIMENT_IDS,
    *INFORMATION_GAIN_EXPERIMENT_IDS,
]
NULL_ORACLE_DISTANCE_EXPERIMENT_IDS = [
    f"{experiment_id}__null" for experiment_id in ORACLE_DISTANCE_EXPERIMENT_IDS
]

SELECTION_ORDER = [
    "random",
    "bald",
    "epig",
    "revealed_targeted_eig",
    "target_eig",
    "dual_balanced",
    "max_score",
    "posterior_predictive_uncertainty",
]
SELECTION_LABELS = {
    "random": "Random",
    "epig": "EPIG",
    "revealed_targeted_eig": "EIG",
    "dual_balanced": "Dual BALD / max-score",
    "max_score": "Max score",
    "posterior_predictive_uncertainty": "PPU",
    "bald": "BALD",
    "target_eig": "Target EIG",
}
SELECTION_COLORS = {
    "random": "#4C78A8",
    "bald": "#F58518",
    "epig": "#72B7B2",
    "revealed_targeted_eig": "#8C564B",
    "target_eig": "#B6992D",
    "dual_balanced": "#E45756",
    "max_score": "#54A24B",
    "posterior_predictive_uncertainty": "#B279A2",
}
DATASET_COLORS = {
    "acoustic": "#4C78A8",
    "iwildcam": "#54A24B",
}
DATASET_LABELS = {
    "acoustic": "Acoustic",
    "iwildcam": "Camera trap",
}
DATASET_CLASS_DISTRIBUTION_FILENAME = "dataset_class_distribution.png"
DATASET_ORDER = ["acoustic", "iwildcam"]
ACOUSTIC_SPECIES_METADATA_PATHS = [
    DATA_DIR / "acoustic_forest_soundscape/perch_v2/dataset_species_perch_mapping.csv",
    DATA_DIR / "acoustic_forest_soundscape/raw/species.csv",
    DATA_DIR / "acoustic/species.csv",
]
IWILDCAM_SPECIES_METADATA_PATH = DATA_DIR / "iwildcam2022/target_species_mapping.csv"
MODEL_FAMILY_LABELS = {
    "reviewed_bernoulli": "Reviewed-only",
    "calibrated_cs": "Calibrated continuous-score",
    "cut_normal_calibrated_cs": "Decoupled continuous-score",
    "original_cs": "Original continuous-score",
}
MODEL_FAMILY_ORDER = [
    "reviewed_bernoulli",
    # "calibrated_cs",
    "cut_normal_calibrated_cs",
    # "original_cs",
]
MODEL_FAMILY_LINESTYLES = {
    "reviewed_bernoulli": "--",
    # "calibrated_cs": ":",
    "cut_normal_calibrated_cs": "-",
    # "original_cs": "-.",
}
MODEL_FAMILY_COLORS = {
    "calibrated_cs": "#4C78A8",
    "cut_normal_calibrated_cs": "#59A14F",
    "original_cs": "#9D755D",
    "reviewed_bernoulli": "#E45756",
}
GROUP_METRICS = [
    ("mean_oracle_distance", "Overall mean $W_1$"),
    ("grouped_oracle_distance__overall_occupancy_psi", "Occupancy probability $\\psi$"),
    (
        "grouped_oracle_distance__occupancy_coefficients",
        "Occupancy coefficients $\\beta$",
    ),
    (
        "grouped_oracle_distance__detection_coefficients",
        "Detection coefficients $\\alpha$",
    ),
]
GROUP_NAME_TO_COLUMN = {
    "Overall occupancy / psi": "grouped_oracle_distance__overall_occupancy_psi",
    "Occupancy coefficients": "grouped_oracle_distance__occupancy_coefficients",
    "Detection coefficients": "grouped_oracle_distance__detection_coefficients",
}
AGREEMENT_EXPERIMENT_IDS = list(ORACLE_DISTANCE_EXPERIMENT_IDS)
NULL_AGREEMENT_EXPERIMENT_IDS = [
    f"{experiment_id}__null" for experiment_id in AGREEMENT_EXPERIMENT_IDS
]
AGREEMENT_PLOT_VARIANTS = [
    ("full", "default"),
    ("null", "null_model"),
]
AGREEMENT_PLOT_MODEL_FAMILY_GROUPS = [
    (["calibrated_cs", "cut_normal_calibrated_cs", "original_cs"], "cs"),
    (["cut_normal_calibrated_cs"], "cut_normal_cs"),
    (["original_cs"], "original_cs"),
    (["reviewed_bernoulli"], "bernoulli"),
]
AGREEMENT_FAMILY_AVERAGE_MODEL_FAMILIES = [
    "reviewed_bernoulli",
    # "calibrated_cs",
    "original_cs",
    "cut_normal_calibrated_cs",
]
AGREEMENT_CURVE_METRICS = [
    ("psi_probability_agreement", "Occupancy $\\psi$ agreement"),
    ("site_priority_spearman", "Site-rank Spearman $\\rho$"),
    (
        "coef_ci_conclusion_agreement__occupancy",
        "Occupancy\ncoefficient conclusions",
    ),
    (
        "coef_ci_conclusion_agreement__detection",
        "Detection\ncoefficient conclusions",
    ),
]
AGREEMENT_TABLE_METRICS = list(AGREEMENT_CURVE_METRICS)
RMSE_METRIC_COLUMNS: set[str] = set()
CAMERA_TRAP_AGREEMENT_THRESHOLD = 0.95
AGREEMENT_THRESHOLD_RULES = {
    "psi_probability_agreement": {
        "direction": "ge",
        "threshold": 0.90,
        "threshold_label": f"$\\ge 0.90$",
    },
    "site_priority_spearman": {
        "direction": "ge",
        "threshold": REVIEW_TARGET_FRACTION,
        "threshold_label": f"$\\ge {REVIEW_TARGET_FRACTION:.2f}$",
    },
    "coef_ci_conclusion_agreement__occupancy": {
        "direction": "ge",
        "threshold": REVIEW_TARGET_FRACTION,
        "threshold_label": f"$\\ge {REVIEW_TARGET_FRACTION:.2f}$",
    },
    "coef_ci_conclusion_agreement__detection": {
        "direction": "ge",
        "threshold": REVIEW_TARGET_FRACTION,
        "threshold_label": f"$\\ge {REVIEW_TARGET_FRACTION:.2f}$",
    },
}
CAMERA_TRAP_AGREEMENT_THRESHOLD_RULES = {
    metric_column: {"threshold": CAMERA_TRAP_AGREEMENT_THRESHOLD}
    for metric_column, _metric_label in AGREEMENT_CURVE_METRICS
}
CAMERA_TRAP_AGREEMENT_THRESHOLD_RULES = {
    "psi_probability_agreement": {
        "direction": "ge",
        "threshold": 0.95,
        "threshold_label": f"$\\ge 0.95$",
    },
    "site_priority_spearman": {
        "direction": "ge",
        "threshold": 0.95,
        "threshold_label": f"$\\ge 0.95$",
    },
    "coef_ci_conclusion_agreement__occupancy": {
        "direction": "ge",
        "threshold": 0.95,
        "threshold_label": f"$\\ge 0.95$",
    },
    "coef_ci_conclusion_agreement__detection": {
        "direction": "ge",
        "threshold": 0.95,
        "threshold_label": f"$\\ge 0.95$",
    },
}
AGREEMENT_THRESHOLD_RULES_BY_DATASET = {
    "acoustic": {},
    "iwildcam": CAMERA_TRAP_AGREEMENT_THRESHOLD_RULES,
    "camera trap": CAMERA_TRAP_AGREEMENT_THRESHOLD_RULES,
    "camera_trap": CAMERA_TRAP_AGREEMENT_THRESHOLD_RULES,
}
AGREEMENT_THRESHOLD_TABLE_FAMILIES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Reviewed-only Bernoulli",
        [
            ("Random", "reviewed_random"),
            ("Max-score", "reviewed_max_score"),
            ("BALD", "reviewed_bald"),
            ("Target EIG", "reviewed_target_eig"),
        ],
    ),
    # (
    #     "Continuous-score",
    #     [
    #         ("Random", "adaptive_calibrated_random"),
    #         ("Max-score", "adaptive_calibrated_max_score"),
    #         ("BALD", "adaptive_calibrated_bald"),
    #     ],
    # ),
    (
        "Decoupled continuous-score",
        [
            ("Random", "cut_normal_calibrated_random"),
            ("Max-score", "cut_normal_calibrated_max_score"),
            ("BALD", "cut_normal_calibrated_bald"),
        ],
    ),
]
AGREEMENT_THRESHOLD_STANDALONE_POLICIES: list[tuple[str, str]] = [
    (r"\methoda\ (ours)", "cut_normal_calibrated_target_eig"),
]
AGREEMENT_THRESHOLD_METRIC_LABELS = {
    "psi_probability_agreement": "$\\overline{\\psi}$ agreement",
    "site_priority_spearman": "Site-rank Spearman $\\rho$",
    "coef_ci_conclusion_agreement__occupancy": "Occ. coeff. conclusions",
    "coef_ci_conclusion_agreement__detection": "Det. coeff. conclusions",
}
AGREEMENT_THRESHOLD_RENDER_REVIEW_FRACTIONS = (
    os.environ.get(
        "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_RENDER_REVIEW_FRACTIONS",
        "0",
    )
    != "0"
)
AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN = (
    os.environ.get(
        "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN",
        "1",
    )
    != "0"
)
AGREEMENT_THRESHOLD_SPEEDUP_COLUMN_LABEL = os.environ.get(
    "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_SPEEDUP_COLUMN_LABEL",
    "Speedup vs. CS max-score",
)
AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_EXPERIMENT_ID = os.environ.get(
    "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_EXPERIMENT_ID",
    "cut_normal_calibrated_max_score",
)
AGREEMENT_THRESHOLD_SPEEDUP_TARGET_EXPERIMENT_ID = os.environ.get(
    "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_SPEEDUP_TARGET_EXPERIMENT_ID",
    "cut_normal_calibrated_target_eig",
)
AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_LABEL = os.environ.get(
    "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_LABEL",
    "decoupled continuous-score max-score",
)
AGREEMENT_THRESHOLD_SPEEDUP_TARGET_LABEL = os.environ.get(
    "OCCUBED_ANALYSIS_AGREEMENT_THRESHOLD_SPEEDUP_TARGET_LABEL",
    r"\methoda{}",
)
AGREEMENT_THRESHOLD_POLICY_HEADERS = {
    ("Reviewed-only Bernoulli", "Random"): "Random",
    ("Reviewed-only Bernoulli", "Max-score"): "Max-score",
    ("Reviewed-only Bernoulli", "BALD"): "BALD",
    # ("Continuous-score", "Random"): "Random",
    # ("Continuous-score", "Max-score"): "Max-score",
    # ("Continuous-score", "BALD"): "BALD",
    ("Decoupled continuous-score", "Random"): "Random",
    ("Decoupled continuous-score", "Max-score"): "Max-score",
    ("Decoupled continuous-score", "BALD"): "BALD",
}
COMPUTATIONAL_COST_MODEL_FAMILIES = [
    "reviewed_bernoulli",
    "cut_normal_calibrated_cs",
    "original_cs",
]
COMPUTATIONAL_COST_MODEL_LABELS = {
    "reviewed_bernoulli": "Reviewed-only Bernoulli",
    "cut_normal_calibrated_cs": "Decoupled continuous-score",
    "original_cs": "Original continuous-score",
}
COMPUTATIONAL_COST_SELECTION_METHODS = [
    "random",
    "max_score",
    "posterior_predictive_uncertainty",
    "bald",
    "target_eig",
]
COMPUTATIONAL_COST_SELECTION_LABELS = {
    "random": "Random",
    "max_score": "Max-score",
    "posterior_predictive_uncertainty": "PPU",
    "bald": "BALD",
    "target_eig": "Target EIG",
}


# %%
def display_markdown(text: str) -> None:
    if Markdown is None:
        print(text)
        return
    display(Markdown(text))


def display_frame(title: str, frame: pd.DataFrame) -> None:
    display_markdown(f"### {title}")
    display(frame)


def metric_display_mean(metric_column: str, mean_value: object) -> float:
    value = safe_float(mean_value)
    if metric_column in RMSE_METRIC_COLUMNS:
        return float(np.sqrt(max(value, 0.0)))
    return value


def metric_display_confidence_interval(
    metric_column: str,
    mean_values: pd.Series | np.ndarray,
    sem_values: pd.Series | np.ndarray,
    *,
    z_score: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean_values, dtype=float)
    sem_array = np.asarray(sem_values, dtype=float)
    return metric_display_interval(
        metric_column,
        mean_array,
        mean_array - z_score * sem_array,
        mean_array + z_score * sem_array,
    )


def metric_display_interval(
    metric_column: str,
    center_values: pd.Series | np.ndarray,
    lower_values: pd.Series | np.ndarray,
    upper_values: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_array = np.asarray(center_values, dtype=float)
    lower_array = np.asarray(lower_values, dtype=float)
    upper_array = np.asarray(upper_values, dtype=float)
    if metric_column in RMSE_METRIC_COLUMNS:
        center_array = np.sqrt(np.clip(center_array, 0.0, None))
        lower_array = np.sqrt(np.clip(lower_array, 0.0, None))
        upper_array = np.sqrt(np.clip(upper_array, 0.0, None))

    if metric_column == "site_priority_spearman":
        center_array = np.clip(center_array, -1.0, 1.0)
        lower_array = np.clip(lower_array, -1.0, 1.0)
        upper_array = np.clip(upper_array, -1.0, 1.0)
    elif metric_column not in RMSE_METRIC_COLUMNS:
        center_array = np.clip(center_array, 0.0, 1.0)
        lower_array = np.clip(lower_array, 0.0, 1.0)
        upper_array = np.clip(upper_array, 0.0, 1.0)

    upper_array = np.maximum(upper_array, lower_array)
    return center_array, lower_array, upper_array


def finite_nanmax(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def finite_nanmin(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def finite_upper_median(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.sort(array[np.isfinite(array)])
    if finite.size == 0:
        return float("nan")
    return float(finite[finite.size // 2])


def slugify(value: object) -> str:
    slug = "".join(char.lower() if str(char).isalnum() else "_" for char in str(value))
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def resolve_sweep_dir(repo_root: Path, sweep_dir: str | Path) -> Path:
    candidate = Path(sweep_dir).expanduser()
    search_paths = [
        candidate,
        repo_root / candidate,
        Path.cwd() / candidate,
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve sweep directory: {sweep_dir}")


def infer_selection_method(experiment_id: str, manifest_row: dict[str, object]) -> str:
    value = str(manifest_row.get("selection_method", "")).strip()
    if value:
        return value
    base_experiment_id = (
        experiment_id[:-6] if experiment_id.endswith("__null") else experiment_id
    )
    if "dual_bald_max_balanced" in base_experiment_id:
        return "dual_balanced"
    if base_experiment_id.endswith("posterior_predictive_uncertainty"):
        return "posterior_predictive_uncertainty"
    if base_experiment_id.endswith("revealed_targeted_eig"):
        return "revealed_targeted_eig"
    if base_experiment_id.endswith(("target_eig", "targeted_eig")):
        return "target_eig"
    if base_experiment_id.endswith("epig"):
        return "epig"
    if base_experiment_id.endswith("max_score"):
        return "max_score"
    if base_experiment_id.endswith("bald"):
        return "bald"
    return "random"


def infer_model_family(experiment_id: str, manifest_row: dict[str, object]) -> str:
    value = str(manifest_row.get("model_family", "")).strip()
    if value:
        return value
    if experiment_id.startswith("original_occu_cs"):
        return "original_cs"
    if experiment_id.startswith("cut_normal_calibrated"):
        return "cut_normal_calibrated_cs"
    return (
        "calibrated_cs"
        if experiment_id.startswith("adaptive_calibrated")
        else "reviewed_bernoulli"
    )


def infer_covariate_variant(experiment_id: str, manifest_row: dict[str, object]) -> str:
    value = str(manifest_row.get("covariate_variant", "")).strip().lower()
    if value and value not in {"nan", "none"}:
        return "null" if "null" in value else value
    return "null" if experiment_id.endswith("__null") else "full"


def experiment_label(model_family: str, selection_method: str) -> str:
    family_label = MODEL_FAMILY_LABELS.get(model_family, model_family)
    selection_label = SELECTION_LABELS.get(
        selection_method, selection_method.replace("_", " ").title()
    )
    return f"{family_label} {selection_label}"


def experiment_sort_key(
    experiment_id: str, experiment_row: pd.Series | dict[str, object]
) -> tuple[int, int, str]:
    selection_method = str(experiment_row["selection_method"])
    model_family = str(experiment_row["model_family"])
    selection_rank = (
        SELECTION_ORDER.index(selection_method)
        if selection_method in SELECTION_ORDER
        else len(SELECTION_ORDER)
    )
    family_rank = (
        MODEL_FAMILY_ORDER.index(model_family)
        if model_family in MODEL_FAMILY_ORDER
        else len(MODEL_FAMILY_ORDER)
    )
    return selection_rank, family_rank, experiment_id


def ordered_values(
    values: pd.Series | pd.Index | list[object],
    preferred_order: list[str],
) -> list[str]:
    value_series = pd.Series(values, dtype=object)
    available = {
        str(value)
        for value in value_series.dropna().astype(str)
        if str(value).strip() and str(value).lower() != "nan"
    }
    ordered = [value for value in preferred_order if value in available]
    ordered.extend(sorted(available - set(preferred_order)))
    return ordered


def dataset_label(dataset_kind: object, sample_size: int | None = None) -> str:
    dataset_key = str(dataset_kind).strip().lower()
    label = DATASET_LABELS.get(
        dataset_key,
        str(dataset_kind).replace("_", " ").title(),
    )
    if sample_size is None:
        return label
    return f"{label} (k={sample_size})"


def dataset_species_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "species_key" not in frame.columns:
        return {}
    return {
        str(dataset_kind): int(species_count)
        for dataset_kind, species_count in frame.groupby("dataset_kind")["species_key"]
        .nunique()
        .dropna()
        .items()
    }


def dataset_sample_sizes_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    if "species_key" in frame.columns:
        return dataset_species_counts(frame)
    for count_column in ("dataset_species_count", "species_count"):
        if count_column not in frame.columns:
            continue
        count_frame = frame[["dataset_kind", count_column]].copy()
        count_frame[count_column] = pd.to_numeric(
            count_frame[count_column],
            errors="coerce",
        )
        count_frame = count_frame.loc[np.isfinite(count_frame[count_column])]
        if count_frame.empty:
            continue
        return {
            str(dataset_kind): int(species_count)
            for dataset_kind, species_count in count_frame.groupby("dataset_kind")[
                count_column
            ]
            .max()
            .items()
        }
    return {}


def maybe_save_figure(fig, output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(filename).with_suffix(".pdf")
    fig.patch.set_facecolor("none")
    for axis in fig.axes:
        axis.set_facecolor("none")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.savefig(
        path,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    return path


def maybe_save_text(
    text: str,
    output_dir: Path | None,
    filename: str,
) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(text, encoding="utf-8")
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


def resolved_parallel_workers(
    task_count: int,
    *,
    configured_workers: int = 0,
    default_cap: int = 8,
) -> int:
    if task_count <= 1:
        return 1
    if configured_workers > 0:
        return max(1, min(task_count, configured_workers))
    return max(1, min(task_count, default_cap, os.cpu_count() or 1))


def create_process_pool(max_workers: int) -> ProcessPoolExecutor | None:
    try:
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("fork"),
        )
    except ValueError:
        return None


# TODO: remove?
def maybe_smooth_aggregated_curve(
    review_counts: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
) -> np.ndarray:
    x_values = np.asarray(review_counts, dtype=float)
    y_values = np.asarray(values, dtype=float)
    if y_values.size < AGGREGATED_SMOOTHING_WINDOW:
        return y_values

    value_range = float(np.nanmax(y_values) - np.nanmin(y_values))
    if not np.isfinite(value_range) or value_range <= 0:
        return y_values

    deltas = np.diff(y_values)
    if deltas.size == 0:
        return y_values

    roughness = float(np.nanstd(deltas) / value_range)
    max_jump = float(np.nanmax(np.abs(deltas)) / value_range)
    x_span = float(np.nanmax(x_values) - np.nanmin(x_values)) if x_values.size else 0.0
    if x_span >= AGGREGATED_SMOOTHING_LONG_HORIZON_MIN_SPAN:
        window = min(AGGREGATED_SMOOTHING_LONG_WINDOW, y_values.size)
        roughness_threshold = AGGREGATED_SMOOTHING_LONG_ROUGHNESS_THRESHOLD
        max_jump_threshold = AGGREGATED_SMOOTHING_LONG_MAX_JUMP_THRESHOLD
    else:
        window = min(AGGREGATED_SMOOTHING_WINDOW, y_values.size)
        roughness_threshold = AGGREGATED_SMOOTHING_ROUGHNESS_THRESHOLD
        max_jump_threshold = AGGREGATED_SMOOTHING_MAX_JUMP_THRESHOLD
    if roughness < roughness_threshold and max_jump < max_jump_threshold:
        return y_values

    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y_values

    series = pd.Series(y_values)
    y_values_smoothed = (
        series.rolling(window, center=True, min_periods=1)
        .median()
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    ).copy()

    # make sure first and last values are unchanged
    if y_values_smoothed.size > 0:
        y_values_smoothed[0] = y_values[0]
        y_values_smoothed[-1] = y_values[-1]

    return y_values_smoothed


def centered_rolling_trend(
    values: pd.Series | np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Return a centered rolling trend while keeping the raw curve plottable."""
    y_values = np.asarray(values, dtype=float)
    if y_values.size < 3 or window <= 1:
        return y_values

    trend_window = min(int(window), y_values.size)
    if trend_window % 2 == 0:
        trend_window -= 1
    if trend_window < 3:
        return y_values

    min_periods = max(1, trend_window // 2)
    return (
        pd.Series(y_values)
        .rolling(trend_window, center=True, min_periods=min_periods)
        .median()
        .rolling(trend_window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def summarize_mean_oracle_distance(
    aggregate_frame: pd.DataFrame,
    *,
    review_count: int,
    experiment_id: str,
) -> float:
    subset = aggregate_frame[
        (aggregate_frame["review_count"] == review_count)
        & (aggregate_frame["experiment_id"] == experiment_id)
    ]
    if subset.empty:
        return float("nan")
    return float(subset["mean_value"].iloc[0])


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalized_max_species_by_dataset(
    max_species_by_dataset: dict[str, int | None],
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for dataset_kind, max_species_count in max_species_by_dataset.items():
        if max_species_count is None:
            continue
        dataset_key = str(dataset_kind).strip().lower()
        if not dataset_key:
            continue
        count = int(max_species_count)
        if count < 0:
            raise ValueError(
                f"Maximum species count for dataset `{dataset_kind}` must be "
                f"nonnegative, got {max_species_count!r}."
            )
        if count > 0:
            limits[dataset_key] = count
    return limits


def default_top_species_keys_by_dataset(
    species_frame: pd.DataFrame,
    max_species_by_dataset: dict[str, int | None],
) -> dict[str, set[str]]:
    if species_frame.empty:
        return {}
    limits = normalized_max_species_by_dataset(max_species_by_dataset)
    if not limits:
        return {}

    selected_species_keys: dict[str, set[str]] = {}
    for dataset_kind in ordered_values(species_frame["dataset_kind"], DATASET_ORDER):
        dataset_key = str(dataset_kind).strip().lower()
        max_species_count = limits.get(dataset_key)
        if max_species_count is None:
            continue
        species_by_dataset = default_species_by_dataset(
            top_k=max_species_count,
            dataset_kinds=[dataset_key],
        )
        selected_species_keys[dataset_key] = {
            f"{dataset_key}::{species_name}"
            for species_name in species_by_dataset.get(dataset_key, [])
        }
    return selected_species_keys


def limit_frame_species_by_dataset(
    frame: pd.DataFrame,
    selected_species_keys_by_dataset: dict[str, set[str]],
) -> pd.DataFrame:
    if frame.empty or not selected_species_keys_by_dataset:
        return frame
    if "dataset_kind" not in frame.columns or "species_key" not in frame.columns:
        return frame

    limited_datasets = set(selected_species_keys_by_dataset)
    selected_species_keys = set().union(*selected_species_keys_by_dataset.values())
    dataset_keys = frame["dataset_kind"].astype("string").str.lower()
    keep_mask = ~dataset_keys.isin(limited_datasets) | frame["species_key"].isin(
        selected_species_keys
    )
    return frame.loc[keep_mask].reset_index(drop=True).copy()


def limit_inventory_species_by_dataset(
    species_frame: pd.DataFrame,
    experiment_frame: pd.DataFrame,
    trace_frame: pd.DataFrame,
    max_species_by_dataset: dict[str, int | None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_species_keys_by_dataset = default_top_species_keys_by_dataset(
        species_frame,
        max_species_by_dataset,
    )
    if not selected_species_keys_by_dataset:
        return species_frame, experiment_frame, trace_frame
    return (
        limit_frame_species_by_dataset(
            species_frame,
            selected_species_keys_by_dataset,
        ),
        limit_frame_species_by_dataset(
            experiment_frame,
            selected_species_keys_by_dataset,
        ),
        limit_frame_species_by_dataset(
            trace_frame,
            selected_species_keys_by_dataset,
        ),
    )


def inverse_logit(value: object) -> float:
    float_value = safe_float(value)
    if not np.isfinite(float_value):
        return float("nan")
    return float(1.0 / (1.0 + np.exp(-np.clip(float_value, -709.0, 709.0))))


def first_nonempty_string(values: pd.Series) -> str:
    for value in values.dropna().astype(str):
        value = value.strip()
        if value and value.lower() != "nan":
            return value
    return ""


def parse_numeric_vector(value: object, separator: str = "|") -> np.ndarray | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    tokens = [token.strip() for token in text.split(separator)]
    try:
        return np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError:
        return None


def load_parameter_mean_vector(
    path: str | Path,
    parameter_name: str,
) -> np.ndarray | None:
    for name, parameter_stats in iter_parameter_summary_arrays(
        path,
        parameters=[parameter_name],
        stats=("mean",),
    ):
        if name != parameter_name:
            continue
        values = parameter_stats.get("mean")
        if values is not None:
            return np.asarray(values, dtype=float).reshape(-1)
    return None


def resolve_species_artifact_dir(
    sweep_dir: Path | None,
    *,
    species_index: object,
    dataset_kind: object,
    species: object,
) -> Path | None:
    if sweep_dir is None:
        return None
    index_value = safe_float(species_index)
    if not np.isfinite(index_value):
        return None
    candidate = (
        Path(sweep_dir)
        / f"{int(index_value):03d}_{str(dataset_kind).lower()}__{slugify(species)}"
    )
    return candidate if candidate.exists() else None


@lru_cache(maxsize=1)
def acoustic_species_common_names() -> dict[str, str]:
    root = find_repo_root()
    for relative_path in ACOUSTIC_SPECIES_METADATA_PATHS:
        path = root / relative_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"AOU_Code", "Common_Name"}.issubset(frame.columns):
            return {
                str(row.AOU_Code).strip().upper(): str(row.Common_Name).strip()
                for row in frame.itertuples(index=False)
                if str(row.AOU_Code).strip() and str(row.Common_Name).strip()
            }
    return {}


def species_name_lookup_key(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().lower())


def sentence_case_label(value: object) -> str:
    label = str(value).replace("_", " ").strip()
    return f"{label[:1].upper()}{label[1:]}" if label else ""


@lru_cache(maxsize=1)
def iwildcam_species_common_names() -> dict[str, str]:
    path = find_repo_root() / IWILDCAM_SPECIES_METADATA_PATH
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "speciesnet_common_name" not in frame.columns:
        return {}

    source_columns = [
        "category_name",
        "normalized_category_name",
        "speciesnet_scientific_name",
        "speciesnet_common_name",
        "match_name",
    ]
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        common_name = sentence_case_label(getattr(row, "speciesnet_common_name", ""))
        if not common_name:
            continue
        for column in source_columns:
            if column not in frame.columns:
                continue
            key = species_name_lookup_key(getattr(row, column, ""))
            if key:
                mapping.setdefault(key, common_name)
    return mapping


def species_display_label(dataset_kind: object, species: object) -> str:
    dataset_key = str(dataset_kind).strip().lower()
    if dataset_key == "acoustic":
        species_code = str(species).strip().upper()
        return acoustic_species_common_names().get(species_code, species_code)
    if dataset_key == "iwildcam":
        species_key = species_name_lookup_key(species)
        label = iwildcam_species_common_names().get(species_key)
        if label:
            return label
    return sentence_case_label(species)


def species_dataset_distribution_frame(
    species_frame: pd.DataFrame,
    sweep_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if species_frame.empty:
        return pd.DataFrame()

    for species_row in species_frame.itertuples(index=False):
        dataset_kind = getattr(species_row, "dataset_kind", "")
        species = getattr(species_row, "species", "")
        species_dir = resolve_species_artifact_dir(
            sweep_dir,
            species_index=getattr(species_row, "species_index", np.nan),
            dataset_kind=dataset_kind,
            species=species,
        )
        if species_dir is None:
            continue

        true_labels_path = species_dir / "inputs" / "true_labels.npy"
        reviewable_mask_path = species_dir / "dataset_metadata" / "reviewable_mask.npy"
        if not true_labels_path.exists() or not reviewable_mask_path.exists():
            continue

        true_labels = np.asarray(np.load(true_labels_path), dtype=float)
        reviewable_mask = np.asarray(np.load(reviewable_mask_path), dtype=bool)
        if true_labels.ndim == reviewable_mask.ndim + 1 and true_labels.shape[0] == 1:
            true_labels = true_labels[0]
        if true_labels.shape != reviewable_mask.shape:
            continue

        valid = reviewable_mask & np.isfinite(true_labels)
        if not np.any(valid):
            continue

        observation_axes = tuple(range(1, valid.ndim))
        site_count = (
            int(valid.any(axis=observation_axes).sum())
            if observation_axes
            else int(np.any(valid))
        )
        rows.append(
            {
                "dataset_kind": str(dataset_kind),
                "species_key": getattr(species_row, "species_key"),
                "species": str(species),
                "species_label": species_display_label(dataset_kind, species),
                "n_sites": site_count,
                "n_samples": int(np.sum(valid)),
                "n_positive_samples": int(np.sum(true_labels[valid] > 0.0)),
                "max_replicates_per_site": (
                    int(true_labels.shape[-1]) if true_labels.ndim else int(valid.size)
                ),
            }
        )

    return pd.DataFrame(rows)


def species_dataset_distribution_summary(
    distribution_frame: pd.DataFrame,
) -> pd.DataFrame:
    if distribution_frame.empty:
        return pd.DataFrame()
    summary = (
        distribution_frame.groupby("dataset_kind")
        .agg(
            n_species=("species_key", "nunique"),
            sites_min=("n_sites", "min"),
            sites_median=("n_sites", "median"),
            sites_max=("n_sites", "max"),
            samples_min=("n_samples", "min"),
            samples_median=("n_samples", "median"),
            samples_max=("n_samples", "max"),
            samples_total=("n_samples", "sum"),
            positives_min=("n_positive_samples", "min"),
            positives_median=("n_positive_samples", "median"),
            positives_max=("n_positive_samples", "max"),
            positives_total=("n_positive_samples", "sum"),
            max_replicates_min=("max_replicates_per_site", "min"),
            max_replicates_median=("max_replicates_per_site", "median"),
            max_replicates_max=("max_replicates_per_site", "max"),
        )
        .reset_index()
    )
    summary["dataset"] = summary["dataset_kind"].map(dataset_label)
    return summary[
        [
            "dataset",
            "n_species",
            "sites_min",
            "sites_median",
            "sites_max",
            "samples_min",
            "samples_median",
            "samples_max",
            "samples_total",
            "positives_min",
            "positives_median",
            "positives_max",
            "positives_total",
            "max_replicates_min",
            "max_replicates_median",
            "max_replicates_max",
        ]
    ]


def plot_dataset_class_distribution(
    distribution_frame: pd.DataFrame,
    *,
    filename: str = DATASET_CLASS_DISTRIBUTION_FILENAME,
) -> None:
    if distribution_frame.empty:
        print("No dataset class-distribution data available.")
        return

    datasets = ordered_values(distribution_frame["dataset_kind"], DATASET_ORDER)
    if not datasets:
        print("No dataset class-distribution data available.")
        return

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(FULL_WIDTH_IN, 6.8),
        squeeze=False,
    )

    for row_index, dataset_kind in enumerate(datasets):
        axis = axes[row_index, 0]
        subset = (
            distribution_frame.loc[distribution_frame["dataset_kind"] == dataset_kind]
            .sort_values(
                ["n_positive_samples", "species_label"], ascending=[False, True]
            )
            .reset_index(drop=True)
        )
        x_positions = np.arange(len(subset))
        axis.bar(
            x_positions,
            subset["n_positive_samples"],
            color=DATASET_COLORS.get(dataset_kind, "#4C78A8"),
            alpha=0.88,
            linewidth=0.0,
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            subset["species_label"],
            rotation=45,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        axis.text(
            0.995,
            0.965,
            f"{dataset_label(dataset_kind)} ({len(subset)} species)",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=AXIS_TITLE_FONTSIZE,
        )
        axis.set_ylabel("Positive replicate labels")
        axis.grid(axis="y", alpha=0.25)
        axis.grid(axis="x", visible=False)
        axis.tick_params(
            axis="x",
            labelsize=5.8 if row_index < len(datasets) - 1 else 6.5,
            pad=1.0,
        )

    apply_tight_layout(
        fig,
        rect=(0.0, 0.02, 1.0, 0.98),
        pad=0.01,
        w_pad=0.02,
        h_pad=0.02,
        wspace=0.20,
        hspace=0.82,
    )
    saved_path = maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)
    if saved_path is not None:
        print(f"Saved {saved_path.relative_to(repo_root)}")


def load_oracle_psi_values_from_artifact(
    sweep_dir: Path | None,
    *,
    species_index: object,
    dataset_kind: object,
    species: object,
) -> str:
    species_dir = resolve_species_artifact_dir(
        sweep_dir,
        species_index=species_index,
        dataset_kind=dataset_kind,
        species=species,
    )
    if species_dir is None:
        return ""
    values = load_parameter_mean_vector(
        species_dir / "oracle" / "parameter_summary.npz",
        "psi",
    )
    if values is None:
        return ""
    return "|".join(
        "NaN" if np.isnan(value) else f"{float(value):.17g}"
        for value in np.asarray(values, dtype=float).reshape(-1)
    )


def limit_trace_reviews_by_dataset(
    trace_frame: pd.DataFrame,
    max_review_count_by_dataset: dict[str, int | None],
) -> pd.DataFrame:
    if trace_frame.empty or not max_review_count_by_dataset:
        return trace_frame

    limits = {
        str(dataset_kind).strip().lower(): int(max_review_count)
        for dataset_kind, max_review_count in max_review_count_by_dataset.items()
        if max_review_count is not None
    }
    if not limits:
        return trace_frame

    dataset_limits = (
        trace_frame["dataset_kind"].astype("string").str.lower().map(limits)
    )
    keep_mask = dataset_limits.isna() | (trace_frame["review_count"] <= dataset_limits)
    return trace_frame.loc[keep_mask].reset_index(drop=True).copy()


# %% [markdown]
# ## Load the sweep inventory


# %%
def resolve_results_csv_path(sweep_dir: Path, results_csv: str | Path) -> Path:
    candidate = Path(results_csv).expanduser()
    search_paths = [
        candidate,
        sweep_dir / candidate,
        Path.cwd() / candidate,
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve combined results CSV: {results_csv}")


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
    frame: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> pd.DataFrame:
    frame = frame.copy()
    for column in list(frame.columns):
        for prefix in prefixes:
            if not column.startswith(prefix):
                continue
            alias = column[len(prefix) :]
            if alias and alias not in frame.columns:
                frame[alias] = frame[column]
    return frame


def normalize_exported_results_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_prefixed_aliases(
        results_frame,
        prefixes=("trace__", "agreement__"),
    )
    frame["dataset_kind"] = coalesce_text_columns(
        frame,
        ["dataset_kind", "run__dataset_kind", "dataset__dataset_kind"],
    ).str.lower()
    frame["dataset_name"] = coalesce_text_columns(
        frame,
        ["run__dataset_name", "dataset__dataset_name"],
    )
    frame["species"] = coalesce_text_columns(
        frame,
        ["species", "run__target_species", "dataset__target_species"],
    )
    frame["target_label"] = coalesce_text_columns(
        frame,
        ["target_label", "run__target_label", "dataset__target_label"],
    )
    frame["species_key"] = frame["dataset_kind"] + "::" + frame["species"]
    frame["selection_method"] = coalesce_text_columns(
        frame,
        ["experiment__selection_method"],
    )
    frame["model_family"] = coalesce_text_columns(
        frame,
        ["experiment__model_family"],
    )
    frame["covariate_variant"] = coalesce_text_columns(
        frame,
        ["experiment__covariate_variant", "experiment__oracle_variant"],
    )

    for row_index, row in frame.iterrows():
        experiment_id = str(row.get("experiment_id", "")).strip()
        manifest_row = row.to_dict()
        if not str(row.get("selection_method", "")).strip():
            frame.at[row_index, "selection_method"] = infer_selection_method(
                experiment_id,
                manifest_row,
            )
        if not str(row.get("model_family", "")).strip():
            frame.at[row_index, "model_family"] = infer_model_family(
                experiment_id,
                manifest_row,
            )
        covariate_variant = str(row.get("covariate_variant", "")).strip().lower()
        if not covariate_variant or covariate_variant in {"nan", "none"}:
            frame.at[row_index, "covariate_variant"] = infer_covariate_variant(
                experiment_id,
                manifest_row,
            )
        else:
            frame.at[row_index, "covariate_variant"] = (
                "null" if "null" in covariate_variant else covariate_variant
            )

    frame["plot_label"] = [
        experiment_label(model_family, selection_method)
        for model_family, selection_method in zip(
            frame["model_family"],
            frame["selection_method"],
        )
    ]

    numeric_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "row_kind",
            "sweep_name",
            "dataset_kind",
            "dataset_name",
            "species",
            "target_label",
            "species_key",
            "experiment_id",
            "selection_method",
            "model_family",
            "covariate_variant",
            "plot_label",
            "experiment__display_name",
            "experiment__model_label",
        }
    ]
    for column in numeric_columns:
        nonempty = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        if not nonempty.any():
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.loc[nonempty].notna().all():
            frame[column] = converted
    return frame


def load_sweep_inventory(
    results_csv_path: Path,
    *,
    dataset_filter: list[str] | None = None,
    sweep_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_frame = normalize_exported_results_frame(
        pd.read_csv(results_csv_path, low_memory=False)
    )
    if dataset_filter:
        results_frame = results_frame.loc[
            results_frame["dataset_kind"].isin(dataset_filter)
        ].copy()

    species_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []
    valid_experiment = (
        results_frame["experiment_id"].astype("string").fillna("").str.strip()
    )
    experiment_source = results_frame.loc[valid_experiment.ne("")].copy()
    if not experiment_source.empty:
        experiment_source = experiment_source.sort_values(
            ["dataset_kind", "species", "experiment_id", "step_index"],
            na_position="last",
        )
        for record in (
            experiment_source.drop_duplicates(["species_key", "experiment_id"])
            .sort_values(["dataset_kind", "species", "experiment_id"])
            .itertuples(index=False)
        ):
            experiment_rows.append(
                {
                    "species_key": record.species_key,
                    "species": record.species,
                    "dataset_kind": record.dataset_kind,
                    "dataset_name": record.dataset_name,
                    "experiment_id": record.experiment_id,
                    "selection_method": record.selection_method,
                    "model_family": record.model_family,
                    "covariate_variant": record.covariate_variant,
                    "plot_label": record.plot_label,
                    "final_review_count": safe_float(
                        getattr(record, "experiment__final_review_count", np.nan)
                    ),
                    "final_mean_oracle_distance": safe_float(
                        getattr(
                            record, "experiment__final_mean_oracle_distance", np.nan
                        )
                    ),
                }
            )

    for (_species_index, _species_key), group in results_frame.groupby(
        ["species_index", "species_key"],
        dropna=False,
        sort=True,
    ):
        first = group.iloc[0]
        oracle_psi_mean_values = (
            first_nonempty_string(group["oracle__psi_mean_values"])
            if "oracle__psi_mean_values" in group.columns
            else ""
        )
        if not oracle_psi_mean_values:
            oracle_psi_mean_values = load_oracle_psi_values_from_artifact(
                sweep_dir,
                species_index=first.get("species_index"),
                dataset_kind=first["dataset_kind"],
                species=first["species"],
            )
        available_experiment_ids = tuple(
            sorted(
                experiment_id
                for experiment_id in group["experiment_id"]
                .dropna()
                .astype(str)
                .unique()
                if experiment_id.strip() and experiment_id.strip().lower() != "nan"
            )
        )
        is_available = bool(
            available_experiment_ids
            or str(first.get("run__dataset_name", "")).strip()
            or np.isfinite(safe_float(first.get("dataset__n_reviewable")))
        )
        species_rows.append(
            {
                "species_index": int(safe_float(first.get("species_index"))),
                "species_key": first["species_key"],
                "dataset_kind": first["dataset_kind"],
                "dataset_name": first["dataset_name"],
                "species": first["species"],
                "target_label": first["target_label"],
                "is_available": is_available,
                "n_available_experiments": len(available_experiment_ids),
                "available_experiment_ids": available_experiment_ids,
                "n_sites": safe_float(first.get("dataset__n_sites")),
                "n_reviewable": safe_float(first.get("dataset__n_reviewable")),
                "num_total_labels": safe_float(first.get("oracle__num_total_labels")),
                "num_positive_labels": safe_float(
                    first.get("oracle__num_positive_labels")
                ),
                "oracle_psi_mean_values": oracle_psi_mean_values,
            }
        )

    species_frame = pd.DataFrame(species_rows)
    experiment_frame = pd.DataFrame(experiment_rows)
    trace_frame = results_frame.loc[
        results_frame["row_kind"].astype("string").fillna("").eq("step")
    ].copy()

    if not species_frame.empty:
        available_sets = {
            row.species_key: set(row.available_experiment_ids)
            for row in species_frame.itertuples()
            if row.is_available
        }
        species_frame["is_complete_full"] = species_frame["species_key"].map(
            lambda key: FULL_EXPERIMENT_IDS
            and set(FULL_EXPERIMENT_IDS).issubset(available_sets.get(key, set()))
        )
        species_frame["is_complete_null"] = species_frame["species_key"].map(
            lambda key: NULL_EXPERIMENT_IDS
            and set(NULL_EXPERIMENT_IDS).issubset(available_sets.get(key, set()))
        )

    if not trace_frame.empty:
        trace_frame["step_index"] = trace_frame["step_index"].astype(int)
        trace_frame["review_count"] = trace_frame["review_count"].astype(int)
        if "selected_count" in trace_frame.columns:
            trace_frame["selected_count"] = trace_frame["selected_count"].astype(int)
        trace_frame = trace_frame.sort_values(
            ["dataset_kind", "species", "experiment_id", "review_count"]
        ).reset_index(drop=True)
        if (
            not species_frame.empty
            and "oracle_psi_mean_values" in species_frame.columns
        ):
            oracle_psi_by_species = species_frame.set_index("species_key")[
                "oracle_psi_mean_values"
            ].to_dict()
            trace_frame["oracle_psi_mean_values"] = trace_frame["species_key"].map(
                oracle_psi_by_species
            )

    return species_frame, experiment_frame, trace_frame


def load_dataset_only_species_frame(
    sweep_dir: Path,
    max_species_by_dataset: dict[str, int | None],
    dataset_filter: list[str] | None = None,
) -> pd.DataFrame:
    limits = normalized_max_species_by_dataset(max_species_by_dataset)
    dataset_filter_set = set(dataset_filter or [])
    counts_by_dataset: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for species_dir in sorted(Path(sweep_dir).iterdir()):
        if not species_dir.is_dir():
            continue
        match = re.match(
            r"^(?P<species_index>\d+)_(?P<dataset_kind>.+?)__(?P<species_slug>.+)$",
            species_dir.name,
        )
        if match is None:
            continue
        scalars_path = species_dir / "dataset_metadata" / "scalars.csv"
        if not scalars_path.exists():
            continue

        dataset_kind = match.group("dataset_kind").lower()
        if dataset_filter_set and dataset_kind not in dataset_filter_set:
            continue
        species_limit = limits.get(dataset_kind)
        if (
            species_limit is not None
            and counts_by_dataset.get(dataset_kind, 0) >= species_limit
        ):
            continue

        scalars = pd.read_csv(scalars_path).iloc[0].to_dict()
        species = str(scalars.get("target_species") or match.group("species_slug"))
        rows.append(
            {
                "species_index": int(match.group("species_index")),
                "species_key": f"{dataset_kind}::{species}",
                "dataset_kind": dataset_kind,
                "dataset_name": str(scalars.get("dataset_name", "")),
                "species": species,
                "target_label": str(scalars.get("target_label", species)),
                "is_available": True,
                "n_available_experiments": 0,
                "available_experiment_ids": tuple(),
                "n_sites": safe_float(scalars.get("n_sites")),
                "n_reviewable": safe_float(scalars.get("n_reviewable")),
                "num_total_labels": float("nan"),
                "num_positive_labels": float("nan"),
                "oracle_psi_mean_values": "",
                "is_complete_full": False,
                "is_complete_null": False,
            }
        )
        counts_by_dataset[dataset_kind] = counts_by_dataset.get(dataset_kind, 0) + 1

    return pd.DataFrame(rows)


repo_root = find_repo_root()
dataset_filter = (
    None
    if DATASET_FILTER is None
    else [str(value).strip().lower() for value in np.atleast_1d(DATASET_FILTER)]
)
sweep_dir = resolve_sweep_dir(repo_root, SWEEP_DIR)
combined_results_path = (
    None if DATASET_ONLY else resolve_results_csv_path(sweep_dir, COMBINED_RESULTS_CSV)
)
figure_output_dir = (repo_root / FIGURE_OUTPUT_DIR) if SAVE_FIGURES else None
table_output_dir = (repo_root / TABLE_OUTPUT_DIR) if SAVE_TABLES else None
parameter_figure_output_dir = (
    figure_output_dir / PARAMETER_TRAJECTORY_SUBDIR
    if figure_output_dir is not None
    else None
)

if DATASET_ONLY:
    species_frame = load_dataset_only_species_frame(
        sweep_dir,
        MAX_SPECIES_BY_DATASET,
        dataset_filter=dataset_filter,
    )
    experiment_frame = pd.DataFrame()
    trace_frame = pd.DataFrame()
else:
    species_frame, experiment_frame, trace_frame = load_sweep_inventory(
        combined_results_path,
        dataset_filter=dataset_filter,
        sweep_dir=sweep_dir,
    )
    species_frame, experiment_frame, trace_frame = limit_inventory_species_by_dataset(
        species_frame,
        experiment_frame,
        trace_frame,
        MAX_SPECIES_BY_DATASET,
    )
    trace_frame = limit_trace_reviews_by_dataset(
        trace_frame,
        MAX_REVIEW_COUNT_BY_DATASET,
    )

print(f"Using sweep directory: {sweep_dir}")
if combined_results_path is not None:
    print(f"Using combined results CSV: {combined_results_path}")
else:
    print("Dataset-only mode: skipping combined results CSV load.")
print(f"Saving figures: {SAVE_FIGURES}")
if figure_output_dir is not None:
    print(f"Figure output directory: {figure_output_dir}")
print(f"Saving tables: {SAVE_TABLES}")
if table_output_dir is not None:
    print(f"Table output directory: {table_output_dir}")
species_limits = normalized_max_species_by_dataset(MAX_SPECIES_BY_DATASET)
if species_limits:
    species_limit_text = ", ".join(
        f"{dataset_kind} <= {max_species_count}"
        for dataset_kind, max_species_count in sorted(species_limits.items())
    )
    print(
        "Limiting species by dataset using submitit species ordering: "
        f"{species_limit_text}"
    )
if MAX_AGREEMENT_SPECIES_PER_DATASET > 0:
    print(
        "Limiting agreement analysis to first "
        f"{MAX_AGREEMENT_SPECIES_PER_DATASET} species per dataset."
    )
if MAX_REVIEW_COUNT_BY_DATASET:
    review_limit_text = ", ".join(
        f"{dataset_kind} <= {max_review_count}"
        for dataset_kind, max_review_count in sorted(
            MAX_REVIEW_COUNT_BY_DATASET.items()
        )
        if max_review_count is not None
    )
    print(f"Limiting review counts by dataset: {review_limit_text}")

display_frame(
    "Sweep coverage",
    species_frame.groupby("dataset_kind")
    .agg(
        submitted_species=("species_key", "nunique"),
        completed_species=("is_available", "sum"),
        full_model_complete=("is_complete_full", "sum"),
        null_model_complete=("is_complete_null", "sum"),
    )
    .sort_index(),
)

display_frame(
    "Available experiments per completed species",
    species_frame.loc[species_frame["is_available"]]
    .groupby(["dataset_kind", "n_available_experiments"])["species_key"]
    .count()
    .unstack(fill_value=0)
    .sort_index(),
)

dataset_distribution_frame = species_dataset_distribution_frame(
    species_frame, sweep_dir
)
display_frame(
    "Dataset class-distribution summary",
    species_dataset_distribution_summary(dataset_distribution_frame).round(1),
)
plot_dataset_class_distribution(dataset_distribution_frame)
if DATASET_ONLY:
    print("Dataset-only analysis requested; stopping after dataset summaries.")
    raise SystemExit(0)


# %% [markdown]
# The main cross-experiment summaries below use only species with all required
# experiment variants for a given comparison.


# %%
def required_experiment_ids(covariate_variant: str) -> list[str]:
    if covariate_variant == "null":
        return list(NULL_EXPERIMENT_IDS)
    return list(FULL_EXPERIMENT_IDS)


def oracle_distance_plot_experiment_ids(covariate_variant: str) -> list[str]:
    if covariate_variant == "null":
        return list(NULL_ORACLE_DISTANCE_EXPERIMENT_IDS)
    return list(ORACLE_DISTANCE_EXPERIMENT_IDS)


def agreement_experiment_ids(covariate_variant: str) -> list[str]:
    if covariate_variant == "null":
        return list(NULL_AGREEMENT_EXPERIMENT_IDS)
    return list(AGREEMENT_EXPERIMENT_IDS)


def all_agreement_candidate_experiment_ids(covariate_variant: str) -> list[str]:
    if trace_frame.empty:
        return []
    return sorted(
        experiment_id
        for experiment_id in trace_frame.loc[
            trace_frame["covariate_variant"] == covariate_variant,
            "experiment_id",
        ]
        .dropna()
        .astype(str)
        .unique()
        if experiment_id.strip()
    )


def covariate_variant_experiment_id(
    base_experiment_id: str,
    covariate_variant: str,
) -> str:
    if covariate_variant == "null" and not base_experiment_id.endswith("__null"):
        return f"{base_experiment_id}__null"
    if covariate_variant != "null" and base_experiment_id.endswith("__null"):
        return base_experiment_id[:-6]
    return base_experiment_id


def covariate_variant_experiment_ids(
    base_experiment_ids: list[str],
    covariate_variant: str,
) -> list[str]:
    return [
        covariate_variant_experiment_id(experiment_id, covariate_variant)
        for experiment_id in base_experiment_ids
    ]


def continuous_score_random_candidates(covariate_variant: str) -> list[str]:
    return covariate_variant_experiment_ids(
        ["adaptive_calibrated_random", "cut_normal_calibrated_random"],
        covariate_variant,
    )


def continuous_score_bald_candidates(covariate_variant: str) -> list[str]:
    return covariate_variant_experiment_ids(
        ["adaptive_calibrated_bald", "cut_normal_calibrated_bald"],
        covariate_variant,
    )


def continuous_score_target_eig_candidates(covariate_variant: str) -> list[str]:
    return covariate_variant_experiment_ids(
        ["adaptive_calibrated_target_eig", "cut_normal_calibrated_target_eig"],
        covariate_variant,
    )


def continuous_score_max_score_candidates(covariate_variant: str) -> list[str]:
    return covariate_variant_experiment_ids(
        ["adaptive_calibrated_max_score", "cut_normal_calibrated_max_score"],
        covariate_variant,
    )


def reviewed_random_experiment_id(covariate_variant: str) -> str:
    return covariate_variant_experiment_id("reviewed_random", covariate_variant)


def reviewed_bald_experiment_id(covariate_variant: str) -> str:
    return covariate_variant_experiment_id("reviewed_bald", covariate_variant)


def complete_species_keys(
    species_frame: pd.DataFrame,
    covariate_variant: str,
    experiment_ids: list[str] | None = None,
) -> list[str]:
    if experiment_ids is not None:
        requested_experiment_ids = {
            str(experiment_id)
            for experiment_id in experiment_ids
            if str(experiment_id).strip()
        }
        if not requested_experiment_ids:
            return []

        def has_any_requested_experiment(available_experiment_ids: object) -> bool:
            return bool(
                requested_experiment_ids.intersection(
                    str(experiment_id)
                    for experiment_id in available_experiment_ids
                    if str(experiment_id).strip()
                )
            )

        subset = species_frame.loc[
            species_frame["available_experiment_ids"].map(has_any_requested_experiment)
        ]
        return sorted(subset["species_key"].tolist())

    column = "is_complete_null" if covariate_variant == "null" else "is_complete_full"
    subset = species_frame.loc[species_frame[column].fillna(False)]
    return sorted(subset["species_key"].tolist())


def filtered_trace(
    trace_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    species_keys: list[str] | None = None,
    experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    frame = trace_frame.loc[
        trace_frame["covariate_variant"] == covariate_variant
    ].copy()
    if species_keys is not None:
        frame = frame.loc[frame["species_key"].isin(species_keys)]
    if experiment_ids is not None:
        frame = frame.loc[frame["experiment_id"].isin(experiment_ids)]
    return frame


def filtered_experiment_frame(
    experiment_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    species_keys: list[str] | None = None,
    experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    frame = experiment_frame.loc[
        experiment_frame["covariate_variant"] == covariate_variant
    ].copy()
    if species_keys is not None:
        frame = frame.loc[frame["species_key"].isin(species_keys)]
    if experiment_ids is not None:
        frame = frame.loc[frame["experiment_id"].isin(experiment_ids)]
    return frame


def aggregate_summary_curves(
    frame: pd.DataFrame,
    *,
    metric_columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    value_frames = []
    for metric_column in metric_columns:
        if metric_column not in frame.columns:
            continue
        metric_input = frame.copy()
        metric_input[metric_column] = pd.to_numeric(
            metric_input[metric_column],
            errors="coerce",
        )
        valid_metric_input = metric_input.loc[np.isfinite(metric_input[metric_column])]
        species_counts = dataset_species_counts(valid_metric_input)
        metric_frame = (
            metric_input.groupby(
                [
                    "dataset_kind",
                    "experiment_id",
                    "selection_method",
                    "model_family",
                    "plot_label",
                    "review_count",
                ]
            )
            .agg(
                mean_value=(metric_column, "mean"),
                median_value=(metric_column, "median"),
                q25_value=(metric_column, lambda values: values.quantile(0.25)),
                q75_value=(metric_column, lambda values: values.quantile(0.75)),
                std_value=(metric_column, "std"),
                species_count=(metric_column, "count"),
            )
            .reset_index()
        )
        metric_frame["dataset_species_count"] = (
            metric_frame["dataset_kind"].astype(str).map(species_counts)
        )
        metric_frame["metric"] = metric_column
        metric_frame["sem_value"] = metric_frame["std_value"] / np.sqrt(
            metric_frame["species_count"].clip(lower=1)
        )
        value_frames.append(metric_frame)
    if not value_frames:
        return pd.DataFrame()
    return pd.concat(value_frames, ignore_index=True)


def aggregate_shared_policy_model_family_curves(
    frame: pd.DataFrame,
    *,
    metric_columns: list[str],
    model_families: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    value_frames = []
    required_families = [str(model_family) for model_family in model_families]
    for metric_column in metric_columns:
        if metric_column not in frame.columns:
            continue
        metric_frame = frame.copy()
        metric_frame[metric_column] = pd.to_numeric(
            metric_frame[metric_column],
            errors="coerce",
        )
        metric_frame = metric_frame.loc[
            metric_frame["model_family"].isin(required_families)
            & np.isfinite(metric_frame[metric_column])
        ].copy()
        if metric_frame.empty:
            continue
        species_counts = dataset_species_counts(metric_frame)

        rows: list[dict[str, object]] = []
        group_columns = ["dataset_kind", "review_count"]
        for (dataset_kind, review_count), group in metric_frame.groupby(
            group_columns,
            sort=True,
        ):
            family_policy_sets = [
                set(
                    group.loc[
                        group["model_family"] == model_family,
                        "selection_method",
                    ]
                    .dropna()
                    .astype(str)
                )
                for model_family in required_families
            ]
            if not family_policy_sets or any(
                not policies for policies in family_policy_sets
            ):
                continue
            shared_policies = set.intersection(*family_policy_sets)
            if not shared_policies:
                continue

            shared_group = group.loc[
                group["selection_method"].astype(str).isin(shared_policies)
            ]
            for model_family, family_group in shared_group.groupby(
                "model_family",
                sort=False,
            ):
                if model_family not in required_families:
                    continue
                policy_values = family_group.groupby("selection_method")[
                    metric_column
                ].mean()
                policy_count = int(policy_values.count())
                if policy_count == 0:
                    continue
                rows.append(
                    {
                        "dataset_kind": dataset_kind,
                        "model_family": model_family,
                        "review_count": review_count,
                        "mean_value": float(policy_values.mean()),
                        "median_value": float(policy_values.median()),
                        "q25_value": float(policy_values.quantile(0.25)),
                        "q75_value": float(policy_values.quantile(0.75)),
                        "std_value": float(policy_values.std()),
                        "dataset_species_count": species_counts.get(str(dataset_kind)),
                        "policy_count": policy_count,
                        "species_count": int(family_group["species_key"].nunique()),
                        "row_count": int(family_group[metric_column].count()),
                        "shared_policy_count": len(shared_policies),
                        "shared_policies": "|".join(sorted(shared_policies)),
                        "metric": metric_column,
                    }
                )

        if not rows:
            continue
        metric_summary = pd.DataFrame.from_records(rows)
        metric_summary["sem_value"] = metric_summary["std_value"] / np.sqrt(
            metric_summary["policy_count"].clip(lower=1)
        )
        value_frames.append(metric_summary)
    if not value_frames:
        return pd.DataFrame()
    return pd.concat(value_frames, ignore_index=True)


def aggregate_metric_curves(
    trace_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    metric_columns: list[str],
    experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    if experiment_ids is None:
        experiment_ids = required_experiment_ids(covariate_variant)
    species_keys = complete_species_keys(
        species_frame,
        covariate_variant,
        experiment_ids=experiment_ids,
    )
    frame = filtered_trace(
        trace_frame,
        covariate_variant=covariate_variant,
        species_keys=species_keys,
        experiment_ids=experiment_ids,
    )
    return aggregate_summary_curves(frame, metric_columns=metric_columns)


def pivot_value(
    pivot: pd.DataFrame,
    dataset_kind: object,
    review_count: int,
    experiment_id: str,
) -> float:
    key = (dataset_kind, review_count)
    if key not in pivot.index or experiment_id not in pivot.columns:
        return float("nan")
    return safe_float(pivot.loc[key, experiment_id])


def first_pivot_value(
    pivot: pd.DataFrame,
    dataset_kind: object,
    review_count: int,
    experiment_ids: list[str],
) -> float:
    for experiment_id in experiment_ids:
        value = pivot_value(pivot, dataset_kind, review_count, experiment_id)
        if np.isfinite(value):
            return value
    return float("nan")


def coalesce_numeric_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    existing_columns = [column for column in columns if column in frame.columns]
    if not existing_columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric_frame = frame[existing_columns].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.bfill(axis=1).iloc[:, 0]


def policy_summary_table(
    aggregate_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    review_count: int = 50,
) -> pd.DataFrame:
    experiment_ids = required_experiment_ids(covariate_variant)
    subset = aggregate_frame[
        (aggregate_frame["metric"] == "mean_oracle_distance")
        & (aggregate_frame["experiment_id"].isin(experiment_ids))
        & (aggregate_frame["review_count"].isin([0, review_count, 100]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index=["dataset_kind", "review_count"],
        columns="experiment_id",
        values="mean_value",
    )
    rows = []
    for dataset_kind in ordered_values(
        pivot.index.get_level_values("dataset_kind"),
        DATASET_ORDER,
    ):
        cs_random_candidates = continuous_score_random_candidates(covariate_variant)
        cs_bald_candidates = continuous_score_bald_candidates(covariate_variant)
        reviewed_random_id = reviewed_random_experiment_id(covariate_variant)
        reviewed_bald_id = reviewed_bald_experiment_id(covariate_variant)
        zero_cs = first_pivot_value(pivot, dataset_kind, 0, cs_random_candidates)
        zero_reviewed = pivot_value(pivot, dataset_kind, 0, reviewed_random_id)
        target_review_count = review_count
        cs_bald = first_pivot_value(
            pivot,
            dataset_kind,
            target_review_count,
            cs_bald_candidates,
        )
        cs_random = first_pivot_value(
            pivot,
            dataset_kind,
            target_review_count,
            cs_random_candidates,
        )
        reviewed_bald = pivot_value(
            pivot,
            dataset_kind,
            target_review_count,
            reviewed_bald_id,
        )
        reviewed_random = pivot_value(
            pivot,
            dataset_kind,
            target_review_count,
            reviewed_random_id,
        )

        rows.append(
            {
                "dataset_kind": dataset_kind,
                "zero_reviewed": zero_reviewed,
                "zero_continuous_score": zero_cs,
                "zero_score_gain_pct": 100.0
                * (zero_reviewed - zero_cs)
                / zero_reviewed,
                f"cs_bald_at_{review_count}": cs_bald,
                f"cs_random_at_{review_count}": cs_random,
                f"cs_bald_gain_vs_random_pct_at_{review_count}": 100.0
                * (cs_random - cs_bald)
                / cs_random,
                f"reviewed_bald_at_{review_count}": reviewed_bald,
                f"reviewed_random_at_{review_count}": reviewed_random,
                f"reviewed_bald_gain_vs_random_pct_at_{review_count}": 100.0
                * (reviewed_random - reviewed_bald)
                / reviewed_random,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_kind")


full_aggregate = aggregate_metric_curves(
    trace_frame,
    covariate_variant="full",
    metric_columns=[metric_column for metric_column, _ in GROUP_METRICS],
    experiment_ids=oracle_distance_plot_experiment_ids("full"),
)
null_aggregate = aggregate_metric_curves(
    trace_frame,
    covariate_variant="null",
    metric_columns=[metric_column for metric_column, _ in GROUP_METRICS],
    experiment_ids=oracle_distance_plot_experiment_ids("null"),
)

display_frame(
    "Full-model policy summary",
    policy_summary_table(
        full_aggregate, covariate_variant="full", review_count=50
    ).round(3),
)
display_frame(
    "Null-model policy summary",
    policy_summary_table(
        null_aggregate, covariate_variant="null", review_count=50
    ).round(3),
)


# %% [markdown]
# ## Averaged oracle-distance curves
#
# The main manuscript claim is visible directly in these averages:
#
# - the continuous-score family starts much closer to the oracle at zero reviews,
# - BALD is the strongest acquisition policy within that family on each dataset,
# - and detection coefficients remain the hardest quantity to recover.


# %%
def plot_aggregated_curves(
    aggregate_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    filename: str,
) -> None:
    if aggregate_frame.empty:
        print(f"No data available for {covariate_variant} aggregated curves.")
        return

    datasets = ordered_values(aggregate_frame["dataset_kind"], DATASET_ORDER)
    sample_sizes = dataset_sample_sizes_from_frame(aggregate_frame)
    experiment_meta = (
        aggregate_frame[
            ["experiment_id", "selection_method", "model_family", "plot_label"]
        ]
        .drop_duplicates()
        .sort_values(
            by="experiment_id",
            key=lambda column: [
                experiment_sort_key(
                    experiment_id,
                    aggregate_frame.loc[
                        aggregate_frame["experiment_id"] == experiment_id
                    ].iloc[0],
                )
                for experiment_id in column
            ],
        )
    )

    fig, axes = plt.subplots(
        len(datasets),
        len(GROUP_METRICS),
        figsize=full_width_grid_size(len(GROUP_METRICS), len(datasets), row_scale=0.98),
        sharex="row",
    )
    if len(datasets) == 1:
        axes = np.asarray([axes])

    for row_index, dataset_kind in enumerate(datasets):
        dataset_frame = aggregate_frame.loc[
            aggregate_frame["dataset_kind"] == dataset_kind
        ]
        x_min = float(dataset_frame["review_count"].min())
        x_max = float(dataset_frame["review_count"].max())
        for col_index, (metric_column, metric_label) in enumerate(GROUP_METRICS):
            axis = axes[row_index, col_index]
            metric_frame = dataset_frame.loc[dataset_frame["metric"] == metric_column]

            for experiment_row in experiment_meta.itertuples(index=False):
                experiment_metric = metric_frame.loc[
                    metric_frame["experiment_id"] == experiment_row.experiment_id
                ].sort_values("review_count")
                if experiment_metric.empty:
                    continue
                color = SELECTION_COLORS.get(experiment_row.selection_method, "#444444")
                linestyle = MODEL_FAMILY_LINESTYLES.get(
                    experiment_row.model_family, "-"
                )
                smoothed_values = maybe_smooth_aggregated_curve(
                    experiment_metric["review_count"],
                    experiment_metric["mean_value"],
                )
                axis.plot(
                    experiment_metric["review_count"],
                    smoothed_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=PLOT_LINEWIDTH,
                )
            if row_index == 0:
                axis.set_title(metric_label, pad=4.0, fontsize=AXIS_TITLE_FONTSIZE)
            if col_index == 0:
                axis.set_ylabel(
                    f"{dataset_label(dataset_kind, sample_sizes.get(str(dataset_kind)))} "
                    "$\\overline{W_1}$"
                )
            if row_index == len(datasets) - 1:
                axis.set_xlabel("$t$")
            axis.set_xlim(x_min, x_max)

    visible_selection_methods = ordered_values(
        experiment_meta["selection_method"],
        SELECTION_ORDER,
    )
    policy_handles = [
        Line2D(
            [0],
            [0],
            color=SELECTION_COLORS.get(name, "#444444"),
            linewidth=PLOT_LINEWIDTH,
            label=SELECTION_LABELS.get(name, name.replace("_", " ").title()),
        )
        for name in visible_selection_methods
    ]
    visible_model_families = ordered_values(
        experiment_meta["model_family"],
        MODEL_FAMILY_ORDER,
    )
    family_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=MODEL_FAMILY_LINESTYLES.get(name, "-"),
            linewidth=PLOT_LINEWIDTH,
            label=MODEL_FAMILY_LABELS.get(name, name.replace("_", " ").title()),
        )
        for name in visible_model_families
    ]
    policy_legend = fig.legend(
        handles=policy_handles,
        ncol=max(len(policy_handles), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.085),
        frameon=False,
    )
    fig.add_artist(policy_legend)
    fig.legend(
        handles=family_handles,
        ncol=max(len(family_handles), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.135),
        frameon=False,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.14, 1.0, 0.94),
        pad=0.01,
        w_pad=0.01,
        h_pad=0.04,
        wspace=0.20,
        hspace=0.24,
    )
    saved_path = maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)
    if saved_path is not None:
        print(f"Saved {saved_path.relative_to(repo_root)}")


plot_aggregated_curves(
    full_aggregate,
    covariate_variant="full",
    filename="aggregated_curve.png",
)
plot_aggregated_curves(
    null_aggregate,
    covariate_variant="null",
    filename="aggregated_curve_null.png",
)


# %% [markdown]
# ## Representative species-level trajectories
#
# By default, the script chooses one species per dataset where calibrated BALD
# gains the most over calibrated random at 50 reviews. Set `SPECIES_EXAMPLES`
# above if you want to inspect particular species instead.


# %%
def species_level_policy_gap_table(
    trace_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    review_count: int,
) -> pd.DataFrame:
    experiment_ids = oracle_distance_plot_experiment_ids(covariate_variant)
    species_keys = complete_species_keys(
        species_frame,
        covariate_variant,
        experiment_ids=experiment_ids,
    )
    subset = filtered_trace(
        trace_frame,
        covariate_variant=covariate_variant,
        species_keys=species_keys,
        experiment_ids=experiment_ids,
    )
    subset = subset.loc[subset["review_count"] == review_count]
    wide = subset.pivot_table(
        index=["dataset_kind", "species_key", "species"],
        columns="experiment_id",
        values="mean_oracle_distance",
    ).reset_index()
    wide.columns.name = None
    wide["continuous_score_random"] = coalesce_numeric_columns(
        wide,
        continuous_score_random_candidates(covariate_variant),
    )
    wide["continuous_score_bald"] = coalesce_numeric_columns(
        wide,
        continuous_score_bald_candidates(covariate_variant),
    )
    reviewed_random_id = reviewed_random_experiment_id(covariate_variant)
    reviewed_bald_id = reviewed_bald_experiment_id(covariate_variant)
    if reviewed_random_id not in wide.columns:
        wide[reviewed_random_id] = np.nan
    if reviewed_bald_id not in wide.columns:
        wide[reviewed_bald_id] = np.nan
    wide["cs_bald_gain_vs_random"] = (
        wide["continuous_score_random"] - wide["continuous_score_bald"]
    )
    wide["reviewed_bald_gain_vs_random"] = (
        wide[reviewed_random_id] - wide[reviewed_bald_id]
    )
    wide = wide.loc[
        np.isfinite(wide["cs_bald_gain_vs_random"])
        & np.isfinite(wide["reviewed_bald_gain_vs_random"])
    ]
    return wide.sort_values(
        ["dataset_kind", "cs_bald_gain_vs_random"], ascending=[True, False]
    )


def choose_species_examples(
    gap_frame: pd.DataFrame,
    *,
    manual_examples: dict[str, str | None],
) -> dict[str, str]:
    examples: dict[str, str] = {}
    for dataset_kind in ordered_values(gap_frame["dataset_kind"], DATASET_ORDER):
        manual_species = manual_examples.get(dataset_kind)
        if manual_species:
            examples[dataset_kind] = manual_species
            continue
        top_row = gap_frame.loc[gap_frame["dataset_kind"] == dataset_kind].iloc[0]
        examples[dataset_kind] = str(top_row["species"])
    return examples


def plot_species_examples(
    trace_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    examples: dict[str, str],
    filename: str,
) -> None:
    experiment_ids = oracle_distance_plot_experiment_ids(covariate_variant)
    if not examples:
        print("No representative species available.")
        return

    fig, axes = plt.subplots(
        1,
        len(examples),
        figsize=full_width_grid_size(len(examples), 1, row_scale=0.82),
        sharey=True,
    )
    if len(examples) == 1:
        axes = [axes]

    for axis, (dataset_kind, species_name) in zip(axes, examples.items()):
        species_subset = trace_frame.loc[
            (trace_frame["covariate_variant"] == covariate_variant)
            & (trace_frame["dataset_kind"] == dataset_kind)
            & (trace_frame["species"] == species_name)
            & (trace_frame["experiment_id"].isin(experiment_ids))
        ].copy()
        if species_subset.empty:
            axis.set_axis_off()
            continue

        experiment_meta = (
            species_subset[
                ["experiment_id", "selection_method", "model_family", "plot_label"]
            ]
            .drop_duplicates()
            .sort_values(
                by="experiment_id",
                key=lambda column: [
                    experiment_sort_key(
                        experiment_id,
                        species_subset.loc[
                            species_subset["experiment_id"] == experiment_id
                        ].iloc[0],
                    )
                    for experiment_id in column
                ],
            )
        )

        for experiment_row in experiment_meta.itertuples(index=False):
            experiment_trace = species_subset.loc[
                species_subset["experiment_id"] == experiment_row.experiment_id
            ].sort_values("review_count")
            axis.plot(
                experiment_trace["review_count"],
                experiment_trace["mean_oracle_distance"],
                color=SELECTION_COLORS.get(experiment_row.selection_method, "#444444"),
                linestyle=MODEL_FAMILY_LINESTYLES.get(experiment_row.model_family, "-"),
                linewidth=PLOT_LINEWIDTH,
                label=experiment_row.plot_label,
            )
        sample_size = int(species_subset["species_key"].nunique())
        axis.set_title(
            f"{dataset_label(dataset_kind, sample_size)}: {species_name}",
            pad=4.0,
        )
        axis.set_xlabel("$t$")
        if axis is axes[0]:
            axis.set_ylabel("$\\overline{W_1}$")

    legend_rows = (
        trace_frame.loc[
            (trace_frame["covariate_variant"] == covariate_variant)
            & (trace_frame["experiment_id"].isin(experiment_ids))
        ][["experiment_id", "selection_method", "model_family", "plot_label"]]
        .drop_duplicates()
        .sort_values(
            by="experiment_id",
            key=lambda column: [
                experiment_sort_key(
                    experiment_id,
                    trace_frame.loc[trace_frame["experiment_id"] == experiment_id].iloc[
                        0
                    ],
                )
                for experiment_id in column
            ],
        )
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=SELECTION_COLORS.get(row.selection_method, "#444444"),
            linestyle=MODEL_FAMILY_LINESTYLES.get(row.model_family, "-"),
            linewidth=PLOT_LINEWIDTH,
            label=row.plot_label,
        )
        for row in legend_rows.itertuples(index=False)
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.17, 1.0, 0.93),
        pad=0.01,
        w_pad=0.02,
        h_pad=0.04,
        wspace=0.18,
        hspace=0.22,
    )
    maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)


species_gap_table = species_level_policy_gap_table(
    trace_frame,
    covariate_variant="full",
    review_count=50,
)
species_examples = choose_species_examples(
    species_gap_table,
    manual_examples=SPECIES_EXAMPLES,
)

display_frame(
    "Largest calibrated BALD gains over calibrated random at 50 reviews",
    species_gap_table[
        [
            "dataset_kind",
            "species",
            "cs_bald_gain_vs_random",
            "continuous_score_bald",
            "continuous_score_random",
            "reviewed_bald_gain_vs_random",
        ]
    ]
    .groupby("dataset_kind")
    .head(8)
    .reset_index(drop=True)
    .round(3),
)

plot_species_examples(
    trace_frame,
    covariate_variant="full",
    examples=species_examples,
    filename="species_examples.png",
)


# %% [markdown]
# ## Species-specific parameter trajectories
#
# These figures track the shared occupancy and detection coefficients over the
# review budget for every species with the required experiment variants. Oracle
# posterior means and credible intervals are shown in black.


# %%
def parameter_display_label(parameter_name: str) -> str:
    if parameter_name.startswith("cov_state_"):
        return f"$\\beta_{{{parameter_name.split('_')[-1]}}}$"
    if parameter_name.startswith("cov_det_"):
        return f"$\\alpha_{{{parameter_name.split('_')[-1]}}}$"
    return parameter_name


def coefficient_parameter_names_from_wide(
    frame: pd.DataFrame,
    *,
    prefix: str = "oracle__parameter_mean__",
) -> list[str]:
    parameter_names = []
    for column in frame.columns:
        if not column.startswith(prefix):
            continue
        parameter_name = column[len(prefix) :]
        if parameter_name.startswith(("cov_state_", "cov_det_")):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                parameter_names.append(parameter_name)
    return sorted(
        set(parameter_names),
        key=lambda name: (
            0 if name.startswith("cov_state_") else 1,
            int(name.split("_")[-1]),
        ),
    )


def extract_wide_parameter_summary(
    row: pd.Series,
    parameter_names: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for parameter_name in parameter_names:
        mean = safe_float(row.get(f"{prefix}mean__{parameter_name}"))
        q05 = safe_float(row.get(f"{prefix}q05__{parameter_name}"))
        q95 = safe_float(row.get(f"{prefix}q95__{parameter_name}"))
        if not any(np.isfinite(value) for value in [mean, q05, q95]):
            continue
        records.append(
            {
                "parameter": parameter_name,
                "parameter_index": "",
                "mean": mean,
                "q05": q05,
                "q95": q95,
            }
        )
    return pd.DataFrame.from_records(records)


def load_species_parameter_trajectories_from_subset(
    subset: pd.DataFrame,
    *,
    covariate_variant: str,
    experiment_ids: list[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = subset.loc[
        (subset["covariate_variant"] == covariate_variant)
        & (subset["experiment_id"].isin(experiment_ids))
    ].copy()
    if subset.empty:
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dataset_kind = str(subset.iloc[0]["dataset_kind"])
    species_name = str(subset.iloc[0]["species"])
    species_key = str(subset.iloc[0]["species_key"])
    subset = subset.sort_values(
        by="experiment_id",
        key=lambda column: [
            experiment_sort_key(
                experiment_id,
                subset.loc[subset["experiment_id"] == experiment_id].iloc[0],
            )
            for experiment_id in column
        ],
    )
    trace_subset = trace_frame.loc[
        (trace_frame["species_key"] == species_key)
        & (trace_frame["covariate_variant"] == covariate_variant)
        & (trace_frame["experiment_id"].isin(experiment_ids))
    ].copy()
    if trace_subset.empty:
        return [], pd.DataFrame(), pd.DataFrame(), subset

    oracle_prefix = (
        "null_oracle__parameter_"
        if covariate_variant == "null"
        else "oracle__parameter_"
    )
    parameter_names = coefficient_parameter_names_from_wide(
        trace_subset,
        prefix=f"{oracle_prefix}mean__",
    )
    oracle_summary = extract_wide_parameter_summary(
        trace_subset.iloc[0],
        parameter_names,
        prefix=oracle_prefix,
    )
    trajectory_rows: list[dict[str, object]] = []

    for trace_row in trace_subset.sort_values(
        ["experiment_id", "review_count"]
    ).itertuples(index=False):
        for parameter_name in parameter_names:
            mean = safe_float(getattr(trace_row, f"parameter_mean__{parameter_name}"))
            q05 = safe_float(getattr(trace_row, f"parameter_q05__{parameter_name}"))
            q95 = safe_float(getattr(trace_row, f"parameter_q95__{parameter_name}"))
            if any(np.isfinite(value) for value in [mean, q05, q95]):
                trajectory_rows.append(
                    {
                        "dataset_kind": dataset_kind,
                        "species": species_name,
                        "experiment_id": trace_row.experiment_id,
                        "plot_label": trace_row.plot_label,
                        "selection_method": trace_row.selection_method,
                        "model_family": trace_row.model_family,
                        "review_count": int(trace_row.review_count),
                        "parameter": parameter_name,
                        "mean": mean,
                        "q05": q05,
                        "q95": q95,
                    }
                )

    return (
        parameter_names,
        pd.DataFrame.from_records(trajectory_rows),
        oracle_summary,
        subset,
    )


def load_species_parameter_trajectories(
    experiment_frame: pd.DataFrame,
    *,
    dataset_kind: str,
    species_name: str,
    covariate_variant: str,
    experiment_ids: list[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = experiment_frame.loc[
        (experiment_frame["dataset_kind"] == dataset_kind)
        & (experiment_frame["species"] == species_name)
        & (experiment_frame["covariate_variant"] == covariate_variant)
        & (experiment_frame["experiment_id"].isin(experiment_ids))
    ].copy()
    return load_species_parameter_trajectories_from_subset(
        subset,
        covariate_variant=covariate_variant,
        experiment_ids=experiment_ids,
    )


def parameter_trajectory_species_table(
    experiment_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    experiment_ids: list[str],
) -> pd.DataFrame:
    subset = experiment_frame.loc[
        (experiment_frame["covariate_variant"] == covariate_variant)
        & (experiment_frame["experiment_id"].isin(experiment_ids))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    required_count = len(set(experiment_ids))
    species_table = (
        subset.groupby(["dataset_kind", "species_key", "species"])["experiment_id"]
        .nunique()
        .reset_index(name="available_experiments")
    )
    return species_table.loc[
        species_table["available_experiments"] >= required_count
    ].sort_values(["dataset_kind", "species"])


def render_species_parameter_trajectory_figure(
    species_subset: pd.DataFrame,
    covariate_variant: str,
    experiment_ids: list[str],
    output_dir: Path | None,
) -> dict[str, object] | None:
    if species_subset.empty:
        return None

    dataset_kind = str(species_subset.iloc[0]["dataset_kind"])
    species_name = str(species_subset.iloc[0]["species"])
    (
        parameter_names,
        trajectory_frame,
        oracle_summary,
        experiment_subset,
    ) = load_species_parameter_trajectories_from_subset(
        species_subset,
        covariate_variant=covariate_variant,
        experiment_ids=experiment_ids,
    )
    if not parameter_names or trajectory_frame.empty or oracle_summary.empty:
        return None

    n_parameters = len(parameter_names)
    n_cols = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=full_width_grid_size(n_cols, n_rows, row_scale=0.98),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    x_values = trajectory_frame["review_count"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    ordered_experiment_ids = [
        experiment_id
        for experiment_id in experiment_ids
        if experiment_id in set(trajectory_frame["experiment_id"])
    ]

    for axis, parameter_name in zip(axes.flat, parameter_names):
        parameter_traces = trajectory_frame.loc[
            trajectory_frame["parameter"] == parameter_name
        ].copy()
        oracle_row = oracle_summary.loc[oracle_summary["parameter"] == parameter_name]
        if parameter_traces.empty or oracle_row.empty:
            axis.set_axis_off()
            continue

        oracle = oracle_row.iloc[0]
        axis.fill_between(
            [x_min, x_max],
            [safe_float(oracle["q05"]), safe_float(oracle["q05"])],
            [safe_float(oracle["q95"]), safe_float(oracle["q95"])],
            color="black",
            alpha=0.10,
        )
        axis.plot(
            [x_min, x_max],
            [safe_float(oracle["mean"]), safe_float(oracle["mean"])],
            color="black",
            linewidth=PLOT_LINEWIDTH,
        )

        for experiment_id in ordered_experiment_ids:
            experiment_trace = parameter_traces.loc[
                parameter_traces["experiment_id"] == experiment_id
            ].sort_values("review_count")
            if experiment_trace.empty:
                continue
            experiment_meta = experiment_subset.loc[
                experiment_subset["experiment_id"] == experiment_id
            ].iloc[0]
            color = SELECTION_COLORS.get(experiment_meta["selection_method"], "#444444")
            linestyle = MODEL_FAMILY_LINESTYLES.get(
                experiment_meta["model_family"], "-"
            )
            axis.fill_between(
                experiment_trace["review_count"],
                experiment_trace["q05"],
                experiment_trace["q95"],
                color=color,
                alpha=0.08,
            )
            axis.plot(
                experiment_trace["review_count"],
                experiment_trace["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=PLOT_LINEWIDTH,
            )

        axis.set_xlabel("$t$")
        axis.set_ylabel(parameter_display_label(parameter_name), labelpad=1.0)

    for axis in axes.flat[n_parameters:]:
        axis.set_axis_off()

    legend_rows = (
        experiment_subset[
            ["experiment_id", "selection_method", "model_family", "plot_label"]
        ]
        .drop_duplicates()
        .set_index("experiment_id")
        .loc[ordered_experiment_ids]
        .reset_index()
    )
    handles = [
        Line2D(
            [0], [0], color="black", linewidth=PLOT_LINEWIDTH, label="Oracle posterior"
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=SELECTION_COLORS.get(row.selection_method, "#444444"),
            linestyle=MODEL_FAMILY_LINESTYLES.get(row.model_family, "-"),
            linewidth=PLOT_LINEWIDTH,
            label=row.plot_label,
        )
        for row in legend_rows.itertuples(index=False)
    )
    fig.legend(
        handles=handles,
        ncol=min(len(handles), 3),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.13, 1.0, 1.0),
        pad=0.01,
        w_pad=0.01,
        h_pad=0.05,
        wspace=0.22,
        hspace=0.20,
    )
    filename = f"{dataset_kind}__{slugify(species_name)}__parameter_trajectories.pdf"
    saved_path = maybe_save_figure(fig, output_dir, filename)
    plt.close(fig)
    if saved_path is None:
        return None
    return {
        "dataset_kind": dataset_kind,
        "species": species_name,
        "path": str(saved_path.relative_to(repo_root)),
    }


def plot_species_parameter_trajectories(
    experiment_frame: pd.DataFrame,
    *,
    species_table: pd.DataFrame,
    experiment_ids: list[str],
    covariate_variant: str,
    output_dir: Path | None,
) -> pd.DataFrame:
    if species_table.empty:
        return pd.DataFrame()

    species_keys = species_table["species_key"].astype(str).unique().tolist()
    task_frame = experiment_frame.loc[
        (experiment_frame["covariate_variant"] == covariate_variant)
        & (experiment_frame["experiment_id"].isin(experiment_ids))
        & (experiment_frame["species_key"].isin(species_keys))
    ].copy()
    species_subsets = [
        group.copy()
        for _, group in task_frame.groupby(["dataset_kind", "species_key"], sort=True)
    ]

    saved_rows: list[dict[str, object]] = []
    worker_count = resolved_parallel_workers(
        len(species_subsets),
        configured_workers=PARAMETER_TRAJECTORY_PLOT_WORKERS,
        default_cap=4,
    )
    process_pool = None if worker_count == 1 else create_process_pool(worker_count)
    if worker_count == 1 or process_pool is None:
        for species_subset in tqdm(
            species_subsets,
            desc="Plotting species parameter trajectories",
            unit="species",
            total=len(species_subsets),
        ):
            result = render_species_parameter_trajectory_figure(
                species_subset,
                covariate_variant,
                experiment_ids,
                output_dir,
            )
            if result is not None:
                print(f"Saved {result['path']}")
                saved_rows.append(result)
    else:
        with process_pool as executor:
            futures = [
                executor.submit(
                    render_species_parameter_trajectory_figure,
                    species_subset,
                    covariate_variant,
                    experiment_ids,
                    output_dir,
                )
                for species_subset in species_subsets
            ]
            for future in tqdm(
                as_completed(futures),
                desc="Plotting species parameter trajectories",
                unit="species",
                total=len(futures),
            ):
                result = future.result()
                if result is not None:
                    print(f"Saved {result['path']}")
                    saved_rows.append(result)
    if not saved_rows:
        return pd.DataFrame()
    return pd.DataFrame(saved_rows).sort_values(["dataset_kind", "species"])


if EXPORT_PARAMETER_TRAJECTORIES:
    species_parameter_species_table = parameter_trajectory_species_table(
        experiment_frame,
        covariate_variant="full",
        experiment_ids=PARAMETER_TRAJECTORY_EXPERIMENT_IDS,
    )

    species_parameter_figure_frame = plot_species_parameter_trajectories(
        experiment_frame,
        species_table=species_parameter_species_table,
        experiment_ids=PARAMETER_TRAJECTORY_EXPERIMENT_IDS,
        covariate_variant="full",
        output_dir=parameter_figure_output_dir,
    )

    if not species_parameter_species_table.empty:
        display_frame(
            "Species with exported parameter trajectories",
            species_parameter_species_table.groupby("dataset_kind")
            .agg(species_count=("species_key", "nunique"))
            .reset_index(),
        )

    if not species_parameter_figure_frame.empty:
        display_frame(
            "Species parameter trajectory figures",
            species_parameter_figure_frame,
        )
else:
    print("Skipping species parameter trajectory export.")


# %% [markdown]
# ## Ecological conclusion agreement
#
# These summaries replace posterior geometry with two manuscript-oriented
# questions:
#
# - do we recover the same occupancy/detection covariate conclusions as the
#   fully reviewed oracle posterior, and
# - do site-level occupancy probabilities and rankings agree with the oracle?


def credible_interval_conclusion_from_bounds(
    q05: float,
    q95: float,
) -> str | None:
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
        recall_name = f"coef_ci_oracle_decisive_recall__{group_name}"
        false_strong_name = f"coef_ci_false_strong_rate__{group_name}"
        if not group_parameters:
            metrics[agreement_name] = float("nan")
            metrics[recall_name] = float("nan")
            metrics[false_strong_name] = float("nan")
            continue

        agreement = [
            states[parameter_name] == oracle_states[parameter_name]
            for parameter_name in group_parameters
        ]
        decisive_recall = [
            states[parameter_name] == oracle_states[parameter_name]
            for parameter_name in group_parameters
            if oracle_states[parameter_name] != "uncertain"
        ]
        false_strong = [
            states[parameter_name] != "uncertain"
            and oracle_states[parameter_name] == "uncertain"
            for parameter_name in group_parameters
        ]
        metrics[agreement_name] = float(np.mean(agreement))
        metrics[recall_name] = (
            float(np.mean(decisive_recall)) if decisive_recall else float("nan")
        )
        metrics[false_strong_name] = float(np.mean(false_strong))
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


def compute_intercept_occupancy_probability_metrics(
    row: pd.Series,
    oracle_parameter_prefix: str,
) -> dict[str, float]:
    psi_estimate = inverse_logit(row.get("parameter_mean__cov_state_0"))
    oracle_psi_estimate = inverse_logit(
        row.get(f"{oracle_parameter_prefix}mean__cov_state_0")
    )
    if not np.isfinite(psi_estimate) or not np.isfinite(oracle_psi_estimate):
        return {"psi_probability_agreement": float("nan")}
    return {
        "psi_probability_agreement": float(
            np.clip(1.0 - abs(psi_estimate - oracle_psi_estimate), 0.0, 1.0)
        )
    }


def build_ecological_agreement_frames(
    experiment_frame: pd.DataFrame,
    *,
    covariate_variant: str,
    experiment_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del experiment_frame
    if experiment_ids is None:
        experiment_ids = agreement_experiment_ids(covariate_variant)
    species_keys = complete_species_keys(
        species_frame,
        covariate_variant,
        experiment_ids=experiment_ids,
    )
    if MAX_AGREEMENT_SPECIES_PER_DATASET > 0:
        limited_species_keys: list[str] = []
        for dataset_kind in ordered_values(
            species_frame["dataset_kind"], DATASET_ORDER
        ):
            dataset_species = (
                species_frame.loc[
                    (species_frame["dataset_kind"] == dataset_kind)
                    & (species_frame["species_key"].isin(species_keys)),
                    "species_key",
                ]
                .astype(str)
                .sort_values()
                .tolist()
            )
            limited_species_keys.extend(
                dataset_species[:MAX_AGREEMENT_SPECIES_PER_DATASET]
            )
        species_keys = limited_species_keys
    subset = filtered_trace(
        trace_frame,
        covariate_variant=covariate_variant,
        species_keys=species_keys,
        experiment_ids=experiment_ids,
    )
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    oracle_parameter_prefix = (
        "null_oracle__parameter_"
        if covariate_variant == "null"
        else "oracle__parameter_"
    )
    base_columns = [
        "dataset_kind",
        "species",
        "species_key",
        "experiment_id",
        "selection_method",
        "model_family",
        "covariate_variant",
        "plot_label",
        "review_count",
    ]
    coefficient_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    parsed_vector_cache: dict[str, np.ndarray | None] = {}

    def cached_parse_vector(value: object) -> np.ndarray | None:
        text = str(value).strip()
        if text not in parsed_vector_cache:
            parsed_vector_cache[text] = parse_numeric_vector(text)
        return parsed_vector_cache[text]

    for _row_index, row in subset.iterrows():
        base_row = {column: row[column] for column in base_columns}
        base_row["n_reviewable"] = safe_float(row.get("dataset__n_reviewable"))
        coefficient_rows.append(
            {
                **base_row,
                **compute_coefficient_conclusion_metrics_from_states(
                    coefficient_states_from_row(row, "parameter_"),
                    coefficient_states_from_row(row, oracle_parameter_prefix),
                ),
            }
        )

        step_psi_mean = (
            cached_parse_vector(row.get("psi_mean_values", ""))
            if covariate_variant == "full"
            else None
        )
        oracle_psi_mean = (
            cached_parse_vector(row.get("oracle_psi_mean_values", ""))
            if covariate_variant == "full"
            else None
        )
        if step_psi_mean is not None and oracle_psi_mean is not None:
            site_rows.append(
                {
                    **base_row,
                    **compute_site_priority_metrics(step_psi_mean, oracle_psi_mean),
                }
            )
        elif covariate_variant == "null":
            site_rows.append(
                {
                    **base_row,
                    **compute_intercept_occupancy_probability_metrics(
                        row,
                        oracle_parameter_prefix,
                    ),
                }
            )

    coefficient_frame = pd.DataFrame.from_records(coefficient_rows)
    if not coefficient_frame.empty:
        coefficient_frame = coefficient_frame.sort_values(
            ["dataset_kind", "species", "experiment_id", "review_count"]
        ).reset_index(drop=True)
    site_frame = pd.DataFrame.from_records(site_rows)
    if not site_frame.empty:
        site_frame = site_frame.sort_values(
            ["dataset_kind", "species", "experiment_id", "review_count"]
        ).reset_index(drop=True)
    return coefficient_frame, site_frame


def focal_metric_summary_table(
    aggregate_frame: pd.DataFrame,
    *,
    review_count: int,
    metric_specs: list[tuple[str, str]],
    experiment_ids: list[str],
) -> pd.DataFrame:
    if aggregate_frame.empty:
        return pd.DataFrame()
    subset = aggregate_frame.loc[
        (aggregate_frame["review_count"] == review_count)
        & (aggregate_frame["metric"].isin([metric for metric, _ in metric_specs]))
        & (aggregate_frame["experiment_id"].isin(experiment_ids))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["metric_label"] = subset["metric"].map(dict(metric_specs))
    subset["display_value"] = [
        metric_display_mean(metric_column, mean_value)
        for metric_column, mean_value in zip(subset["metric"], subset["mean_value"])
    ]
    pivot = subset.pivot_table(
        index=["dataset_kind", "metric_label"],
        columns="experiment_id",
        values="display_value",
    )
    available_columns = [
        experiment_id
        for experiment_id in experiment_ids
        if experiment_id in pivot.columns
    ]
    label_map = (
        aggregate_frame[
            ["experiment_id", "plot_label", "selection_method", "model_family"]
        ]
        .drop_duplicates()
        .sort_values(
            by="experiment_id",
            key=lambda column: [
                experiment_sort_key(
                    experiment_id,
                    aggregate_frame.loc[
                        aggregate_frame["experiment_id"] == experiment_id
                    ].iloc[0],
                )
                for experiment_id in column
            ],
        )
        .set_index("experiment_id")["plot_label"]
        .to_dict()
    )
    pivot = pivot.loc[:, available_columns].rename(columns=label_map)
    return pivot.reset_index()


def agreement_threshold_label(rule: dict[str, object]) -> str:
    direction = str(rule["direction"])
    threshold = float(rule["threshold"])
    operator = "\\ge" if direction == "ge" else "\\le"
    return f"${operator} {threshold:.2f}$"


def normalized_dataset_kind(dataset_kind: object) -> str | None:
    if dataset_kind is None:
        return None
    dataset_key = str(dataset_kind).strip().lower()
    if not dataset_key or dataset_key == "nan":
        return None
    return dataset_key


def agreement_threshold_rule(
    metric_column: str,
    dataset_kind: object = None,
) -> dict[str, object]:
    rule = dict(AGREEMENT_THRESHOLD_RULES[metric_column])
    dataset_key = normalized_dataset_kind(dataset_kind)
    if dataset_key is not None:
        rule.update(
            AGREEMENT_THRESHOLD_RULES_BY_DATASET.get(dataset_key, {}).get(
                metric_column,
                {},
            )
        )
    rule["threshold_label"] = agreement_threshold_label(rule)
    return rule


def agreement_threshold_frame_dataset_kind(
    species_metric_frame: pd.DataFrame,
) -> str | None:
    if species_metric_frame.empty or "dataset_kind" not in species_metric_frame.columns:
        return None
    dataset_keys = [
        dataset_key
        for dataset_key in (
            normalized_dataset_kind(value)
            for value in species_metric_frame["dataset_kind"].dropna().unique()
        )
        if dataset_key is not None
    ]
    if len(dataset_keys) != 1:
        return None
    return dataset_keys[0]


def agreement_threshold_species_hit(
    species_metric_frame: pd.DataFrame,
    *,
    metric_column: str,
    dataset_kind: object = None,
) -> dict[str, object]:
    subset = species_metric_frame.sort_values("review_count").copy()
    rule = agreement_threshold_rule(
        metric_column,
        dataset_kind=(
            normalized_dataset_kind(dataset_kind)
            or agreement_threshold_frame_dataset_kind(subset)
        ),
    )
    if subset.empty or metric_column not in subset.columns:
        return {
            "step": None,
            "fraction": float("nan"),
            "reached": False,
            "max_review": None,
            "n_reviewable": float("nan"),
            "threshold": float(rule["threshold"]),
            "threshold_label": rule["threshold_label"],
        }

    metric_values = np.asarray(
        [
            metric_display_mean(metric_column, mean_value)
            for mean_value in subset[metric_column]
        ],
        dtype=float,
    )
    valid_metric_mask = np.isfinite(metric_values)
    if not np.any(valid_metric_mask):
        return {
            "step": None,
            "fraction": float("nan"),
            "reached": False,
            "max_review": None,
            "n_reviewable": float("nan"),
            "threshold": float(rule["threshold"]),
            "threshold_label": rule["threshold_label"],
        }

    subset = subset.loc[valid_metric_mask].copy()
    metric_values = metric_values[valid_metric_mask]
    if str(rule["direction"]) == "ge":
        hit_mask = metric_values >= float(rule["threshold"])
    else:
        hit_mask = metric_values <= float(rule["threshold"])

    max_review = int(subset["review_count"].max())
    if np.any(hit_mask):
        step = int(subset.loc[hit_mask, "review_count"].iloc[0])
        reached = True
    else:
        step = max_review
        reached = False

    n_reviewable = finite_nanmax(
        pd.to_numeric(
            subset.get("n_reviewable", pd.Series(dtype=float)),
            errors="coerce",
        )
    )
    fraction = (
        float(step) / n_reviewable
        if np.isfinite(n_reviewable) and n_reviewable > 0
        else float("nan")
    )
    return {
        "step": step,
        "fraction": fraction,
        "reached": reached,
        "max_review": max_review,
        "n_reviewable": n_reviewable,
        "threshold": float(rule["threshold"]),
        "threshold_label": rule["threshold_label"],
    }


def agreement_threshold_summary(
    agreement_frame: pd.DataFrame,
    *,
    dataset_kind: str,
    metric_column: str,
    experiment_id: str,
) -> dict[str, object]:
    if agreement_frame.empty or metric_column not in agreement_frame.columns:
        return {"available": False}

    subset = agreement_frame.loc[
        (agreement_frame["dataset_kind"] == dataset_kind)
        & (agreement_frame["experiment_id"] == experiment_id)
    ].copy()
    if subset.empty:
        return {"available": False}

    hits = [
        agreement_threshold_species_hit(
            species_group,
            metric_column=metric_column,
            dataset_kind=dataset_kind,
        )
        for _species_key, species_group in subset.groupby("species_key", sort=True)
    ]
    hits = [
        hit
        for hit in hits
        if hit.get("step") is not None and np.isfinite(safe_float(hit.get("step")))
    ]
    if not hits:
        return {"available": False}

    fractions = np.asarray([safe_float(hit["fraction"]) for hit in hits], dtype=float)
    steps = np.asarray([safe_float(hit["step"]) for hit in hits], dtype=float)
    reached_count = int(sum(bool(hit.get("reached")) for hit in hits))
    species_count = len(hits)
    median_resolved = reached_count > species_count / 2
    return {
        "available": True,
        "mean_fraction": (
            float(np.nanmean(fractions))
            if np.any(np.isfinite(fractions))
            else float("nan")
        ),
        "mean_review_count": float(np.nanmean(steps)),
        "median_fraction": finite_upper_median(fractions),
        "median_review_count": finite_upper_median(steps),
        "reached_count": reached_count,
        "species_count": species_count,
        "median_resolved": median_resolved,
        "all_reached": reached_count == species_count,
    }


def format_review_fraction(value: object) -> str:
    fraction = safe_float(value)
    if not np.isfinite(fraction):
        return "--"
    return f"{fraction:.3f}"


def format_review_count(value: object) -> str:
    count = safe_float(value)
    if not np.isfinite(count):
        return "--"
    return f"{count:.0f}"


def format_threshold_summary(
    summary: dict[str, object],
    *,
    bold: bool = False,
    render_review_fraction: bool | None = None,
) -> str:
    if not bool(summary.get("available")):
        return "--"

    if render_review_fraction is None:
        render_review_fraction = AGREEMENT_THRESHOLD_RENDER_REVIEW_FRACTIONS

    count_text = format_review_count(summary.get("median_review_count"))
    if count_text == "--":
        return "--"

    if render_review_fraction:
        fraction_text = format_review_fraction(summary.get("median_fraction"))
        if fraction_text == "--":
            return "--"
        if bool(summary.get("median_resolved")):
            text = f"{fraction_text} ({count_text})"
        else:
            text = f"$>{fraction_text}$ ($>{count_text}$)"
    elif bool(summary.get("median_resolved")):
        text = count_text
    else:
        text = f"$>{count_text}$"
    return f"\\textbf{{{text}}}" if bold else text


def format_speedup_factor(value: object) -> str:
    speedup = safe_float(value)
    if np.isposinf(speedup):
        return "$\\infty$"
    if not np.isfinite(speedup):
        return "--"
    return f"{speedup:.1f}$\\times$"


def agreement_threshold_speedup_factor(
    *,
    baseline_summary: dict[str, object] | None,
    target_summary: dict[str, object] | None,
) -> float:
    if not baseline_summary or not target_summary:
        return float("nan")
    if not bool(baseline_summary.get("available")) or not bool(
        target_summary.get("available")
    ):
        return float("nan")

    baseline_count = safe_float(baseline_summary.get("median_review_count"))
    target_count = safe_float(target_summary.get("median_review_count"))
    if not np.isfinite(baseline_count) or not np.isfinite(target_count):
        return float("nan")
    if target_count == 0:
        if baseline_count == 0:
            return float("nan")
        return float("inf")
    return baseline_count / target_count


def agreement_threshold_table_column(
    family_label: str,
    policy_label: str,
) -> str:
    return f"{family_label} {policy_label}"


def agreement_threshold_best_value(summary: dict[str, object]) -> float:
    key = (
        "median_fraction"
        if AGREEMENT_THRESHOLD_RENDER_REVIEW_FRACTIONS
        else "median_review_count"
    )
    return safe_float(summary.get(key))


def table_metric_label(metric_label: str) -> str:
    return " ".join(str(metric_label).split())


def agreement_threshold_results_table(
    agreement_frame: pd.DataFrame,
) -> pd.DataFrame:
    if agreement_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    dataset_kinds = ordered_values(
        agreement_frame["dataset_kind"],
        DATASET_ORDER,
    )
    for dataset_kind in dataset_kinds:
        for metric_column, metric_label in AGREEMENT_CURVE_METRICS:
            rule = agreement_threshold_rule(
                metric_column,
                dataset_kind=dataset_kind,
            )
            table_metric = AGREEMENT_THRESHOLD_METRIC_LABELS.get(
                metric_column,
                metric_label,
            )
            row: dict[str, object] = {
                "Dataset": dataset_label(dataset_kind),
                "Metric": (
                    f"{table_metric_label(table_metric)} {rule['threshold_label']}"
                ),
            }
            policy_summaries: list[tuple[str, str, dict[str, object]]] = []
            for family_label, review_policies in AGREEMENT_THRESHOLD_TABLE_FAMILIES:
                for policy_label, experiment_id in review_policies:
                    summary = agreement_threshold_summary(
                        agreement_frame,
                        dataset_kind=str(dataset_kind),
                        metric_column=metric_column,
                        experiment_id=experiment_id,
                    )
                    policy_summaries.append(
                        (
                            agreement_threshold_table_column(
                                family_label,
                                policy_label,
                            ),
                            experiment_id,
                            summary,
                        )
                    )
            for column_label, experiment_id in AGREEMENT_THRESHOLD_STANDALONE_POLICIES:
                policy_summaries.append(
                    (
                        column_label,
                        experiment_id,
                        agreement_threshold_summary(
                            agreement_frame,
                            dataset_kind=str(dataset_kind),
                            metric_column=metric_column,
                            experiment_id=experiment_id,
                        ),
                    )
                )
            summary_by_experiment_id = {
                experiment_id: summary
                for _column, experiment_id, summary in policy_summaries
            }

            resolved_medians = [
                agreement_threshold_best_value(summary)
                for _column, _experiment_id, summary in policy_summaries
                if bool(summary.get("available"))
                and bool(summary.get("median_resolved"))
            ]
            best_candidates = (
                resolved_medians
                if resolved_medians
                else [
                    agreement_threshold_best_value(summary)
                    for _column, _experiment_id, summary in policy_summaries
                    if bool(summary.get("available"))
                ]
            )
            best_candidates = [value for value in best_candidates if np.isfinite(value)]
            best_value = min(best_candidates, default=float("nan"))

            for column, _experiment_id, summary in policy_summaries:
                candidate_value = agreement_threshold_best_value(summary)
                is_best = bool(summary.get("available")) and np.isfinite(best_value)
                if resolved_medians:
                    is_best = (
                        is_best
                        and bool(summary.get("median_resolved"))
                        and np.isclose(candidate_value, best_value)
                    )
                else:
                    is_best = is_best and np.isclose(candidate_value, best_value)
                row[column] = format_threshold_summary(
                    summary,
                    bold=is_best,
                )
            if AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN:
                row[AGREEMENT_THRESHOLD_SPEEDUP_COLUMN_LABEL] = format_speedup_factor(
                    agreement_threshold_speedup_factor(
                        baseline_summary=summary_by_experiment_id.get(
                            AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_EXPERIMENT_ID
                        ),
                        target_summary=summary_by_experiment_id.get(
                            AGREEMENT_THRESHOLD_SPEEDUP_TARGET_EXPERIMENT_ID
                        ),
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def agreement_threshold_results_latex(
    results_table: pd.DataFrame,
) -> str:
    if results_table.empty:
        return ""

    policy_columns = [
        (family_label, policy_label)
        for family_label, review_policies in AGREEMENT_THRESHOLD_TABLE_FAMILIES
        for policy_label, _experiment_id in review_policies
    ]
    standalone_columns = [
        column_label
        for column_label, _experiment_id in AGREEMENT_THRESHOLD_STANDALONE_POLICIES
    ]
    family_counts = [
        len(review_policies)
        for _family_label, review_policies in AGREEMENT_THRESHOLD_TABLE_FAMILIES
    ]
    right_column_count = (
        sum(family_counts[1:])
        + len(standalone_columns)
        + int(AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN)
    )
    column_format = "ll"
    if family_counts:
        column_format += "r" * family_counts[0]
    if right_column_count:
        column_format += "|" + "r" * right_column_count

    family_header_cells = [
        "\\multirow{2}{4em}{Dataset}",
        "\\multirow{2}{4em}{Metric}",
    ]
    policy_header_cells = ["", ""]
    cmidrules: list[str] = []
    start_column = 3
    for family_label, review_policies in AGREEMENT_THRESHOLD_TABLE_FAMILIES:
        end_column = start_column + len(review_policies) - 1
        latex_family_label = str(family_label).replace("_", "\\_")
        family_header_cells.append(
            f"\\multicolumn{{{len(review_policies)}}}{{c}}{{{latex_family_label}}}"
        )
        policy_header_cells.extend(
            AGREEMENT_THRESHOLD_POLICY_HEADERS.get(
                (family_label, policy_label),
                policy_label,
            )
            for policy_label, _experiment_id in review_policies
        )
        cmidrules.append(f"\\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column = end_column + 1

    for column_label in standalone_columns:
        family_header_cells.append(
            "\\multicolumn{1}{c}{"
            f"\\multirow{{2}}{{6em}}{{\\vspace{{-2mm}}{column_label}}}"
            "}"
        )
        policy_header_cells.append("")
        start_column += 1

    if AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN:
        family_header_cells.append(
            "\\multicolumn{1}{c}{"
            "\\multirow{2}{6em}{\\vspace{-2mm}"
            f"{AGREEMENT_THRESHOLD_SPEEDUP_COLUMN_LABEL}"
            "}"
            "}"
        )
        policy_header_cells.append("")
        start_column += 1

    tabular_lines = [
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(family_header_cells) + r" \\",
        "".join(cmidrules),
        " & ".join(policy_header_cells) + r" \\",
        "\\midrule",
    ]
    for dataset_index, (_dataset, dataset_rows) in enumerate(
        results_table.groupby("Dataset", sort=False)
    ):
        if dataset_index > 0:
            tabular_lines.append("\\midrule")
        dataset_row_count = len(dataset_rows)
        for row_offset, (_row_index, row) in enumerate(dataset_rows.iterrows()):
            dataset_cell = (
                f"\\multirow{{{dataset_row_count}}}{{4em}}{{{row['Dataset']}}}"
                if row_offset == 0
                else ""
            )
            row_cells = [dataset_cell, str(row["Metric"])]
            row_cells.extend(
                str(row[agreement_threshold_table_column(family_label, policy_label)])
                for family_label, policy_label in policy_columns
            )
            row_cells.extend(str(row[column]) for column in standalone_columns)
            if AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN:
                row_cells.append(str(row[AGREEMENT_THRESHOLD_SPEEDUP_COLUMN_LABEL]))
            tabular_lines.append(" & ".join(row_cells) + r" \\")
    tabular_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tabular = "\n".join(tabular_lines)
    if AGREEMENT_THRESHOLD_RENDER_REVIEW_FRACTIONS:
        caption_start = (
            "Median fraction of reviews required per species to reach "
            "dataset-specific agreement thresholds, with the median number of "
            "reviews in parentheses. Review fractions are computed relative to "
            "each species' number of reviewable replicates."
        )
    else:
        caption_start = (
            "Median number of reviews required per species to reach "
            "dataset-specific agreement thresholds."
        )
    speedup_caption = (
        f" The speedup column reports the {AGREEMENT_THRESHOLD_SPEEDUP_BASELINE_LABEL} "
        "median review count divided by the "
        f"{AGREEMENT_THRESHOLD_SPEEDUP_TARGET_LABEL} median review count."
        if AGREEMENT_THRESHOLD_RENDER_SPEEDUP_COLUMN
        else ""
    )
    caption = (
        f"\\caption{{\\footnotesize {caption_start} Results of the "
        "best-performing method(s) are bolded. Thresholds are shown beside each "
        "metric name and are higher for camera-trap datasets, where lower "
        "thresholds are frequently reached at zero reviews by continuous-score "
        f"models.{speedup_caption}}}"
    )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\resizebox{\\textwidth}{!}{",
            tabular,
            "}",
            caption,
            "\\label{tab:agreement_table}",
            "\\end{table}",
        ]
    )


def timing_numeric_series(
    frame: pd.DataFrame,
    *columns: str,
) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def timing_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    z_score: float = 1.96,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if finite.size < 2:
        return mean, mean, mean
    half_width = float(z_score * np.std(finite, ddof=1) / np.sqrt(finite.size))
    return mean, max(0.0, mean - half_width), mean + half_width


def computational_cost_iteration_table(trace_frame: pd.DataFrame) -> pd.DataFrame:
    if trace_frame.empty:
        return pd.DataFrame()

    frame = trace_frame.copy()
    if "covariate_variant" in frame.columns:
        frame = frame.loc[frame["covariate_variant"].astype(str).str.lower() == "full"]
    selected_count = timing_numeric_series(
        frame,
        "selected_count",
        "trace__selected_count",
    )
    if selected_count.notna().any():
        frame = frame.loc[selected_count.fillna(0) > 0].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["fit_seconds"] = timing_numeric_series(
        frame,
        "mcmc_wall_time_seconds",
        "trace__mcmc_wall_time_seconds",
    ).fillna(0.0) + timing_numeric_series(
        frame,
        "score_calibration_wall_time_seconds",
        "trace__score_calibration_wall_time_seconds",
    ).fillna(
        0.0
    )
    frame["ranking_seconds"] = timing_numeric_series(
        frame,
        "selection_wall_time_seconds",
        "trace__selection_wall_time_seconds",
    )
    frame["iteration_seconds"] = timing_numeric_series(
        frame,
        "step_wall_time_seconds",
        "trace__step_wall_time_seconds",
    )
    frame = frame.loc[
        frame["model_family"].isin(COMPUTATIONAL_COST_MODEL_FAMILIES)
        & frame["selection_method"].isin(COMPUTATIONAL_COST_SELECTION_METHODS)
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    experiment_summary = (
        frame.groupby(
            [
                "dataset_kind",
                "species_key",
                "model_family",
                "selection_method",
                "experiment_id",
            ],
            dropna=False,
        )
        .agg(
            review_iterations=("review_count", "count"),
            fit_seconds=("fit_seconds", "mean"),
            ranking_seconds=("ranking_seconds", "mean"),
            iteration_seconds=("iteration_seconds", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (
        dataset_kind,
        model_family,
        selection_method,
    ), group in experiment_summary.groupby(
        ["dataset_kind", "model_family", "selection_method"],
        dropna=False,
        sort=False,
    ):
        row: dict[str, object] = {
            "dataset_kind": dataset_kind,
            "model_family": model_family,
            "selection_method": selection_method,
            "species": group["species_key"].nunique(),
            "review_iterations": int(group["review_iterations"].sum()),
        }
        for column in ("fit_seconds", "ranking_seconds", "iteration_seconds"):
            mean, lower, upper = timing_mean_ci(group[column])
            row[column] = mean
            row[f"{column}_lower"] = lower
            row[f"{column}_upper"] = upper
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["Dataset"] = (
        summary["dataset_kind"]
        .map(DATASET_LABELS)
        .fillna(summary["dataset_kind"].astype(str).str.replace("_", " ").str.title())
    )
    summary["Model"] = summary["model_family"].map(COMPUTATIONAL_COST_MODEL_LABELS)
    summary["Review method"] = summary["selection_method"].map(
        COMPUTATIONAL_COST_SELECTION_LABELS
    )
    summary["sort_key"] = (
        summary["dataset_kind"]
        .map({value: index for index, value in enumerate(DATASET_ORDER)})
        .fillna(len(DATASET_ORDER))
        .astype(int)
        * 100
        + summary["model_family"]
        .map(
            {
                value: index
                for index, value in enumerate(COMPUTATIONAL_COST_MODEL_FAMILIES)
            }
        )
        .fillna(len(COMPUTATIONAL_COST_MODEL_FAMILIES))
        .astype(int)
        * 10
        + summary["selection_method"]
        .map(
            {
                value: index
                for index, value in enumerate(COMPUTATIONAL_COST_SELECTION_METHODS)
            }
        )
        .fillna(len(COMPUTATIONAL_COST_SELECTION_METHODS))
        .astype(int)
    )
    return summary.sort_values("sort_key").reset_index(drop=True)


def format_timing_seconds(row: pd.Series, column: str) -> str:
    seconds = safe_float(row.get(column))
    if not np.isfinite(seconds):
        return "--"
    if 0.0 < abs(seconds) < 0.005:
        return "$<0.01$"
    lower = safe_float(row.get(f"{column}_lower"))
    upper = safe_float(row.get(f"{column}_upper"))
    if np.isfinite(lower) and np.isfinite(upper) and upper >= lower:
        half_width = max(seconds - lower, upper - seconds)
        return f"{seconds:.2f} $\\pm$ {half_width:.2f}"
    return f"{seconds:.2f}"


def computational_cost_iteration_latex(cost_table: pd.DataFrame) -> str:
    if cost_table.empty:
        return ""

    tabular_lines = [
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        (
            "Dataset & Model & Review method & Species & "
            "Model fitting (s, mean $\\pm$ 95\\% CI) & "
            "Ranking (s, mean $\\pm$ 95\\% CI) & "
            "Total (s, mean $\\pm$ 95\\% CI) \\\\"
        ),
        r"\midrule",
    ]
    for dataset_index, (_dataset, dataset_rows) in enumerate(
        cost_table.groupby("Dataset", sort=False)
    ):
        if dataset_index > 0:
            tabular_lines.append(r"\midrule")
        model_groups = list(dataset_rows.groupby("Model", sort=False))
        for model_index, (_model, model_rows) in enumerate(model_groups):
            model_row_count = len(model_rows)
            for row_offset, (_row_index, row) in enumerate(model_rows.iterrows()):
                dataset_cell = (
                    row["Dataset"] if model_index == 0 and row_offset == 0 else ""
                )
                model_cell = (
                    f"\\multirow{{{model_row_count}}}{{*}}{{{row['Model']}}}"
                    if row_offset == 0
                    else ""
                )
                row_cells = [
                    dataset_cell,
                    model_cell,
                    str(row["Review method"]),
                    str(int(row["species"])),
                    format_timing_seconds(row, "fit_seconds"),
                    format_timing_seconds(row, "ranking_seconds"),
                    format_timing_seconds(row, "iteration_seconds"),
                ]
                tabular_lines.append(" & ".join(row_cells) + r" \\")
            if model_index < len(model_groups) - 1:
                tabular_lines.append(r"\cmidrule(lr){2-7}")
    tabular_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(tabular_lines)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\resizebox{\textwidth}{!}{",
            tabular,
            "}",
            (
                r"\caption{\footnotesize Mean wall-clock time per active-review "
                r"iteration, averaged over species-experiment trajectories after "
                r"excluding terminal iterations with no selected reviews. Model "
                r"fitting is the occupancy-model MCMC time plus score-mixture "
                r"calibration for the decoupled continuous-score model. Ranking "
                r"is the time used to score and order candidate reviews. Total "
                r"includes recorded per-iteration overhead in addition to these "
                r"two components. Plus-minus values are 95\% confidence-interval "
                r"half-widths across species-experiment trajectory means; intervals "
                r"are omitted for entries below 0.01 s.}"
            ),
            r"\label{tab:computational_cost_iteration_times}",
            r"\end{table}",
        ]
    )


def plot_agreement_curves(
    aggregate_frame: pd.DataFrame,
    *,
    filename: str,
    summary_statistic: str = "mean",
    model_families: list[str],
    metric_specs: list[tuple[str, str]] | None = None,
) -> None:
    if aggregate_frame.empty:
        print("No agreement data available.")
        return
    if summary_statistic not in {"mean", "median"}:
        raise ValueError(
            "summary_statistic must be either 'mean' or 'median', "
            f"got {summary_statistic!r}."
        )

    model_families = [str(model_family) for model_family in model_families]
    aggregate_frame = aggregate_frame.loc[
        aggregate_frame["model_family"].isin(model_families)
    ].copy()
    if aggregate_frame.empty:
        family_label = " / ".join(
            MODEL_FAMILY_LABELS.get(
                model_family,
                model_family.replace("_", " ").title(),
            )
            for model_family in model_families
        )
        print(f"No {family_label} agreement data available.")
        return

    value_column = "median_value" if summary_statistic == "median" else "mean_value"
    metric_specs = list(
        AGREEMENT_CURVE_METRICS if metric_specs is None else metric_specs
    )
    metric_specs = [
        (metric_column, metric_label)
        for metric_column, metric_label in metric_specs
        if metric_column in set(aggregate_frame["metric"])
        and np.isfinite(
            aggregate_frame.loc[
                aggregate_frame["metric"] == metric_column,
                value_column,
            ].to_numpy(dtype=float)
        ).any()
    ]
    if not metric_specs:
        family_label = " / ".join(
            MODEL_FAMILY_LABELS.get(
                model_family,
                model_family.replace("_", " ").title(),
            )
            for model_family in model_families
        )
        print(f"No plottable {family_label} agreement metrics available.")
        return

    datasets = ordered_values(aggregate_frame["dataset_kind"], DATASET_ORDER)
    experiment_meta = (
        aggregate_frame[
            ["experiment_id", "selection_method", "model_family", "plot_label"]
        ]
        .drop_duplicates()
        .sort_values(
            by="experiment_id",
            key=lambda column: [
                experiment_sort_key(
                    experiment_id,
                    aggregate_frame.loc[
                        aggregate_frame["experiment_id"] == experiment_id
                    ].iloc[0],
                )
                for experiment_id in column
            ],
        )
    )

    fig, axes = plt.subplots(
        len(datasets),
        len(metric_specs),
        figsize=full_width_grid_size(
            len(metric_specs),
            len(datasets),
            row_scale=0.92,
        ),
        sharex="row",
        sharey=False,
    )
    axes = np.asarray(axes).reshape(len(datasets), len(metric_specs))

    for row_index, dataset_kind in enumerate(datasets):
        dataset_frame = aggregate_frame.loc[
            aggregate_frame["dataset_kind"] == dataset_kind
        ]
        x_min = float(dataset_frame["review_count"].min())
        x_max = float(dataset_frame["review_count"].max())
        positive_x = pd.to_numeric(
            dataset_frame["review_count"],
            errors="coerce",
        )
        positive_x = positive_x.loc[positive_x > 0]
        x_min_log = float(positive_x.min()) if not positive_x.empty else x_min
        for col_index, (metric_column, metric_label) in enumerate(metric_specs):
            axis = axes[row_index, col_index]
            metric_frame = dataset_frame.loc[dataset_frame["metric"] == metric_column]
            metric_sample_sizes = dataset_sample_sizes_from_frame(metric_frame)
            for experiment_row in experiment_meta.itertuples(index=False):
                experiment_metric = metric_frame.loc[
                    metric_frame["experiment_id"] == experiment_row.experiment_id
                ].sort_values("review_count")
                if experiment_metric.empty:
                    continue
                if summary_statistic == "median":
                    (
                        display_mean,
                        display_lower,
                        display_upper,
                    ) = metric_display_interval(
                        metric_column,
                        experiment_metric["median_value"],
                        experiment_metric["q25_value"],
                        experiment_metric["q75_value"],
                    )
                else:
                    (
                        display_mean,
                        display_lower,
                        display_upper,
                    ) = metric_display_confidence_interval(
                        metric_column,
                        experiment_metric["mean_value"],
                        experiment_metric["sem_value"],
                    )
                trend_mean = centered_rolling_trend(
                    display_mean,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_lower = centered_rolling_trend(
                    display_lower,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_upper = centered_rolling_trend(
                    display_upper,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_upper = np.maximum(trend_upper, trend_lower)
                color = SELECTION_COLORS.get(
                    experiment_row.selection_method,
                    "#444444",
                )
                linestyle = MODEL_FAMILY_LINESTYLES.get(
                    experiment_row.model_family,
                    "-",
                )
                if np.isfinite(trend_lower).any() and np.isfinite(trend_upper).any():
                    axis.fill_between(
                        experiment_metric["review_count"],
                        trend_lower,
                        trend_upper,
                        color=color,
                        alpha=0.10,
                        linewidth=0.0,
                    )
                # The faint line is the unsmoothed aggregate; the heavier line
                # is a visual trend, not a replacement for the raw variability.
                axis.plot(
                    experiment_metric["review_count"],
                    display_mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=AGREEMENT_RAW_TRACE_LINEWIDTH,
                    alpha=AGREEMENT_RAW_TRACE_ALPHA,
                )
                axis.plot(
                    experiment_metric["review_count"],
                    trend_mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=AGREEMENT_TREND_LINEWIDTH,
                )
                # axis.set_yscale("log")
                # axis.set_xscale("log")
            if row_index == 0:
                axis.set_title(metric_label, pad=4.0, fontsize=AXIS_TITLE_FONTSIZE)
            if col_index == 0:
                axis.set_ylabel(
                    dataset_label(
                        dataset_kind,
                        metric_sample_sizes.get(str(dataset_kind)),
                    )
                )
            if row_index == len(datasets) - 1:
                axis.set_xlabel("$t$")
            axis.set_xlim(x_min_log, x_max)

    visible_selection_methods = ordered_values(
        experiment_meta["selection_method"],
        SELECTION_ORDER,
    )
    policy_handles = [
        Line2D(
            [0],
            [0],
            color=SELECTION_COLORS.get(name, "#444444"),
            linewidth=PLOT_LINEWIDTH,
            label=SELECTION_LABELS.get(name, name.replace("_", " ").title()),
        )
        for name in visible_selection_methods
    ]
    visible_model_families = ordered_values(
        experiment_meta["model_family"],
        MODEL_FAMILY_ORDER,
    )
    family_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=MODEL_FAMILY_LINESTYLES.get(name, "-"),
            linewidth=PLOT_LINEWIDTH,
            label=MODEL_FAMILY_LABELS.get(name, name.replace("_", " ").title()),
        )
        for name in visible_model_families
    ]
    policy_legend = fig.legend(
        handles=policy_handles,
        ncol=max(len(policy_handles), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.085),
        frameon=False,
    )
    fig.add_artist(policy_legend)
    fig.legend(
        handles=family_handles,
        ncol=max(len(family_handles), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.135),
        frameon=False,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.14, 1.0, 0.94),
        pad=0.01,
        w_pad=0.01,
        h_pad=0.04,
        wspace=0.20,
        hspace=0.24,
    )
    maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)


def plot_average_model_family_agreement_curves(
    aggregate_frame: pd.DataFrame,
    *,
    filename: str,
    metric_specs: list[tuple[str, str]] | None = None,
) -> None:
    if aggregate_frame.empty:
        print("No shared-policy agreement data available.")
        return

    metric_specs = list(
        AGREEMENT_CURVE_METRICS if metric_specs is None else metric_specs
    )
    metric_specs = [
        (metric_column, metric_label)
        for metric_column, metric_label in metric_specs
        if metric_column in set(aggregate_frame["metric"])
        and np.isfinite(
            aggregate_frame.loc[
                aggregate_frame["metric"] == metric_column,
                "mean_value",
            ].to_numpy(dtype=float)
        ).any()
    ]
    if not metric_specs:
        print("No plottable shared-policy agreement metrics available.")
        return

    datasets = ordered_values(aggregate_frame["dataset_kind"], DATASET_ORDER)
    model_families = ordered_values(
        aggregate_frame["model_family"],
        MODEL_FAMILY_ORDER,
    )
    fig, axes = plt.subplots(
        len(datasets),
        len(metric_specs),
        figsize=full_width_grid_size(
            len(metric_specs),
            len(datasets),
            row_scale=0.92,
        ),
        sharex="row",
        sharey=False,
    )
    axes = np.asarray(axes).reshape(len(datasets), len(metric_specs))

    for row_index, dataset_kind in enumerate(datasets):
        dataset_frame = aggregate_frame.loc[
            aggregate_frame["dataset_kind"] == dataset_kind
        ]
        x_min = float(dataset_frame["review_count"].min())
        x_max = float(dataset_frame["review_count"].max())
        positive_x = pd.to_numeric(
            dataset_frame["review_count"],
            errors="coerce",
        )
        positive_x = positive_x.loc[positive_x > 0]
        x_min_log = float(positive_x.min()) if not positive_x.empty else x_min
        for col_index, (metric_column, metric_label) in enumerate(metric_specs):
            axis = axes[row_index, col_index]
            metric_frame = dataset_frame.loc[dataset_frame["metric"] == metric_column]
            metric_sample_sizes = dataset_sample_sizes_from_frame(metric_frame)
            for model_family in model_families:
                family_metric = metric_frame.loc[
                    metric_frame["model_family"] == model_family
                ].sort_values("review_count")
                if family_metric.empty:
                    continue
                (
                    display_mean,
                    display_lower,
                    display_upper,
                ) = metric_display_confidence_interval(
                    metric_column,
                    family_metric["mean_value"],
                    family_metric["sem_value"],
                )
                trend_mean = centered_rolling_trend(
                    display_mean,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_lower = centered_rolling_trend(
                    display_lower,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_upper = centered_rolling_trend(
                    display_upper,
                    window=AGREEMENT_TREND_WINDOW,
                )
                trend_upper = np.maximum(trend_upper, trend_lower)
                color = MODEL_FAMILY_COLORS.get(model_family, "#444444")
                if np.isfinite(trend_lower).any() and np.isfinite(trend_upper).any():
                    axis.fill_between(
                        family_metric["review_count"],
                        trend_lower,
                        trend_upper,
                        color=color,
                        alpha=0.10,
                        linewidth=0.0,
                    )
                axis.plot(
                    family_metric["review_count"],
                    display_mean,
                    color=color,
                    linestyle=MODEL_FAMILY_LINESTYLES.get(model_family, "-"),
                    linewidth=AGREEMENT_RAW_TRACE_LINEWIDTH,
                    alpha=AGREEMENT_RAW_TRACE_ALPHA,
                )
                axis.plot(
                    family_metric["review_count"],
                    trend_mean,
                    color=color,
                    linestyle=MODEL_FAMILY_LINESTYLES.get(model_family, "-"),
                    linewidth=AGREEMENT_TREND_LINEWIDTH,
                )
                # axis.set_xscale("log")
            if row_index == 0:
                axis.set_title(metric_label, pad=4.0, fontsize=AXIS_TITLE_FONTSIZE)
            if col_index == 0:
                axis.set_ylabel(
                    dataset_label(
                        dataset_kind,
                        metric_sample_sizes.get(str(dataset_kind)),
                    )
                )
            if row_index == len(datasets) - 1:
                axis.set_xlabel("$t$")
            axis.set_xlim(x_min_log, x_max)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_FAMILY_COLORS.get(model_family, "#444444"),
            linestyle=MODEL_FAMILY_LINESTYLES.get(model_family, "-"),
            linewidth=PLOT_LINEWIDTH,
            label=MODEL_FAMILY_LABELS.get(
                model_family,
                model_family.replace("_", " ").title(),
            ),
        )
        for model_family in model_families
    ]
    fig.legend(
        handles=family_handles,
        ncol=max(len(family_handles), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.085),
        frameon=False,
    )
    apply_tight_layout(
        fig,
        rect=(0.0, 0.10, 1.0, 0.94),
        pad=0.01,
        w_pad=0.01,
        h_pad=0.04,
        wspace=0.20,
        hspace=0.24,
    )
    maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)


def agreement_plot_filename(
    *,
    variant_slug: str,
    family_slug: str,
    summary_suffix: str,
) -> str:
    if variant_slug == "default" and family_slug == "cs":
        return f"coefficient_conclusion_agreement{summary_suffix}.png"
    return (
        "coefficient_conclusion_agreement_"
        f"{variant_slug}_{family_slug}{summary_suffix}.png"
    )


(
    coefficient_conclusion_frame,
    site_priority_frame,
) = build_ecological_agreement_frames(experiment_frame, covariate_variant="full")
agreement_frame = pd.concat(
    [coefficient_conclusion_frame, site_priority_frame],
    ignore_index=True,
    sort=False,
)
agreement_aggregate = aggregate_summary_curves(
    agreement_frame,
    metric_columns=[metric_column for metric_column, _ in AGREEMENT_CURVE_METRICS],
)
(
    null_coefficient_conclusion_frame,
    null_site_priority_frame,
) = build_ecological_agreement_frames(experiment_frame, covariate_variant="null")
null_agreement_frame = pd.concat(
    [null_coefficient_conclusion_frame, null_site_priority_frame],
    ignore_index=True,
    sort=False,
)
null_agreement_aggregate = aggregate_summary_curves(
    null_agreement_frame,
    metric_columns=[metric_column for metric_column, _ in AGREEMENT_CURVE_METRICS],
)
agreement_aggregates = {
    "full": agreement_aggregate,
    "null": null_agreement_aggregate,
}
(
    full_family_average_coefficient_frame,
    full_family_average_site_frame,
) = build_ecological_agreement_frames(
    experiment_frame,
    covariate_variant="full",
    experiment_ids=all_agreement_candidate_experiment_ids("full"),
)
full_family_average_agreement_frame = pd.concat(
    [full_family_average_coefficient_frame, full_family_average_site_frame],
    ignore_index=True,
    sort=False,
)
(
    null_family_average_coefficient_frame,
    null_family_average_site_frame,
) = build_ecological_agreement_frames(
    experiment_frame,
    covariate_variant="null",
    experiment_ids=all_agreement_candidate_experiment_ids("null"),
)
null_family_average_agreement_frame = pd.concat(
    [null_family_average_coefficient_frame, null_family_average_site_frame],
    ignore_index=True,
    sort=False,
)
family_average_agreement_aggregates = {
    "full": aggregate_shared_policy_model_family_curves(
        full_family_average_agreement_frame,
        metric_columns=[metric_column for metric_column, _ in AGREEMENT_CURVE_METRICS],
        model_families=AGREEMENT_FAMILY_AVERAGE_MODEL_FAMILIES,
    ),
    "null": aggregate_shared_policy_model_family_curves(
        null_family_average_agreement_frame,
        metric_columns=[metric_column for metric_column, _ in AGREEMENT_CURVE_METRICS],
        model_families=AGREEMENT_FAMILY_AVERAGE_MODEL_FAMILIES,
    ),
}
agreement_threshold_table = agreement_threshold_results_table(agreement_frame)
agreement_threshold_latex = agreement_threshold_results_latex(agreement_threshold_table)
computational_cost_table = computational_cost_iteration_table(trace_frame)
computational_cost_latex = computational_cost_iteration_latex(computational_cost_table)

display_frame(
    "Agreement metrics at 50 reviews",
    focal_metric_summary_table(
        agreement_aggregate,
        review_count=50,
        metric_specs=AGREEMENT_TABLE_METRICS,
        experiment_ids=AGREEMENT_EXPERIMENT_IDS,
    ).round(3),
)
display_frame(
    "Null-model agreement metrics at 50 reviews",
    focal_metric_summary_table(
        null_agreement_aggregate,
        review_count=50,
        metric_specs=AGREEMENT_TABLE_METRICS,
        experiment_ids=NULL_AGREEMENT_EXPERIMENT_IDS,
    ).round(3),
)

if not agreement_threshold_table.empty:
    display_frame(
        "Agreement threshold results",
        agreement_threshold_table,
    )
    saved_table_path = maybe_save_text(
        agreement_threshold_latex,
        table_output_dir,
        "agreement_threshold_steps.tex",
    )
    if saved_table_path is not None:
        print(f"Saved {saved_table_path.relative_to(repo_root)}")

if not computational_cost_table.empty:
    display_frame(
        "Computational cost per review iteration",
        computational_cost_table[
            [
                "Dataset",
                "Model",
                "Review method",
                "species",
                "fit_seconds",
                "ranking_seconds",
                "iteration_seconds",
            ]
        ].round(3),
    )
    saved_table_path = maybe_save_text(
        computational_cost_latex,
        table_output_dir,
        "computational_cost_iteration_times.tex",
    )
    if saved_table_path is not None:
        print(f"Saved {saved_table_path.relative_to(repo_root)}")

for covariate_variant, variant_slug in AGREEMENT_PLOT_VARIANTS:
    for model_families, family_slug in AGREEMENT_PLOT_MODEL_FAMILY_GROUPS:
        for summary_statistic, summary_suffix in [
            ("mean", ""),
            ("median", "_median"),
        ]:
            plot_agreement_curves(
                agreement_aggregates[covariate_variant],
                filename=agreement_plot_filename(
                    variant_slug=variant_slug,
                    family_slug=family_slug,
                    summary_suffix=summary_suffix,
                ),
                summary_statistic=summary_statistic,
                model_families=model_families,
            )

plot_average_model_family_agreement_curves(
    family_average_agreement_aggregates["full"],
    filename="coefficient_conclusion_agreement_family_average.png",
)
plot_average_model_family_agreement_curves(
    family_average_agreement_aggregates["null"],
    filename="coefficient_conclusion_agreement_null_model_family_average.png",
)


# %% [markdown]
# ## Species-level classifier quality and review burden
#
# This section separates three mechanisms:
#
# - classifier quality and the zero-review starting point,
# - prevalence and detection as constraints on how quickly review can help, and
# - when uncertainty-directed review reaches oracle-agreement thresholds faster
#   than positive-label harvesting.
#
# Average precision, empirical prevalence, and empirical detection probability
# are computed directly from the saved scores and true labels when those raw
# artifacts are available. Review-time summaries reuse the same oracle-agreement
# threshold logic as the manuscript threshold table.


# %%
def species_mean_agreement_frame(
    agreement_frame: pd.DataFrame,
    *,
    experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    if agreement_frame.empty:
        return pd.DataFrame()
    metric_columns = [
        metric_column
        for metric_column, _metric_label in AGREEMENT_CURVE_METRICS
        if metric_column in agreement_frame.columns
    ]
    if not metric_columns:
        return pd.DataFrame()

    frame = agreement_frame.copy()
    if experiment_ids is not None:
        frame = frame.loc[frame["experiment_id"].isin(experiment_ids)]
    if frame.empty:
        return pd.DataFrame()

    group_columns = [
        "dataset_kind",
        "species",
        "species_key",
        "experiment_id",
        "selection_method",
        "model_family",
        "covariate_variant",
        "plot_label",
        "review_count",
    ]
    grouped = (
        frame.groupby(group_columns, dropna=False, as_index=False)[metric_columns]
        .mean()
        .sort_values(["dataset_kind", "species_key", "experiment_id", "review_count"])
        .reset_index(drop=True)
    )
    grouped["mean_agreement"] = np.nanmean(
        grouped[metric_columns].to_numpy(dtype=float),
        axis=1,
    )
    return grouped


def reviews_to_fraction_of_best_agreement(
    trace: pd.DataFrame,
    target_fraction: float,
) -> int:
    trace = trace.sort_values("review_count")
    baseline = float(trace["mean_agreement"].iloc[0])
    best = float(trace["mean_agreement"].max())
    target = baseline + target_fraction * (best - baseline)
    hits = trace.loc[trace["mean_agreement"] >= target, "review_count"]
    if hits.empty:
        return int(trace["review_count"].max())
    return int(hits.iloc[0])


def score_calibration_dprime(row: pd.Series) -> float:
    pooled_sigma = np.sqrt(
        0.5
        * (
            safe_float(row.get("score_calibration__sigma0")) ** 2
            + safe_float(row.get("score_calibration__sigma1")) ** 2
        )
    )
    if not np.isfinite(pooled_sigma) or pooled_sigma <= 0.0:
        return float("nan")
    return float(
        (
            safe_float(row.get("score_calibration__mu1"))
            - safe_float(row.get("score_calibration__mu0"))
        )
        / pooled_sigma
    )


def build_score_quality_frame(
    agreement_frame: pd.DataFrame,
    *,
    experiment_id: str | None = None,
    experiment_ids: list[str] | None = None,
    target_fraction: float,
) -> pd.DataFrame:
    rows = []
    if experiment_ids is None:
        experiment_ids = [str(experiment_id)] if experiment_id is not None else []
    experiment_ids = [
        experiment_id for experiment_id in experiment_ids if experiment_id
    ]
    mean_agreement = species_mean_agreement_frame(
        agreement_frame,
        experiment_ids=experiment_ids,
    )
    if mean_agreement.empty:
        return pd.DataFrame()

    calibration_subset = trace_frame.loc[
        (trace_frame["experiment_id"].isin(experiment_ids))
        & (trace_frame["covariate_variant"] == "full")
        & (trace_frame["review_count"] == 0)
    ].copy()
    calibration_lookup = {
        (row.dataset_kind, row.species_key, row.experiment_id): row
        for row in calibration_subset.itertuples(index=False)
    }

    metric_columns = [
        metric_column
        for metric_column, _metric_label in AGREEMENT_CURVE_METRICS
        if metric_column in mean_agreement.columns
    ]
    for (_dataset_kind, _species_key), trace in mean_agreement.groupby(
        ["dataset_kind", "species_key"],
        sort=True,
    ):
        for candidate_experiment_id in experiment_ids:
            candidate_trace = trace.loc[
                trace["experiment_id"] == candidate_experiment_id
            ].sort_values("review_count")
            if (
                not candidate_trace.empty
                and (candidate_trace["review_count"] == 0).any()
            ):
                trace = candidate_trace
                break
        else:
            continue

        if trace.empty or not (trace["review_count"] == 0).any():
            continue

        zero_row = trace.loc[trace["review_count"] == 0].iloc[0]
        nonzero_trace = trace.loc[trace["review_count"] > 0]
        best_row = trace.loc[trace["mean_agreement"].astype(float).idxmax()]
        final_row = trace.loc[trace["review_count"].astype(int).idxmax()]
        calibration = calibration_lookup.get(
            (
                zero_row["dataset_kind"],
                zero_row["species_key"],
                zero_row["experiment_id"],
            )
        )
        base_row = {
            "dataset_kind": zero_row["dataset_kind"],
            "species": zero_row["species"],
            "species_key": zero_row["species_key"],
            "experiment_id": zero_row["experiment_id"],
            "dprime": (
                score_calibration_dprime(pd.Series(calibration._asdict()))
                if calibration is not None
                else float("nan")
            ),
            "zero_review_mean_agreement": float(zero_row["mean_agreement"]),
            "trajectory_mean_agreement": (
                float(nonzero_trace["mean_agreement"].mean())
                if not nonzero_trace.empty
                else float("nan")
            ),
            "best_mean_agreement": float(best_row["mean_agreement"]),
            "final_review_count": int(final_row["review_count"]),
            "final_mean_agreement": float(final_row["mean_agreement"]),
            "best_review_gain": float(
                best_row["mean_agreement"] - zero_row["mean_agreement"]
            ),
            "final_review_gain": float(
                final_row["mean_agreement"] - zero_row["mean_agreement"]
            ),
            "reviews_to_target": reviews_to_fraction_of_best_agreement(
                trace,
                target_fraction,
            ),
        }
        for metric_column in metric_columns:
            base_row[f"zero_review__{metric_column}"] = safe_float(
                zero_row.get(metric_column)
            )
            base_row[f"trajectory_mean__{metric_column}"] = (
                float(nonzero_trace[metric_column].mean())
                if not nonzero_trace.empty
                else float("nan")
            )
            base_row[f"final__{metric_column}"] = safe_float(
                final_row.get(metric_column)
            )
        rows.append(base_row)
    return pd.DataFrame(rows)


def score_quality_summary_table(quality_frame: pd.DataFrame) -> pd.DataFrame:
    if quality_frame.empty:
        return pd.DataFrame()

    rows = []
    for dataset_kind in ordered_values(quality_frame["dataset_kind"], DATASET_ORDER):
        subset = quality_frame.loc[quality_frame["dataset_kind"] == dataset_kind].copy()
        rows.append(
            {
                "dataset_kind": dataset_kind,
                "species": subset["species_key"].nunique(),
                "mean_dprime": float(subset["dprime"].mean()),
                "mean_zero_review_agreement": float(
                    subset["zero_review_mean_agreement"].mean()
                ),
                "mean_trajectory_agreement": float(
                    subset["trajectory_mean_agreement"].mean()
                ),
                "mean_best_review_gain": float(subset["best_review_gain"].mean()),
                "mean_reviews_to_target": float(subset["reviews_to_target"].mean()),
                "corr_dprime_vs_zero_review_agreement": finite_correlation(
                    subset,
                    "dprime",
                    "zero_review_mean_agreement",
                ),
                "corr_dprime_vs_trajectory_agreement": finite_correlation(
                    subset,
                    "dprime",
                    "trajectory_mean_agreement",
                ),
                "corr_dprime_vs_best_gain": finite_correlation(
                    subset,
                    "dprime",
                    "best_review_gain",
                ),
                "corr_dprime_vs_reviews_to_target": finite_correlation(
                    subset,
                    "dprime",
                    "reviews_to_target",
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_kind")


def plot_score_quality_vs_process(
    quality_frame: pd.DataFrame,
    *,
    filename: str,
) -> None:
    if quality_frame.empty:
        print("No score-quality data available.")
        return

    datasets = ordered_values(quality_frame["dataset_kind"], DATASET_ORDER)
    sample_sizes = dataset_sample_sizes_from_frame(quality_frame)
    fig, axes = plt.subplots(1, 2, figsize=full_width_grid_size(2, 1, row_scale=1.05))
    left_axis, right_axis = axes
    dprime_count = int(
        numeric_optional_column(quality_frame, "dprime")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .shape[0]
    )
    quality_column = "dprime" if dprime_count >= 4 else "average_precision"
    quality_label = (
        "Average precision (AP)" if quality_column == "average_precision" else "$d'$"
    )

    for dataset_kind in datasets:
        subset = quality_frame.loc[quality_frame["dataset_kind"] == dataset_kind].copy()
        color = DATASET_COLORS.get(dataset_kind, "#444444")
        label = dataset_label(dataset_kind, sample_sizes.get(str(dataset_kind)))

        left_axis.scatter(
            subset[quality_column],
            subset["zero_review_mean_agreement"],
            color=color,
            alpha=0.8,
            edgecolors="white",
            linewidth=PLOT_LINEWIDTH,
            label=label,
        )
        right_axis.scatter(
            subset[quality_column],
            subset["trajectory_mean_agreement"],
            color=color,
            alpha=0.8,
            edgecolors="white",
            linewidth=PLOT_LINEWIDTH,
        )

        if len(subset) >= 2:
            left_x = subset[quality_column].to_numpy(dtype=float)
            left_y = subset["zero_review_mean_agreement"].to_numpy(dtype=float)
            left_slope, left_intercept = np.polyfit(left_x, left_y, deg=1)
            left_x_line = np.linspace(np.nanmin(left_x), np.nanmax(left_x), 100)
            left_axis.plot(
                left_x_line,
                left_slope * left_x_line + left_intercept,
                color=color,
                linewidth=PLOT_LINEWIDTH,
            )

            right_x = subset[quality_column].to_numpy(dtype=float)
            right_y = subset["trajectory_mean_agreement"].to_numpy(dtype=float)
            right_slope, right_intercept = np.polyfit(right_x, right_y, deg=1)
            right_x_line = np.linspace(np.nanmin(right_x), np.nanmax(right_x), 100)
            right_axis.plot(
                right_x_line,
                right_slope * right_x_line + right_intercept,
                color=color,
                linewidth=PLOT_LINEWIDTH,
            )

    left_lines = []
    right_lines = []
    for dataset_kind in datasets:
        subset = quality_frame.loc[quality_frame["dataset_kind"] == dataset_kind]
        label = dataset_label(dataset_kind, sample_sizes.get(str(dataset_kind)))
        left_corr = finite_correlation(
            subset,
            quality_column,
            "zero_review_mean_agreement",
        )
        right_corr = finite_correlation(
            subset,
            quality_column,
            "trajectory_mean_agreement",
        )
        left_lines.append(f"{label}: r = {left_corr:.2f}")
        right_lines.append(f"{label}: r = {right_corr:.2f}")

    left_axis.text(
        0.98,
        0.96,
        "\n".join(left_lines),
        transform=left_axis.transAxes,
        ha="right",
        va="top",
    )
    right_axis.text(
        0.98,
        0.96,
        "\n".join(right_lines),
        transform=right_axis.transAxes,
        ha="right",
        va="top",
    )
    left_axis.set_xlabel(quality_label)
    left_axis.set_ylabel("Mean oracle agreement at $t=0$")
    right_axis.set_xlabel(quality_label)
    right_axis.set_ylabel("Mean reviewed-trajectory agreement")
    handles, labels = left_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        frameon=False,
    )

    apply_tight_layout(
        fig,
        rect=(0.0, 0.14, 1.0, 1.0),
        pad=0.01,
        w_pad=0.06,
        h_pad=0.04,
        wspace=0.18,
        hspace=0.18,
    )
    maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)


def binary_average_precision(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> float:
    labels_array = np.asarray(labels, dtype=float).reshape(-1)
    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(labels_array) & np.isfinite(scores_array)
    if not np.any(finite):
        return float("nan")

    labels_binary = labels_array[finite] > 0
    scores_finite = scores_array[finite]
    positive_count = int(labels_binary.sum())
    if positive_count == 0 or positive_count == labels_binary.size:
        return 1.0 if positive_count == labels_binary.size else float("nan")

    order = np.argsort(-scores_finite, kind="mergesort")
    sorted_labels = labels_binary[order]
    cumulative_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, sorted_labels.size + 1, dtype=float)
    precision_at_positive = cumulative_positives[sorted_labels] / ranks[sorted_labels]
    return float(np.mean(precision_at_positive))


def load_species_classifier_quality_metrics(
    species_frame: pd.DataFrame,
    sweep_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if species_frame.empty:
        return pd.DataFrame()

    for species_row in species_frame.itertuples(index=False):
        species_dir = resolve_species_artifact_dir(
            sweep_dir,
            species_index=getattr(species_row, "species_index", np.nan),
            dataset_kind=getattr(species_row, "dataset_kind", ""),
            species=getattr(species_row, "species", ""),
        )
        if species_dir is None:
            continue

        true_labels_path = species_dir / "inputs" / "true_labels.npy"
        reviewable_mask_path = species_dir / "dataset_metadata" / "reviewable_mask.npy"
        score_values_path = species_dir / "dataset_metadata" / "score_values.npy"
        if not (
            true_labels_path.exists()
            and reviewable_mask_path.exists()
            and score_values_path.exists()
        ):
            continue

        true_labels = np.asarray(np.load(true_labels_path), dtype=float)
        reviewable_mask = np.asarray(np.load(reviewable_mask_path), dtype=bool)
        score_values = np.asarray(np.load(score_values_path), dtype=float)
        if true_labels.ndim == reviewable_mask.ndim + 1 and true_labels.shape[0] == 1:
            true_labels = true_labels[0]
        if true_labels.shape != reviewable_mask.shape:
            continue
        if score_values.shape != reviewable_mask.shape:
            continue
        if reviewable_mask.ndim == 0:
            continue

        valid = reviewable_mask & np.isfinite(true_labels) & np.isfinite(score_values)
        if not np.any(valid):
            continue

        labels = true_labels[valid] > 0
        scores = score_values[valid]
        label_prevalence = float(np.mean(labels))
        observation_axes = tuple(range(1, valid.ndim))
        site_has_reviewable = valid.any(axis=observation_axes)
        site_has_positive = (valid & (true_labels > 0)).any(axis=observation_axes)
        site_occupancy_prevalence = (
            float(np.mean(site_has_positive[site_has_reviewable]))
            if np.any(site_has_reviewable)
            else float("nan")
        )
        site_shape = (site_has_positive.shape[0],) + (1,) * (valid.ndim - 1)
        occupied_replicates = valid & site_has_positive.reshape(site_shape)
        empirical_detection = (
            float(np.mean(true_labels[occupied_replicates] > 0))
            if np.any(occupied_replicates)
            else float("nan")
        )
        average_precision = binary_average_precision(labels, scores)

        rows.append(
            {
                "species_key": getattr(species_row, "species_key"),
                "dataset_kind": getattr(species_row, "dataset_kind"),
                "species": getattr(species_row, "species"),
                "n_reviewable_raw": int(np.sum(valid)),
                "positive_labels_raw": int(np.sum(labels)),
                "label_prevalence": label_prevalence,
                "site_occupancy_prevalence": site_occupancy_prevalence,
                "empirical_detection_probability": empirical_detection,
                "average_precision": average_precision,
                "average_precision_lift": (
                    average_precision / label_prevalence
                    if np.isfinite(average_precision) and label_prevalence > 0.0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def agreement_threshold_time_frame(
    agreement_frame: pd.DataFrame,
    *,
    metric_column: str,
    experiment_ids: list[str],
) -> pd.DataFrame:
    if agreement_frame.empty or metric_column not in agreement_frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (_dataset_kind, _species_key), species_group in agreement_frame.groupby(
        ["dataset_kind", "species_key"],
        sort=True,
    ):
        first_row = species_group.iloc[0]
        row: dict[str, object] = {
            "dataset_kind": first_row["dataset_kind"],
            "species": first_row["species"],
            "species_key": first_row["species_key"],
        }
        for experiment_id in experiment_ids:
            experiment_group = species_group.loc[
                species_group["experiment_id"] == experiment_id
            ]
            hit = agreement_threshold_species_hit(
                experiment_group,
                metric_column=metric_column,
                dataset_kind=_dataset_kind,
            )
            prefix = f"{experiment_id}__{metric_column}"
            row[f"{prefix}__review_count"] = safe_float(hit.get("step"))
            row[f"{prefix}__review_fraction"] = safe_float(hit.get("fraction"))
            row[f"{prefix}__reached"] = bool(hit.get("reached"))
            row[f"{prefix}__max_review"] = safe_float(hit.get("max_review"))
            row[f"{prefix}__threshold"] = safe_float(hit.get("threshold"))
        rows.append(row)

    return pd.DataFrame(rows)


def build_species_triage_frame(
    quality_frame: pd.DataFrame,
    classifier_quality_frame: pd.DataFrame,
    agreement_frame: pd.DataFrame,
) -> pd.DataFrame:
    if quality_frame.empty:
        return pd.DataFrame()

    frame = quality_frame.copy()
    if not classifier_quality_frame.empty:
        classifier_columns = [
            column
            for column in classifier_quality_frame.columns
            if column not in {"dataset_kind", "species"}
        ]
        frame = frame.merge(
            classifier_quality_frame[classifier_columns],
            on="species_key",
            how="left",
        )

    threshold_frame = agreement_threshold_time_frame(
        agreement_frame,
        metric_column="psi_probability_agreement",
        experiment_ids=[
            *continuous_score_bald_candidates("full"),
            *continuous_score_max_score_candidates("full"),
            *continuous_score_random_candidates("full"),
        ],
    )
    if not threshold_frame.empty:
        threshold_columns = [
            column
            for column in threshold_frame.columns
            if column not in {"dataset_kind", "species"}
        ]
        frame = frame.merge(
            threshold_frame[threshold_columns],
            on="species_key",
            how="left",
        )

    bald_fraction_columns = [
        f"{experiment_id}__psi_probability_agreement__review_fraction"
        for experiment_id in continuous_score_bald_candidates("full")
    ]
    max_score_fraction_columns = [
        f"{experiment_id}__psi_probability_agreement__review_fraction"
        for experiment_id in continuous_score_max_score_candidates("full")
    ]
    frame["continuous_score_bald__psi_probability_agreement__review_fraction"] = (
        coalesce_numeric_columns(frame, bald_fraction_columns)
    )
    frame["continuous_score_max_score__psi_probability_agreement__review_fraction"] = (
        coalesce_numeric_columns(frame, max_score_fraction_columns)
    )
    frame["psi_threshold_fraction_gap_max_score_minus_bald"] = pd.to_numeric(
        frame["continuous_score_max_score__psi_probability_agreement__review_fraction"],
        errors="coerce",
    ) - pd.to_numeric(
        frame["continuous_score_bald__psi_probability_agreement__review_fraction"],
        errors="coerce",
    )
    return frame


def finite_correlation(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> float:
    if left_column not in frame.columns or right_column not in frame.columns:
        return float("nan")
    subset = frame[[left_column, right_column]].copy()
    subset[left_column] = pd.to_numeric(subset[left_column], errors="coerce")
    subset[right_column] = pd.to_numeric(subset[right_column], errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return float("nan")
    if subset[left_column].nunique() < 2 or subset[right_column].nunique() < 2:
        return float("nan")
    return float(subset.corr().iloc[0, 1])


def numeric_optional_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def species_triage_summary_table(triage_frame: pd.DataFrame) -> pd.DataFrame:
    if triage_frame.empty:
        return pd.DataFrame()

    rows = []
    bald_fraction_column = (
        "continuous_score_bald__psi_probability_agreement__review_fraction"
    )
    for dataset_kind in ordered_values(triage_frame["dataset_kind"], DATASET_ORDER):
        subset = triage_frame.loc[triage_frame["dataset_kind"] == dataset_kind].copy()
        rows.append(
            {
                "dataset_kind": dataset_kind,
                "species": subset["species_key"].nunique(),
                "mean_average_precision": float(
                    numeric_optional_column(subset, "average_precision").mean()
                ),
                "mean_dprime": float(numeric_optional_column(subset, "dprime").mean()),
                "mean_label_prevalence": float(
                    numeric_optional_column(subset, "label_prevalence").mean()
                ),
                "mean_empirical_detection": float(
                    numeric_optional_column(
                        subset,
                        "empirical_detection_probability",
                    ).mean()
                ),
                "median_bald_psi_threshold_fraction": float(
                    numeric_optional_column(subset, bald_fraction_column).median()
                ),
                "mean_zero_review_agreement": float(
                    numeric_optional_column(
                        subset,
                        "zero_review_mean_agreement",
                    ).mean()
                ),
                "mean_trajectory_agreement": float(
                    numeric_optional_column(
                        subset,
                        "trajectory_mean_agreement",
                    ).mean()
                ),
                "mean_final_agreement": float(
                    numeric_optional_column(subset, "final_mean_agreement").mean()
                ),
                "mean_best_review_gain": float(
                    numeric_optional_column(subset, "best_review_gain").mean()
                ),
                "corr_ap_vs_zero_review_agreement": finite_correlation(
                    subset,
                    "average_precision",
                    "zero_review_mean_agreement",
                ),
                "corr_ap_vs_trajectory_agreement": finite_correlation(
                    subset,
                    "average_precision",
                    "trajectory_mean_agreement",
                ),
                "corr_dprime_vs_zero_review_agreement": finite_correlation(
                    subset,
                    "dprime",
                    "zero_review_mean_agreement",
                ),
                "corr_dprime_vs_trajectory_agreement": finite_correlation(
                    subset,
                    "dprime",
                    "trajectory_mean_agreement",
                ),
                "corr_detection_vs_bald_psi_threshold_fraction": finite_correlation(
                    subset,
                    "empirical_detection_probability",
                    bald_fraction_column,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_kind")


def marker_sizes_from_prevalence(values: pd.Series | np.ndarray) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    finite_values = value_array[np.isfinite(value_array)]
    if finite_values.size == 0:
        return np.full(value_array.shape, 34.0, dtype=float)
    max_value = max(float(np.nanmax(finite_values)), 1e-12)
    scaled_values = np.clip(value_array / max_value, 0.0, 1.0)
    scaled_values = np.where(np.isfinite(scaled_values), scaled_values, 0.25)
    return 22.0 + 72.0 * np.sqrt(scaled_values)


def add_scatter_trend_line(axis, x_values, y_values, *, color: str) -> None:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if finite.sum() < 3:
        return
    x_finite = x_array[finite]
    y_finite = y_array[finite]
    if np.nanmin(x_finite) == np.nanmax(x_finite):
        return
    slope, intercept = np.polyfit(x_finite, y_finite, deg=1)
    x_line = np.linspace(np.nanmin(x_finite), np.nanmax(x_finite), 100)
    axis.plot(
        x_line,
        slope * x_line + intercept,
        color=color,
        linewidth=PLOT_LINEWIDTH,
    )


def plot_species_classifier_triage(
    triage_frame: pd.DataFrame,
    *,
    filename: str,
) -> None:
    if triage_frame.empty:
        print("No species-triage data available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=full_width_grid_size(3, 1, row_scale=0.95))
    zero_axis, ap_axis, trajectory_axis = axes
    datasets = ordered_values(triage_frame["dataset_kind"], DATASET_ORDER)
    size_source = triage_frame.get(
        "site_occupancy_prevalence",
        triage_frame.get("label_prevalence", pd.Series(index=triage_frame.index)),
    )
    triage_frame = triage_frame.copy()
    triage_frame["_marker_size"] = marker_sizes_from_prevalence(size_source)
    dprime_count = int(
        numeric_optional_column(triage_frame, "dprime")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .shape[0]
    )
    quality_column = "dprime" if dprime_count >= 4 else "average_precision"
    quality_label = (
        "Average precision (AP)" if quality_column == "average_precision" else "$d'$"
    )
    ap_label = "Average precision (AP)"
    legend_handles = []

    for dataset_kind in datasets:
        subset = triage_frame.loc[triage_frame["dataset_kind"] == dataset_kind].copy()
        color = DATASET_COLORS.get(dataset_kind, "#444444")
        quality_values = numeric_optional_column(subset, quality_column).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        label_count = int(quality_values.dropna().shape[0])
        if label_count == 0:
            label_count = int(subset["species_key"].nunique())
        label = dataset_label(dataset_kind, label_count)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                label=label,
            )
        )

        zero_scatter = subset[
            np.isfinite(pd.to_numeric(subset[quality_column], errors="coerce"))
            & np.isfinite(
                pd.to_numeric(
                    subset["zero_review_mean_agreement"],
                    errors="coerce",
                )
            )
        ]
        zero_axis.scatter(
            zero_scatter[quality_column],
            zero_scatter["zero_review_mean_agreement"],
            s=zero_scatter["_marker_size"],
            color=color,
            alpha=0.80,
            edgecolors="white",
            linewidth=PLOT_LINEWIDTH,
        )
        add_scatter_trend_line(
            zero_axis,
            subset[quality_column],
            subset["zero_review_mean_agreement"],
            color=color,
        )

        ap_scatter = subset[
            np.isfinite(pd.to_numeric(subset["average_precision"], errors="coerce"))
            & np.isfinite(
                pd.to_numeric(
                    subset["zero_review_mean_agreement"],
                    errors="coerce",
                )
            )
        ]
        ap_axis.scatter(
            ap_scatter["average_precision"],
            ap_scatter["zero_review_mean_agreement"],
            s=ap_scatter["_marker_size"],
            color=color,
            alpha=0.80,
            edgecolors="white",
            linewidth=PLOT_LINEWIDTH,
        )
        add_scatter_trend_line(
            ap_axis,
            subset["average_precision"],
            subset["zero_review_mean_agreement"],
            color=color,
        )

        trajectory_scatter = subset[
            np.isfinite(pd.to_numeric(subset[quality_column], errors="coerce"))
            & np.isfinite(
                pd.to_numeric(
                    subset["trajectory_mean_agreement"],
                    errors="coerce",
                )
            )
        ]
        trajectory_axis.scatter(
            trajectory_scatter[quality_column],
            trajectory_scatter["trajectory_mean_agreement"],
            s=trajectory_scatter["_marker_size"],
            color=color,
            alpha=0.80,
            edgecolors="white",
            linewidth=PLOT_LINEWIDTH,
        )
        add_scatter_trend_line(
            trajectory_axis,
            subset[quality_column],
            subset["trajectory_mean_agreement"],
            color=color,
        )

    zero_lines = []
    ap_lines = []
    trajectory_lines = []
    for dataset_kind in datasets:
        subset = triage_frame.loc[triage_frame["dataset_kind"] == dataset_kind]
        corr = finite_correlation(
            subset,
            quality_column,
            "zero_review_mean_agreement",
        )
        if np.isfinite(corr):
            zero_lines.append(f"{dataset_label(dataset_kind)}: r = {corr:.2f}")
        corr = finite_correlation(
            subset,
            "average_precision",
            "zero_review_mean_agreement",
        )
        if np.isfinite(corr):
            ap_lines.append(f"{dataset_label(dataset_kind)}: r = {corr:.2f}")
        corr = finite_correlation(
            subset,
            quality_column,
            "trajectory_mean_agreement",
        )
        if np.isfinite(corr):
            trajectory_lines.append(f"{dataset_label(dataset_kind)}: r = {corr:.2f}")
    if zero_lines:
        zero_axis.text(
            0.98,
            0.96,
            "\n".join(zero_lines),
            transform=zero_axis.transAxes,
            ha="right",
            va="top",
        )
    if ap_lines:
        ap_axis.text(
            0.98,
            0.96,
            "\n".join(ap_lines),
            transform=ap_axis.transAxes,
            ha="right",
            va="top",
        )
    if trajectory_lines:
        trajectory_axis.text(
            0.98,
            0.96,
            "\n".join(trajectory_lines),
            transform=trajectory_axis.transAxes,
            ha="right",
            va="top",
        )

    zero_axis.set_title("Zero-review start")
    zero_axis.set_xlabel(quality_label)
    zero_axis.set_ylabel("Mean oracle agreement")
    ap_axis.set_title("Zero-review start")
    ap_axis.set_xlabel(ap_label)
    ap_axis.set_ylabel("Mean oracle agreement")
    trajectory_axis.set_title("Target EIG trajectory")
    trajectory_axis.set_xlabel(quality_label)
    trajectory_axis.set_ylabel("Mean oracle agreement")

    fig.legend(
        handles=legend_handles,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.065),
        frameon=False,
    )

    apply_tight_layout(
        fig,
        rect=(0.0, 0.14, 1.0, 1.0),
        pad=0.01,
        w_pad=0.04,
        h_pad=0.04,
        wspace=0.24,
        hspace=0.18,
    )
    maybe_save_figure(fig, figure_output_dir, filename)
    plt.show(block=False)
    plt.close(fig)


score_quality_frame = build_score_quality_frame(
    agreement_frame,
    experiment_ids=continuous_score_target_eig_candidates("full"),
    target_fraction=REVIEW_TARGET_FRACTION,
)
classifier_quality_frame = load_species_classifier_quality_metrics(
    species_frame,
    sweep_dir,
)
species_triage_frame = build_species_triage_frame(
    score_quality_frame,
    classifier_quality_frame,
    agreement_frame,
)

display_frame(
    "Species triage summary",
    species_triage_summary_table(species_triage_frame).round(3),
)

plot_species_classifier_triage(
    species_triage_frame,
    filename="classifier_quality_agreement.png",
)


# %% [markdown]
# ## Stopping criteria
#
# These post-hoc sweeps evaluate the three stopping rules emphasized in the
# manuscript on the cut-normal continuous-score Target EIG trajectories: an
# MCMC diagnostic, posterior-shift stabilization, and diminishing expected
# acquisition gain. Regret is computed from the same four agreement metrics
# used in the agreement plots.


# %%
if RUN_STOPPING_ANALYSIS:
    try:
        import stopping_analysis
    except ImportError:
        from bed import stopping_analysis

    stopping_dataset_filter = (
        None
        if DATASET_FILTER is None
        else [str(value).strip().lower() for value in np.atleast_1d(DATASET_FILTER)]
    )
    stopping_experiment_ids = list(STOPPING_EXPERIMENT_IDS)
    stopping_trace_frame = stopping_analysis.read_stopping_trace_export(
        combined_results_path,
        dataset_filter=stopping_dataset_filter,
        experiment_ids=stopping_experiment_ids,
    )
    stopping_frame = stopping_analysis.evaluate_stopping_sweeps(
        stopping_trace_frame,
        experiment_ids=stopping_experiment_ids,
        max_workers=stopping_analysis.STOPPING_MAX_WORKERS,
    )
    stopping_summary_frame = stopping_analysis.summarize_stopping_sweeps(stopping_frame)
    if not stopping_summary_frame.empty:
        display_frame(
            "Stopping criteria summary",
            stopping_summary_frame[
                [
                    "dataset_kind",
                    "criterion_label",
                    "parameter_value",
                    "mean_budget_fraction",
                    "mean_agreement_regret",
                    "trigger_rate",
                ]
            ]
            .sort_values(["dataset_kind", "criterion_label", "mean_agreement_regret"])
            .groupby(["dataset_kind", "criterion_label"], sort=False)
            .head(1)
            .round(3),
        )
        stopping_analysis.plot_stopping_tradeoff_sweeps(
            stopping_summary_frame,
            output_dir=figure_output_dir,
            filename="budget_regret_tradeoff_sweep",
        )
else:
    print("Skipping stopping-criteria analysis.")
