# %%
import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from occubed.paths import DEFAULT_SWEEP_DIR, FIGURE_DIR

# %%
plt.style.use("seaborn-v0_8-whitegrid")

WIDTH_SCALE = 2
NEURIPS_TEXT_WIDTH_PT = 397.48499 * WIDTH_SCALE
TEX_POINTS_PER_INCH = 72.27
FULL_WIDTH_IN = NEURIPS_TEXT_WIDTH_PT / TEX_POINTS_PER_INCH
BASE_FONT_SIZE_PT = 10.0

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
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.alpha": 0.25,
    }
)

DATASET_LABELS = {
    "acoustic": "Acoustic",
}
MODEL_FAMILY_LABELS = {
    "calibrated_cs": "Continuous-score",
    "reviewed_bernoulli": "Reviewed-only",
}
SELECTION_LABELS = {
    "random": "Random",
    "bald": "BALD",
    "bald_site_diverse": "Site-diverse BALD",
    "dual_balanced": "Dual BALD / max-score",
    "max_score": "Max score",
    "posterior_predictive_uncertainty": "PPU",
}

ITERATION_COMPONENTS = [
    ("mcmc_wall_time_seconds", "MCMC", "#4C78A8"),
    ("score_calibration_wall_time_seconds", "Score calibration", "#F58518"),
    ("selection_wall_time_seconds", "Review selection", "#54A24B"),
    ("overhead_wall_time_seconds", "Other step work", "#79706E"),
]

# %% [markdown]
# ## Configuration

# %%
SWEEP_DIR = os.environ.get(
    "OCCUBED_TIMING_SWEEP_DIR",
    str(DEFAULT_SWEEP_DIR),
)
COMBINED_RESULTS_CSV = os.environ.get(
    "OCCUBED_TIMING_RESULTS_CSV",
    "combined_species_results.csv",
)
COVARIATE_VARIANT = os.environ.get(
    "OCCUBED_TIMING_COVARIATE_VARIANT",
    "full",
)
SAVE_FIGURES = os.environ.get("OCCUBED_TIMING_WRITE_FIGURES", "1") != "0"
FIGURE_OUTPUT_DIR = os.environ.get(
    "OCCUBED_TIMING_FIGURE_OUTPUT_DIR",
    str(FIGURE_DIR),
)
MIN_SEGMENT_LABEL_FRACTION = 0.08

# %% [markdown]
# ## Loading helpers


# %%
def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def resolve_existing_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    for path in (candidate, repo_root / candidate, Path.cwd() / candidate):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve path: {value}")


def resolve_results_csv_path(sweep_dir: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    for path in (candidate, sweep_dir / candidate, Path.cwd() / candidate):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve combined results CSV: {value}")


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


def infer_selection_method(experiment_id: object) -> str:
    text = str(experiment_id).strip()
    if "dual_bald_max_balanced" in text:
        return "dual_balanced"
    if "bald_site_diverse" in text:
        return "bald_site_diverse"
    if text.replace("__null", "").endswith("posterior_predictive_uncertainty"):
        return "posterior_predictive_uncertainty"
    if text.replace("__null", "").endswith("max_score"):
        return "max_score"
    if text.replace("__null", "").endswith("bald"):
        return "bald"
    return "random"


def infer_model_family(experiment_id: object) -> str:
    text = str(experiment_id).strip()
    if text.startswith("adaptive_calibrated"):
        return "calibrated_cs"
    if text.startswith("reviewed"):
        return "reviewed_bernoulli"
    return ""


def infer_covariate_variant(experiment_id: object) -> str:
    text = str(experiment_id).strip().lower()
    if text.endswith("__null"):
        return "null"
    return "full"


def read_timing_export(path: Path) -> pd.DataFrame:
    desired_columns = [
        "row_kind",
        "species_index",
        "dataset_kind",
        "species",
        "target_label",
        "run__dataset_kind",
        "run__target_species",
        "dataset__dataset_kind",
        "dataset__target_species",
        "experiment_id",
        "experiment__model_family",
        "experiment__selection_method",
        "experiment__covariate_variant",
        "experiment__oracle_variant",
        "experiment__total_score_calibration_wall_time_seconds",
        "experiment__total_mcmc_wall_time_seconds",
        "experiment__total_selection_wall_time_seconds",
        "experiment__total_step_wall_time_seconds",
        "experiment__experiment_wall_time_seconds",
        "trace__step_index",
        "trace__review_count",
        "trace__selected_count",
        "trace__score_calibration_wall_time_seconds",
        "trace__mcmc_wall_time_seconds",
        "trace__selection_wall_time_seconds",
        "trace__step_wall_time_seconds",
        "run__dataset_load_wall_time_seconds",
        "run__experiments_wall_time_seconds",
        "run__dataset_compute_wall_time_seconds",
        "run__oracle_mcmc_wall_time_seconds",
        "run__null_oracle_mcmc_wall_time_seconds",
        "run__artifact_write_wall_time_seconds",
        "dataset__n_sites",
        "dataset__n_replicates",
        "dataset__n_reviewable",
    ]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in desired_columns if column in available]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)

    frame["dataset_kind"] = coalesce_text_columns(
        frame,
        ["dataset_kind", "run__dataset_kind", "dataset__dataset_kind"],
    ).str.lower()
    frame["species"] = coalesce_text_columns(
        frame,
        ["species", "run__target_species", "dataset__target_species"],
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

    missing_selection = (
        frame["selection_method"].eq("") & frame["experiment_id"].notna()
    )
    frame.loc[missing_selection, "selection_method"] = frame.loc[
        missing_selection,
        "experiment_id",
    ].map(infer_selection_method)

    missing_model = frame["model_family"].eq("") & frame["experiment_id"].notna()
    frame.loc[missing_model, "model_family"] = frame.loc[
        missing_model,
        "experiment_id",
    ].map(infer_model_family)

    missing_covariates = (
        frame["covariate_variant"].eq("") & frame["experiment_id"].notna()
    )
    frame.loc[missing_covariates, "covariate_variant"] = frame.loc[
        missing_covariates,
        "experiment_id",
    ].map(infer_covariate_variant)
    frame["covariate_variant"] = frame["covariate_variant"].map(
        lambda value: "null" if "null" in str(value).lower() else str(value).lower()
    )

    numeric_columns = [
        column
        for column in frame.columns
        if "seconds" in column
        or column.endswith("_count")
        or column.endswith("_index")
        or column.startswith("dataset__n_")
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def maybe_save_figure(fig, output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(filename).with_suffix(".pdf")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    return path


# %% [markdown]
# ## Load timing data

# %%
repo_root = find_repo_root()
sweep_dir = resolve_existing_path(repo_root, SWEEP_DIR)
results_csv_path = resolve_results_csv_path(sweep_dir, COMBINED_RESULTS_CSV)
figure_output_dir = (repo_root / FIGURE_OUTPUT_DIR) if SAVE_FIGURES else None

timing_frame = read_timing_export(results_csv_path)
trace_frame = timing_frame.loc[timing_frame["row_kind"].eq("step")].copy()
trace_frame = trace_frame.loc[
    trace_frame["covariate_variant"].eq(COVARIATE_VARIANT)
].copy()
if "trace__selected_count" in trace_frame.columns:
    iteration_trace_frame = trace_frame.loc[
        trace_frame["trace__selected_count"].fillna(0) > 0
    ].copy()
else:
    iteration_trace_frame = trace_frame.copy()

print(f"Using sweep directory: {sweep_dir}")
print(f"Using combined results CSV: {results_csv_path}")
print(f"Covariate variant: {COVARIATE_VARIANT}")
print(f"Saving figures: {SAVE_FIGURES}")
if figure_output_dir is not None:
    print(f"Figure output directory: {figure_output_dir}")

display(
    pd.DataFrame(
        {
            "rows": timing_frame["row_kind"].value_counts(dropna=False),
        }
    )
)
display(
    iteration_trace_frame.groupby(["dataset_kind", "model_family", "selection_method"])
    .agg(
        species=("species_key", "nunique"),
        review_iterations=("trace__step_index", "count"),
        median_review_count=("trace__review_count", "median"),
    )
    .sort_index()
)

# %% [markdown]
# The per-step timings show that the acquisition step itself is tiny relative to posterior fitting. The summaries below exclude terminal rows with zero selected reviews, so the reported units are actual review iterations rather than whole experiments.

# %%
step_timing_columns = [
    "trace__score_calibration_wall_time_seconds",
    "trace__mcmc_wall_time_seconds",
    "trace__selection_wall_time_seconds",
    "trace__step_wall_time_seconds",
]
step_summary = iteration_trace_frame[step_timing_columns].describe(
    percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
)
display(step_summary.round(3))

# %% [markdown]
# ## Build cost tables


# %%
def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def build_iteration_costs(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    trace = trace.sort_values(
        ["dataset_kind", "species", "experiment_id", "trace__review_count"],
        na_position="last",
    )
    for row in trace.itertuples(index=False):
        mcmc_seconds = finite_float(
            getattr(row, "trace__mcmc_wall_time_seconds", np.nan)
        )
        calibration_seconds = finite_float(
            getattr(row, "trace__score_calibration_wall_time_seconds", np.nan)
        )
        selection_seconds = finite_float(
            getattr(row, "trace__selection_wall_time_seconds", np.nan)
        )
        step_seconds = finite_float(
            getattr(row, "trace__step_wall_time_seconds", np.nan)
        )
        recorded_other = max(
            0.0,
            step_seconds - mcmc_seconds - calibration_seconds - selection_seconds,
        )
        rows.append(
            {
                "dataset_kind": getattr(row, "dataset_kind"),
                "species": getattr(row, "species"),
                "species_key": getattr(row, "species_key"),
                "experiment_id": getattr(row, "experiment_id"),
                "model_family": getattr(row, "model_family"),
                "selection_method": getattr(row, "selection_method"),
                "step_index": finite_float(getattr(row, "trace__step_index", np.nan)),
                "review_count": finite_float(
                    getattr(row, "trace__review_count", np.nan)
                ),
                "selected_count": finite_float(
                    getattr(row, "trace__selected_count", np.nan)
                ),
                "mcmc_wall_time_seconds": mcmc_seconds,
                "score_calibration_wall_time_seconds": calibration_seconds,
                "selection_wall_time_seconds": selection_seconds,
                "overhead_wall_time_seconds": recorded_other,
                "step_wall_time_seconds": step_seconds,
            }
        )
    return pd.DataFrame(rows)


def summarize_experiment_iterations(iterations: pd.DataFrame) -> pd.DataFrame:
    return (
        iterations.groupby(
            [
                "dataset_kind",
                "species",
                "species_key",
                "experiment_id",
                "model_family",
                "selection_method",
            ],
            dropna=False,
        )
        .agg(
            review_iterations=("step_index", "count"),
            median_review_count=("review_count", "median"),
            **{column: (column, "median") for column, _, _ in ITERATION_COMPONENTS},
            step_wall_time_seconds=("step_wall_time_seconds", "median"),
        )
        .reset_index()
    )


iteration_costs = build_iteration_costs(iteration_trace_frame)
experiment_iteration_costs = summarize_experiment_iterations(iteration_costs)

display(
    experiment_iteration_costs.groupby(["dataset_kind", "model_family"])
    .agg(
        species=("species_key", "nunique"),
        experiments=("experiment_id", "nunique"),
        review_iterations=("review_iterations", "sum"),
        median_iteration_seconds=(
            "step_wall_time_seconds",
            lambda values: np.nanmedian(values),
        ),
        p90_iteration_seconds=(
            "step_wall_time_seconds",
            lambda values: np.nanpercentile(values, 90),
        ),
    )
    .round(2)
)

# %% [markdown]
# ## Plot helpers


# %%
def stacked_barh(
    summary: pd.DataFrame,
    *,
    label_column: str,
    components: list[tuple[str, str, str]],
    scale: float,
    xlabel: str,
    filename: str,
    title: str | None = None,
) -> Path | None:
    if summary.empty:
        raise ValueError("No rows available to plot.")
    plot_frame = summary.copy()
    component_columns = [column for column, _, _ in components]
    values = plot_frame[component_columns].fillna(0.0).clip(lower=0.0) / scale
    totals = values.sum(axis=1)
    y_positions = np.arange(len(plot_frame))
    fig_height = max(1.9, 0.48 * len(plot_frame) + 1.0)
    fig, axis = plt.subplots(figsize=(FULL_WIDTH_IN, fig_height))
    left = np.zeros(len(plot_frame), dtype=float)

    for column, label, color in components:
        segment = values[column].to_numpy(dtype=float)
        axis.barh(
            y_positions,
            segment,
            left=left,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )
        for y_position, start, width, total in zip(y_positions, left, segment, totals):
            if total > 0 and width / total >= MIN_SEGMENT_LABEL_FRACTION:
                axis.text(
                    start + width / 2.0,
                    y_position,
                    f"{width:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7.5,
                )
        left += segment

    offset = max(float(totals.max()) * 0.012, 0.02)
    for y_position, total in zip(y_positions, totals):
        axis.text(
            total + offset,
            y_position,
            f"{total:.1f}",
            ha="left",
            va="center",
            fontsize=8.0,
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(plot_frame[label_column].tolist())
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    if title:
        axis.set_title(title)
    axis.set_xlim(0.0, max(float(totals.max()) * 1.14, 1.0))
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=min(len(components), 4),
        frameon=False,
    )
    fig.tight_layout(pad=0.05)
    saved_path = maybe_save_figure(fig, figure_output_dir, filename)
    plt.show()
    return saved_path


# %% [markdown]
# ## Iteration cost by dataset
#
# The dataset-level plot summarizes a typical review iteration after first taking
# the median iteration cost within each species and experiment. This keeps
# longer-running experiments from dominating the aggregate only because they have
# more recorded iterations.

# %%
dataset_iteration_summary = (
    experiment_iteration_costs.groupby("dataset_kind")[
        [column for column, _, _ in ITERATION_COMPONENTS]
    ]
    .median()
    .reset_index()
)
dataset_iteration_summary["label"] = dataset_iteration_summary["dataset_kind"].map(
    DATASET_LABELS
)
dataset_iteration_summary["sort_key"] = dataset_iteration_summary["dataset_kind"].map(
    {"acoustic": 0}
)
dataset_iteration_summary = dataset_iteration_summary.sort_values(
    "sort_key"
).reset_index(drop=True)

dataset_iteration_display = dataset_iteration_summary[
    ["label"] + [column for column, _, _ in ITERATION_COMPONENTS]
].copy()
display(dataset_iteration_display.set_index("label").round(2))
dataset_iteration_figure_path = stacked_barh(
    dataset_iteration_summary,
    label_column="label",
    components=ITERATION_COMPONENTS,
    scale=1.0,
    xlabel="Median wall-clock time per review iteration (seconds)",
    filename="computational_cost_iteration_dataset_breakdown",
    title="Per-iteration cost is dominated by posterior fitting",
)
dataset_iteration_figure_path

# %% [markdown]
# ## Iteration cost by model family
#
# Selection times are orders of magnitude smaller than fitting times, so the
# model-family view is cleaner than a policy-by-policy plot. The overhead segment
# is the non-fit, non-calibration, non-selection part of the recorded iteration,
# including posterior-distance summaries and parameter summaries.

# %%
model_iteration_summary = (
    experiment_iteration_costs.groupby(["dataset_kind", "model_family"])[
        [column for column, _, _ in ITERATION_COMPONENTS]
    ]
    .median()
    .reset_index()
)
model_iteration_summary["dataset_label"] = model_iteration_summary["dataset_kind"].map(
    DATASET_LABELS
)
model_iteration_summary["model_label"] = model_iteration_summary["model_family"].map(
    MODEL_FAMILY_LABELS
)
model_iteration_summary["label"] = (
    model_iteration_summary["dataset_label"]
    + "\n"
    + model_iteration_summary["model_label"]
)
model_iteration_summary["sort_key"] = model_iteration_summary["dataset_kind"].map(
    {"acoustic": 0}
) * 10 + model_iteration_summary["model_family"].map(
    {"calibrated_cs": 0, "reviewed_bernoulli": 1}
)
model_iteration_summary = model_iteration_summary.sort_values("sort_key").reset_index(
    drop=True
)

model_iteration_display = model_iteration_summary[
    ["label"] + [column for column, _, _ in ITERATION_COMPONENTS]
].copy()
display(model_iteration_display.set_index("label").round(2))
model_iteration_figure_path = stacked_barh(
    model_iteration_summary,
    label_column="label",
    components=ITERATION_COMPONENTS,
    scale=1.0,
    xlabel="Median wall-clock time per review iteration (seconds)",
    filename="computational_cost_iteration_model_breakdown",
    title="Acquisition is negligible within each review iteration",
)
model_iteration_figure_path

# %% [markdown]
# ## Optional policy detail
#
# This table keeps the policy-level detail available without making it the main
# figure. It is useful for checking that the model-family aggregation is not
# hiding a large acquisition-time difference.

# %%
policy_summary = (
    experiment_iteration_costs.groupby(
        ["dataset_kind", "model_family", "selection_method"]
    )
    .agg(
        species=("species_key", "nunique"),
        experiments=("experiment_id", "nunique"),
        review_iterations=("review_iterations", "sum"),
        median_iteration_seconds=("step_wall_time_seconds", "median"),
        median_mcmc_seconds=("mcmc_wall_time_seconds", "median"),
        median_selection_seconds=("selection_wall_time_seconds", "median"),
        median_calibration_seconds=("score_calibration_wall_time_seconds", "median"),
        median_overhead_seconds=("overhead_wall_time_seconds", "median"),
    )
    .reset_index()
)
policy_summary["dataset"] = policy_summary["dataset_kind"].map(DATASET_LABELS)
policy_summary["model"] = policy_summary["model_family"].map(MODEL_FAMILY_LABELS)
policy_summary["policy"] = policy_summary["selection_method"].map(SELECTION_LABELS)
display(
    policy_summary[
        [
            "dataset",
            "model",
            "policy",
            "species",
            "experiments",
            "review_iterations",
            "median_iteration_seconds",
            "median_mcmc_seconds",
            "median_selection_seconds",
            "median_calibration_seconds",
            "median_overhead_seconds",
        ]
    ].round(3)
)
