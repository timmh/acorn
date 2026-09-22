import csv
import io
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from occubed.paths import SWEEP_ROOT

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable

    class tqdm:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            return False

        def update(self, n=1):
            del n

        def close(self):
            return None

        @staticmethod
        def write(message):
            print(message, file=sys.stderr)


from occubed.summaries import (
    flatten_parameter_summary,
    iter_parameter_summary_arrays,
    mean_parameter_stat,
    parameter_summary_exists,
)

DEFAULT_SWEEP_ROOT = SWEEP_ROOT
DEFAULT_OUTPUT_NAME = "combined_species_results.csv"
DEFAULT_MAX_WORKERS = 8
EXCLUDED_SWEEP_FIELDS = {
    "output_root",
    "sweep_dir",
    "env_file",
    "experiment_script",
    "debug_mode",
    "continue_mode",
}
PARAMETER_SUMMARY_HEADER = (
    "parameter,parameter_index,mean,std,median,q05,q95,n_eff,r_hat"
)
COEFFICIENT_PARAMETER_PREFIXES = ("cov_state_", "cov_det_")
SPECIES_DIR_PATTERN = re.compile(
    r"^(?P<species_index>\d+)_(?P<dataset_kind>.+?)__(?P<species_slug>.+)$"
)
REQUIRED_COMPLETION_FILES = (
    "run_manifest.csv",
    "dataset_metadata/scalars.csv",
    "oracle/scalar_metrics.csv",
    "null_oracle/scalar_metrics.csv",
)


def slugify(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "item"


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_single_csv_row(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


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


def parse_species_dir_name(path):
    match = SPECIES_DIR_PATTERN.fullmatch(Path(path).name)
    if not match:
        return None
    return {
        "species_index": int(match.group("species_index")),
        "dataset_kind": match.group("dataset_kind"),
        "species_slug": match.group("species_slug"),
    }


def species_artifact_dirs(sweep_dir):
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        return []
    return sorted(
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and parse_species_dir_name(path) is not None
    )


def find_most_recent_sweep_dir(output_root):
    output_root = Path(output_root)
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and ((path / "submitted_jobs.csv").exists() or species_artifact_dirs(path))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No prior sweeps with species artifacts were found under {output_root}."
        )
    return max(
        candidates,
        key=lambda path: (
            path.name.rsplit("_sweep_", maxsplit=1)[-1],
            path.name,
        ),
    )


def with_prefix(row, prefix):
    merged = {}
    for key, value in row.items():
        key = str(key).strip()
        if not key:
            continue
        merged[f"{prefix}{key}"] = value
    return merged


def exclude_fields(row, excluded_fields):
    return {
        key: value
        for key, value in row.items()
        if str(key).strip() and str(key) not in excluded_fields
    }


def flatten_named_values(rows, name_field, value_field, prefix):
    flattened = {}
    for row in rows:
        name = str(row.get(name_field, "")).strip()
        if not name:
            continue
        flattened[f"{prefix}{slugify(name)}"] = row.get(value_field, "")
    return flattened


def flatten_parameter_summary_file(
    path,
    prefix,
    *,
    parameters=None,
    parameter_prefixes=(),
):
    if (
        parameters is None
        and tuple(parameter_prefixes) == COEFFICIENT_PARAMETER_PREFIXES
    ):
        return flatten_coefficient_summary_file(path, prefix)

    if parameters is None and not parameter_prefixes:
        return flatten_parameter_summary(path, prefix, slugify_func=slugify)

    flattened = {}
    for parameter_name, parameter_stats in iter_parameter_summary_arrays(
        path,
        parameters=parameters,
        parameter_prefixes=parameter_prefixes,
        stats=("mean", "q05", "q95"),
    ):
        parameter_key = slugify(parameter_name)
        for stat_name, values in parameter_stats.items():
            finite_values = np.asarray(values, dtype=float)
            if finite_values.size == 0 or not np.isfinite(finite_values).any():
                continue
            flattened[f"{prefix}{stat_name}__{parameter_key}"] = float(
                np.nanmean(finite_values)
            )
    return flattened


def legacy_csv_path_for_parameter_summary(path):
    path = Path(path)
    if path.suffix == ".csv":
        return path
    if path.suffix == ".npz":
        return path.with_suffix(".csv")
    return path / "parameter_summary.csv"


def flatten_coefficient_summary_file(path, prefix, tail_bytes=1_048_576):
    npz_path = Path(path)
    if npz_path.suffix != ".npz":
        npz_path = npz_path / "parameter_summary.npz"
    if npz_path.exists():
        flattened = {}
        for parameter_name, parameter_stats in iter_parameter_summary_arrays(
            npz_path,
            parameter_prefixes=COEFFICIENT_PARAMETER_PREFIXES,
            stats=("mean", "q05", "q95"),
        ):
            parameter_key = slugify(parameter_name)
            for stat_name, values in parameter_stats.items():
                finite_values = np.asarray(values, dtype=float)
                if finite_values.size == 0 or not np.isfinite(finite_values).any():
                    continue
                flattened[f"{prefix}{stat_name}__{parameter_key}"] = float(
                    np.nanmean(finite_values)
                )
        return flattened

    csv_path = legacy_csv_path_for_parameter_summary(path)
    if not csv_path.exists():
        return {}

    file_size = csv_path.stat().st_size
    with csv_path.open("rb") as handle:
        handle.seek(max(0, file_size - tail_bytes))
        text = handle.read().decode("utf-8", errors="ignore")

    lines = text.splitlines()
    if not lines:
        return {}
    if file_size > tail_bytes:
        lines = lines[1:]
    if lines and lines[0].startswith("parameter,"):
        lines = lines[1:]

    rows = csv.DictReader(
        io.StringIO(PARAMETER_SUMMARY_HEADER + "\n" + "\n".join(lines))
    )
    flattened = {}
    for row in rows:
        parameter_name = str(row.get("parameter", "")).strip()
        if not parameter_name.startswith(COEFFICIENT_PARAMETER_PREFIXES):
            continue
        parameter_key = slugify(parameter_name)
        for stat_name in ("mean", "q05", "q95"):
            value = safe_float(row.get(stat_name))
            if np.isfinite(value):
                flattened[f"{prefix}{stat_name}__{parameter_key}"] = value
    return flattened


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_numeric_vector(array, separator="|"):
    if array is None:
        return ""
    values = np.asarray(array, dtype=float).reshape(-1)
    formatted_values = []
    for value in values:
        if np.isnan(value):
            formatted_values.append("NaN")
        elif np.isposinf(value):
            formatted_values.append("Inf")
        elif np.isneginf(value):
            formatted_values.append("-Inf")
        else:
            formatted_values.append(f"{value:.17g}")
    return separator.join(formatted_values)


def load_parameter_mean_array(path, parameter_name):
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


def load_npy_array(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.asarray(np.load(path, allow_pickle=False), dtype=float)


def infer_covariate_variant(experiment_id, manifest_row):
    value = str(manifest_row.get("covariate_variant", "")).strip().lower()
    if value and value not in {"nan", "none"}:
        return "null" if "null" in value else value
    return "null" if str(experiment_id).endswith("__null") else "full"


def recommended_worker_count():
    return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))


def should_show_progress(show_progress):
    if show_progress is not None:
        return bool(show_progress)
    return sys.stderr.isatty()


def resolve_species_dir(sweep_dir, species_index, submitted_row):
    sweep_dir = Path(sweep_dir)
    dataset_kind = str(submitted_row.get("dataset_kind", "")).strip()
    species_name = str(submitted_row.get("species", "")).strip()
    expected_dir = (
        sweep_dir / f"{species_index:03d}_{dataset_kind}__{slugify(species_name)}"
    )
    recorded_output_dir = str(submitted_row.get("output_dir", "")).strip()

    candidates = [expected_dir]
    if recorded_output_dir:
        recorded_path = Path(recorded_output_dir)
        candidates.append(sweep_dir / recorded_path.name)
        candidates.append(recorded_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return expected_dir


def relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def submitted_rows_by_species_dir(sweep_dir, submitted_rows):
    sweep_dir = Path(sweep_dir)
    rows_by_dir_name = {}
    for submitted_row in submitted_rows:
        recorded_output_dir = str(submitted_row.get("output_dir", "")).strip()
        if not recorded_output_dir:
            continue
        recorded_path = Path(recorded_output_dir)
        rows_by_dir_name[recorded_path.name] = submitted_row
        rows_by_dir_name[(sweep_dir / recorded_path.name).name] = submitted_row
    return rows_by_dir_name


def missing_completion_artifacts(species_dir):
    species_dir = Path(species_dir)
    missing = [
        relative_path(species_dir / relative_file, species_dir)
        for relative_file in REQUIRED_COMPLETION_FILES
        if not (species_dir / relative_file).exists()
    ]
    for relative_dir in ("oracle", "null_oracle"):
        if not parameter_summary_exists(species_dir / relative_dir):
            missing.append(f"{relative_dir}/parameter_summary.npz or .csv")
    return missing


def submitted_row_for_species_dir(species_dir, submitted_rows_by_dir_name):
    species_dir = Path(species_dir)
    parsed = parse_species_dir_name(species_dir)
    if parsed is None:
        return {}

    submitted_row = dict(submitted_rows_by_dir_name.get(species_dir.name, {}))
    run_manifest = read_single_csv_row(species_dir / "run_manifest.csv")
    submitted_row.setdefault("dataset_kind", run_manifest.get("dataset_kind", ""))
    submitted_row.setdefault("species", run_manifest.get("target_species", ""))
    submitted_row.setdefault("output_dir", str(species_dir))
    if not str(submitted_row.get("dataset_kind", "")).strip():
        submitted_row["dataset_kind"] = parsed["dataset_kind"]
    if not str(submitted_row.get("species", "")).strip():
        submitted_row["species"] = parsed["species_slug"].replace("_", " ")
    return submitted_row


def discover_species_work_items(sweep_dir):
    sweep_dir = Path(sweep_dir)
    submitted_rows = read_csv_rows(sweep_dir / "submitted_jobs.csv")
    submitted_by_dir_name = submitted_rows_by_species_dir(sweep_dir, submitted_rows)
    completed_work_items = []
    skipped = []

    for species_dir in species_artifact_dirs(sweep_dir):
        parsed = parse_species_dir_name(species_dir)
        missing = missing_completion_artifacts(species_dir)
        if missing:
            skipped.append((species_dir, missing))
            continue
        submitted_row = submitted_row_for_species_dir(
            species_dir,
            submitted_by_dir_name,
        )
        completed_work_items.append(
            {
                "work_index": len(completed_work_items),
                "species_index": parsed["species_index"],
                "species_dir": species_dir,
                "submitted_row": submitted_row,
            }
        )

    for species_dir, missing in skipped:
        tqdm.write(
            "Skipping incomplete species directory "
            f"{relative_path(species_dir, sweep_dir)}: missing {', '.join(missing)}"
        )

    if not completed_work_items:
        raise FileNotFoundError(
            f"No completed species directories were found in {sweep_dir}."
        )
    return completed_work_items


def load_grouped_trace_rows(experiment_dir):
    grouped_rows = read_csv_rows(
        Path(experiment_dir) / "trace_grouped_oracle_distances.csv"
    )
    rows_by_step = {}
    for row in grouped_rows:
        step_index = str(row.get("step_index", "")).strip()
        rows_by_step.setdefault(step_index, []).append(row)
    return rows_by_step


def infer_experiment_metadata(experiment_id, base_row):
    base_experiment_id = (
        experiment_id[:-6] if str(experiment_id).endswith("__null") else experiment_id
    )
    covariate_variant = "null" if str(experiment_id).endswith("__null") else "full"
    if base_experiment_id.startswith("original_occu_cs"):
        model_family = "original_cs"
        model_label = "occu_cs"
        display_family = "Original occu_cs"
    elif base_experiment_id.startswith("adaptive_calibrated"):
        model_family = "calibrated_cs"
        model_label = "occu_cs_calibrated"
        display_family = "CS semi-supervised calibrated"
    else:
        model_family = "reviewed_bernoulli"
        model_label = "occu"
        display_family = "Reviewed-only Bernoulli"

    if "dual_bald_max_balanced" in base_experiment_id:
        selection_method = "dual_balanced"
        selection_label = "dual BALD/max-score"
    elif base_experiment_id.endswith("posterior_predictive_uncertainty"):
        selection_method = "posterior_predictive_uncertainty"
        selection_label = "posterior predictive uncertainty"
    elif base_experiment_id.endswith("revealed_targeted_eig"):
        selection_method = "revealed_targeted_eig"
        selection_label = "revealed targeted EIG"
    elif base_experiment_id.endswith(("target_eig", "targeted_eig")):
        selection_method = "target_eig"
        selection_label = "target EIG"
    elif base_experiment_id.endswith("epig"):
        selection_method = "epig"
        selection_label = "EPIG"
    elif base_experiment_id.endswith("max_score"):
        selection_method = "max_score"
        selection_label = "max score"
    elif base_experiment_id.endswith("bald_site_diverse"):
        selection_method = "bald_site_diverse"
        selection_label = "BALD (site diverse)"
    elif base_experiment_id.endswith("bald"):
        selection_method = "bald"
        selection_label = "BALD"
    else:
        selection_method = "random"
        selection_label = "random"

    n_site_covariates = safe_float(base_row.get("dataset__n_site_covariates"))
    n_obs_covariates = safe_float(base_row.get("dataset__n_obs_covariates"))
    if covariate_variant == "null":
        n_site_covariates = 0
        n_obs_covariates = 0

    suffix = " (null covariates)" if covariate_variant == "null" else ""
    return {
        "experiment_id": experiment_id,
        "display_name": f"{display_family} {selection_label}{suffix}",
        "model_family": model_family,
        "model_label": model_label,
        "selection_method": selection_method,
        "covariate_variant": covariate_variant,
        "oracle_variant": covariate_variant,
        "n_site_covariates": n_site_covariates,
        "n_obs_covariates": n_obs_covariates,
    }


def load_trace_rows(experiment_dir):
    experiment_dir = Path(experiment_dir)
    trace_rows = read_csv_rows(experiment_dir / "trace.csv")
    if trace_rows:
        return trace_rows

    rows = []
    steps_dir = experiment_dir / "steps"
    if not steps_dir.exists():
        return rows
    step_dirs = sorted(
        path
        for path in steps_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"step_\d+", path.name)
    )
    for step_dir in step_dirs:
        scalar_metrics = read_single_csv_row(step_dir / "scalar_metrics.csv")
        if not scalar_metrics:
            continue
        scalar_metrics = dict(scalar_metrics)
        scalar_metrics["step_index"] = str(int(step_dir.name.rsplit("_", 1)[-1]))
        rows.append(scalar_metrics)
    return rows


def grouped_rows_for_step(grouped_rows_by_step, step_dir, step_index):
    grouped_rows = grouped_rows_by_step.get(str(step_index), [])
    if grouped_rows:
        return grouped_rows
    return read_csv_rows(Path(step_dir) / "grouped_oracle_distances.csv")


def build_base_row(
    sweep_dir,
    species_index,
    submitted_row,
    sweep_manifest,
    species_dir=None,
):
    species_dir = (
        Path(species_dir)
        if species_dir is not None
        else resolve_species_dir(sweep_dir, species_index, submitted_row)
    )
    run_manifest = read_single_csv_row(species_dir / "run_manifest.csv")
    dataset_metadata = read_single_csv_row(
        species_dir / "dataset_metadata" / "scalars.csv"
    )
    oracle_metrics = read_single_csv_row(species_dir / "oracle" / "scalar_metrics.csv")
    oracle_parameter_summary = flatten_parameter_summary_file(
        species_dir / "oracle" / "parameter_summary.npz",
        "oracle__parameter_",
        parameter_prefixes=COEFFICIENT_PARAMETER_PREFIXES,
    )
    null_oracle_parameter_summary = flatten_parameter_summary_file(
        species_dir / "null_oracle" / "parameter_summary.npz",
        "null_oracle__parameter_",
        parameter_prefixes=COEFFICIENT_PARAMETER_PREFIXES,
    )

    row = dict(
        row_kind="species",
        sweep_name=Path(sweep_dir).name,
        species_index=species_index,
        dataset_kind=str(
            submitted_row.get("dataset_kind")
            or run_manifest.get("dataset_kind")
            or dataset_metadata.get("dataset_kind", "")
        ),
        species=str(
            submitted_row.get("species")
            or run_manifest.get("target_species")
            or dataset_metadata.get("target_species", "")
        ),
        target_label=str(
            run_manifest.get("target_label") or dataset_metadata.get("target_label", "")
        ),
    )
    row.update(
        with_prefix(exclude_fields(sweep_manifest, EXCLUDED_SWEEP_FIELDS), "sweep__")
    )
    row.update(with_prefix(run_manifest, "run__"))
    row.update(with_prefix(dataset_metadata, "dataset__"))
    row.update(with_prefix(oracle_metrics, "oracle__"))
    row.update(oracle_parameter_summary)
    row.update(null_oracle_parameter_summary)
    return row, species_dir


def build_species_row(base_row, species_dir):
    row = dict(base_row)
    row["oracle__psi_mean_values"] = format_numeric_vector(
        load_parameter_mean_array(
            Path(species_dir) / "oracle" / "parameter_summary.npz",
            "psi",
        )
    )
    return row


def build_step_rows(base_row, sweep_dir, experiment_dir):
    del sweep_dir
    experiment_dir = Path(experiment_dir)
    experiment_manifest = read_single_csv_row(
        experiment_dir / "experiment_manifest.csv"
    )
    experiment_id = str(experiment_manifest.get("experiment_id", experiment_dir.name))
    if not experiment_manifest:
        experiment_manifest = infer_experiment_metadata(experiment_id, base_row)
    covariate_variant = infer_covariate_variant(experiment_id, experiment_manifest)
    trace_rows = load_trace_rows(experiment_dir)
    grouped_rows_by_step = load_grouped_trace_rows(experiment_dir)

    if not trace_rows:
        row = dict(base_row)
        row.update(
            row_kind="experiment",
            experiment_id=experiment_id,
        )
        row.update(with_prefix(experiment_manifest, "experiment__"))
        return [row]

    rows = []
    for trace_row in trace_rows:
        step_index = str(trace_row.get("step_index", "")).strip()
        step_dir = experiment_dir / "steps" / f"step_{int(step_index):03d}"
        parameter_oracle_distances = flatten_named_values(
            read_csv_rows(step_dir / "parameter_oracle_distances.csv"),
            "parameter",
            "distance",
            "parameter_oracle_distance__",
        )
        parameter_summary = flatten_parameter_summary_file(
            step_dir / "parameter_summary.npz",
            "parameter_",
            parameter_prefixes=COEFFICIENT_PARAMETER_PREFIXES,
        )
        parameter_summary["parameter_std__psi"] = mean_parameter_stat(
            step_dir / "parameter_summary.npz",
            "psi",
            "std",
        )
        score_calibration = with_prefix(
            read_single_csv_row(step_dir / "score_calibration.csv"),
            "score_calibration__",
        )
        raw_psi_values = {}
        if covariate_variant == "full":
            raw_psi_values["trace__psi_mean_values"] = format_numeric_vector(
                load_npy_array(step_dir / "psi_mean.npy")
            )
        grouped_oracle_distances = flatten_named_values(
            grouped_rows_for_step(grouped_rows_by_step, step_dir, step_index),
            "group_name",
            "distance",
            "trace__grouped_oracle_distance__",
        )

        row = dict(base_row)
        row.update(
            row_kind="step",
            experiment_id=experiment_id,
            step_index=step_index,
        )
        row.update(with_prefix(experiment_manifest, "experiment__"))
        row.update(with_prefix(trace_row, "trace__"))
        for key, value in grouped_oracle_distances.items():
            row.setdefault(key, value)
        row.update(parameter_summary)
        row.update(parameter_oracle_distances)
        row.update(score_calibration)
        row.update(raw_psi_values)
        rows.append(row)
    return rows


def collect_species_export_rows(
    sweep_dir,
    species_index,
    submitted_row,
    sweep_manifest,
    species_dir=None,
):
    base_row, species_dir = build_base_row(
        sweep_dir=sweep_dir,
        species_index=species_index,
        submitted_row=submitted_row,
        sweep_manifest=sweep_manifest,
        species_dir=species_dir,
    )
    species_row = build_species_row(base_row, species_dir)
    experiments_dir = species_dir / "experiments"
    experiment_dirs = (
        sorted(path for path in experiments_dir.iterdir() if path.is_dir())
        if experiments_dir.exists()
        else []
    )
    if not experiment_dirs:
        return species_index, [species_row]

    rows = [species_row]
    for experiment_dir in experiment_dirs:
        rows.extend(
            build_step_rows(
                base_row,
                sweep_dir,
                experiment_dir,
            )
        )
    return species_index, rows


def collect_species_work_item_export_rows(
    sweep_dir,
    sweep_manifest,
    work_item,
):
    _species_index, rows = collect_species_export_rows(
        sweep_dir=sweep_dir,
        species_index=work_item["species_index"],
        submitted_row=work_item["submitted_row"],
        sweep_manifest=sweep_manifest,
        species_dir=work_item["species_dir"],
    )
    return work_item["work_index"], rows


def collect_export_rows(sweep_dir):
    sweep_dir = Path(sweep_dir)
    sweep_manifest = read_single_csv_row(sweep_dir / "sweep_manifest.csv")
    work_items = discover_species_work_items(sweep_dir)

    export_rows = []
    for work_item in work_items:
        _work_index, rows = collect_species_work_item_export_rows(
            sweep_dir=sweep_dir,
            sweep_manifest=sweep_manifest,
            work_item=work_item,
        )
        export_rows.extend(rows)
    return export_rows


def collect_export_rows_parallel(sweep_dir, workers=1, show_progress=None):
    sweep_dir = Path(sweep_dir)
    sweep_manifest = read_single_csv_row(sweep_dir / "sweep_manifest.csv")
    work_items = discover_species_work_items(sweep_dir)
    workers = max(1, int(workers))
    results_by_work_index = {}
    show_progress = should_show_progress(show_progress)

    with tqdm(
        total=len(work_items),
        desc="Exporting species",
        unit="species",
        disable=not show_progress,
    ) as progress:
        if workers == 1:
            for work_item in work_items:
                work_index, rows = collect_species_work_item_export_rows(
                    sweep_dir,
                    sweep_manifest,
                    work_item,
                )
                results_by_work_index[work_index] = rows
                progress.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        collect_species_work_item_export_rows,
                        sweep_dir,
                        sweep_manifest,
                        work_item,
                    )
                    for work_item in work_items
                ]
                for future in as_completed(futures):
                    work_index, rows = future.result()
                    results_by_work_index[work_index] = rows
                    progress.update(1)

    export_rows = []
    for work_index in range(len(work_items)):
        export_rows.extend(results_by_work_index[work_index])
    return export_rows


def main(
    sweep_dir=None,
    output_path=None,
    output_root=DEFAULT_SWEEP_ROOT,
    workers=None,
    show_progress=None,
):
    sweep_dir = (
        find_most_recent_sweep_dir(output_root)
        if sweep_dir is None
        else Path(sweep_dir)
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(sweep_dir) / DEFAULT_OUTPUT_NAME
    )
    rows = collect_export_rows_parallel(
        sweep_dir,
        workers=recommended_worker_count() if workers is None else workers,
        show_progress=show_progress,
    )
    write_csv_rows(output_path, rows)
    return output_path
