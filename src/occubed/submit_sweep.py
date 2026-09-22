import csv
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # pragma: no cover - species helpers do not need dotenv.
    dotenv_values = None
    load_dotenv = None

try:
    import submitit
except ImportError:  # pragma: no cover - species helpers do not need submitit.
    submitit = None

from occubed.paths import (
    ACOUSTIC_DATA_DIR,
    DATA_DIR,
    PACKAGE_DIR,
    SCRIPT_DIR,
    SWEEP_ROOT,
)

EXPERIMENT_SCRIPT = SCRIPT_DIR / "run_reviews.py"
DEFAULT_SWEEP_ROOT = SWEEP_ROOT
SUPPORTED_DATASETS = ("acoustic", "iwildcam")
DEFAULT_DATASETS = ("acoustic", "iwildcam")
DEFAULT_TOP_SPECIES_COUNT = 50
DEFAULT_ACOUSTIC_REVIEWS_PER_STEP = 5
DEFAULT_ACOUSTIC_MAX_REVIEWS = sys.maxsize
DEFAULT_IWILDCAM_REVIEWS_PER_STEP = 25
DEFAULT_IWILDCAM_MAX_REVIEWS = 1000
ACOUSTIC_ANNOTATION_ROOT = ACOUSTIC_DATA_DIR / "annotations"
ACOUSTIC_PERCH_LOGIT_PARTS = ACOUSTIC_DATA_DIR / "perch_v2" / "window_logits_parts"
IWILDCAM_TARGET_SPECIES_MAPPING = (
    DATA_DIR / "iwildcam2022" / "target_species_mapping.csv"
)
REQUEUE_RECOVERY_TIMEOUT_SECONDS = 900
REQUEUE_RECOVERY_POLL_INTERVAL_SECONDS = 15


def slugify(value):
    return (
        "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip(
            "_"
        )
        or "item"
    )


def normalize_species(species):
    normalized = []
    for item in species:
        if isinstance(item, (list, tuple)):
            normalized.extend(normalize_species(item))
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def normalize_datasets(dataset):
    if dataset is None:
        return list(DEFAULT_DATASETS)
    if isinstance(dataset, (list, tuple)):
        values = dataset
    else:
        values = str(dataset).split(",")
    normalized = []
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_DATASETS:
            raise ValueError(
                "Unsupported dataset "
                f"`{value}`. Expected one of: {', '.join(SUPPORTED_DATASETS)}."
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("Provide at least one dataset.")
    return normalized


def default_species_by_dataset(top_k=DEFAULT_TOP_SPECIES_COUNT, dataset_kinds=None):
    if dataset_kinds is None:
        dataset_kinds = DEFAULT_DATASETS
    return {
        dataset_kind: top_species_for_dataset(dataset_kind, top_k=top_k)
        for dataset_kind in dataset_kinds
    }


def top_species_for_dataset(dataset_kind, top_k=DEFAULT_TOP_SPECIES_COUNT):
    if dataset_kind == "acoustic":
        return top_acoustic_species(top_k=top_k)
    if dataset_kind == "iwildcam":
        return top_iwildcam_species(top_k=top_k)
    raise ValueError(
        "Unsupported dataset "
        f"`{dataset_kind}`. Expected one of: {', '.join(SUPPORTED_DATASETS)}."
    )


def top_acoustic_species(top_k=DEFAULT_TOP_SPECIES_COUNT):
    supported_species = acoustic_supported_species()
    counts = {}
    for path in ACOUSTIC_ANNOTATION_ROOT.rglob("*.txt"):
        try:
            frame = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if "Species" not in frame.columns:
            continue
        for species_name in sorted(set(frame["Species"].dropna().astype(str))):
            if species_name not in supported_species:
                continue
            counts[species_name] = counts.get(species_name, 0) + 1
    return [
        species_name
        for species_name, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:top_k]
    ]


def acoustic_supported_species():
    csv_paths = sorted(ACOUSTIC_PERCH_LOGIT_PARTS.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No Perch logit shards found under {ACOUSTIC_PERCH_LOGIT_PARTS}."
        )
    header = pd.read_csv(csv_paths[0], nrows=0)
    return {
        column.removeprefix("logit_")
        for column in header.columns
        if column.startswith("logit_")
    }


def top_iwildcam_species(top_k=DEFAULT_TOP_SPECIES_COUNT):
    if not IWILDCAM_TARGET_SPECIES_MAPPING.exists():
        raise FileNotFoundError(
            "No iWildCam SpeciesNet target mapping found at "
            f"{IWILDCAM_TARGET_SPECIES_MAPPING}. Run "
            "scripts/preprocess_iwildcam.py first."
        )
    mapping = pd.read_csv(IWILDCAM_TARGET_SPECIES_MAPPING)
    required_columns = {"category_name", "annotation_count"}
    missing_columns = sorted(required_columns - set(mapping.columns))
    if missing_columns:
        raise ValueError(
            "iWildCam target mapping is missing required columns: "
            + ", ".join(missing_columns)
        )
    mapping = mapping.copy()
    mapping["annotation_count"] = pd.to_numeric(
        mapping["annotation_count"],
        errors="coerce",
    )
    mapping = mapping[
        mapping["category_name"].notna() & (mapping["annotation_count"] > 0)
    ].copy()
    if mapping.empty:
        raise ValueError("iWildCam target mapping does not contain any species rows.")
    mapping = mapping.sort_values(
        ["annotation_count", "category_name"],
        ascending=[False, True],
    )
    return [str(species_name) for species_name in mapping["category_name"].head(top_k)]


def build_job_specs(dataset, species, top_k=DEFAULT_TOP_SPECIES_COUNT):
    datasets = normalize_datasets(dataset)
    species = normalize_species(species)
    if species:
        if len(datasets) != 1:
            raise ValueError(
                "When specifying species explicitly, provide exactly one dataset."
            )
        return [dict(dataset_kind=datasets[0], species_name=name) for name in species]

    defaults = default_species_by_dataset(top_k=top_k, dataset_kinds=datasets)
    job_specs = []
    for dataset_kind in datasets:
        for species_name in defaults[dataset_kind]:
            job_specs.append(dict(dataset_kind=dataset_kind, species_name=species_name))
    return job_specs


def review_limits_for_dataset(
    dataset_kind,
    acoustic_reviews_per_step=DEFAULT_ACOUSTIC_REVIEWS_PER_STEP,
    acoustic_max_reviews=DEFAULT_ACOUSTIC_MAX_REVIEWS,
    iwildcam_reviews_per_step=DEFAULT_IWILDCAM_REVIEWS_PER_STEP,
    iwildcam_max_reviews=DEFAULT_IWILDCAM_MAX_REVIEWS,
):
    if dataset_kind == "acoustic":
        return int(acoustic_reviews_per_step), int(acoustic_max_reviews)
    if dataset_kind == "iwildcam":
        return int(iwildcam_reviews_per_step), int(iwildcam_max_reviews)
    raise ValueError(
        "Unsupported dataset "
        f"`{dataset_kind}`. Expected one of: {', '.join(SUPPORTED_DATASETS)}."
    )


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


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_single_csv_row(path):
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def read_text(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_submitit_result(path):
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None, None
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and payload[0] in {"success", "error"}
    ):
        return payload
    return None, None


def build_completed_job_row(spec, result):
    row = dict(
        dataset_kind=result["dataset_kind"],
        species=result["species"],
        output_dir=result["output_dir"],
        job_id=result["job_id"],
        status="completed",
        returncode=result["returncode"],
        job_wall_time_seconds=result["job_wall_time_seconds"],
    )
    for key in ("ran_subprocess", "skipped_existing_result"):
        if key in result:
            row[key] = int(coerce_env_value(result[key]))
    return row


def build_completed_job_row_from_metadata(spec, job_id, job_metadata):
    returncode = coerce_env_value(job_metadata.get("returncode", 0))
    job_wall_time_seconds = coerce_env_value(
        job_metadata.get("job_wall_time_seconds", 0.0)
    )
    if returncode in {"", None}:
        returncode = 0
    if job_wall_time_seconds in {"", None}:
        job_wall_time_seconds = 0.0
    row = dict(
        dataset_kind=spec["dataset_kind"],
        species=spec["species"],
        output_dir=spec["output_dir"],
        job_id=job_metadata.get("job_id", job_id),
        status="completed",
        returncode=int(returncode),
        job_wall_time_seconds=float(job_wall_time_seconds),
    )
    for key in ("ran_subprocess", "skipped_existing_result"):
        if key in job_metadata:
            row[key] = int(coerce_env_value(job_metadata[key]))
    return row


def build_failed_job_row(spec, job_id, job_metadata, error_text):
    row = dict(
        dataset_kind=spec["dataset_kind"],
        species=spec["species"],
        output_dir=spec["output_dir"],
        job_id=job_metadata.get("job_id", job_id),
        status="failed",
        returncode=job_metadata.get("returncode", ""),
        job_wall_time_seconds=job_metadata.get("job_wall_time_seconds", ""),
        error=error_text,
    )
    for key in ("ran_subprocess", "skipped_existing_result"):
        if key in job_metadata:
            row[key] = int(coerce_env_value(job_metadata[key]))
    return row


def reconcile_job_outcome(
    job_id,
    spec,
    submitit_dir,
    observed_result=None,
    observed_error=None,
):
    if observed_result is not None:
        return build_completed_job_row(spec, observed_result), None

    metadata_path = Path(spec["output_dir"]) / "submitit_job.csv"
    result_path = Path(submitit_dir) / f"{job_id}_0_result.pkl"
    log_err_path = Path(submitit_dir) / f"{job_id}_0_log.err"

    error_text = "" if observed_error is None else str(observed_error)
    log_err_text = read_text(log_err_path)
    retryable_failure = (
        "PREEMPTION" in log_err_text
        or "requeue" in error_text.lower()
        or read_single_csv_row(metadata_path).get("returncode", "") == "-15"
    )
    deadline = (
        time.time() + REQUEUE_RECOVERY_TIMEOUT_SECONDS
        if retryable_failure
        else time.time()
    )

    while True:
        result_state, result_payload = read_submitit_result(result_path)
        job_metadata = read_single_csv_row(metadata_path)
        if result_state == "success" and isinstance(result_payload, dict):
            return build_completed_job_row(spec, result_payload), None
        if (
            str(job_metadata.get("status", "")).strip().lower() == "completed"
            and str(job_metadata.get("returncode", "")).strip() == "0"
        ):
            return (
                build_completed_job_row_from_metadata(spec, job_id, job_metadata),
                None,
            )
        if time.time() >= deadline:
            break
        time.sleep(REQUEUE_RECOVERY_POLL_INTERVAL_SECONDS)

    if result_state == "error" and isinstance(result_payload, str):
        error_text = result_payload
    elif not error_text:
        error_text = read_text(log_err_path).strip()
    if not error_text:
        error_text = f"{spec['species']} failed without a recorded submitit error."
    job_metadata = read_single_csv_row(metadata_path)
    return build_failed_job_row(spec, job_id, job_metadata, error_text), error_text


def coerce_env_value(value):
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
        return int(lowered)
    try:
        return float(lowered)
    except ValueError:
        return value


def coerce_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_submitit_config(env_file):
    env_file = Path(env_file)
    dotenv_config = {}
    if env_file.exists():
        if dotenv_values is None or load_dotenv is None:
            raise ImportError(
                "python-dotenv is required to load submitit configuration from "
                f"{env_file}."
            )
        dotenv_config = {
            key: value
            for key, value in dotenv_values(env_file).items()
            if value not in (None, "")
        }
        load_dotenv(env_file, override=False)

    merged = dict(dotenv_config)
    merged.update(
        {key: value for key, value in os.environ.items() if key.startswith("SUBMITIT_")}
    )
    debug = coerce_bool(merged.pop("SUBMITIT_DEBUG", "0"))
    parameters = {}
    for key, value in merged.items():
        if not key.startswith("SUBMITIT_"):
            continue
        if key.startswith("SUBMITIT_SLURM_"):
            parameter_name = "slurm_" + key[len("SUBMITIT_SLURM_") :].lower()
        else:
            parameter_name = key[len("SUBMITIT_") :].lower()
        parameters[parameter_name] = coerce_env_value(value)
    return debug, parameters


def make_sweep_dir(output_root, dataset_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_root) / f"{dataset_label}_sweep_{timestamp}"
    sweep_dir = base_dir
    suffix = 1
    while sweep_dir.exists():
        sweep_dir = Path(f"{base_dir}_{suffix:02d}")
        suffix += 1
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def find_most_recent_sweep_dir(output_root):
    output_root = Path(output_root)
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "submitted_jobs.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No prior sweeps with submitted jobs were found under {output_root}."
        )
    return max(
        candidates,
        key=lambda path: (
            path.name.rsplit("_sweep_", maxsplit=1)[-1],
            path.name,
        ),
    )


def load_existing_sweep_job_specs(sweep_dir):
    submitted_rows = read_csv_rows(Path(sweep_dir) / "submitted_jobs.csv")
    if not submitted_rows:
        raise FileNotFoundError(
            f"No submitted_jobs.csv rows were found in {sweep_dir}."
        )
    job_specs = []
    for index, row in enumerate(submitted_rows):
        dataset_kind = str(row["dataset_kind"])
        species_name = str(row["species"])
        expected_output_dir = (
            Path(sweep_dir) / f"{index:03d}_{dataset_kind}__{slugify(species_name)}"
        )
        recorded_output_dir = str(row.get("output_dir", "")).strip()
        output_dir = expected_output_dir
        if (
            not expected_output_dir.exists()
            and recorded_output_dir
            and Path(recorded_output_dir).exists()
        ):
            output_dir = Path(recorded_output_dir)
        job_specs.append(
            dict(
                dataset_kind=dataset_kind,
                species_name=species_name,
                output_dir=str(output_dir),
                submitted_job_id=str(row.get("job_id", "")),
            )
        )
    return job_specs


def job_spec_key(job_spec):
    return (str(job_spec["dataset_kind"]), str(job_spec["species_name"]))


def merge_continue_job_specs(sweep_dir, desired_job_specs):
    existing_specs = load_existing_sweep_job_specs(sweep_dir)
    existing_specs_by_key = {
        job_spec_key(existing_spec): existing_spec for existing_spec in existing_specs
    }
    merged_specs = []
    for index, desired_spec in enumerate(desired_job_specs):
        dataset_kind = str(desired_spec["dataset_kind"])
        species_name = str(desired_spec["species_name"])
        merged_spec = dict(dataset_kind=dataset_kind, species_name=species_name)
        existing_spec = existing_specs_by_key.get(job_spec_key(merged_spec))
        if existing_spec is None:
            output_dir = (
                Path(sweep_dir) / f"{index:03d}_{dataset_kind}__{slugify(species_name)}"
            )
            merged_spec.update(output_dir=str(output_dir), submitted_job_id="")
        else:
            merged_spec.update(
                output_dir=existing_spec["output_dir"],
                submitted_job_id=str(existing_spec.get("submitted_job_id", "")),
            )
        merged_specs.append(merged_spec)
    return merged_specs


_SUBMITIT_CHECKPOINTABLE = (
    submitit.helpers.Checkpointable if submitit is not None else object
)


class SpeciesExperimentJob(_SUBMITIT_CHECKPOINTABLE):
    def __init__(
        self,
        dataset_kind,
        species,
        output_dir,
        num_warmup,
        num_samples,
        num_chains,
        random_seed,
        reviews_per_step,
        max_reviews,
    ):
        self.dataset_kind = dataset_kind
        self.species = species
        self.output_dir = Path(output_dir)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.random_seed = int(random_seed)
        self.reviews_per_step = int(reviews_per_step)
        self.max_reviews = int(max_reviews)

    def command(self):
        return [
            sys.executable,
            str(EXPERIMENT_SCRIPT),
            "--dataset",
            self.dataset_kind,
            "--species",
            self.species,
            "--output_dir",
            str(self.output_dir),
            "--num_warmup",
            str(self.num_warmup),
            "--num_samples",
            str(self.num_samples),
            "--num_chains",
            str(self.num_chains),
            "--random_seed",
            str(self.random_seed),
            "--reviews_per_step",
            str(self.reviews_per_step),
            "--max_reviews",
            str(self.max_reviews),
            "--continue=True",
        ]

    def __call__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.output_dir / "job_stdout.log"
        stderr_path = self.output_dir / "job_stderr.log"
        submitit_job_path = self.output_dir / "submitit_job.csv"
        job_metadata = dict(
            dataset_kind=self.dataset_kind,
            species=self.species,
            output_dir=str(self.output_dir),
            command=" ".join(self.command()),
        )
        try:
            job_env = submitit.JobEnvironment()
            job_metadata.update(
                job_id=str(job_env.job_id),
                local_rank=int(job_env.local_rank),
                global_rank=int(job_env.global_rank),
                num_tasks=int(job_env.num_tasks),
            )
        except RuntimeError:
            job_env = None
            job_metadata.update(job_id="", local_rank=0, global_rank=0, num_tasks=1)
        job_start_time = time.perf_counter()

        from occubed.run_reviews import real_data_output_is_complete

        if real_data_output_is_complete(self.output_dir):
            job_wall_time_seconds = time.perf_counter() - job_start_time
            job_metadata.update(
                status="completed",
                returncode=0,
                job_wall_time_seconds=float(job_wall_time_seconds),
                ran_subprocess=0,
                skipped_existing_result=1,
            )
            write_csv_rows(submitit_job_path, [job_metadata])
            return dict(
                dataset_kind=self.dataset_kind,
                species=self.species,
                output_dir=str(self.output_dir),
                job_id="" if job_env is None else str(job_env.job_id),
                returncode=0,
                job_wall_time_seconds=float(job_wall_time_seconds),
                ran_subprocess=0,
                skipped_existing_result=1,
            )
        write_csv_rows(submitit_job_path, [job_metadata])

        completed = None
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            completed = subprocess.run(
                self.command(),
                cwd=PACKAGE_DIR,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        job_wall_time_seconds = time.perf_counter() - job_start_time
        job_metadata.update(
            status="completed" if completed.returncode == 0 else "failed",
            returncode=int(completed.returncode),
            job_wall_time_seconds=float(job_wall_time_seconds),
            ran_subprocess=1,
            skipped_existing_result=0,
        )
        write_csv_rows(submitit_job_path, [job_metadata])
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.species} failed with exit code {completed.returncode}. "
                f"See {stderr_path}."
            )
        return dict(
            dataset_kind=self.dataset_kind,
            species=self.species,
            output_dir=str(self.output_dir),
            job_id="" if job_env is None else str(job_env.job_id),
            returncode=int(completed.returncode),
            job_wall_time_seconds=float(job_wall_time_seconds),
            ran_subprocess=1,
            skipped_existing_result=0,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def main(
    dataset=None,
    *species,
    output_root=DEFAULT_SWEEP_ROOT,
    env_file=PACKAGE_DIR / ".env",
    job_name="real-data-sweep",
    debug=False,
    num_warmup=500,
    num_samples=500,
    num_chains=3,
    random_seed=0,
    reviews_per_step=DEFAULT_ACOUSTIC_REVIEWS_PER_STEP,
    max_reviews=DEFAULT_ACOUSTIC_MAX_REVIEWS,
    iwildcam_reviews_per_step=DEFAULT_IWILDCAM_REVIEWS_PER_STEP,
    iwildcam_max_reviews=DEFAULT_IWILDCAM_MAX_REVIEWS,
    top_k=DEFAULT_TOP_SPECIES_COUNT,
    continue_=False,
):
    if submitit is None:
        raise ImportError(
            "submitit is required to run real-data sweeps. Install submitit or run "
            "this script with uv so the script dependencies are available."
        )

    sweep_start_time = time.perf_counter()
    output_root = Path(output_root)
    env_file = Path(env_file)
    debug_from_env, executor_parameters = load_submitit_config(env_file)
    debug_mode = bool(debug or debug_from_env)
    existing_sweep_manifest = {}
    if continue_:
        sweep_dir = find_most_recent_sweep_dir(output_root)
        existing_sweep_manifest = read_single_csv_row(sweep_dir / "sweep_manifest.csv")
        num_warmup = int(
            coerce_env_value(existing_sweep_manifest.get("num_warmup", num_warmup))
        )
        num_samples = int(
            coerce_env_value(existing_sweep_manifest.get("num_samples", num_samples))
        )
        num_chains = int(
            coerce_env_value(existing_sweep_manifest.get("num_chains", num_chains))
        )
        random_seed = int(
            coerce_env_value(existing_sweep_manifest.get("random_seed", random_seed))
        )
        reviews_per_step = int(
            coerce_env_value(
                existing_sweep_manifest.get(
                    "acoustic_reviews_per_step",
                    existing_sweep_manifest.get("reviews_per_step", reviews_per_step),
                )
            )
        )
        max_reviews = int(
            coerce_env_value(
                existing_sweep_manifest.get(
                    "acoustic_max_reviews",
                    existing_sweep_manifest.get("max_reviews", max_reviews),
                )
            )
        )
        iwildcam_reviews_per_step = int(
            coerce_env_value(
                existing_sweep_manifest.get(
                    "iwildcam_reviews_per_step",
                    iwildcam_reviews_per_step,
                )
            )
        )
        iwildcam_max_reviews = int(
            coerce_env_value(
                existing_sweep_manifest.get(
                    "iwildcam_max_reviews",
                    iwildcam_max_reviews,
                )
            )
        )
        top_k = int(coerce_env_value(existing_sweep_manifest.get("top_k", top_k)))
        job_spec_build_start_time = time.perf_counter()
        desired_job_spec_list = build_job_specs(dataset, species, top_k=int(top_k))
        job_spec_list = merge_continue_job_specs(sweep_dir, desired_job_spec_list)
        job_spec_build_wall_time_seconds = (
            time.perf_counter() - job_spec_build_start_time
        )
    else:
        job_spec_build_start_time = time.perf_counter()
        job_spec_list = build_job_specs(dataset, species, top_k=int(top_k))
        job_spec_build_wall_time_seconds = (
            time.perf_counter() - job_spec_build_start_time
        )
        dataset_values = sorted(
            {job_spec["dataset_kind"] for job_spec in job_spec_list}
        )
        dataset_label = (
            dataset_values[0] if len(dataset_values) == 1 else "multi_dataset"
        )
        sweep_dir = make_sweep_dir(output_root, dataset_label)
    dataset_values = sorted({job_spec["dataset_kind"] for job_spec in job_spec_list})
    submitit_dir = sweep_dir / "submitit"
    submitit_dir.mkdir(parents=True, exist_ok=True)
    filtered_parameters = (
        {
            key: value
            for key, value in executor_parameters.items()
            if not key.startswith("slurm_")
        }
        if debug_mode
        else dict(executor_parameters)
    )
    sweep_manifest_row = dict(existing_sweep_manifest)
    sweep_manifest_row.update(
        dataset_kind=",".join(dataset_values),
        dataset_count=int(len(dataset_values)),
        debug_mode=int(debug_mode),
        species_count=int(len(job_spec_list)),
        output_root=str(output_root),
        sweep_dir=str(sweep_dir),
        env_file=str(env_file),
        experiment_script=str(EXPERIMENT_SCRIPT),
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        random_seed=int(random_seed),
        reviews_per_step=int(reviews_per_step),
        max_reviews=int(max_reviews),
        acoustic_reviews_per_step=int(reviews_per_step),
        acoustic_max_reviews=int(max_reviews),
        iwildcam_reviews_per_step=int(iwildcam_reviews_per_step),
        iwildcam_max_reviews=int(iwildcam_max_reviews),
        top_k=int(top_k),
        continue_mode=int(bool(continue_)),
        job_spec_build_wall_time_seconds=float(job_spec_build_wall_time_seconds),
    )

    write_csv_rows(
        sweep_dir / "sweep_manifest.csv",
        [sweep_manifest_row],
    )
    write_csv_rows(
        sweep_dir / "submitit_parameters.csv",
        [filtered_parameters],
    )

    ordered_specs = []
    completed_rows_by_output_dir = {}
    for index, job_spec in enumerate(job_spec_list):
        species_name = job_spec["species_name"]
        dataset_kind = job_spec["dataset_kind"]
        species_dir = Path(
            job_spec.get(
                "output_dir",
                sweep_dir / f"{index:03d}_{dataset_kind}__{slugify(species_name)}",
            )
        )
        spec = dict(
            dataset_kind=dataset_kind,
            species=species_name,
            output_dir=str(species_dir),
            submitted_job_id=str(job_spec.get("submitted_job_id", "")),
        )
        ordered_specs.append(spec)

    job_specs = {}
    jobs = []
    submitted_rows = []
    submission_start_time = time.perf_counter()
    if ordered_specs:
        # Submitit captures these environment variables if they are still present.
        for key in list(os.environ.keys()):
            if key.startswith("SLURM_"):
                del os.environ[key]
        executor = (
            submitit.DebugExecutor(folder=submitit_dir)
            if debug_mode
            else submitit.AutoExecutor(folder=submitit_dir)
        )
        executor.update_parameters(
            name=job_name,
            **filtered_parameters,
            slurm_additional_parameters={"requeue": True, "gpus": 0},
        )
        with executor.batch():
            for spec in ordered_specs:
                spec_reviews_per_step, spec_max_reviews = review_limits_for_dataset(
                    spec["dataset_kind"],
                    acoustic_reviews_per_step=reviews_per_step,
                    acoustic_max_reviews=max_reviews,
                    iwildcam_reviews_per_step=iwildcam_reviews_per_step,
                    iwildcam_max_reviews=iwildcam_max_reviews,
                )
                job = executor.submit(
                    SpeciesExperimentJob(
                        dataset_kind=spec["dataset_kind"],
                        species=spec["species"],
                        output_dir=spec["output_dir"],
                        num_warmup=int(num_warmup),
                        num_samples=int(num_samples),
                        num_chains=int(num_chains),
                        random_seed=int(random_seed),
                        reviews_per_step=int(spec_reviews_per_step),
                        max_reviews=int(spec_max_reviews),
                    )
                )
                jobs.append(job)
                job_specs[job] = spec
        submitted_rows = [
            dict(
                dataset_kind=job_specs[job]["dataset_kind"],
                species=job_specs[job]["species"],
                output_dir=job_specs[job]["output_dir"],
                job_id=str(job.job_id),
            )
            for job in jobs
        ]
    batch_submission_wall_time_seconds = time.perf_counter() - submission_start_time
    if not continue_:
        write_csv_rows(sweep_dir / "submitted_jobs.csv", submitted_rows)
    elif submitted_rows:
        write_csv_rows(sweep_dir / "resubmitted_jobs.csv", submitted_rows)

    observed_results = {}
    observed_errors = {}
    completion_wait_start_time = time.perf_counter()
    if jobs:
        with tqdm(total=len(jobs), desc="Completed jobs") as progress:
            for job in submitit.helpers.as_completed(jobs):
                try:
                    observed_results[job] = job.result()
                except Exception as exc:
                    observed_errors[job] = exc
                progress.update(1)
    completion_wait_wall_time_seconds = time.perf_counter() - completion_wait_start_time

    failures = []
    for job in jobs:
        spec = job_specs[job]
        completed_row, failure = reconcile_job_outcome(
            job_id=str(job.job_id),
            spec=spec,
            submitit_dir=submitit_dir,
            observed_result=observed_results.get(job),
            observed_error=observed_errors.get(job),
        )
        completed_rows_by_output_dir[spec["output_dir"]] = completed_row
        if failure is not None:
            failures.append((spec["species"], failure))

    completed_rows = [
        completed_rows_by_output_dir[spec["output_dir"]] for spec in ordered_specs
    ]

    write_csv_rows(sweep_dir / "jobs.csv", completed_rows)
    job_wall_times = [
        float(row["job_wall_time_seconds"])
        for row in completed_rows
        if str(row.get("job_wall_time_seconds", "")).strip() != ""
    ]
    sweep_manifest_row.update(
        batch_submission_wall_time_seconds=float(batch_submission_wall_time_seconds),
        completion_wait_wall_time_seconds=float(completion_wait_wall_time_seconds),
        skipped_completed_job_count=int(
            sum(
                1
                for row in completed_rows
                if coerce_bool(row.get("skipped_existing_result", 0))
            )
        ),
        resubmitted_job_count=int(len(jobs)),
        sweep_wall_time_seconds=float(time.perf_counter() - sweep_start_time),
        completed_job_count=int(
            sum(1 for row in completed_rows if row["status"] == "completed")
        ),
        failed_job_count=int(
            sum(1 for row in completed_rows if row["status"] == "failed")
        ),
        total_job_wall_time_seconds=(
            float(sum(job_wall_times)) if job_wall_times else 0.0
        ),
        mean_job_wall_time_seconds=(
            float(sum(job_wall_times) / len(job_wall_times)) if job_wall_times else 0.0
        ),
        max_job_wall_time_seconds=float(max(job_wall_times)) if job_wall_times else 0.0,
    )
    write_csv_rows(sweep_dir / "sweep_manifest.csv", [sweep_manifest_row])
    if failures:
        failed_species = ", ".join(species_name for species_name, _ in failures)
        raise RuntimeError(f"One or more jobs failed: {failed_species}")
    print(sweep_dir)
