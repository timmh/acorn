"""Full-replicate versus top-example review, using the main experiment runner."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import submitit

from occubed import export_sweep
from occubed.characterization import normalized_auc
from occubed.export_sweep import read_csv_rows, read_single_csv_row, write_csv_rows
from occubed.paths import PACKAGE_DIR
from occubed.submit_sweep import load_submitit_config, slugify

EXPERIMENT = "cut_normal_calibrated_target_eig"
PROTOCOLS = {"acoustic": ("full_replicate", "top_window"),
             "iwildcam": ("full_replicate", "top_image", "top_sequence")}
METRICS = ("occupancy_probability_agreement", "site_rank_spearman",
           "occupancy_coefficient_agreement", "detection_coefficient_agreement",
           "mean_ecological_agreement")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_array(path):
    # These are trusted, locally generated research artifacts, including legacy
    # object arrays of site IDs and recording filenames.
    value = np.load(path, allow_pickle=True)
    return value.astype(str) if value.dtype.kind == "O" else value


def load_inputs(root):
    root = Path(root)
    data = {key: load_array(root / "inputs" / f"{key}.npy")
            for key in ("obs", "site_covs", "obs_covs")}
    metadata = {path.stem: load_array(path) for path in (root / "dataset_metadata").glob("*.npy")}
    truth = dict(metadata, f=load_array(root / "inputs/true_labels.npy"))
    for key in ("dataset_name", "target_label"):
        truth[key] = str(metadata[key].item())
    return data, truth, metadata


def replicate_frame(root):
    data, truth, metadata = load_inputs(root)
    mask = metadata["reviewable_mask"].astype(bool)
    if mask.ndim != 3 or mask.shape[1] != 1:
        raise ValueError("Expected single-season source artifacts.")
    site, _, replicate = np.where(mask)
    frame = pd.DataFrame(dict(site_index=site, replicate_index=replicate,
        site_id=metadata["site_ids"][site].astype(str),
        replicate_id=metadata["replicate_ids"][site, replicate].astype(str),
        max_score=data["obs"][0, site, 0, replicate],
        full_label=truth["f"][0, site, 0, replicate]))
    if frame.duplicated(["site_id", "replicate_id"]).any():
        raise ValueError("Replicate IDs must be unique within each site.")
    return frame


def window_label(annotation, species, start, end):
    """Presence in a half-open window, including point annotations."""
    selected = annotation.loc[annotation.Species.eq(species)]
    if selected.empty:
        return 0.0
    begin = pd.to_numeric(selected["Begin Time (s)"], errors="raise").to_numpy()
    finish = pd.to_numeric(selected["End Time (s)"], errors="raise").to_numpy()
    duration = finish - begin
    if "File Offset (s)" in selected:
        offsets = pd.to_numeric(selected["File Offset (s)"], errors="raise").to_numpy()
        finish, begin = offsets + duration, offsets
    if not np.isfinite(np.r_[begin, finish]).all() or np.any(duration < 0):
        raise ValueError("Invalid annotation intervals.")
    overlap = (begin < end) & (finish > start) & (duration > 0)
    point = (duration == 0) & (begin >= start) & (begin < end)
    return float(np.any(overlap | point))


def top_camera_examples(images, score_column, category):
    """Select by scores alone; use labels only after deterministic selection."""
    keys = ["site_id", "replicate_id"]
    images = images.sort_values("image_id", kind="stable").copy()
    if images.duplicated("image_id").any():
        raise ValueError("Duplicate camera image IDs.")
    normalized = images.category_name.str.split().str.join(' ').str.lower()
    images["label"] = normalized.eq(' '.join(category.lower().split())).astype(float)
    if 'annotation_repaired' not in images:
        images['annotation_repaired'] = False
    # A missing sequence ID must not merge unrelated images into one sequence.
    images["sequence_key"] = images.seq_id.where(images.seq_id.ne(""), "image:" + images.image_id)
    full = images.groupby(keys, as_index=False).agg(full_label=("label", "max"), full_units=("label", "size"),
                                                  resolved_annotation_count=("annotation_repaired", "sum"))
    finite = images.loc[np.isfinite(images[score_column])]
    top = finite.loc[finite.groupby(keys)[score_column].idxmax()].copy()
    sequence = images.groupby(keys + ["sequence_key"], as_index=False).agg(
        top_sequence_label=("label", "max"), top_sequence_units=("label", "size"))
    top = top.merge(sequence, on=keys + ["sequence_key"], validate="one_to_one")
    top = top.rename(columns={score_column: "max_score", "label": "top_image_label", "image_id": "top_unit_id"})
    top["top_image_units"] = 1
    return full.merge(top[keys + ["max_score", "top_image_label", "top_sequence_label", "top_image_units",
                                 "top_sequence_units", "top_unit_id", "sequence_key"]], on=keys, validate="one_to_one")


def resolve_camera_annotations(images, raw_metadata):
    images = images.copy()
    images['annotation_repaired'] = False
    annotations = {str(row['image_id']): row for row in raw_metadata['annotations']}
    categories = {int(row['id']): row['name'] for row in raw_metadata['categories']}
    for index, row in images.loc[pd.to_numeric(images.has_ground_truth).ne(1)].iterrows():
        # One original image ID has a typo; its actual file stem matches an
        # explicitly annotated image ID. Never infer a label from its neighbors.
        annotation = annotations.get(str(row.image_id), annotations.get(Path(row.file_name).stem))
        if annotation is None:
            raise ValueError(f'No annotation for camera image {row.image_id}')
        images.loc[index, 'category_name'] = categories[int(annotation['category_id'])]
        images.loc[index, 'has_ground_truth'] = 1
        images.loc[index, 'annotation_repaired'] = True
    return images


def align_examples(reference, examples):
    keys = ["site_id", "replicate_id"]
    result = reference.merge(examples, on=keys, how="left", suffixes=("", "_raw"), validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Missing example-level data for source replicates.")
    if not np.array_equal(result.full_label, result.full_label_raw):
        raise ValueError("Raw labels disagree with the manuscript source cohort.")
    if not np.allclose(result.max_score, result.max_score_raw, atol=1e-6, rtol=1e-6):
        raise ValueError("Raw max-pooled scores disagree with the manuscript source cohort.")
    return result.drop(columns=["full_label_raw", "max_score_raw", "_merge"])


def acoustic_examples(cohort, source):
    from occubed import datasets

    base = datasets.ACOUSTIC_FOREST_SOUNDSCAPE_DIR
    columns = ["Recording_FileName", "SubCollection_ID", "Annotation_FileName", "window_index",
               "window_start_s", "window_end_s"] + [f"logit_{row['species']}" for row in cohort]
    windows = pd.read_csv(base / "perch_v2/perch_v2_window_logits.csv", usecols=columns)
    windows = windows.sort_values(["Recording_FileName", "window_index"], kind="stable")
    if windows.duplicated(["Recording_FileName", "window_index"]).any():
        raise ValueError("Duplicate acoustic window indices.")
    lookup = datasets._build_acoustic_annotation_filename_lookup(
        pd.read_csv(base / "perch_v2/recording_filename_mismatches.csv"))
    annotations = {}
    for name, subcollection, filename in windows[
            ["Recording_FileName", "SubCollection_ID", "Annotation_FileName"]].drop_duplicates().itertuples(index=False, name=None):
        path = datasets._resolve_acoustic_annotation_path(base / "annotations", subcollection, filename, lookup)
        annotations[name] = pd.read_csv(path, sep="\t")
    counts = windows.groupby("Recording_FileName").agg(
        full_units=("window_index", "size"), full_seconds=("window_end_s", "max"))
    for row in cohort:
        species, key = row["species"], row["species_key"]
        reference = replicate_frame(Path(row["reference_dir"]))
        finite = windows.loc[np.isfinite(windows[f"logit_{species}"])]
        top = finite.loc[finite.groupby("Recording_FileName")[f"logit_{species}"].idxmax()]
        examples = []
        for item in top.to_dict("records"):
            name = item["Recording_FileName"]
            annotation = annotations[name]
            examples.append(dict(site_id="", replicate_id=name,
                max_score=item[f"logit_{species}"], full_label=float(annotation.Species.eq(species).any()),
                top_window_label=window_label(annotation, species, item["window_start_s"], item["window_end_s"]),
                full_units=int(counts.loc[name, "full_units"]), full_seconds=float(counts.loc[name, "full_seconds"]),
                top_window_units=1, top_window_seconds=item["window_end_s"] - item["window_start_s"],
                top_unit_id=f"{name}#window={item['window_index']}",
                window_start_s=item["window_start_s"], window_end_s=item["window_end_s"]))
        examples = pd.DataFrame(examples).drop(columns="site_id")
        examples = examples.merge(reference[["site_id", "replicate_id"]], on="replicate_id", validate="one_to_one")
        yield row, align_examples(reference, examples)


def camera_examples(cohort, source):
    from occubed import datasets

    base = datasets.IWILDCAM2022_DIR
    images = pd.read_csv(base / "metadata/images.csv", usecols=["split", "image_id", "category_name", "location_id",
        "seq_id", "capture_datetime", "latitude", "longitude", "has_ground_truth", "file_name"],
        dtype={"image_id": str, "location_id": str, "seq_id": str}, keep_default_na=False)
    images = images.loc[images.split.eq("train")].copy()
    coordinates = images[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    images = images.loc[np.isfinite(coordinates).all(axis=1)].copy()
    dates = pd.to_datetime(images.capture_datetime, errors="coerce")
    images = images.loc[dates.notna()].copy()
    images["site_id"] = images.location_id.astype(str)
    images["replicate_id"] = dates.loc[images.index].dt.strftime("%Y-%m-%d")
    targets = [datasets._find_iwildcam2022_target_mapping(row["species"]) for row in cohort]
    all_sites = set()
    for row in cohort:
        all_sites.update(replicate_frame(Path(row["reference_dir"])).site_id)
    images = images.loc[images.site_id.isin(all_sites)].copy()
    images['annotation_repaired'] = False
    if pd.to_numeric(images.has_ground_truth).ne(1).any():
        from occubed.paths import IWILDCAM_RAW_DIR
        raw = json.loads((IWILDCAM_RAW_DIR / 'metadata/iwildcam2022_train_annotations.json').read_text())
        images = resolve_camera_annotations(images, raw)
    score_rows = []
    for shard in sorted((base / "outputs").glob("shard_*")):
        path = shard / "manifest.csv"
        if not path.exists() or not (shard / "logits.npy").exists():
            continue
        part = pd.read_csv(path, usecols=["image_id", "row_idx", "status"], dtype={"image_id": str}, keep_default_na=False)
        part = part.loc[part.image_id.isin(set(images.image_id))].copy()
        logits = np.load(shard / "logits.npy", mmap_mode="r")
        for index, target in enumerate(targets):
            values = np.asarray(logits[part.row_idx.to_numpy(int), int(target.target_logit_index)], dtype=float)
            part[f"score_{index}"] = np.where(part.status.eq("ok"), values, np.nan)
        score_rows.append(part.drop(columns=["row_idx", "status"]))
    scores = pd.concat(score_rows, ignore_index=True)
    images = images.merge(scores, on="image_id", how="left", validate="one_to_one")
    for index, (row, target) in enumerate(zip(cohort, targets)):
        reference = replicate_frame(Path(row["reference_dir"]))
        subset = images.merge(reference[["site_id", "replicate_id"]], on=["site_id", "replicate_id"], validate="many_to_one")
        if not pd.to_numeric(subset.has_ground_truth, errors="raise").eq(1).all():
            raise ValueError("Selected camera images lack ground-truth annotations.")
        examples = top_camera_examples(subset, f"score_{index}", target.category_name)
        yield row, align_examples(reference, examples)


def audit_labels(frame, dataset, species, key):
    rows = []
    for protocol in PROTOCOLS[dataset]:
        label = frame.full_label if protocol == "full_replicate" else frame[f"{protocol}_label"]
        units = frame.full_units if protocol == "full_replicate" else frame[f"{protocol}_units"]
        if not np.isin(label, [0, 1]).all() or np.any(label > frame.full_label):
            raise ValueError("Subset labels must be binary and cannot introduce new true positives.")
        positives = frame.full_label.eq(1)
        found_sites = frame.loc[label.eq(1), "site_id"].nunique()
        positive_sites = frame.loc[positives, "site_id"].nunique()
        row = dict(dataset_kind=dataset, species=species, species_key=key, protocol=protocol,
            replicates=len(frame), true_positive_replicates=int(positives.sum()),
            missed_positive_replicates=int((positives & label.eq(0)).sum()),
            positive_replicate_recall=float(label[positives].mean()),
            positive_site_recall=found_sites/positive_sites, total_review_units=int(units.sum()),
            full_review_units=int(frame.full_units.sum()), review_unit_fraction=float(units.sum()/frame.full_units.sum()))
        if dataset == "acoustic":
            seconds = frame.full_seconds if protocol == "full_replicate" else frame[f"{protocol}_seconds"]
            row.update(review_seconds=float(seconds.sum()), review_time_fraction=float(seconds.sum()/frame.full_seconds.sum()))
        rows.append(row)
    return rows




def label_tensor(truth, examples, protocol):
    labels = np.full_like(truth, np.nan, dtype=float)
    values = examples.full_label if protocol == "full_replicate" else examples[f"{protocol}_label"]
    labels[0, examples.site_index.to_numpy(int), 0, examples.replicate_index.to_numpy(int)] = values
    if not np.array_equal(np.isfinite(labels), np.isfinite(truth)):
        raise ValueError("Review labels and original truth have different support.")
    return labels


def validate_trace(root, protocol, examples, budget, batch):
    directory = Path(root) / protocol / "experiments" / EXPERIMENT
    trace = read_csv_rows(directory / "trace.csv")
    expected = list(range(0, budget, batch)) + [budget]
    if [int(row["review_count"]) for row in trace] != expected:
        raise ValueError(f"Incomplete trajectory: {directory}")
    costs, units, missed, cumulative = [], 0.0, 0, 0
    seen = None
    expected_labels = examples.full_label if protocol == "full_replicate" else examples[f"{protocol}_label"]
    full_units, full_seconds, seconds = 0.0, 0.0, 0.0
    for i, row in enumerate(trace):
        step = directory / "steps" / f"step_{int(row['step_index']):03d}"
        recorded = np.load(step / "reviewed_mask.npy", allow_pickle=False).squeeze(axis=1)
        if seen is None:
            seen = np.zeros_like(recorded)
        if recorded.sum() != expected[i] or not np.array_equal(recorded, seen):
            raise ValueError("Saved reviewed mask and trace disagree.")
        values = np.load(step / "reviewed_values.npy", allow_pickle=False).squeeze(axis=1)
        point_seen = recorded[examples.site_index.to_numpy(int), examples.replicate_index.to_numpy(int)]
        point_values = values[examples.site_index.to_numpy(int), examples.replicate_index.to_numpy(int)]
        if not np.array_equal(point_values[point_seen], expected_labels.to_numpy()[point_seen]):
            raise ValueError("Saved review responses differ from the requested label protocol.")
        costs.append(dict(review_count=expected[i], review_units=units, full_review_units_for_selected=full_units,
                          review_seconds=seconds, full_review_seconds_for_selected=full_seconds,
                          missed_positive_replicates=missed, selected_replicates=cumulative))
        selected = np.load(step / "selected_mask.npy", allow_pickle=False).squeeze(axis=1)
        expected_selected = expected[i+1]-expected[i] if i+1 < len(expected) else 0
        if selected.sum() != expected_selected:
            raise ValueError("Unexpected selected batch size.")
        point_mask = selected[examples.site_index.to_numpy(int), examples.replicate_index.to_numpy(int)]
        points = examples.loc[point_mask]
        if len(points) != expected_selected or np.any(seen & selected):
            raise ValueError("Review repeats or selects padded observations.")
        seen |= selected
        label = points.full_label if protocol == "full_replicate" else points[f"{protocol}_label"]
        units += float((points.full_units if protocol == "full_replicate" else points[f"{protocol}_units"]).sum())
        full_units += float(points.full_units.sum())
        missed += int((points.full_label.eq(1) & label.eq(0)).sum())
        cumulative += len(points)
        if "full_seconds" in points:
            seconds += float((points.full_seconds if protocol == "full_replicate" else points[f"{protocol}_seconds"]).sum())
            full_seconds += float(points.full_seconds.sum())
    return pd.DataFrame(costs)


def prepare(source, output):
    """Construct subset labels from raw media annotations and fresh main-run inputs."""
    source, output = Path(source), Path(output)
    cohort, seen = [], set()
    for root in export_sweep.species_artifact_dirs(source):
        run = read_single_csv_row(root / "run_manifest.csv")
        if not run:
            continue
        identity = (run["dataset_kind"], run["target_species"])
        if identity in seen:
            continue
        seen.add(identity)
        cohort.append(dict(dataset_kind=identity[0], species=identity[1],
                           species_key=f"{identity[0]}__{slugify(identity[1])}",
                           reference_dir=str(root.resolve())))
    if not cohort:
        raise ValueError("No fresh main-experiment inputs found in the source sweep.")
    output.mkdir(parents=True, exist_ok=False)
    audit = []
    for dataset, generator in (("acoustic", acoustic_examples), ("iwildcam", camera_examples)):
        subset = [row for row in cohort if row["dataset_kind"] == dataset]
        if not subset:
            continue
        for row, examples in generator(subset, source):
            directory = output / row["species_key"]
            directory.mkdir()
            row["examples_file"] = str((directory / "examples.csv").resolve())
            examples.to_csv(row["examples_file"], index=False)
            audit.extend(audit_labels(examples, dataset, row["species"], row["species_key"]))
    write_csv_rows(output / "species.csv", cohort)
    write_csv_rows(output / "label_audit_by_species.csv", audit)
    return output


def run_worker(spec):
    from occubed import run_reviews as reviews

    if reviews.jax.config.jax_enable_x64:
        raise ValueError("Review-granularity experiments use manuscript float32 settings.")
    root = Path(spec["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    data, truth, _ = load_inputs(spec["reference_dir"])
    examples = pd.read_csv(spec["examples_file"], dtype={"site_id": str, "replicate_id": str})
    seed = int(spec["seed"])
    mcmc = dict(num_warmup=int(spec["num_warmup"]), num_samples=int(spec["num_samples"]),
                num_chains=int(spec["num_chains"]), random_seed=seed)
    calibration = dict(reviews.DEFAULT_CUT_NORMAL_SCORE_CALIBRATION_KWARGS,
                       score_calibration_random_seed=seed,
                       score_calibration_num_warmup=int(spec["score_num_warmup"]),
                       score_calibration_num_samples=int(spec["score_num_samples"]))
    oracle = reviews.fit_oracle_variant(spec["dataset_kind"], spec["species"], "full",
                                        data["site_covs"], data["obs_covs"], truth, mcmc)
    budget = min(int(spec["max_reviews"]), len(examples))
    frames = []
    for protocol in PROTOCOLS[spec["dataset_kind"]]:
        mode = root / protocol
        reviews.run_real_data_experiment(
            spec["dataset_kind"], spec["species"], output_dir=mode,
            mcmc_kwargs=mcmc, score_calibration_kwargs=calibration,
            n_reviews_per_step=int(spec["reviews_per_step"]), max_reviews=budget,
            continue_existing=True, experiment_ids=[EXPERIMENT], include_null=False,
            compact_selection_diagnostics=True, dataset_bundle=(data, truth),
            oracle_bundles={"full": oracle}, review_labels=label_tensor(truth["f"], examples, protocol),
            review_protocol=protocol)
        costs = validate_trace(root, protocol, examples, budget, int(spec["reviews_per_step"]))
        metrics = pd.read_csv(mode / "characterization/task_metric_rows.csv")
        metrics = metrics.merge(costs, on="review_count", validate="one_to_one")
        diagnostics = pd.read_csv(mode / "experiments" / EXPERIMENT / "trace.csv",
                                  usecols=["review_count", "num_diverging", "posterior_draws"])
        metrics = metrics.merge(diagnostics, on="review_count", validate="one_to_one")
        metrics["protocol"], metrics["seed"] = protocol, seed
        metrics["species_key"] = spec["species_key"]
        frames.append(metrics)
    metrics = pd.concat(frames, ignore_index=True)
    initial = metrics.loc[metrics.review_count.eq(0), list(METRICS)].to_numpy()
    if not np.allclose(initial, initial[0], atol=1e-12, rtol=0):
        raise ValueError("Paired protocols do not have identical zero-review metrics.")
    metrics.to_csv(root / "metrics.csv", index=False)
    write_json(root / "status.json", dict(status="completed", seed=seed, budget=budget))
    return str(root)


class GranularityJob(submitit.helpers.Checkpointable):
    def __init__(self, spec):
        self.spec = spec

    def __call__(self):
        return run_worker(self.spec)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def submit(prepared, output, seeds=(0, 1, 2), env_file=PACKAGE_DIR / ".env",
           num_warmup=500, num_samples=500, num_chains=3,
           score_num_warmup=100, score_num_samples=100, max_reviews=None, limit_per_dataset=None):
    """One species-seed job per array task; each shares one fresh oracle across protocols."""
    cohort = read_csv_rows(Path(prepared) / "species.csv")
    if isinstance(seeds, str):
        seeds = seeds.split(",")
    seeds = [int(seed) for seed in (seeds if isinstance(seeds, (list, tuple)) else [seeds])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct inference seeds.")
    if limit_per_dataset is not None:
        cohort = [row for dataset in PROTOCOLS for row in
                  [item for item in cohort if item["dataset_kind"] == dataset][:int(limit_per_dataset)]]
    if not cohort:
        raise ValueError("No prepared species selected.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = [dict(row, seed=seed, output_dir=str(output / f"seed_{seed}" / row["species_key"]),
                  num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                  score_num_warmup=score_num_warmup, score_num_samples=score_num_samples,
                  reviews_per_step=5 if row["dataset_kind"] == "acoustic" else 25,
                  max_reviews=(sys.maxsize if row["dataset_kind"] == "acoustic" else 1000)
                              if max_reviews is None else int(max_reviews))
             for seed in seeds for row in cohort]
    write_csv_rows(output / "job_specs.csv", specs)
    debug, parameters = load_submitit_config(env_file)
    parameters = {key: value for key, value in parameters.items()
                  if key != "slurm_array_parallelism" and (not debug or not key.startswith("slurm_"))}
    additional = dict(parameters.pop("slurm_additional_parameters", {}))
    additional.update(requeue=True, gpus=0)
    if not debug and len(specs) > 1:
        additional["array"] = f"0-{len(specs) - 1}"
    else:
        additional.pop("array", None)
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            del os.environ[key]
    executor = (submitit.DebugExecutor(folder=output / "submitit") if debug
                else submitit.AutoExecutor(folder=output / "submitit"))
    executor.update_parameters(name="review-granularity", slurm_additional_parameters=additional, **parameters)
    with executor.batch():
        jobs = [executor.submit(GranularityJob(spec)) for spec in specs]
    write_csv_rows(output / "submitted_jobs.csv", [dict(spec, job_id=str(job.job_id)) for spec, job in zip(specs, jobs)])
    for job in submitit.helpers.as_completed(jobs):
        job.result()
    analyze(output)
    return output
def effort_checkpoints(metrics):
    """Compare attained fitted checkpoints at matched media-inspection budgets."""
    rows = []
    for (dataset, species, seed), group in metrics.groupby(["dataset_kind", "species_key", "seed"]):
        unit = "review_seconds" if dataset == "acoustic" else "review_units"
        budgets = (60, 300, 600, 1800, 3600) if dataset == "acoustic" else (25, 50, 100, 500, 1000)
        maximum = group.groupby("protocol")[unit].max().min()
        for budget in budgets:
            if budget > maximum:
                continue
            for protocol, trajectory in group.groupby("protocol"):
                available = trajectory.loc[trajectory[unit].le(budget)].sort_values("review_count")
                if available.empty:
                    raise ValueError("Missing zero-effort checkpoint.")
                last = available.iloc[-1]
                rows.append(dict(dataset_kind=dataset, species_key=species, seed=seed, protocol=protocol,
                    effort_unit=unit, effort_budget=budget, attained_effort=float(last[unit]),
                    attained_review_count=int(last.review_count), **{metric: last[metric] for metric in METRICS}))
    return pd.DataFrame(rows)


def paired_summary(frame, value, group_columns):
    rng, rows = np.random.default_rng(0), []
    for keys, group in frame.groupby(["dataset_kind"] + group_columns):
        dataset, *rest = keys
        pivot = group.pivot(index=["species_key", "seed"], columns="protocol", values=value)
        for protocol in PROTOCOLS[dataset][1:]:
            values = (pivot[protocol]-pivot.full_replicate).groupby("species_key").mean().dropna().to_numpy()
            if not len(values):
                continue
            samples = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
            rows.append(dict(dataset_kind=dataset, protocol=protocol, **dict(zip(group_columns, rest)),
                species=len(values), mean_difference=float(values.mean()),
                ci_low=float(np.quantile(samples, .025)), ci_high=float(np.quantile(samples, .975))))
    return pd.DataFrame(rows)




def analyze(output):
    output = Path(output)
    specs = read_csv_rows(output / "job_specs.csv")
    frames = []
    for spec in specs:
        root = Path(spec["output_dir"])
        if not (root / "status.json").exists() or json.loads((root / "status.json").read_text())["status"] != "completed":
            raise ValueError(f"Cannot aggregate incomplete worker: {root}")
        frames.append(pd.read_csv(root / "metrics.csv"))
    metrics = pd.concat(frames, ignore_index=True)
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    metrics.to_csv(destination / "metrics_by_species_seed.csv", index=False)
    rows = []
    # Integrate each trajectory before seed averaging. Do not extrapolate beyond
    # measured effort or interpret species-seed pairs as independent species.
    for spec in specs:
        frame = metrics.loc[metrics.species_key.eq(spec["species_key"]) & metrics.seed.eq(int(spec["seed"]))]
        final_budget = int(frame.review_count.max())
        limits = [min(75 if spec['dataset_kind'] == 'acoustic' else 50, final_budget), min(100, final_budget), final_budget]
        for protocol, group in frame.groupby("protocol"):
            group = group.sort_values("review_count")
            for name, endpoint in zip(("early", "100_reviews", "full"), limits):
                for metric in METRICS:
                    rows.append(dict(dataset_kind=spec["dataset_kind"], species_key=spec["species_key"], seed=spec["seed"],
                        protocol=protocol, window=name, metric=metric, budget=endpoint,
                        auc=normalized_auc(group.review_count, group[metric], endpoint),
                        final_agreement=float(group[metric].iloc[-1])))
    auc = pd.DataFrame(rows)
    auc.to_csv(destination / "auc_by_species_seed.csv", index=False)
    differences = paired_summary(auc, "auc", ["window", "metric"]).rename(columns={"mean_difference": "mean_auc_difference"})
    differences.to_csv(destination / "paired_auc_differences.csv", index=False)
    effort = effort_checkpoints(metrics)
    effort.to_csv(destination / "matched_effort_checkpoints.csv", index=False)
    if not effort.empty:
        long = effort.melt(id_vars=["dataset_kind", "species_key", "seed", "protocol", "effort_unit", "effort_budget"],
                           value_vars=list(METRICS), var_name="metric", value_name="agreement")
        paired_summary(long, "agreement", ["effort_unit", "effort_budget", "metric"]).to_csv(
            destination / "paired_matched_effort_differences.csv", index=False)
    diagnostic_summary = metrics.groupby(["dataset_kind", "protocol"])[["num_diverging", "posterior_draws"]].sum()
    diagnostic_summary['divergence_fraction'] = diagnostic_summary.num_diverging/diagnostic_summary.posterior_draws
    diagnostic_summary.to_csv(destination / "sampling_diagnostics.csv")
    keys = ["dataset_kind", "species_key", "protocol", "review_count"]
    species = metrics.groupby(keys, as_index=False)[list(METRICS) + ["review_units", "review_seconds", "missed_positive_replicates"]].mean()
    summary = species.groupby(["dataset_kind", "protocol", "review_count"])[list(METRICS) + ["review_units", "review_seconds", "missed_positive_replicates"]].agg(['mean', 'sem'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index().to_csv(destination / "curves_species_summary.csv", index=False)
    plot_results(species, destination)
    return destination


def plot_results(species, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(full_replicate="#287c6f", top_window="#c13953", top_image="#c13953", top_sequence="#3469af")
    labels = dict(full_replicate="Full replicate", top_window="Top window", top_image="Top image", top_sequence="Top sequence")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, dataset in enumerate(PROTOCOLS):
        for protocol in PROTOCOLS[dataset]:
            group = species.loc[species.dataset_kind.eq(dataset) & species.protocol.eq(protocol)]
            for column, (metric, title) in enumerate((("occupancy_probability_agreement", "Occupancy agreement"),
                                                     ("site_rank_spearman", "Site-rank correlation"),
                                                     ("occupancy_coefficient_agreement", "Occupancy conclusions"),
                                                     ("detection_coefficient_agreement", "Detection conclusions"))):
                summary = group.groupby("review_count")[metric].agg(['mean', 'sem'])
                axes[row, column].plot(summary.index, summary['mean'], label=labels[protocol], color=colors[protocol])
                axes[row, column].fill_between(summary.index, summary['mean']-1.96*summary['sem'],
                    summary['mean']+1.96*summary['sem'], color=colors[protocol], alpha=.14)
                axes[row, column].set(title=title, xlabel="Replicates reviewed")
        axes[row, 0].set_ylabel("Acoustic" if dataset == 'acoustic' else "Camera trap")
        axes[row, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "single_example_agreement.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for column, dataset in enumerate(PROTOCOLS):
        for protocol in PROTOCOLS[dataset]:
            group = species.loc[species.dataset_kind.eq(dataset) & species.protocol.eq(protocol)]
            unit = "review_seconds" if dataset == 'acoustic' else "review_units"
            summary = group.groupby("review_count")[[unit, "mean_ecological_agreement"]].mean()
            axes[column].plot(summary[unit]/60 if dataset == 'acoustic' else summary[unit],
                summary.mean_ecological_agreement, label=labels[protocol], color=colors[protocol])
        axes[column].set(xlabel="Audio minutes inspected" if dataset == 'acoustic' else "Images inspected",
            ylabel="Mean ecological agreement", title="Acoustic" if dataset == 'acoustic' else "Camera trap")
        axes[column].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "single_example_effort.pdf")
    plt.close(fig)




def main(action="submit", **kwargs):
    actions = dict(prepare=prepare, submit=submit, analyze=analyze)
    if action not in actions:
        raise ValueError(f"Expected one of {tuple(actions)}")
    return actions[action](**kwargs)
