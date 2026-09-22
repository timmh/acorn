import csv
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import soundfile as sf
from perch_hoplite.zoo import model_configs

from occubed.paths import ACOUSTIC_DATA_DIR

WINDOW_SECONDS = 5.0
SAMPLE_RATE = 32000
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)


@dataclass(frozen=True)
class TargetSpecies:
    """A dataset label that maps cleanly onto a Perch output class."""

    aou_code: str
    common_name: str
    scientific_name: str
    is_bird: str
    perch_label_index: int
    annotation_count: int
    recording_count: int

    @property
    def logit_column(self) -> str:
        return f"logit_{self.aou_code}"


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def collect_ground_truth_stats(
    annotation_dir: Path,
) -> tuple[Counter[str], Counter[str]]:
    """Count annotation boxes and recording presence by dataset label code."""

    annotation_counts: Counter[str] = Counter()
    recording_counts: Counter[str] = Counter()
    for annotation_path in sorted(annotation_dir.glob("**/*.txt")):
        species_in_recording: set[str] = set()
        with annotation_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                species_code = (row.get("Species") or "").strip()
                if not species_code:
                    continue
                annotation_counts[species_code] += 1
                species_in_recording.add(species_code)
        recording_counts.update(species_in_recording)
    return annotation_counts, recording_counts


def load_perch_label_index(model_path: Path) -> dict[str, int]:
    labels_csv = model_path / "assets" / "labels.csv"
    require_path(labels_csv, "Perch labels.csv")
    with labels_csv.open("r", newline="") as handle:
        reader = csv.reader(handle)
        namespace_row = next(reader, None)
        if namespace_row is None:
            raise ValueError(f"Perch labels file is empty: {labels_csv}")
        return {
            row[0].strip(): index
            for index, row in enumerate(reader)
            if row and row[0].strip()
        }


def build_target_species(
    annotation_dir: Path,
    species_csv_path: Path,
    perch_label_index: dict[str, int],
) -> tuple[list[TargetSpecies], pd.DataFrame]:
    """Map dataset labels from the annotations onto Perch label indices."""

    annotation_counts, recording_counts = collect_ground_truth_stats(annotation_dir)
    species_table = pd.read_csv(species_csv_path).copy()
    species_lookup = species_table.drop_duplicates(subset=["AOU_Code"]).set_index(
        "AOU_Code"
    )

    target_species: list[TargetSpecies] = []
    excluded_rows: list[dict[str, object]] = []
    for aou_code in sorted(annotation_counts):
        if aou_code not in species_lookup.index:
            excluded_rows.append(
                {
                    "AOU_Code": aou_code,
                    "Common_Name": None,
                    "Scientific_Name": None,
                    "IsBird": None,
                    "annotation_count": annotation_counts[aou_code],
                    "recording_count": recording_counts[aou_code],
                    "reason": "missing_from_species_csv",
                }
            )
            continue

        row = species_lookup.loc[aou_code]
        scientific_name = row["Scientific_Name"]
        common_name = row["Common_Name"]
        is_bird = row["IsBird"]
        if pd.isna(scientific_name) or not str(scientific_name).strip():
            excluded_rows.append(
                {
                    "AOU_Code": aou_code,
                    "Common_Name": common_name,
                    "Scientific_Name": scientific_name,
                    "IsBird": is_bird,
                    "annotation_count": annotation_counts[aou_code],
                    "recording_count": recording_counts[aou_code],
                    "reason": "missing_scientific_name",
                }
            )
            continue

        scientific_name = str(scientific_name).strip()
        if scientific_name not in perch_label_index:
            excluded_rows.append(
                {
                    "AOU_Code": aou_code,
                    "Common_Name": common_name,
                    "Scientific_Name": scientific_name,
                    "IsBird": is_bird,
                    "annotation_count": annotation_counts[aou_code],
                    "recording_count": recording_counts[aou_code],
                    "reason": "scientific_name_not_in_perch_labels",
                }
            )
            continue

        target_species.append(
            TargetSpecies(
                aou_code=aou_code,
                common_name=str(common_name),
                scientific_name=scientific_name,
                is_bird=str(is_bird),
                perch_label_index=perch_label_index[scientific_name],
                annotation_count=annotation_counts[aou_code],
                recording_count=recording_counts[aou_code],
            )
        )

    excluded_columns = [
        "AOU_Code",
        "Common_Name",
        "Scientific_Name",
        "IsBird",
        "annotation_count",
        "recording_count",
        "reason",
    ]
    excluded_table = pd.DataFrame(excluded_rows, columns=excluded_columns)
    if not excluded_table.empty:
        excluded_table = excluded_table.sort_values(
            ["reason", "AOU_Code"], ignore_index=True
        )
    return sorted(target_species, key=lambda item: item.aou_code), excluded_table


def write_species_mapping(
    output_dir: Path,
    target_species: list[TargetSpecies],
    excluded_table: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped_rows = [
        {
            "AOU_Code": item.aou_code,
            "Common_Name": item.common_name,
            "Scientific_Name": item.scientific_name,
            "IsBird": item.is_bird,
            "perch_label_index": item.perch_label_index,
            "annotation_count": item.annotation_count,
            "recording_count": item.recording_count,
        }
        for item in target_species
    ]
    pd.DataFrame(mapped_rows).to_csv(
        output_dir / "dataset_species_perch_mapping.csv", index=False
    )
    excluded_table.to_csv(output_dir / "excluded_ground_truth_labels.csv", index=False)


def normalize_audio(audio_batch: np.ndarray, target_peak: float | None) -> np.ndarray:
    """Match perch-hoplite's audio normalization for fixed-size windows."""

    if target_peak is None:
        return audio_batch

    audio_batch = audio_batch.copy()
    audio_batch -= np.mean(audio_batch, axis=-1, keepdims=True)
    peak_norm = np.max(np.abs(audio_batch), axis=-1, keepdims=True)
    normalized = np.zeros_like(audio_batch)
    np.divide(audio_batch, peak_norm, out=normalized, where=(peak_norm > 0.0))
    normalized *= target_peak
    return normalized


def resolve_recording_paths(
    metadata: pd.DataFrame,
    recordings_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame, list[str]]:
    """Resolve metadata rows onto extracted FLAC paths.

    The Zenodo metadata currently contains a small number of filename mismatches
    relative to the extracted archives. We resolve those rows by exact filename
    first and then by unique ItemID, which is unique across the entire dataset.
    """

    paths = sorted(recordings_dir.glob("**/*.flac"))
    if not paths:
        raise FileNotFoundError(
            f"No FLAC recordings were found under {recordings_dir}. "
            "Download and extract the recording archives first."
        )

    by_name: dict[str, Path] = {}
    by_item_id: dict[str, Path] = {}
    duplicate_names: list[str] = []
    duplicate_item_ids: list[str] = []
    for path in paths:
        if path.name in by_name:
            duplicate_names.append(path.name)
        by_name[path.name] = path

        item_id = path.name.split(".", maxsplit=1)[0]
        if item_id in by_item_id:
            duplicate_item_ids.append(item_id)
        by_item_id[item_id] = path

    if duplicate_names:
        duplicate_list = ", ".join(sorted(set(duplicate_names))[:10])
        raise ValueError(f"Duplicate recording filenames found: {duplicate_list}")
    if duplicate_item_ids:
        duplicate_list = ", ".join(sorted(set(duplicate_item_ids))[:10])
        raise ValueError(f"Duplicate recording ItemIDs found: {duplicate_list}")

    resolved_paths: dict[str, Path] = {}
    mismatch_rows: list[dict[str, object]] = []
    missing_rows: list[str] = []
    for row in metadata.itertuples(index=False):
        expected_name = getattr(row, "Recording_FileName")
        item_id = str(getattr(row, "ItemID"))
        if expected_name in by_name:
            resolved_paths[expected_name] = by_name[expected_name]
            continue
        if item_id not in by_item_id:
            missing_rows.append(expected_name)
            continue

        actual_path = by_item_id[item_id]
        resolved_paths[expected_name] = actual_path
        mismatch_rows.append(
            {
                "ItemID": item_id,
                "expected_recording_filename": expected_name,
                "actual_recording_filename": actual_path.name,
                "actual_audio_path": str(actual_path),
            }
        )

    mismatch_columns = [
        "ItemID",
        "expected_recording_filename",
        "actual_recording_filename",
        "actual_audio_path",
    ]
    mismatch_table = pd.DataFrame(mismatch_rows, columns=mismatch_columns)
    if not mismatch_table.empty:
        mismatch_table = mismatch_table.sort_values(["ItemID"], ignore_index=True)
    return resolved_paths, mismatch_table, missing_rows


def iter_window_batches(
    audio_path: Path,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, list[int], list[float], float]]:
    """Yield normalized 5-second windows from a recording in mini-batches."""

    with sf.SoundFile(audio_path) as audio_file:
        if audio_file.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate for {audio_path}: "
                f"{audio_file.samplerate} != {SAMPLE_RATE}"
            )

        duration_s = audio_file.frames / audio_file.samplerate
        batch_windows: list[np.ndarray] = []
        batch_window_indices: list[int] = []
        batch_starts: list[float] = []
        for window_index, block in enumerate(
            audio_file.blocks(
                blocksize=WINDOW_SAMPLES,
                overlap=0,
                dtype="float32",
                always_2d=True,
            )
        ):
            mono = block.mean(axis=1)
            if mono.shape[0] < WINDOW_SAMPLES:
                mono = np.pad(mono, (0, WINDOW_SAMPLES - mono.shape[0]))
            batch_windows.append(mono)
            batch_window_indices.append(window_index)
            batch_starts.append(window_index * WINDOW_SECONDS)
            if len(batch_windows) == batch_size:
                yield (
                    np.stack(batch_windows, axis=0),
                    batch_window_indices,
                    batch_starts,
                    duration_s,
                )
                batch_windows = []
                batch_window_indices = []
                batch_starts = []

        if batch_windows:
            yield (
                np.stack(batch_windows, axis=0),
                batch_window_indices,
                batch_starts,
                duration_s,
            )


def infer_recording(
    audio_path: Path,
    metadata: dict[str, object],
    infer_fn,
    target_peak: float | None,
    target_species: list[TargetSpecies],
    batch_size: int,
) -> list[dict[str, object]]:
    """Run Perch logits for a single recording and return CSV-ready rows."""

    label_indices = np.asarray(
        [item.perch_label_index for item in target_species], dtype=np.int64
    )
    rows: list[dict[str, object]] = []
    for batch_audio, window_indices, window_starts, duration_s in iter_window_batches(
        audio_path, batch_size=batch_size
    ):
        normalized = normalize_audio(batch_audio, target_peak=target_peak)
        outputs = infer_fn(inputs=normalized)
        logits = np.asarray(outputs["label"], dtype=np.float32)[:, label_indices]
        for row_offset, window_index in enumerate(window_indices):
            row = dict(metadata)
            row["audio_path"] = str(audio_path)
            row["window_index"] = window_index
            row["window_start_s"] = window_starts[row_offset]
            row["window_end_s"] = min(
                window_starts[row_offset] + WINDOW_SECONDS, duration_s
            )
            for species_offset, item in enumerate(target_species):
                row[item.logit_column] = float(logits[row_offset, species_offset])
            rows.append(row)
    return rows


def write_part_csv(
    part_path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    """Write one recording's window-level logits atomically."""

    if not rows:
        raise ValueError(f"No inference rows were produced for {part_path.name}")
    tmp_path = part_path.parent / f"{part_path.name}.tmp"
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(part_path)


def concatenate_part_csvs(
    part_paths: list[Path],
    output_csv: Path,
    fieldnames: list[str],
) -> None:
    """Concatenate per-recording CSV parts into a single final CSV."""

    tmp_output = output_csv.parent / f"{output_csv.name}.tmp"
    with tmp_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for part_path in part_paths:
            with part_path.open("r", newline="") as part_handle:
                reader = csv.reader(part_handle)
                header = next(reader)
                if header != fieldnames:
                    raise ValueError(f"Unexpected header in part file: {part_path}")
                writer.writerows(reader)
    tmp_output.replace(output_csv)


def main(
    data_dir: str | Path = ACOUSTIC_DATA_DIR,
    recordings_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str = "perch_v2_cpu",
    infer_batch_size: int = 32,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    data_dir = Path(data_dir).expanduser().resolve()
    recordings_dir = (
        Path(recordings_dir).expanduser().resolve()
        if recordings_dir is not None
        else (data_dir / "recordings").resolve()
    )
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (data_dir / "perch_v2").resolve()
    )
    parts_dir = output_dir / "window_logits_parts"
    output_csv = output_dir / "perch_v2_window_logits.csv"

    raw_dir = data_dir / "raw"
    annotation_dir = data_dir / "annotations"
    metadata_csv = raw_dir / "recording_metadata.csv"
    species_csv = raw_dir / "species.csv"

    require_path(metadata_csv, "recording metadata CSV")
    require_path(species_csv, "species CSV")
    require_path(annotation_dir, "annotation directory")
    require_path(recordings_dir, "recordings directory")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model preset {model_name}...", flush=True)
    model = model_configs.load_model_by_name(model_name)
    infer_fn = model.model.signatures["serving_default"]
    perch_label_index = load_perch_label_index(Path(model.model_path))

    print("Building dataset-to-Perch species mapping from annotations...", flush=True)
    target_species, excluded_table = build_target_species(
        annotation_dir=annotation_dir,
        species_csv_path=species_csv,
        perch_label_index=perch_label_index,
    )
    if not target_species:
        raise ValueError("No ground-truth species mapped to Perch labels.")
    write_species_mapping(output_dir, target_species, excluded_table)
    print(
        f"Mapped {len(target_species)} species to Perch labels; "
        f"excluded {len(excluded_table)} non-species or unmapped labels.",
        flush=True,
    )

    metadata = pd.read_csv(metadata_csv)
    if limit is not None:
        metadata = metadata.head(int(limit)).copy()
    recording_lookup, mismatch_table, missing_recordings = resolve_recording_paths(
        metadata=metadata,
        recordings_dir=recordings_dir,
    )
    mismatch_table.to_csv(output_dir / "recording_filename_mismatches.csv", index=False)
    if not mismatch_table.empty:
        print(
            f"Resolved {len(mismatch_table)} metadata/archive filename mismatches by ItemID.",
            flush=True,
        )
    if missing_recordings:
        preview = ", ".join(missing_recordings[:10])
        raise FileNotFoundError(
            f"Missing extracted recordings for {len(missing_recordings)} metadata rows. "
            f"First few: {preview}"
        )

    base_fieldnames = list(metadata.columns) + [
        "audio_path",
        "window_index",
        "window_start_s",
        "window_end_s",
    ]
    fieldnames = base_fieldnames + [item.logit_column for item in target_species]

    total_recordings = len(metadata)
    part_paths: list[Path] = []
    for idx, row in enumerate(metadata.itertuples(index=False), start=1):
        recording_name = getattr(row, "Recording_FileName")
        part_path = parts_dir / f"{recording_name}.csv"
        part_paths.append(part_path)
        if part_path.exists():
            print(
                f"[{idx}/{total_recordings}] Skipping existing part {recording_name}",
                flush=True,
            )
            continue

        audio_path = recording_lookup[recording_name]
        metadata_dict = row._asdict()
        print(f"[{idx}/{total_recordings}] Scoring {recording_name}", flush=True)
        rows = infer_recording(
            audio_path=audio_path,
            metadata=metadata_dict,
            infer_fn=infer_fn,
            target_peak=model.target_peak,
            target_species=target_species,
            batch_size=int(infer_batch_size),
        )
        expected_windows = math.ceil(sf.info(audio_path).frames / WINDOW_SAMPLES)
        if len(rows) != expected_windows:
            raise ValueError(
                f"Unexpected window count for {audio_path}: "
                f"{len(rows)} != {expected_windows}"
            )
        write_part_csv(part_path=part_path, rows=rows, fieldnames=fieldnames)

    print("Concatenating per-recording CSV parts...", flush=True)
    concatenate_part_csvs(
        part_paths=part_paths,
        output_csv=output_csv,
        fieldnames=fieldnames,
    )
    print(f"Wrote {output_csv}", flush=True)
    return 0
