import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import kagglehub
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import submitit
from dotenv import dotenv_values, load_dotenv
from tqdm import tqdm

from occubed.paths import IWILDCAM2022_DIR, IWILDCAM_RAW_DIR, PACKAGE_DIR

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_RAW_DATA_DIR = IWILDCAM_RAW_DIR
DEFAULT_OUTPUT_DIR = IWILDCAM2022_DIR
DEFAULT_MODEL_HANDLE = "kaggle:google/speciesnet/keras/v4.0.0b"

SPECIESNET_IMAGE_SIZE = 480
SPECIESNET_MAX_CROP_RATIO = 0.3
SPECIESNET_MAX_CROP_PIXELS = 400

IMAGE_METADATA_COLUMNS = [
    "split",
    "image_id",
    "annotation_id",
    "category_id",
    "category_name",
    "has_ground_truth",
    "is_empty",
    "file_name",
    "relative_path",
    "location_id",
    "sub_location",
    "seq_id",
    "seq_frame_num",
    "seq_num_frames",
    "capture_datetime",
    "width",
    "height",
    "latitude",
    "longitude",
]

# A small set of scientific-name updates, spelling fixes, and coarse unknowns
# needed for the iWildCam 2022 labels to land on the SpeciesNet v4 taxonomy.
GROUND_TRUTH_NAME_ALIASES = {
    "agouti paca": "cuniculus paca",
    "aguila sp": "accipitridae family",
    "andropadus latirostris": "eurillas latirostris",
    "andropadus virens": "eurillas virens",
    "aramides cajanea": "aramides cajaneus",
    "ave desconocida": "bird",
    "cercopithecus lhoesti": "allochrocebus lhoesti",
    "dioptrornis fischeri": "melaenornis fischeri",
    "erithacus cyane": "larvivora cyane",
    "eurocephalus rueppelli": "eurocephalus ruppelli",
    "francolinus africanus": "scleroptila afra",
    "francolinus nobilis": "pternistis nobilis",
    "mesopicos griseocephalus": "dendropicos griseocephalus",
    "myiophoneus caeruleus": "myophonus caeruleus",
    "myiophoneus glaucinus": "myophonus glaucinus",
    "myiophoneus melanurus": "myophonus melanurus",
    "pardofelis temminckii": "catopuma temminckii",
    "phaetornis sp": "phaethornis sp",
    "puma yagoroundi": "herpailurus yagouaroundi",
    "puma yagouaroundi": "herpailurus yagouaroundi",
    "streptopilia senegalensis": "spilopelia senegalensis",
    "turtur calcospilos": "turtur chalcospilos",
    "unknown bat": "bat",
    "unknown bird": "bird",
}

NON_TARGET_CATEGORIES = {"empty", "motorcycle"}


@dataclass(frozen=True)
class ModelArtifacts:
    model_dir: Path
    classifier_path: Path
    labels_path: Path
    info_path: Path
    info: dict[str, Any]


def normalize_name(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def stable_shard_index(value: str, num_shards: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(num_shards)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def resolve_model_artifacts(model_handle: str) -> ModelArtifacts:
    if model_handle.startswith("kaggle:"):
        model_dir = Path(kagglehub.model_download(model_handle[len("kaggle:") :]))
    else:
        model_dir = Path(model_handle).expanduser()

    info_path = model_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"SpeciesNet model info.json not found: {info_path}")
    info = load_json(info_path)
    if info.get("type") != "full_image":
        raise ValueError(
            f"Expected a SpeciesNet full_image model, found type={info.get('type')!r}."
        )

    classifier_path = model_dir / str(info["classifier"])
    labels_path = model_dir / str(info["classifier_labels"])
    if not classifier_path.exists():
        saved_model_candidates = sorted(model_dir.glob("**/saved_model.pb"))
        if saved_model_candidates:
            classifier_path = saved_model_candidates[0].parent
    if not classifier_path.exists():
        raise FileNotFoundError(f"SpeciesNet classifier not found: {classifier_path}")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"SpeciesNet classifier labels not found: {labels_path}"
        )

    return ModelArtifacts(
        model_dir=model_dir,
        classifier_path=classifier_path,
        labels_path=labels_path,
        info_path=info_path,
        info=info,
    )


def load_speciesnet_labels(labels_path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise ValueError(f"No SpeciesNet labels found in {labels_path}")
    return labels


def parse_speciesnet_label(label: str) -> dict[str, str]:
    fields = label.split(";")
    padded = fields + [""] * max(0, 7 - len(fields))
    return {
        "speciesnet_label": label,
        "speciesnet_taxon_id": padded[0],
        "speciesnet_class": padded[1],
        "speciesnet_order": padded[2],
        "speciesnet_family": padded[3],
        "speciesnet_genus": padded[4],
        "speciesnet_species": padded[5],
        "speciesnet_common_name": padded[6],
    }


def build_label_lookup(labels: list[str]) -> dict[str, dict[str, tuple[int, str]]]:
    scientific: dict[str, tuple[int, str]] = {}
    genus_level: dict[str, tuple[int, str]] = {}
    common: dict[str, tuple[int, str]] = {}

    for index, label in enumerate(labels):
        parsed = parse_speciesnet_label(label)
        genus = normalize_name(parsed["speciesnet_genus"])
        species = normalize_name(parsed["speciesnet_species"])
        common_name = normalize_name(parsed["speciesnet_common_name"])
        family = normalize_name(parsed["speciesnet_family"])
        if genus and species:
            scientific.setdefault(f"{genus} {species}", (index, label))
        if genus and not species:
            genus_level.setdefault(f"{genus} sp", (index, label))
            genus_level.setdefault(f"{genus} species", (index, label))
        if common_name:
            common.setdefault(common_name, (index, label))
        if family and not genus and "family" in common_name:
            common.setdefault(f"{family} family", (index, label))
    return {
        "scientific": scientific,
        "genus_level": genus_level,
        "common": common,
    }


def match_speciesnet_label(
    category_name: str,
    lookup: dict[str, dict[str, tuple[int, str]]],
) -> tuple[int, str, str, str] | None:
    normalized = normalize_name(category_name)
    aliased = normalize_name(GROUND_TRUTH_NAME_ALIASES.get(normalized, normalized))
    for method, table in (
        ("scientific_name", lookup["scientific"]),
        ("genus_level", lookup["genus_level"]),
        ("common_name", lookup["common"]),
    ):
        if aliased in table:
            label_index, label = table[aliased]
            return label_index, label, aliased, method
    return None


def load_iwildcam_metadata(
    raw_data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata_dir = raw_data_dir / "metadata"
    train = load_json(metadata_dir / "iwildcam2022_train_annotations.json")
    test = load_json(metadata_dir / "iwildcam2022_test_information.json")
    gps_locations = load_json(metadata_dir / "gps_locations.json")
    return train, test, gps_locations


def write_category_tables(
    output_dir: Path,
    train_metadata: dict[str, Any],
    labels: list[str],
) -> list[dict[str, Any]]:
    categories = sorted(train_metadata["categories"], key=lambda row: int(row["id"]))
    annotations = train_metadata["annotations"]
    annotation_counts: Counter[int] = Counter(
        int(row["category_id"]) for row in annotations
    )
    lookup = build_label_lookup(labels)

    category_rows = []
    target_rows = []
    excluded_rows = []
    for category in categories:
        category_id = int(category["id"])
        category_name = str(category["name"])
        count = int(annotation_counts.get(category_id, 0))
        category_rows.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "annotation_count": count,
            }
        )
        if count <= 0:
            continue

        normalized = normalize_name(category_name)
        if normalized in NON_TARGET_CATEGORIES:
            excluded_rows.append(
                {
                    "category_id": category_id,
                    "category_name": category_name,
                    "annotation_count": count,
                    "reason": "non_species_category",
                }
            )
            continue

        match = match_speciesnet_label(category_name, lookup)
        if match is None:
            excluded_rows.append(
                {
                    "category_id": category_id,
                    "category_name": category_name,
                    "annotation_count": count,
                    "reason": "not_found_in_speciesnet_labels",
                }
            )
            continue

        speciesnet_label_index, speciesnet_label, match_name, match_method = match
        parsed = parse_speciesnet_label(speciesnet_label)
        scientific_name = " ".join(
            value
            for value in [
                parsed["speciesnet_genus"].strip(),
                parsed["speciesnet_species"].strip(),
            ]
            if value
        )
        target_rows.append(
            {
                "target_logit_index": len(target_rows),
                "category_id": category_id,
                "category_name": category_name,
                "normalized_category_name": normalized,
                "annotation_count": count,
                "speciesnet_label_index": speciesnet_label_index,
                "speciesnet_label": speciesnet_label,
                "speciesnet_scientific_name": scientific_name,
                "speciesnet_common_name": parsed["speciesnet_common_name"],
                "match_name": match_name,
                "match_method": match_method,
            }
        )

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(category_rows).to_csv(metadata_dir / "categories.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        output_dir / "target_species_mapping.csv",
        index=False,
    )
    pd.DataFrame(excluded_rows).to_csv(
        output_dir / "excluded_ground_truth_categories.csv",
        index=False,
    )
    (output_dir / "target_labels.txt").write_text(
        "\n".join(row["speciesnet_label"] for row in target_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "target_category_names.txt").write_text(
        "\n".join(row["category_name"] for row in target_rows) + "\n",
        encoding="utf-8",
    )

    if not target_rows:
        raise ValueError(
            "No iWildCam ground-truth species mapped to SpeciesNet labels."
        )
    return target_rows


def image_record(
    *,
    split: str,
    row: dict[str, Any],
    annotation: dict[str, Any] | None,
    category_names: dict[int, str],
    gps_locations: dict[str, dict[str, float]],
) -> dict[str, Any]:
    location_id = str(row.get("location", ""))
    gps = gps_locations.get(location_id, {})
    category_id = "" if annotation is None else int(annotation["category_id"])
    category_name = (
        "" if annotation is None else category_names.get(int(category_id), "")
    )
    return {
        "split": split,
        "image_id": str(row.get("id", "")),
        "annotation_id": "" if annotation is None else str(annotation.get("id", "")),
        "category_id": category_id,
        "category_name": category_name,
        "has_ground_truth": int(annotation is not None),
        "is_empty": int(normalize_name(category_name) == "empty"),
        "file_name": str(row.get("file_name", "")),
        "relative_path": f"{split}/{row.get('file_name', '')}",
        "location_id": location_id,
        "sub_location": str(row.get("sub_location", "")),
        "seq_id": str(row.get("seq_id", "")),
        "seq_frame_num": row.get("seq_frame_num", ""),
        "seq_num_frames": row.get("seq_num_frames", ""),
        "capture_datetime": str(row.get("datetime", "")),
        "width": row.get("width", ""),
        "height": row.get("height", ""),
        "latitude": gps.get("latitude", np.nan),
        "longitude": gps.get("longitude", np.nan),
    }


def build_image_records(
    train_metadata: dict[str, Any],
    test_metadata: dict[str, Any],
    gps_locations: dict[str, dict[str, float]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    category_names = {
        int(category["id"]): str(category["name"])
        for category in train_metadata["categories"]
    }
    annotations_by_image_id = {
        str(annotation["image_id"]): annotation
        for annotation in train_metadata["annotations"]
    }

    records = [
        image_record(
            split="train",
            row=row,
            annotation=annotations_by_image_id.get(str(row.get("id", ""))),
            category_names=category_names,
            gps_locations=gps_locations,
        )
        for row in train_metadata["images"]
    ]
    records.extend(
        image_record(
            split="test",
            row=row,
            annotation=None,
            category_names=category_names,
            gps_locations=gps_locations,
        )
        for row in test_metadata["images"]
    )
    if limit is not None:
        records = records[: int(limit)]
    return records


def write_image_metadata_and_shards(
    output_dir: Path,
    records: list[dict[str, Any]],
    num_shards: int,
    overwrite_plan: bool,
) -> list[int]:
    metadata_dir = output_dir / "metadata"
    inputs_dir = output_dir / "inputs"
    if overwrite_plan and inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records, columns=IMAGE_METADATA_COLUMNS).to_csv(
        metadata_dir / "images.csv",
        index=False,
    )

    shard_counts = [0] * int(num_shards)
    shard_paths = [
        inputs_dir / f"shard_{idx:05d}_of_{num_shards:05d}.csv"
        for idx in range(int(num_shards))
    ]
    handles = [path.open("w", newline="", encoding="utf-8") for path in shard_paths]
    try:
        writers = [
            csv.DictWriter(handle, fieldnames=IMAGE_METADATA_COLUMNS)
            for handle in handles
        ]
        for writer in writers:
            writer.writeheader()
        for record in records:
            shard_idx = stable_shard_index(str(record["image_id"]), int(num_shards))
            writers[shard_idx].writerow(record)
            shard_counts[shard_idx] += 1
    finally:
        for handle in handles:
            handle.close()
    return shard_counts


def prepare_plan(
    *,
    raw_data_dir: Path,
    output_dir: Path,
    num_shards: int,
    model_handle: str,
    limit: int | None,
    overwrite_plan: bool,
) -> dict[str, Any]:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and not overwrite_plan:
        return load_json(plan_path)

    artifacts = resolve_model_artifacts(model_handle)
    shutil.copyfile(artifacts.labels_path, output_dir / "labels.txt")
    shutil.copyfile(artifacts.info_path, output_dir / "info.json")
    train_metadata, test_metadata, gps_locations = load_iwildcam_metadata(raw_data_dir)
    labels = load_speciesnet_labels(output_dir / "labels.txt")
    target_rows = write_category_tables(output_dir, train_metadata, labels)
    gps_rows = [
        {
            "location_id": str(location_id),
            "latitude": values.get("latitude", np.nan),
            "longitude": values.get("longitude", np.nan),
        }
        for location_id, values in sorted(
            gps_locations.items(),
            key=lambda item: str(item[0]),
        )
    ]
    pd.DataFrame(gps_rows).to_csv(
        output_dir / "metadata" / "gps_locations.csv",
        index=False,
    )

    records = build_image_records(
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        gps_locations=gps_locations,
        limit=limit,
    )
    shard_counts = write_image_metadata_and_shards(
        output_dir=output_dir,
        records=records,
        num_shards=int(num_shards),
        overwrite_plan=bool(overwrite_plan),
    )
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    split_counts = Counter(record["split"] for record in records)
    missing_paths = sum(
        not (raw_data_dir / str(record["relative_path"])).exists() for record in records
    )
    plan = {
        "plan_name": "iwildcam2022_speciesnet_v4_0_0b_all_image",
        "raw_data_dir": str(raw_data_dir),
        "output_dir": str(output_dir),
        "inputs_dir": str(output_dir / "inputs"),
        "outputs_dir": str(outputs_dir),
        "model_handle": model_handle,
        "model_dir": str(artifacts.model_dir),
        "classifier_path": str(artifacts.classifier_path),
        "labels_path": str(output_dir / "labels.txt"),
        "target_labels_path": str(output_dir / "target_labels.txt"),
        "target_species_mapping_path": str(output_dir / "target_species_mapping.csv"),
        "num_shards": int(num_shards),
        "num_rows": int(len(records)),
        "num_train_rows": int(split_counts.get("train", 0)),
        "num_test_rows": int(split_counts.get("test", 0)),
        "num_target_species": int(len(target_rows)),
        "limit": None if limit is None else int(limit),
        "shard_counts": shard_counts,
        "missing_image_paths": int(missing_paths),
        "created_at_unix": time.time(),
    }
    write_json(plan_path, plan)
    return plan


def require_tensorflow_gpu(device: str | None = None):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "SpeciesNet iWildCam inference requires a CUDA GPU, but TensorFlow "
            "does not see any GPU devices."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if device is not None:
        logical_gpus = tf.config.list_logical_devices("GPU")
        if not logical_gpus:
            raise RuntimeError(
                "TensorFlow has physical GPUs but no logical GPU devices."
            )
    return tf


def load_keras_classifier(classifier_path: Path):
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(classifier_path, compile=False)
    except Exception:
        return tf.saved_model.load(str(classifier_path))


def extract_logits(output: Any, expected_batch_size: int) -> np.ndarray:
    import tensorflow as tf

    if isinstance(output, dict):
        tensors = list(output.values())
    elif isinstance(output, (list, tuple)):
        tensors = list(output)
    else:
        tensors = [output]

    candidates = []
    for tensor in tensors:
        arr = tensor.numpy() if isinstance(tensor, tf.Tensor) else np.asarray(tensor)
        if arr.ndim == 2 and arr.shape[0] == expected_batch_size:
            candidates.append(arr)
    if not candidates:
        shapes = [np.asarray(tensor).shape for tensor in tensors]
        raise ValueError(
            f"Could not identify a 2D logits output. Output shapes: {shapes}"
        )
    return np.asarray(
        max(candidates, key=lambda arr: arr.shape[1]),
        dtype=np.float32,
    )


def predict_logits(model: Any, batch: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
    if callable(model):
        try:
            return extract_logits(
                model(tensor, training=False),
                expected_batch_size=len(batch),
            )
        except TypeError:
            return extract_logits(model(tensor), expected_batch_size=len(batch))
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        return extract_logits(
            model.signatures["serving_default"](tensor),
            expected_batch_size=len(batch),
        )
    raise TypeError(
        "Loaded SpeciesNet classifier is not callable and has no serving_default "
        "signature."
    )


def preprocess_full_image(image: PIL.Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = PIL.ImageOps.exif_transpose(image)
    crop_pixels = min(
        int(image.height * SPECIESNET_MAX_CROP_RATIO),
        SPECIESNET_MAX_CROP_PIXELS,
    )
    top = crop_pixels // 2
    bottom = image.height - (crop_pixels - top)
    image = image.crop((0, top, image.width, bottom))
    image = image.resize(
        (SPECIESNET_IMAGE_SIZE, SPECIESNET_IMAGE_SIZE),
        resample=PIL.Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def load_preprocessed_image(
    raw_data_dir: Path,
    row_idx: int,
    relative_path: str,
) -> dict[str, Any]:
    image_path = raw_data_dir / relative_path
    try:
        with PIL.Image.open(image_path) as image:
            array = preprocess_full_image(image)
        return {"row_idx": row_idx, "image": array}
    except Exception as exc:  # pragma: no cover - data dependent
        return {"row_idx": row_idx, "error": f"{type(exc).__name__}: {exc}"}


def iter_preprocessed_batches(
    shard_df: pd.DataFrame,
    raw_data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Iterable[dict[str, Any]]:
    records = [
        (idx, str(row.relative_path))
        for idx, row in enumerate(shard_df.itertuples(index=False))
    ]
    total = len(records)
    executor = (
        ThreadPoolExecutor(max_workers=int(num_workers))
        if int(num_workers) > 1
        else None
    )
    try:
        for start in range(0, total, int(batch_size)):
            batch_records = records[start : start + int(batch_size)]
            if executor is not None:
                loaded = list(
                    executor.map(
                        lambda item: load_preprocessed_image(
                            raw_data_dir,
                            item[0],
                            item[1],
                        ),
                        batch_records,
                    )
                )
            else:
                loaded = [
                    load_preprocessed_image(raw_data_dir, row_idx, relative_path)
                    for row_idx, relative_path in batch_records
                ]

            row_indices = []
            images = []
            failures = []
            for item in loaded:
                if "error" in item:
                    failures.append((int(item["row_idx"]), str(item["error"])))
                else:
                    row_indices.append(int(item["row_idx"]))
                    images.append(item["image"])
            yield {
                "row_indices": np.asarray(row_indices, dtype=np.int64),
                "images": np.stack(images, axis=0) if images else None,
                "failures": failures,
            }
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def run_shard(
    *,
    shard_index: int,
    plan_dir: str | Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 128,
    num_workers: int = 8,
    overwrite: bool = False,
    save_full_logits: bool = False,
) -> dict[str, Any]:
    tf = require_tensorflow_gpu()
    del tf

    plan_dir = Path(plan_dir).expanduser().resolve()
    plan = load_json(plan_dir / "plan.json")
    raw_data_dir = Path(plan["raw_data_dir"])
    inputs_dir = Path(plan["inputs_dir"])
    outputs_dir = Path(plan["outputs_dir"])
    num_shards = int(plan["num_shards"])
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    shard_name = f"shard_{shard_index:05d}_of_{num_shards:05d}"
    input_path = inputs_dir / f"{shard_name}.csv"
    output_dir = outputs_dir / shard_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    logits_path = output_dir / "logits.npy"
    full_logits_path = output_dir / "full_logits.npy"
    summary_path = output_dir / "summary.json"
    if (
        not overwrite
        and manifest_path.exists()
        and logits_path.exists()
        and summary_path.exists()
        and (not save_full_logits or full_logits_path.exists())
    ):
        return load_json(summary_path)

    shard_df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    target_mapping = pd.read_csv(plan["target_species_mapping_path"])
    target_label_indices = target_mapping["speciesnet_label_index"].to_numpy(
        dtype=np.int64
    )
    labels = load_speciesnet_labels(Path(plan["labels_path"]))

    n_rows = len(shard_df)
    n_targets = len(target_label_indices)
    target_logits = np.lib.format.open_memmap(
        logits_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, n_targets),
    )
    target_logits[:] = np.nan
    full_logits = None
    if save_full_logits:
        full_logits = np.lib.format.open_memmap(
            full_logits_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, len(labels)),
        )
        full_logits[:] = np.nan

    status = np.full(n_rows, "pending", dtype=object)
    error_messages = np.full(n_rows, "", dtype=object)

    model = load_keras_classifier(Path(plan["classifier_path"]))
    started = time.time()
    progress = tqdm(
        total=n_rows,
        desc=f"SpeciesNet shard {shard_index}/{num_shards}",
        unit="img",
    )
    try:
        for batch in iter_preprocessed_batches(
            shard_df=shard_df,
            raw_data_dir=raw_data_dir,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
        ):
            for row_idx, message in batch["failures"]:
                status[int(row_idx)] = "error"
                error_messages[int(row_idx)] = message
            if batch["failures"]:
                progress.update(len(batch["failures"]))

            images = batch["images"]
            if images is None:
                continue
            logits = predict_logits(model, images)
            if logits.shape[1] != len(labels):
                raise ValueError(
                    f"SpeciesNet logits have {logits.shape[1]} classes, but "
                    f"labels.txt has {len(labels)} labels."
                )
            row_indices = batch["row_indices"]
            target_logits[row_indices] = logits[:, target_label_indices]
            if full_logits is not None:
                full_logits[row_indices] = logits
            status[row_indices] = "ok"
            progress.update(len(row_indices))
    finally:
        progress.close()

    del target_logits
    if full_logits is not None:
        del full_logits
    duration_sec = time.time() - started

    manifest = shard_df.copy()
    manifest["row_idx"] = np.arange(n_rows, dtype=np.int64)
    manifest["status"] = status
    manifest["error"] = error_messages
    manifest.to_csv(manifest_path, index=False)

    n_ok = int((status == "ok").sum())
    n_error = int((status == "error").sum())
    summary = {
        "plan_dir": str(plan_dir),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "logits_path": str(logits_path),
        "full_logits_path": str(full_logits_path) if save_full_logits else None,
        "device": "GPU",
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "n_rows": int(n_rows),
        "n_ok": n_ok,
        "n_error": n_error,
        "n_target_species": int(n_targets),
        "n_model_labels": int(len(labels)),
        "duration_sec": float(duration_sec),
        "images_per_sec": 0.0 if duration_sec <= 0 else float(n_ok / duration_sec),
    }
    write_json(summary_path, summary)
    return summary


def coerce_env_value(value: str) -> Any:
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text


def load_submitit_parameters(env_file: Path) -> tuple[bool, dict[str, Any]]:
    dotenv_config = {}
    if env_file.exists():
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

    debug = str(merged.pop("SUBMITIT_DEBUG", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    parameters: dict[str, Any] = {}
    for key, value in merged.items():
        if key.startswith("SUBMITIT_SLURM_"):
            parameter_name = "slurm_" + key[len("SUBMITIT_SLURM_") :].lower()
        else:
            parameter_name = key[len("SUBMITIT_") :].lower()
        parameters[parameter_name] = coerce_env_value(str(value))
    return debug, parameters


def unset_slurm_environment() -> None:
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            os.environ.pop(key, None)


def _run_shard_job(
    plan_dir: str,
    shard_index: int,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    save_full_logits: bool,
) -> dict[str, Any]:
    return run_shard(
        plan_dir=Path(plan_dir),
        shard_index=int(shard_index),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        overwrite=bool(overwrite),
        save_full_logits=bool(save_full_logits),
    )


def submit_jobs(args: SimpleNamespace) -> dict[str, Any]:
    unset_slurm_environment()
    plan = prepare_plan(
        raw_data_dir=args.raw_data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        num_shards=int(args.num_jobs),
        model_handle=args.model_handle,
        limit=args.limit,
        overwrite_plan=bool(args.overwrite_plan),
    )
    debug, parameters = load_submitit_parameters(args.env_file)
    parameters.setdefault("name", args.job_name)
    parameters.setdefault("cpus_per_task", int(args.cpus_per_task))
    parameters.setdefault("timeout_min", int(args.timeout_min))
    if args.gpus_per_job > 0:
        additional = dict(parameters.get("slurm_additional_parameters", {}) or {})
        additional.setdefault("gpus", int(args.gpus_per_job))
        parameters["slurm_additional_parameters"] = additional

    submitit_dir = args.output_dir / "submitit"
    submitit_dir.mkdir(parents=True, exist_ok=True)
    executor: submitit.Executor
    if debug:
        executor = submitit.DebugExecutor(folder=submitit_dir)
    else:
        executor = submitit.AutoExecutor(folder=submitit_dir)
    executor.update_parameters(**parameters)

    shard_indices = list(range(int(plan["num_shards"])))
    jobs = executor.map_array(
        _run_shard_job,
        [plan["output_dir"]] * len(shard_indices),
        shard_indices,
        [int(args.batch_size)] * len(shard_indices),
        [int(args.num_workers)] * len(shard_indices),
        [bool(args.overwrite_outputs)] * len(shard_indices),
        [bool(args.save_full_logits)] * len(shard_indices),
    )
    summary = {
        "plan_dir": plan["output_dir"],
        "num_jobs": int(args.num_jobs),
        "job_ids": [job.job_id for job in jobs],
        "submitit_dir": str(submitit_dir),
        "executor_parameters": parameters,
    }
    write_json(args.output_dir / "submitit_jobs.json", summary)

    if args.wait:
        completed = 0
        with tqdm(total=len(jobs), desc="Completed jobs", unit="job") as progress:
            while completed < len(jobs):
                done = [job for job in jobs if job.done()]
                if len(done) > completed:
                    progress.update(len(done) - completed)
                    completed = len(done)
                if completed < len(jobs):
                    time.sleep(float(args.poll_interval_sec))
        summary["states"] = [job.state for job in jobs]
        write_json(args.output_dir / "submitit_jobs.json", summary)

    return summary


def prepare(
    num_jobs: int,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_handle: str = DEFAULT_MODEL_HANDLE,
    limit: int | None = None,
    overwrite_plan: bool = False,
) -> dict[str, Any]:
    return prepare_plan(
        raw_data_dir=Path(raw_data_dir).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        num_shards=int(num_jobs),
        model_handle=model_handle,
        limit=limit,
        overwrite_plan=bool(overwrite_plan),
    )


def submit(
    num_jobs: int,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_handle: str = DEFAULT_MODEL_HANDLE,
    limit: int | None = None,
    overwrite_plan: bool = False,
    batch_size: int = 128,
    num_workers: int = 8,
    gpus_per_job: int = 1,
    cpus_per_task: int = 16,
    timeout_min: int = 1440,
    env_file: str | Path = PACKAGE_DIR / ".env",
    job_name: str = "iwildcam-speciesnet",
    overwrite_outputs: bool = False,
    save_full_logits: bool = False,
    wait: bool = False,
    poll_interval_sec: float = 30.0,
) -> dict[str, Any]:
    args = SimpleNamespace(
        raw_data_dir=Path(raw_data_dir).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        model_handle=model_handle,
        limit=limit,
        overwrite_plan=bool(overwrite_plan),
        num_jobs=int(num_jobs),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        gpus_per_job=int(gpus_per_job),
        cpus_per_task=int(cpus_per_task),
        timeout_min=int(timeout_min),
        env_file=Path(env_file).expanduser(),
        job_name=job_name,
        overwrite_outputs=bool(overwrite_outputs),
        save_full_logits=bool(save_full_logits),
        wait=bool(wait),
        poll_interval_sec=float(poll_interval_sec),
    )
    return submit_jobs(args)
