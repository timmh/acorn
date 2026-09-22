import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

SUMMARY_STAT_NAMES = ("mean", "std", "median", "q05", "q95", "n_eff", "r_hat")
PARAMETER_SUMMARY_NPZ_NAME = "parameter_summary.npz"
PARAMETER_SUMMARY_CSV_NAME = "parameter_summary.csv"


def parameter_summary_npz_path(path: str | Path) -> Path:
    """Return the `.npz` summary path corresponding to `path`."""
    path = Path(path)
    if path.suffix == ".npz":
        return path
    if path.suffix == ".csv":
        return path.with_suffix(".npz")
    return path / PARAMETER_SUMMARY_NPZ_NAME


def parameter_summary_csv_path(path: str | Path) -> Path:
    """Return the legacy `.csv` summary path corresponding to `path`."""
    path = Path(path)
    if path.suffix == ".csv":
        return path
    if path.suffix == ".npz":
        return path.with_suffix(".csv")
    return path / PARAMETER_SUMMARY_CSV_NAME


def resolve_parameter_summary_path(path: str | Path) -> Path | None:
    """Resolve a parameter summary, preferring compact `.npz` over legacy CSV."""
    npz_path = parameter_summary_npz_path(path)
    if npz_path.exists():
        return npz_path

    csv_path = parameter_summary_csv_path(path)
    if csv_path.exists():
        return csv_path

    return None


def parameter_summary_exists(path: str | Path) -> bool:
    """Return whether either the compact or legacy parameter summary exists."""
    return resolve_parameter_summary_path(path) is not None


def _parameter_stat_key(parameter_index: int, stat_name: str) -> str:
    return f"parameter_{parameter_index:04d}__{stat_name}"


def _parse_parameter_index(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _format_parameter_index(index: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in index)


def _parameter_matches(
    parameter_name: str,
    *,
    parameters: set[str] | None = None,
    parameter_prefixes: tuple[str, ...] = (),
) -> bool:
    if parameters is not None and parameter_name in parameters:
        return True
    if parameter_prefixes and parameter_name.startswith(parameter_prefixes):
        return True
    return parameters is None and not parameter_prefixes


def write_parameter_summary_npz(
    path: str | Path,
    rows: Iterable[dict[str, object]],
    *,
    stat_names: Iterable[str] = SUMMARY_STAT_NAMES,
) -> Path:
    """Write flattened summary rows to a compact `.npz` artifact.

    Each parameter is stored as shaped arrays per statistic instead of one row
    per element. Readers can still reconstruct the legacy tabular view when
    needed.
    """
    path = parameter_summary_npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stat_names = tuple(stat_names)

    parameter_names: list[str] = []
    grouped_rows: list[list[tuple[tuple[int, ...], dict[str, object]]]] = []
    parameter_positions: dict[str, int] = {}

    for row in rows:
        parameter_name = str(row.get("parameter", "")).strip()
        if not parameter_name:
            continue
        parameter_position = parameter_positions.get(parameter_name)
        if parameter_position is None:
            parameter_position = len(parameter_names)
            parameter_positions[parameter_name] = parameter_position
            parameter_names.append(parameter_name)
            grouped_rows.append([])
        grouped_rows[parameter_position].append(
            (_parse_parameter_index(row.get("parameter_index", "")), row)
        )

    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int16),
        "parameter_names": np.asarray(parameter_names, dtype=str),
        "stat_names": np.asarray(stat_names, dtype=str),
    }

    for parameter_position, parameter_rows in enumerate(grouped_rows):
        indexed_rows = [(index, row) for index, row in parameter_rows if index]
        if indexed_rows:
            ndim = len(indexed_rows[0][0])
            shape = tuple(
                max(index[axis] for index, _row in indexed_rows) + 1
                for axis in range(ndim)
            )
        else:
            shape = tuple()

        for stat_name in stat_names:
            values = np.full(shape, np.nan, dtype=float)
            for index, row in parameter_rows:
                raw_value = row.get(stat_name, np.nan)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    value = float("nan")
                if shape:
                    values[index] = value
                else:
                    values[...] = value
            arrays[_parameter_stat_key(parameter_position, stat_name)] = values

    np.savez_compressed(path, **arrays)
    return path


def iter_parameter_summary_arrays(
    path: str | Path,
    *,
    parameters: Iterable[str] | None = None,
    parameter_prefixes: Iterable[str] = (),
    stats: Iterable[str] = ("mean", "q05", "q95"),
):
    """Yield `(parameter_name, {stat_name: array})` from `.npz` or legacy CSV."""
    resolved_path = resolve_parameter_summary_path(path)
    if resolved_path is None:
        return

    parameters_set = None if parameters is None else {str(name) for name in parameters}
    prefixes = tuple(str(prefix) for prefix in parameter_prefixes)
    stats = tuple(stats)

    if resolved_path.suffix == ".npz":
        with np.load(resolved_path, allow_pickle=False) as data:
            parameter_names = [str(name) for name in data["parameter_names"]]
            available_keys = set(data.files)
            for parameter_position, parameter_name in enumerate(parameter_names):
                if not _parameter_matches(
                    parameter_name,
                    parameters=parameters_set,
                    parameter_prefixes=prefixes,
                ):
                    continue
                parameter_stats = {}
                for stat_name in stats:
                    key = _parameter_stat_key(parameter_position, stat_name)
                    if key in available_keys:
                        parameter_stats[stat_name] = np.asarray(
                            data[key],
                            dtype=float,
                        )
                if parameter_stats:
                    yield parameter_name, parameter_stats
        return

    yield from _iter_legacy_parameter_summary_arrays(
        resolved_path,
        parameters=parameters_set,
        parameter_prefixes=prefixes,
        stats=stats,
    )


def _iter_legacy_parameter_summary_arrays(
    path: Path,
    *,
    parameters: set[str] | None,
    parameter_prefixes: tuple[str, ...],
    stats: tuple[str, ...],
):
    grouped_rows: dict[str, list[tuple[tuple[int, ...], dict[str, str]]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parameter_name = str(row.get("parameter", "")).strip()
            if not parameter_name:
                continue
            if not _parameter_matches(
                parameter_name,
                parameters=parameters,
                parameter_prefixes=parameter_prefixes,
            ):
                continue
            grouped_rows.setdefault(parameter_name, []).append(
                (_parse_parameter_index(row.get("parameter_index", "")), row)
            )

    for parameter_name, parameter_rows in grouped_rows.items():
        indexed_rows = [(index, row) for index, row in parameter_rows if index]
        if indexed_rows:
            ndim = len(indexed_rows[0][0])
            shape = tuple(
                max(index[axis] for index, _row in indexed_rows) + 1
                for axis in range(ndim)
            )
        else:
            shape = tuple()

        parameter_stats = {}
        for stat_name in stats:
            values = np.full(shape, np.nan, dtype=float)
            saw_value = False
            for index, row in parameter_rows:
                raw_value = str(row.get(stat_name, "")).strip()
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if shape:
                    values[index] = value
                else:
                    values[...] = value
                saw_value = True
            if saw_value:
                parameter_stats[stat_name] = values
        if parameter_stats:
            yield parameter_name, parameter_stats


def load_parameter_summary_frame(
    path: str | Path,
    *,
    parameters: Iterable[str] | None = None,
    parameter_prefixes: Iterable[str] = (),
    stats: Iterable[str] = ("mean", "q05", "q95"),
):
    """Load selected parameter summary rows as a DataFrame."""
    import pandas as pd  # Imported lazily to keep the writer lightweight.

    stats = tuple(stats)
    records: list[dict[str, object]] = []
    for parameter_name, parameter_stats in iter_parameter_summary_arrays(
        path,
        parameters=parameters,
        parameter_prefixes=parameter_prefixes,
        stats=stats,
    ):
        shape = next(iter(parameter_stats.values())).shape
        indices = [tuple()] if shape == tuple() else np.ndindex(shape)
        for index in indices:
            record: dict[str, object] = {
                "parameter": parameter_name,
                "parameter_index": _format_parameter_index(index),
            }
            for stat_name in stats:
                values = parameter_stats.get(stat_name)
                if values is None:
                    record[stat_name] = np.nan
                elif values.shape == tuple():
                    record[stat_name] = float(values)
                else:
                    record[stat_name] = float(values[index])
            records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=["parameter", "parameter_index", *stats],
    )


def flatten_parameter_summary(
    path: str | Path,
    prefix: str,
    *,
    stats: Iterable[str] = ("mean", "q05", "q95"),
    slugify_func: Callable[[object], str],
) -> dict[str, float]:
    """Flatten a parameter summary into one aggregate value per parameter/stat."""
    resolved_path = resolve_parameter_summary_path(path)
    if resolved_path is None:
        return {}

    stats = tuple(stats)
    if resolved_path.suffix == ".csv":
        return _flatten_legacy_parameter_summary(
            resolved_path,
            prefix,
            stats=stats,
            slugify_func=slugify_func,
        )

    flattened: dict[str, float] = {}
    for parameter_name, parameter_stats in iter_parameter_summary_arrays(
        resolved_path,
        stats=stats,
    ):
        parameter_key = slugify_func(parameter_name)
        for stat_name in stats:
            values = parameter_stats.get(stat_name)
            if values is None:
                continue
            finite_values = np.asarray(values, dtype=float)
            if finite_values.size == 0 or not np.isfinite(finite_values).any():
                continue
            flattened[f"{prefix}{stat_name}__{parameter_key}"] = float(
                np.nanmean(finite_values)
            )
    return flattened


def _flatten_legacy_parameter_summary(
    path: Path,
    prefix: str,
    *,
    stats: tuple[str, ...],
    slugify_func: Callable[[object], str],
) -> dict[str, float]:
    grouped_statistics: dict[str, dict[str, tuple[float, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        required_fields = {"parameter", *stats}
        if not required_fields.issubset(set(reader.fieldnames)):
            return {}

        for row in reader:
            parameter_name = str(row.get("parameter", "")).strip()
            if not parameter_name:
                continue
            parameter_key = slugify_func(parameter_name)
            parameter_statistics = grouped_statistics.setdefault(parameter_key, {})
            for stat_name in stats:
                raw_value = str(row.get(stat_name, "")).strip()
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if np.isnan(value):
                    continue
                statistic_sum, statistic_count = parameter_statistics.get(
                    stat_name,
                    (0.0, 0),
                )
                parameter_statistics[stat_name] = (
                    statistic_sum + value,
                    statistic_count + 1,
                )

    flattened: dict[str, float] = {}
    for parameter_key, parameter_statistics in grouped_statistics.items():
        for stat_name, (statistic_sum, statistic_count) in parameter_statistics.items():
            if statistic_count:
                flattened[f"{prefix}{stat_name}__{parameter_key}"] = (
                    statistic_sum / statistic_count
                )
    return flattened


def mean_parameter_stat(
    path: str | Path,
    parameter_name: str,
    stat_name: str,
) -> float:
    """Return the mean of one statistic for one parameter."""
    for _parameter_name, parameter_stats in iter_parameter_summary_arrays(
        path,
        parameters=[parameter_name],
        stats=[stat_name],
    ):
        values = parameter_stats.get(stat_name)
        if values is None:
            return float("nan")
        finite_values = np.asarray(values, dtype=float)
        if finite_values.size == 0 or not np.isfinite(finite_values).any():
            return float("nan")
        return float(np.nanmean(finite_values))
    return float("nan")
