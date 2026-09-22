import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import ee
import numpy as np
import pandas as pd
from joblib import Memory

from occubed.paths import CACHE_DIR, DATA_DIR

CACHE_ROOT_DIR = CACHE_DIR
GEE_CACHE_DIR = CACHE_ROOT_DIR / "gee_covariates"
ACOUSTIC_FOREST_SOUNDSCAPE_DIR = DATA_DIR / "acoustic_forest_soundscape"
IWILDCAM2022_DIR = DATA_DIR / "iwildcam2022"
GEE_MEMORY = Memory(GEE_CACHE_DIR, verbose=0)
GEE_PROJECT = "zorrilla"
LOCAL_BUFFER_METERS = 500
LAND_COVER_YEAR = 2020
POPULATION_YEAR = 2020
GEE_CACHE_VERSION = 2
ACOUSTIC_SITE_COVARIATE_NAMES = (
    "mean_tree_canopy_cover_pct",
    "median_canopy_height_m",
    "mean_elevation_m",
)
ACOUSTIC_SUBCOLLECTION_LEVELS = ("ACAD", "MABI", "SIMR")
ACOUSTIC_ANNOTATION_DIRS = {
    "ACAD": "DatasetACAD",
    "MABI": "DatasetMABI",
    "SIMR": "DatasetSIMR",
}
ACOUSTIC_TIMEZONE = "America/New_York"
IWILDCAM_SITE_COVARIATE_NAMES = (
    "mean_elevation_m",
    "prop_forest",
    "mean_human_population_density_per_km2",
    "annual_mean_temperature_c",
    "annual_mean_precipitation_mm",
)
IWILDCAM_OBS_COVARIATE_NAMES = ("log_daily_effort",)
IWILDCAM_GEO_CLUSTER_RADIUS_KM = 250.0
EARTH_RADIUS_KM = 6371.0


def _initialize_earth_engine(project=GEE_PROJECT):
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # pragma: no cover - depends on local credentials
        raise RuntimeError(
            "Google Earth Engine initialization failed for project "
            f"`{project}`. Ensure `earthengine-api` is installed and the "
            "local credentials are configured for this project."
        ) from exc


def _normalize_covariates(covariates, covariate_names, covariate_kinds):
    covariates = np.asarray(covariates, dtype=float)
    if covariates.size == 0:
        empty = covariates.reshape(covariates.shape[0], 0)
        return empty, np.zeros((0,), dtype=float), np.ones((0,), dtype=float)
    if covariates.shape[1] != len(covariate_names):
        raise ValueError("Covariate names must match the number of columns.")
    if covariates.shape[1] != len(covariate_kinds):
        raise ValueError("Covariate kinds must match the number of columns.")

    normalized = covariates.copy()
    offsets = np.zeros(covariates.shape[1], dtype=float)
    scales = np.ones(covariates.shape[1], dtype=float)

    for column_index, (covariate_name, covariate_kind) in enumerate(
        zip(covariate_names, covariate_kinds)
    ):
        column = covariates[:, column_index]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            raise ValueError(f"Covariate `{covariate_name}` is missing for all rows.")

        if covariate_kind == "binary":
            if not np.isin(column[finite_mask], [0.0, 1.0]).all():
                raise ValueError(
                    f"Binary covariate `{covariate_name}` must be encoded as 0/1."
                )
            continue

        if covariate_kind != "continuous":
            raise ValueError(
                f"Unsupported covariate kind `{covariate_kind}` for `{covariate_name}`."
            )

        mean = float(np.nanmean(column))
        std = float(np.nanstd(column))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        normalized[:, column_index] = (column - mean) / std
        offsets[column_index] = mean
        scales[column_index] = std

    return normalized, offsets, scales


def _impute_missing_covariates(frame, covariate_names, covariate_kinds):
    frame = frame.copy()
    for covariate_name, covariate_kind in zip(covariate_names, covariate_kinds):
        values = pd.to_numeric(frame[covariate_name], errors="coerce")
        if covariate_kind == "continuous":
            fill_value = float(values.median()) if values.notna().any() else 0.0
        elif covariate_kind == "binary":
            fill_value = (
                float(values.mode(dropna=True).iloc[0]) if values.notna().any() else 0.0
            )
        else:
            raise ValueError(f"Unsupported covariate kind `{covariate_kind}`.")
        frame[covariate_name] = values.fillna(fill_value)
    return frame


def _gee_feature_collection(site_records):
    return ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point([longitude, latitude]), {"site_id": site_id})
            for site_id, latitude, longitude in site_records
        ]
    )


def _buffer_site_feature(feature, buffer_meters=LOCAL_BUFFER_METERS):
    feature = ee.Feature(feature)
    return ee.Feature(feature.geometry().buffer(buffer_meters)).copyProperties(feature)


def _reduce_regions_to_frame(image, regions, scale, reducer=None):
    reduced = image.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.mean() if reducer is None else reducer,
        scale=scale,
    ).getInfo()
    records = [feature["properties"] for feature in reduced["features"]]
    return pd.DataFrame.from_records(records)


@GEE_MEMORY.cache
def _fetch_iwildcam_site_covariates_cached(cache_version, site_records):
    del cache_version  # cache invalidation marker
    _initialize_earth_engine()
    regions = _gee_feature_collection(site_records).map(_buffer_site_feature)

    land_cover = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate(f"{LAND_COVER_YEAR}-01-01", f"{LAND_COVER_YEAR + 1}-01-01")
        .first()
        .select("LC_Type1")
    )
    forest_cover = land_cover.remap(
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        0,
    ).rename("prop_forest")
    climate_and_population_covariates = ee.Image.cat(
        [
            ee.Image("WORLDCLIM/V1/BIO")
            .select("bio01")
            .multiply(0.1)
            .rename("annual_mean_temperature_c"),
            ee.Image("WORLDCLIM/V1/BIO")
            .select("bio12")
            .rename("annual_mean_precipitation_mm"),
            ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
            .filterDate(f"{POPULATION_YEAR}-01-01", f"{POPULATION_YEAR + 1}-01-01")
            .first()
            .select("population_density")
            .rename("mean_human_population_density_per_km2"),
        ]
    )
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")

    forest_cover_frame = _reduce_regions_to_frame(
        forest_cover,
        regions,
        scale=500,
    ).rename(columns={"mean": "prop_forest"})
    climate_and_population_frame = _reduce_regions_to_frame(
        climate_and_population_covariates,
        regions,
        scale=1_000,
    )
    elevation_frame = _reduce_regions_to_frame(
        elevation,
        regions,
        scale=30,
    ).rename(columns={"mean": "mean_elevation_m"})
    covariates = forest_cover_frame.merge(
        climate_and_population_frame,
        on="site_id",
        how="outer",
    ).merge(
        elevation_frame,
        on="site_id",
        how="outer",
    )
    missing_columns = sorted(
        set(IWILDCAM_SITE_COVARIATE_NAMES) - set(covariates.columns)
    )
    if missing_columns:
        raise ValueError(
            "Earth Engine did not return the requested iWildCam covariates: "
            + ", ".join(missing_columns)
        )
    for covariate_name in IWILDCAM_SITE_COVARIATE_NAMES:
        covariates[covariate_name] = pd.to_numeric(
            covariates[covariate_name],
            errors="coerce",
        )
    return covariates[["site_id", *IWILDCAM_SITE_COVARIATE_NAMES]]


def _attach_iwildcam_gee_site_covariates(sites):
    site_records = tuple(
        (str(site_id), float(latitude), float(longitude))
        for site_id, latitude, longitude in sites[
            ["site_id", "latitude", "longitude"]
        ].itertuples(index=False, name=None)
    )
    gee_covariates = _fetch_iwildcam_site_covariates_cached(
        GEE_CACHE_VERSION,
        site_records,
    )
    return sites.merge(gee_covariates, on="site_id", how="left", sort=False)


def _mask_gedi_quality(image):
    return image.updateMask(image.select("quality_flag").eq(1)).updateMask(
        image.select("degrade_flag").eq(0)
    )


@GEE_MEMORY.cache
def _fetch_acoustic_site_covariates_cached(cache_version, site_records):
    del cache_version  # cache invalidation marker
    _initialize_earth_engine()

    regions_150m = _gee_feature_collection(site_records).map(
        lambda feature: _buffer_site_feature(feature, buffer_meters=150)
    )
    regions_250m = _gee_feature_collection(site_records).map(
        lambda feature: _buffer_site_feature(feature, buffer_meters=250)
    )

    tree_canopy_cover = (
        ee.ImageCollection("MODIS/061/MOD44B")
        .filterDate("2022-01-01", "2024-01-01")
        .select("Percent_Tree_Cover")
        .median()
    )
    canopy_height = (
        ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        .filterDate("2022-01-01", "2024-01-01")
        .map(_mask_gedi_quality)
        .select("rh98")
        .median()
    )
    elevation = (
        ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
    )

    tree_canopy_cover_frame = _reduce_regions_to_frame(
        tree_canopy_cover,
        regions_150m,
        scale=250,
    ).rename(columns={"mean": "mean_tree_canopy_cover_pct"})
    canopy_height_frame = _reduce_regions_to_frame(
        canopy_height,
        regions_150m,
        scale=25,
        reducer=ee.Reducer.median(),
    ).rename(columns={"median": "median_canopy_height_m"})
    elevation_frame = _reduce_regions_to_frame(
        elevation,
        regions_250m,
        scale=10,
    ).rename(columns={"mean": "mean_elevation_m"})

    covariates = tree_canopy_cover_frame.merge(
        canopy_height_frame,
        on="site_id",
        how="outer",
    ).merge(
        elevation_frame,
        on="site_id",
        how="outer",
    )
    for covariate_name in ACOUSTIC_SITE_COVARIATE_NAMES:
        covariates[covariate_name] = pd.to_numeric(
            covariates[covariate_name], errors="coerce"
        )
    return covariates[["site_id", *ACOUSTIC_SITE_COVARIATE_NAMES]]


def _attach_acoustic_gee_site_covariates(sites):
    site_records = tuple(
        (str(site_id), float(latitude), float(longitude))
        for site_id, latitude, longitude in sites[
            ["site_id", "latitude", "longitude"]
        ].itertuples(index=False, name=None)
    )
    gee_covariates = _fetch_acoustic_site_covariates_cached(
        GEE_CACHE_VERSION,
        site_records,
    )
    return sites.merge(gee_covariates, on="site_id", how="left", sort=False)


def _filter_sites_with_complete_covariates(
    observations, sites, site_covariate_names, obs_covariate_names
):
    site_required_columns = [
        "latitude",
        "longitude",
        *site_covariate_names,
    ]
    missing_site_columns = sorted(set(site_required_columns) - set(sites.columns))
    if missing_site_columns:
        raise ValueError(
            "Site table is missing required covariates: "
            + ", ".join(missing_site_columns)
        )

    valid_sites = np.isfinite(sites[site_required_columns].to_numpy(dtype=float)).all(
        axis=1
    )
    filtered_sites = sites.loc[valid_sites].copy()
    filtered_observations = observations[
        observations["site_id"].isin(filtered_sites["site_id"])
    ].copy()

    obs_covariate_names = tuple(obs_covariate_names)
    if obs_covariate_names:
        missing_obs_columns = sorted(
            set(obs_covariate_names) - set(filtered_observations.columns)
        )
        if missing_obs_columns:
            raise ValueError(
                "Observation table is missing required covariates: "
                + ", ".join(missing_obs_columns)
            )
        valid_observations = np.isfinite(
            filtered_observations[list(obs_covariate_names)].to_numpy(dtype=float)
        ).all(axis=1)
        filtered_observations = filtered_observations.loc[valid_observations].copy()
        filtered_sites = filtered_sites[
            filtered_sites["site_id"].isin(filtered_observations["site_id"])
        ].copy()

    if filtered_sites.empty or filtered_observations.empty:
        raise ValueError("No sites remain after filtering missing covariates.")

    ordered_site_ids = filtered_sites["site_id"].tolist()
    filtered_sites["site_id"] = pd.Categorical(
        filtered_sites["site_id"],
        categories=ordered_site_ids,
        ordered=True,
    )
    filtered_sites = filtered_sites.sort_values("site_id").reset_index(drop=True)
    filtered_observations["site_id"] = pd.Categorical(
        filtered_observations["site_id"],
        categories=filtered_sites["site_id"].astype(str).tolist(),
        ordered=True,
    )
    filtered_observations = filtered_observations.sort_values(
        ["site_id", "replicate_time"]
    ).reset_index(drop=True)
    return filtered_observations, filtered_sites


def _sunrise_time_local(observation_timestamp, latitude, longitude):
    if (
        pd.isna(observation_timestamp)
        or not np.isfinite(latitude)
        or not np.isfinite(longitude)
    ):
        return None

    if observation_timestamp.tzinfo is None:
        observation_timestamp = observation_timestamp.tz_localize(ACOUSTIC_TIMEZONE)
    else:
        observation_timestamp = observation_timestamp.tz_convert(ACOUSTIC_TIMEZONE)

    observation_date = observation_timestamp.date()
    day_of_year = observation_date.timetuple().tm_yday
    longitude_hours = longitude / 15.0
    approximate_time = day_of_year + ((6.0 - longitude_hours) / 24.0)
    solar_mean_anomaly = (0.9856 * approximate_time) - 3.289
    sun_true_longitude = (
        solar_mean_anomaly
        + 1.916 * math.sin(math.radians(solar_mean_anomaly))
        + 0.020 * math.sin(math.radians(2.0 * solar_mean_anomaly))
        + 282.634
    ) % 360.0
    right_ascension = (
        math.degrees(math.atan(0.91764 * math.tan(math.radians(sun_true_longitude))))
        % 360.0
    )
    true_longitude_quadrant = math.floor(sun_true_longitude / 90.0) * 90.0
    right_ascension_quadrant = math.floor(right_ascension / 90.0) * 90.0
    right_ascension = (
        right_ascension + true_longitude_quadrant - right_ascension_quadrant
    ) / 15.0

    sun_declination = math.asin(0.39782 * math.sin(math.radians(sun_true_longitude)))
    cosine_hour_angle = (
        math.cos(math.radians(90.833))
        - math.sin(sun_declination) * math.sin(math.radians(latitude))
    ) / (math.cos(sun_declination) * math.cos(math.radians(latitude)))
    if cosine_hour_angle < -1.0 or cosine_hour_angle > 1.0:
        return None

    local_hour_angle = (360.0 - math.degrees(math.acos(cosine_hour_angle))) / 15.0
    local_mean_time = (
        local_hour_angle + right_ascension - (0.06571 * approximate_time) - 6.622
    )
    universal_time = (local_mean_time - longitude_hours) % 24.0
    sunrise_utc = datetime.combine(
        observation_date,
        time.min,
        tzinfo=timezone.utc,
    ) + timedelta(hours=universal_time)
    return sunrise_utc.astimezone(ZoneInfo(ACOUSTIC_TIMEZONE))


def _build_acoustic_detection_covariates(observations):
    observations = observations.copy()
    observations["day_of_year"] = observations["replicate_time"].dt.dayofyear.astype(
        float
    )

    sunrise_times = [
        _sunrise_time_local(replicate_time, latitude, longitude)
        for replicate_time, latitude, longitude in zip(
            observations["replicate_time"],
            observations["latitude"],
            observations["longitude"],
        )
    ]
    observations["time_since_sunrise_hours"] = [
        (
            (
                (
                    replicate_time.tz_localize(ACOUSTIC_TIMEZONE) - sunrise_time
                ).total_seconds()
                / 3600.0
            )
            if sunrise_time is not None and pd.notna(replicate_time)
            else np.nan
        )
        for replicate_time, sunrise_time in zip(
            observations["replicate_time"],
            sunrise_times,
        )
    ]
    for subcollection_level in ACOUSTIC_SUBCOLLECTION_LEVELS[1:]:
        observations[f"subcollection_{subcollection_level.lower()}"] = (
            observations["SubCollection_ID"] == subcollection_level
        ).astype(float)
    return observations


def _build_acoustic_annotation_filename_lookup(filename_mismatches):
    annotation_filename_lookup = {}
    for expected_name, actual_name in filename_mismatches[
        ["expected_recording_filename", "actual_recording_filename"]
    ].itertuples(index=False, name=None):
        expected_stem = Path(expected_name).stem
        actual_stem = Path(actual_name).stem
        for suffix in (".txt", ".Table.1.selections.txt"):
            annotation_filename_lookup[f"{expected_stem}{suffix}"] = (
                f"{actual_stem}{suffix}"
            )
    return annotation_filename_lookup


def _resolve_acoustic_annotation_path(
    annotation_root,
    subcollection_id,
    annotation_filename,
    annotation_filename_lookup,
):
    annotation_subdir = ACOUSTIC_ANNOTATION_DIRS.get(subcollection_id)
    if annotation_subdir is None:
        raise ValueError(
            f"No annotation directory configured for `{subcollection_id}`."
        )

    annotation_dir = annotation_root / annotation_subdir
    annotation_path = annotation_dir / annotation_filename
    if annotation_path.exists():
        return annotation_path

    alternate_filename = annotation_filename_lookup.get(annotation_filename)
    if alternate_filename is not None:
        alternate_path = annotation_dir / alternate_filename
        if alternate_path.exists():
            return alternate_path

    raise FileNotFoundError(
        f"Could not find annotation file `{annotation_filename}` for "
        f"subcollection `{subcollection_id}`."
    )


def _load_acoustic_recording_scores(perch_dir, target_species):
    score_column = f"logit_{target_species}"
    available_columns = pd.read_csv(
        perch_dir / "perch_v2_window_logits.csv",
        nrows=0,
    ).columns
    if score_column not in available_columns:
        raise ValueError(
            f"Perch logits do not include `{target_species}`. "
            "Check `dataset_species_perch_mapping.csv` for supported species."
        )

    window_logits = pd.read_csv(
        perch_dir / "perch_v2_window_logits.csv",
        usecols=["Recording_FileName", score_column],
    )
    return (
        window_logits.groupby("Recording_FileName", as_index=False)[score_column]
        .max()
        .rename(columns={score_column: "score"})
    )


def _pack_single_season_dataset(
    observations,
    sites,
    dataset_name,
    target_label,
    site_covariate_names=None,
    site_covariate_kinds=None,
    obs_covariate_names=None,
    obs_covariate_kinds=None,
    obs_covariate_source="site",
):
    """Pack site/replicate tables into the occupancy-model tensor layout.

    The returned tensors mirror the simulated single-species, single-season
    layout:
    ``obs`` and ``f`` are ``(1, n_sites, 1, n_replicates)`` and ``obs_covs`` is
    ``(n_sites, 1, n_replicates, n_obs_covs)``.
    """

    sites = sites.copy()
    observations = observations.copy().sort_values(["site_id", "replicate_time"])
    site_ids = sites["site_id"].tolist()
    max_replicates = int(observations.groupby("site_id").size().max())
    obs = np.full((1, len(site_ids), 1, max_replicates), np.nan, dtype=float)
    verified_f = np.full_like(obs, np.nan)
    replicate_ids = np.full((len(site_ids), max_replicates), "", dtype=object)
    replicate_times = np.full((len(site_ids), max_replicates), "", dtype=object)

    for site_index, site_id in enumerate(site_ids):
        site_observations = observations[observations["site_id"] == site_id]
        n_site_observations = len(site_observations)
        obs[0, site_index, 0, :n_site_observations] = site_observations[
            "score"
        ].to_numpy(dtype=float)
        verified_f[0, site_index, 0, :n_site_observations] = site_observations[
            "label"
        ].to_numpy(dtype=float)
        replicate_ids[site_index, :n_site_observations] = (
            site_observations["replicate_id"].astype(str).to_numpy()
        )
        replicate_times[site_index, :n_site_observations] = (
            site_observations["replicate_time"].astype(str).to_numpy()
        )

    if site_covariate_names is None:
        site_covariate_names = ("latitude", "longitude")
    if site_covariate_kinds is None:
        site_covariate_kinds = ("continuous",) * len(site_covariate_names)
    site_covariate_names = tuple(site_covariate_names)
    site_covariate_kinds = tuple(site_covariate_kinds)
    site_covariates_raw = sites[list(site_covariate_names)].to_numpy(dtype=float)
    site_covs, site_covariate_means, site_covariate_scales = _normalize_covariates(
        site_covariates_raw,
        site_covariate_names,
        site_covariate_kinds,
    )

    if obs_covariate_names is None:
        obs_covariate_names = ()
    if obs_covariate_kinds is None:
        obs_covariate_kinds = ()
    obs_covariate_names = tuple(obs_covariate_names)
    obs_covariate_kinds = tuple(obs_covariate_kinds)
    if len(obs_covariate_names) != len(obs_covariate_kinds):
        raise ValueError("Observation covariate names and kinds must align.")

    n_obs_covs = len(obs_covariate_names)
    if n_obs_covs > 0:
        obs_covs = np.full(
            (len(site_ids), 1, max_replicates, n_obs_covs),
            np.nan,
            dtype=float,
        )
        obs_covariates_raw = np.full_like(obs_covs, np.nan)

        if obs_covariate_source == "site":
            obs_covariates_source_raw = sites[list(obs_covariate_names)].to_numpy(
                dtype=float
            )
            obs_covariates_source, obs_covariate_offsets, obs_covariate_scales = (
                _normalize_covariates(
                    obs_covariates_source_raw,
                    obs_covariate_names,
                    obs_covariate_kinds,
                )
            )
            for site_index, site_id in enumerate(site_ids):
                n_site_observations = int((observations["site_id"] == site_id).sum())
                obs_covs[site_index, 0, :n_site_observations, :] = (
                    obs_covariates_source[site_index]
                )
                obs_covariates_raw[site_index, 0, :n_site_observations, :] = (
                    obs_covariates_source_raw[site_index]
                )
        elif obs_covariate_source == "observations":
            obs_covariates_source_raw = observations[
                list(obs_covariate_names)
            ].to_numpy(dtype=float)
            (
                obs_covariates_source,
                obs_covariate_offsets,
                obs_covariate_scales,
            ) = _normalize_covariates(
                obs_covariates_source_raw,
                obs_covariate_names,
                obs_covariate_kinds,
            )
            normalized_observations = observations.copy()
            normalized_observations[list(obs_covariate_names)] = obs_covariates_source
            for site_index, site_id in enumerate(site_ids):
                site_mask = observations["site_id"] == site_id
                site_observations = observations.loc[
                    site_mask, list(obs_covariate_names)
                ]
                site_observations_normalized = normalized_observations.loc[
                    site_mask, list(obs_covariate_names)
                ]
                n_site_observations = len(site_observations)
                obs_covs[site_index, 0, :n_site_observations, :] = (
                    site_observations_normalized.to_numpy(dtype=float)
                )
                obs_covariates_raw[site_index, 0, :n_site_observations, :] = (
                    site_observations.to_numpy(dtype=float)
                )
        else:
            raise ValueError(
                "obs_covariate_source must be either `site` or `observations`."
            )
    else:
        obs_covs = np.zeros((len(site_ids), 1, max_replicates, 0), dtype=float)
        obs_covariates_raw = np.zeros((0, 0), dtype=float)
        obs_covariate_offsets = np.zeros((0,), dtype=float)
        obs_covariate_scales = np.ones((0,), dtype=float)

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
    ), dict(
        f=verified_f,
        site_ids=np.asarray(site_ids, dtype=object),
        replicate_ids=replicate_ids,
        replicate_times=replicate_times,
        dataset_name=dataset_name,
        target_label=target_label,
        site_coordinates=sites[["latitude", "longitude"]].to_numpy(dtype=float),
        site_covariates_raw=site_covariates_raw,
        site_covariate_names=site_covariate_names,
        site_covariate_means=site_covariate_means,
        site_covariate_scales=site_covariate_scales,
        obs_covariates_raw=(
            obs_covariates_raw.reshape(-1, n_obs_covs)
            if n_obs_covs > 0
            else obs_covariates_raw
        ),
        obs_covariate_names=np.asarray(obs_covariate_names, dtype=object),
        obs_covariate_offsets=obs_covariate_offsets,
        obs_covariate_scales=obs_covariate_scales,
    )


def load_acoustic_perch_dataset(target_species="HETH", subcollection_ids=None):
    """Load the acoustic Perch-v2 benchmark at the recording level.

    Mapping:
    - site: ``SiteID`` from ``recording_metadata.csv``
    - replicate: one 10-minute recording
    - score: maximum target-species Perch v2 logit across a recording's windows
    - label: whether the paired annotation file contains ``target_species``
    """

    base_dir = ACOUSTIC_FOREST_SOUNDSCAPE_DIR
    raw_dir = base_dir / "raw"
    perch_dir = base_dir / "perch_v2"
    annotation_root = base_dir / "annotations"
    recording_metadata = pd.read_csv(raw_dir / "recording_metadata.csv")
    site_metadata = pd.read_csv(raw_dir / "site_metadata.csv")
    filename_mismatches = pd.read_csv(perch_dir / "recording_filename_mismatches.csv")
    annotation_filename_lookup = _build_acoustic_annotation_filename_lookup(
        filename_mismatches
    )
    perch_scores = _load_acoustic_recording_scores(perch_dir, target_species)

    if subcollection_ids is not None:
        observations = recording_metadata[
            recording_metadata["SubCollection_ID"].isin(subcollection_ids)
        ].copy()
    else:
        observations = recording_metadata.copy()

    observations = observations.merge(
        perch_scores,
        on="Recording_FileName",
        how="inner",
    )
    if observations.empty:
        raise ValueError(
            "No scored acoustic recordings matched the requested subcollections."
        )

    labels = []
    for subcollection_id, annotation_file in observations[
        ["SubCollection_ID", "Annotation_FileName"]
    ].itertuples(index=False, name=None):
        annotation_path = _resolve_acoustic_annotation_path(
            annotation_root,
            subcollection_id,
            annotation_file,
            annotation_filename_lookup,
        )
        annotation = pd.read_csv(annotation_path, sep="\t")
        labels.append(float((annotation["Species"] == target_species).any()))
    observations["label"] = labels
    observations["score"] = observations["score"].astype(float)
    observations["site_id"] = observations["SiteID"]
    observations["replicate_time"] = pd.to_datetime(
        observations["Date"] + " " + observations["Time"].astype(str),
        errors="coerce",
    )
    observations["replicate_id"] = observations["Recording_FileName"]
    sites = (
        observations[["site_id"]]
        .drop_duplicates()
        .merge(
            site_metadata.rename(
                columns={
                    "SiteID": "site_id",
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                }
            )[["site_id", "latitude", "longitude"]],
            on="site_id",
            how="left",
        )
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    sites = _attach_acoustic_gee_site_covariates(sites)
    sites = _impute_missing_covariates(
        sites,
        ACOUSTIC_SITE_COVARIATE_NAMES,
        ("continuous",) * len(ACOUSTIC_SITE_COVARIATE_NAMES),
    )

    observations = observations.merge(
        sites[["site_id", "latitude", "longitude"]],
        on="site_id",
        how="left",
    )
    observations = _build_acoustic_detection_covariates(observations)
    acoustic_obs_covariate_names = (
        "time_since_sunrise_hours",
        "day_of_year",
        "subcollection_mabi",
        "subcollection_simr",
    )
    acoustic_obs_covariate_kinds = (
        "continuous",
        "continuous",
        "binary",
        "binary",
    )
    observations = _impute_missing_covariates(
        observations,
        acoustic_obs_covariate_names,
        acoustic_obs_covariate_kinds,
    )

    observed_subcollections = tuple(sorted(observations["SubCollection_ID"].unique()))
    dataset_name = "Acoustic forest soundscape / Perch v2"
    if observed_subcollections != ACOUSTIC_SUBCOLLECTION_LEVELS:
        dataset_name += f" ({', '.join(observed_subcollections)})"
    return _pack_single_season_dataset(
        observations[
            [
                "site_id",
                "replicate_time",
                "replicate_id",
                "score",
                "label",
                *acoustic_obs_covariate_names,
            ]
        ],
        sites,
        dataset_name=dataset_name,
        target_label=target_species,
        site_covariate_names=ACOUSTIC_SITE_COVARIATE_NAMES,
        site_covariate_kinds=("continuous",) * len(ACOUSTIC_SITE_COVARIATE_NAMES),
        obs_covariate_names=acoustic_obs_covariate_names,
        obs_covariate_kinds=acoustic_obs_covariate_kinds,
        obs_covariate_source="observations",
    )


def _normalize_species_name(value):
    return " ".join(str(value).strip().lower().split())


def _find_iwildcam2022_target_mapping(target_species):
    mapping_path = IWILDCAM2022_DIR / "target_species_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing iWildCam SpeciesNet target mapping: {mapping_path}. "
            "Run scripts/preprocess_iwildcam.py first."
        )
    mapping = pd.read_csv(mapping_path)
    normalized_target = _normalize_species_name(target_species)
    candidate_columns = [
        "category_name",
        "speciesnet_common_name",
        "speciesnet_scientific_name",
    ]
    mask = np.zeros(len(mapping), dtype=bool)
    for column in candidate_columns:
        if column in mapping.columns:
            mask |= (
                mapping[column]
                .fillna("")
                .map(_normalize_species_name)
                .eq(normalized_target)
                .to_numpy()
            )
    if not mask.any():
        supported = ", ".join(mapping["category_name"].astype(str).head(20))
        raise ValueError(
            f"iWildCam SpeciesNet outputs do not include `{target_species}`. "
            f"First supported category names: {supported}"
        )
    return mapping.loc[mask].sort_values("annotation_count", ascending=False).iloc[0]


def _load_iwildcam2022_daily_scores(target_species, selected_image_ids):
    target_row = _find_iwildcam2022_target_mapping(target_species)
    target_logit_index = int(target_row["target_logit_index"])
    selected_image_ids = {str(image_id) for image_id in selected_image_ids}
    output_dir = IWILDCAM2022_DIR / "outputs"
    score_rows = []
    for shard_dir in sorted(output_dir.glob("shard_*")):
        manifest_path = shard_dir / "manifest.csv"
        logits_path = shard_dir / "logits.npy"
        if not manifest_path.exists() or not logits_path.exists():
            continue
        manifest = pd.read_csv(
            manifest_path,
            usecols=[
                "split",
                "image_id",
                "location_id",
                "capture_datetime",
                "status",
                "row_idx",
            ],
            dtype={"image_id": str, "location_id": str},
            keep_default_na=False,
            low_memory=False,
        )
        manifest = manifest[
            manifest["split"].eq("train")
            & manifest["image_id"].astype(str).isin(selected_image_ids)
        ].copy()
        if manifest.empty:
            continue
        logits = np.load(logits_path, mmap_mode="r")
        target_scores = np.asarray(
            logits[manifest["row_idx"].to_numpy(dtype=int), target_logit_index],
            dtype=float,
        )
        manifest["score"] = np.where(
            manifest["status"].eq("ok"),
            target_scores,
            np.nan,
        )
        manifest["replicate_time"] = pd.to_datetime(
            manifest["capture_datetime"],
            errors="coerce",
        ).dt.normalize()
        manifest = manifest[manifest["replicate_time"].notna()].copy()
        if manifest.empty:
            continue
        score_rows.append(manifest[["location_id", "replicate_time", "score"]])
    if not score_rows:
        raise ValueError(
            "No iWildCam SpeciesNet outputs matched the selected geocoded days."
        )
    return (
        pd.concat(score_rows, ignore_index=True)
        .groupby(["location_id", "replicate_time"], as_index=False)["score"]
        .max()
    )


def _haversine_distance_km(point_a, point_b):
    lat_a, lon_a = np.radians(np.asarray(point_a, dtype=float))
    lat_b, lon_b = np.radians(np.asarray(point_b, dtype=float))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine)))


def _attach_iwildcam_geo_clusters(sites, radius_km=IWILDCAM_GEO_CLUSTER_RADIUS_KM):
    """Assign deterministic connected-component clusters by geographic radius."""

    radius_km = float(radius_km)
    if radius_km <= 0.0:
        raise ValueError("iWildCam geo_cluster_radius_km must be positive.")

    sites = sites.copy().reset_index(drop=True)
    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")
    finite_mask = np.isfinite(
        sites[["latitude", "longitude"]].to_numpy(dtype=float)
    ).all(axis=1)
    sites = sites.loc[finite_mask].copy().reset_index(drop=True)
    if sites.empty:
        raise ValueError("No iWildCam locations have finite latitude/longitude values.")

    coordinates = sites[["latitude", "longitude"]].to_numpy(dtype=float)
    parent = list(range(len(sites)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(sites)):
        for right in range(left + 1, len(sites)):
            distance_km = _haversine_distance_km(
                coordinates[left],
                coordinates[right],
            )
            if distance_km <= radius_km:
                union(left, right)

    components = {}
    for index in range(len(sites)):
        components.setdefault(find(index), []).append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda indices: (
            -len(indices),
            min(str(value) for value in sites.loc[indices, "site_id"]),
        ),
    )
    cluster_ids = np.full(len(sites), -1, dtype=int)
    for cluster_id, indices in enumerate(ordered_components):
        cluster_ids[indices] = cluster_id
    sites["geo_cluster_id"] = cluster_ids
    return sites


def load_iwildcam2022_speciesnet_dataset(
    target_species="odocoileus virginianus",
    max_sites=None,
    min_positive_days=0,
    min_negative_days=0,
    min_positive_sequences=None,
    min_negative_sequences=None,
    filter_to_positive_geo_clusters=True,
    geo_cluster_radius_km=IWILDCAM_GEO_CLUSTER_RADIUS_KM,
):
    """Load the iWildCam 2022 / SpeciesNet benchmark at the deployment-day level.

    Mapping:
    - site: ``location_id`` from iWildCam metadata
    - replicate: one location-day
    - label: whether any train image that day matches ``target_species``
    - score: maximum target SpeciesNet logit across train images that day
    - detection covariate: log(number of sequences) at the site that day
    - split: train images only, excluding images without finite coordinates
    - geography: keep only spatial location clusters containing target positives
    """

    if min_positive_sequences is not None:
        min_positive_days = min_positive_sequences
    if min_negative_sequences is not None:
        min_negative_days = min_negative_sequences

    metadata_path = IWILDCAM2022_DIR / "metadata" / "images.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing iWildCam image metadata: {metadata_path}. "
            "Run scripts/preprocess_iwildcam.py first."
        )

    target_row = _find_iwildcam2022_target_mapping(target_species)
    target_category_name = str(target_row["category_name"])
    images = pd.read_csv(
        metadata_path,
        usecols=[
            "split",
            "image_id",
            "category_name",
            "location_id",
            "seq_id",
            "capture_datetime",
            "latitude",
            "longitude",
        ],
        dtype={"location_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    train_images = images[images["split"].eq("train")].copy()
    train_images["latitude"] = pd.to_numeric(
        train_images["latitude"],
        errors="coerce",
    )
    train_images["longitude"] = pd.to_numeric(
        train_images["longitude"],
        errors="coerce",
    )
    finite_coordinates = np.isfinite(
        train_images[["latitude", "longitude"]].to_numpy(dtype=float)
    ).all(axis=1)
    train_images = train_images.loc[finite_coordinates].copy()
    if train_images.empty:
        raise ValueError(
            "No iWildCam train images have finite latitude/longitude values."
        )

    observations = train_images
    observations["label"] = (
        observations["category_name"]
        .map(_normalize_species_name)
        .eq(_normalize_species_name(target_category_name))
        .astype(float)
    )
    observations["replicate_time"] = pd.to_datetime(
        observations["capture_datetime"],
        errors="coerce",
    )
    observations = observations[observations["replicate_time"].notna()].copy()
    observations["replicate_time"] = observations["replicate_time"].dt.normalize()
    if observations.empty:
        raise ValueError(
            "No geocoded iWildCam train images have parseable capture datetimes."
        )
    daily_observations = (
        observations.groupby(["location_id", "replicate_time"], as_index=False)
        .agg(
            label=("label", "max"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            sequence_count=("seq_id", "nunique"),
            image_count=("image_id", "count"),
        )
        .copy()
    )
    effort = daily_observations["sequence_count"].to_numpy(dtype=float)
    daily_observations["log_daily_effort"] = np.log(np.maximum(effort, 1.0))
    daily_observations["replicate_id"] = daily_observations[
        "replicate_time"
    ].dt.strftime("%Y-%m-%d")
    daily_observations["site_id"] = daily_observations["location_id"].astype(str)

    sites = (
        daily_observations[["site_id", "latitude", "longitude"]]
        .drop_duplicates(subset=["site_id"])
        .copy()
    )
    sites = _attach_iwildcam_geo_clusters(
        sites,
        radius_km=float(geo_cluster_radius_km),
    )
    daily_observations = daily_observations[
        daily_observations["site_id"].isin(sites["site_id"])
    ].merge(sites[["site_id", "geo_cluster_id"]], on="site_id", how="left")

    if filter_to_positive_geo_clusters:
        positive_geo_cluster_ids = tuple(
            sorted(
                daily_observations.loc[
                    daily_observations["label"] > 0.0,
                    "geo_cluster_id",
                ].unique()
            )
        )
        if not positive_geo_cluster_ids:
            raise ValueError(
                "No geocoded iWildCam days contain observations of "
                f"`{target_category_name}`."
            )
        sites = sites[sites["geo_cluster_id"].isin(positive_geo_cluster_ids)].copy()
        daily_observations = daily_observations[
            daily_observations["geo_cluster_id"].isin(positive_geo_cluster_ids)
        ].copy()

    location_summary = (
        daily_observations.groupby("location_id")["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    location_summary["neg"] = location_summary["count"] - location_summary["sum"]
    selected_locations = location_summary[
        (location_summary["sum"] >= min_positive_days)
        & (location_summary["neg"] >= min_negative_days)
    ].sort_values(["sum", "count"], ascending=[False, False])
    if max_sites is not None:
        selected_locations = selected_locations.sample(
            n=max_sites,
            random_state=0,
        )["location_id"]
    else:
        selected_locations = selected_locations["location_id"]
    selected_locations = selected_locations.astype(str).tolist()
    if not selected_locations:
        raise ValueError(
            "No iWildCam locations satisfy the requested target-species filters."
        )

    selected_image_ids = observations.loc[
        observations["location_id"].astype(str).isin(selected_locations),
        "image_id",
    ].astype(str)
    scores = _load_iwildcam2022_daily_scores(
        target_category_name,
        selected_image_ids,
    )
    observations = daily_observations[
        daily_observations["location_id"].astype(str).isin(selected_locations)
    ].merge(scores, on=["location_id", "replicate_time"], how="left")
    sites = sites[sites["site_id"].isin(selected_locations)].copy()
    sites = _attach_iwildcam_gee_site_covariates(sites)
    sites = _impute_missing_covariates(
        sites,
        IWILDCAM_SITE_COVARIATE_NAMES,
        ("continuous",) * len(IWILDCAM_SITE_COVARIATE_NAMES),
    )
    observations, sites = _filter_sites_with_complete_covariates(
        observations,
        sites,
        IWILDCAM_SITE_COVARIATE_NAMES,
        IWILDCAM_OBS_COVARIATE_NAMES,
    )
    data, metadata = _pack_single_season_dataset(
        observations[
            [
                "site_id",
                "replicate_time",
                "replicate_id",
                "score",
                "label",
                "log_daily_effort",
            ]
        ],
        sites,
        dataset_name="iWildCam 2022 / SpeciesNet v4.0.0b",
        target_label=target_category_name,
        site_covariate_names=IWILDCAM_SITE_COVARIATE_NAMES,
        site_covariate_kinds=("continuous",) * len(IWILDCAM_SITE_COVARIATE_NAMES),
        obs_covariate_names=IWILDCAM_OBS_COVARIATE_NAMES,
        obs_covariate_kinds=("continuous",) * len(IWILDCAM_OBS_COVARIATE_NAMES),
        obs_covariate_source="observations",
    )
    metadata["site_geo_cluster_ids"] = sites["geo_cluster_id"].to_numpy(dtype=int)
    metadata["geo_cluster_radius_km"] = float(geo_cluster_radius_km)
    metadata["retained_geo_cluster_ids"] = np.asarray(
        sorted(sites["geo_cluster_id"].unique()),
        dtype=int,
    )
    metadata["filter_to_positive_geo_clusters"] = bool(filter_to_positive_geo_clusters)
    metadata["replicate_construction"] = "location_day"
    return data, metadata
