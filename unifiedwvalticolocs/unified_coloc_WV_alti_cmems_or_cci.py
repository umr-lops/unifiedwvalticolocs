"""Create NetCDF colocation datasets from altimeters and S1 WV OCN data.

Author: Antoine Grouazel

Colocates Sentinel-1 WV (wave) OCN Level-2 measurements with CMEMS WAVE L3
or CCI SeaState L2P altimeter observations, in space and time.
"""

from typing import Any

import argparse
import copy
import datetime
import glob
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import timezone
from multiprocessing import get_context
from resource import RUSAGE_SELF, getrusage

import numpy as np
import xarray as xr
from dateutil import rrule
from scipy.spatial import cKDTree
from tqdm import tqdm

import unifiedwvalticolocs
from unifiedwvalticolocs.utils import (
    POSSIBLES_CCI_ALTI,
    POSSIBLES_CMEMS_ALTI,
    all_altimeters,
    date_window_for,
    get_conf_content,
)

EARTH_RADIUS_KM = 6371.0088
logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning
)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(
    action="ignore", message="invalid value encountered in true_divide"
)
warnings.filterwarnings(
    action="ignore",
    message="Degrees of freedom <= 0 \
    for slice",
)

MAX_NB_MATCHUPS_DEV_MODE = 3
error_altidb = "altidb %s not handled"

VAR_NAMES = {
    "cci": {
        "swh_varname": "swh_denoised",
        "lon_varname": "lon",
        "lat_varname": "lat",
    },
    "cmems": {
        "swh_varname": "VAVH",
        "lon_varname": "longitude",
        "lat_varname": "latitude",
    },
}


def from_npdt64_to_dt(dt64: np.datetime64) -> datetime.datetime:
    """Convert numpy.datetime64 to a timezone-aware datetime in UTC.

    Args:
        dt64: numpy.datetime64 value.

    Returns:
        timezone-aware datetime in UTC.
    """
    ref_date = np.datetime64("1970-01-01T00:00:00")
    ts = (dt64 - ref_date) / np.timedelta64(1, "s")
    return datetime.datetime.fromtimestamp(ts, datetime.UTC)


def step_0_get_sar_dt(sards: xr.Dataset) -> list[datetime.datetime]:
    """Get the datetime of the first measurement of the SAR file.

    Args:
        sards: SAR dataset containing 'time_sar' variable.

    Returns:
        List of datetime objects for each SAR measurement.
    """
    t0 = time.time()
    list_date_sar_dt = []
    logger.debug("step 0: get SAR dates")
    for xtimewv in range(len(sards["time_sar"])):
        date_sar = sards["time_sar"].values[xtimewv]
        dt = from_npdt64_to_dt(date_sar)
        list_date_sar_dt.append(dt)
    elapsed = time.time() - t0
    logger.debug("step0 done in %1.2f sec", elapsed)
    return list_date_sar_dt


def step_1_temp_match_cci(
    date_sar_dt: datetime.datetime,
    path_altimeters: str,
    acro_alti: str,
) -> list[str]:
    """Get all altimeter files in the range [D-1, D+1] around SAR acquisition.

    Args:
        date_sar_dt: SAR acquisition time.
        path_altimeters: Path to altimeter dataset.
        acro_alti: Altimeter acronym (e.g., 'jason-3').

    Returns:
        List of paths to matching altimeter files.
    """
    final_list_alti = []
    sta = date_sar_dt - datetime.timedelta(days=1)
    sto = date_sar_dt + datetime.timedelta(days=1)
    for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
        path_glob = os.path.join(
            path_altimeters,
            "data",
            "satellite",
            "altimeter",
            "l2p-swh",
            POSSIBLES_CCI_ALTI[acro_alti][0],
            dd.strftime("%Y"),
            dd.strftime("%j"),
            f"ESACCI-SEASTATE-L2P-SWH-"
            f"{POSSIBLES_CCI_ALTI[acro_alti][1]}-{dd.strftime('%Y%m%d')}T*-fv01.nc",
        )
        logger.debug("pattern alti : %s", path_glob)
        final_list_alti += sorted(glob.glob(path_glob))
    logger.debug("nb CCI files alti to read: %s", len(final_list_alti))
    logger.debug("output listing of alti: %s", final_list_alti)
    return final_list_alti


def step_1_temp_match(
    date_sar_dt: datetime.datetime,
    path_altimeters: str,
    acro_alti: str,
    altidb: str,
) -> list[str]:
    """Wrapper to handle both CMEMS and CCI altimeter databases.

    Args:
        date_sar_dt: SAR acquisition time.
        path_altimeters: Path to altimeter dataset.
        acro_alti: Altimeter acronym.
        altidb: 'cci' or 'cmems'.

    Returns:
        List of paths to matching altimeter files.

    Raises:
        ValueError: If altidb is not 'cci' or 'cmems'.
    """
    if altidb == "cci":
        return step_1_temp_match_cci(date_sar_dt, path_altimeters, acro_alti)
    elif altidb == "cmems":
        return step_1_temp_match_cmems(date_sar_dt, path_altimeters, acro_alti)
    else:
        raise ValueError(error_altidb % altidb)


def is_cmems_file_matching_in_time(
    one_nc_file_alti: str,
    lst_nc_files_alti_timematchup: list[str],
    groups_dates: dict,
    sta: datetime.datetime,
    sto: datetime.datetime,
) -> tuple[list[str], dict]:
    """Check if an altimeter file matches a time window.

    Args:
        one_nc_file_alti: Path to the altimeter file.
        lst_nc_files_alti_timematchup: List of matching files.
        groups_dates: Dictionary of dates.
        sta: Start time.
        sto: Stop time.

    Returns:
        Tuple of updated list and dictionary.
    """
    ymdthms = "%Y%m%dT%H%M%S"
    ymdth = "%Y%m%dT%H"
    date_alt_sta = datetime.datetime.strptime(
        os.path.basename(one_nc_file_alti).split("_")[5], ymdthms
    )
    date_alt_sto = datetime.datetime.strptime(
        os.path.basename(one_nc_file_alti).split("_")[6], ymdthms
    )
    generation_date_alt_sto = datetime.datetime.strptime(
        os.path.basename(one_nc_file_alti).split("_")[7].replace(".nc", ""), ymdthms
    )
    if date_alt_sta.strftime(ymdth) not in groups_dates:
        groups_dates[date_alt_sta.strftime(ymdth)] = [generation_date_alt_sto]
    else:
        groups_dates[date_alt_sta.strftime(ymdth)].append(generation_date_alt_sto)
    date_alt_sta = date_alt_sta.replace(tzinfo=timezone.utc)
    date_alt_sto = date_alt_sto.replace(tzinfo=timezone.utc)
    if (
        (date_alt_sta >= sta and date_alt_sto <= sto)
        or (sta <= date_alt_sta <= sto)
        or (sta <= date_alt_sto <= sto)
        or (sta >= date_alt_sta and sto <= date_alt_sto)
    ):
        if (
            datetime.datetime.strptime(
                os.path.basename(one_nc_file_alti).split("_")[5], ymdthms
            )
            not in lst_nc_files_alti_timematchup
        ):
            lst_nc_files_alti_timematchup.append(one_nc_file_alti)
    return lst_nc_files_alti_timematchup, groups_dates


def step_1_temp_match_cmems(
    date_sar_dt: datetime.datetime,
    path_altimeters: str,
    acro_alti: str,
) -> list[str]:
    """Find CMEMS L3 altimeter files in the range [D-1, D+1] around SAR acquisition.

    Picks the latest generated files in case of duplicates.

    Args:
        date_sar_dt: SAR acquisition time.
        path_altimeters: Path to altimeter dataset.
        acro_alti: Altimeter acronym (e.g., 'al').

    Returns:
        List of paths to matching altimeter files.
    """
    ymdthms = "%Y%m%dT%H%M%S"
    ymdth = "%Y%m%dT%H"
    ymd = "%Y%m%d"
    lst_nc_files_alti_timematchup = []
    lst_nc_files_alti_sorted = []
    sta = date_sar_dt - datetime.timedelta(days=1)
    sto = date_sar_dt + datetime.timedelta(days=1)
    sta = sta.replace(tzinfo=timezone.utc)
    sto = sto.replace(tzinfo=timezone.utc)
    for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
        path_glob = os.path.join(
            path_altimeters,
            dd.strftime("%Y"),
            dd.strftime("%m"),
            f"global_vavh_l3_rt_{acro_alti}_{dd.strftime(ymd)}T*.nc",
        )
        lst_nc_files_alti_sorted += sorted(glob.glob(path_glob))
    groups_dates = {}
    for gg in lst_nc_files_alti_sorted:
        lst_nc_files_alti_timematchup, groups_dates = is_cmems_file_matching_in_time(
            one_nc_file_alti=gg,
            lst_nc_files_alti_timematchup=lst_nc_files_alti_timematchup,
            groups_dates=groups_dates,
            sta=sta,
            sto=sto,
        )
    logger.debug(
        "lst_nc_files_alti_timematchup : %s", len(lst_nc_files_alti_timematchup)
    )
    final_list_alti = []
    for uu in lst_nc_files_alti_timematchup:
        date_alt_sta = datetime.datetime.strptime(
            os.path.basename(uu).split("_")[5], ymdthms
        )
        max_group = np.amax(np.array(groups_dates[date_alt_sta.strftime(ymdth)]))
        generation_date_alt_sto = datetime.datetime.strptime(
            os.path.basename(uu).split("_")[7].replace(".nc", ""), ymdthms
        )
        if max_group == generation_date_alt_sto:
            final_list_alti.append(uu)
    logger.debug("output listing of alti: %s", final_list_alti)
    return final_list_alti


def preproc_cmems_alti_files(ds: xr.Dataset) -> xr.Dataset:
    """Add fname variable associated with each time to keep filenames.

    Args:
        ds: Input dataset.

    Returns:
        Dataset with added 'fname' variable.
    """
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    pct_good = 100.0  # already edited product
    return ds, pct_good


def preproc_cciseastate_alti_files(ds: xr.Dataset) -> tuple[xr.Dataset, float]:
    """Preprocess CCI sea state altimeter files.

    Adds fname variable, filters by quality flag, and computes percentage of good data.

    Args:
        ds: Input dataset.

    Returns:
        Tuple of processed dataset and percentage of good data.
    """
    ds.load()
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    mask_good = ds["swh_quality_level"] == 3
    pct_good = 100.0 * mask_good.sum() / mask_good.size
    ds = ds.where(mask_good, drop=True)
    assert "fname" in ds
    return ds, pct_good


def read_all_alti_files(
    liste_altimeter_files: list[str],
    altidatabase: str,
    conf: dict,
) -> tuple[xr.Dataset, cKDTree]:
    """Read altimeter files and prepare for spatial matching.

    Args:
        liste_altimeter_files: List of altimeter file paths.
        altidatabase: 'cci' or 'cmems'.
        conf: Configuration dictionary.

    Returns:
        Tuple of altimeter dataset and KDTree for spatial queries.

    Raises:
        ValueError: If altidatabase is not 'cci' or 'cmems'.
    """
    counter = defaultdict(int)
    if altidatabase == "cci":
        lon_varname = "lon"
        lat_varname = "lat"
    elif altidatabase == "cmems":
        lon_varname = "longitude"
        lat_varname = "latitude"
    else:
        raise ValueError(error_altidb % altidatabase)

    if altidatabase == "cmems":
        fctpreprocess = preproc_cmems_alti_files
    else:
        fctpreprocess = preproc_cciseastate_alti_files

    tmp_cat_alti_ds = []
    all_pct_good = []
    for ii in tqdm(range(len(liste_altimeter_files)), desc="Reading altimeter files"):
        counter["total_alti_file_read"] += 1
        if ii == 0:
            logger.info("example of altimeter file used: %s", liste_altimeter_files[ii])
        tmpds, pct_good = fctpreprocess(xr.open_dataset(liste_altimeter_files[ii]))
        all_pct_good.append(pct_good)
        if len(tmpds["time"]) > 0:
            counter["total_alti_file_with_data"] += 1
            tmp_cat_alti_ds.append(tmpds)
        else:
            counter["total_alti_file_filteredout_on_swh_quality"] += 1
    logger.info(
        "average percentage of alti data good overall: %1.1f%%", np.mean(all_pct_good)
    )

    ds_alti = xr.concat(tmp_cat_alti_ds, dim="time", combine_attrs="override")
    logger.info("nb good alti points kept: %i", len(ds_alti["time"]))

    if altidatabase == "cci":
        version_database = os.path.basename(conf["cci_alti_dir"].rstrip("/"))
    elif altidatabase == "cmems":
        tmpds = xr.open_dataset(liste_altimeter_files[0])
        version_database = tmpds.attrs.get("software_version", "unknown")
    ds_alti = ds_alti.drop_duplicates(dim="time")
    ds_alti.attrs["altimeter_version_database"] = version_database
    ds_alti[lon_varname].load()
    ds_alti[lat_varname].load()
    tmp_lons = copy.copy(ds_alti[lon_varname].values)
    mask_bad_lon = tmp_lons > 180
    tmp_lons[mask_bad_lon] -= 360.0
    super_bad = tmp_lons > 360
    tmp_lons[super_bad] = np.nan
    logger.debug("tmp_lons : %s %s", np.nanmax(tmp_lons), np.nanmin(tmp_lons))
    ds_alti[lon_varname] = xr.DataArray(
        tmp_lons,
        dims=["time"],
        coords={"time": ds_alti["time"].values},
        attrs=ds_alti[lon_varname].attrs,
    )
    subset_alti1 = ds_alti.where(
        np.isfinite(ds_alti[lon_varname]) & np.isfinite(ds_alti[lat_varname]), drop=True
    )
    points_alti_xyz = latlon_to_xyz(
        subset_alti1[lat_varname],
        subset_alti1[lon_varname],
    )
    tree_alti = cKDTree(points_alti_xyz)
    logger.debug("alti files loaded, number of points: %s", len(subset_alti1["time"]))
    logging.info("counter: %s", dict(counter))
    return subset_alti1, tree_alti


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude in degrees to unit-sphere XYZ.

    Args:
        lat: Latitude array in degrees.
        lon: Longitude array in degrees.

    Returns:
        Array of XYZ coordinates on unit sphere.
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    cos_lat = np.cos(lat)
    return np.column_stack(
        (
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        )
    )


def step_2_geographic_match(
    sards: xr.Dataset,
    ds_alti: xr.Dataset,
    tree_alti: cKDTree,
    delta_dist_km: float,
) -> xr.Dataset | None:
    """Get altimeter points within a Haversine distance around SAR points.

    Args:
        sards: Dataset of a given WV image.
        ds_alti: Altimeter dataset.
        tree_alti: KDTree built from altimeter coordinates in XYZ.
        delta_dist_km: Maximum geographic distance in km.

    Returns:
        Altimeter measurements within delta_dist_km of at least one SAR point,
        or None if no matches.
    """
    points_sar_xyz = latlon_to_xyz(
        sards["oswLat"].values,
        sards["oswLon"].values,
    )
    r_tree = 2 * np.sin(delta_dist_km / (2 * EARTH_RADIUS_KM))
    queryballpoint = tree_alti.query_ball_point(points_sar_xyz, r=r_tree)
    queryballpoint = np.unique(np.concatenate(queryballpoint))
    if len(queryballpoint) > 0:
        return ds_alti.isel(time=queryballpoint)
    return None


def get_distances_v2(
    sar_dataset: xr.Dataset,
    subset_ok_match_alti: xr.Dataset,
    lon_varname: str,
    lat_varname: str,
) -> np.ndarray:
    """Compute distances between SAR center and altimeter points.

    Args:
        sar_dataset: SAR dataset.
        subset_ok_match_alti: Altimeter dataset subset.
        lon_varname: Variable name for longitude in altimeter dataset.
        lat_varname: Variable name for latitude in altimeter dataset.

    Returns:
        Array of distances in km.
    """
    t0 = time.time()
    lons_alt = subset_ok_match_alti[lon_varname]
    lats_alt = subset_ok_match_alti[lat_varname]
    lonsar = sar_dataset["oswLon"].values
    latsar = sar_dataset["oswLat"].values
    lonssartiled = np.tile(lonsar, (lons_alt.size))
    latssartiled = np.tile(latsar, (lons_alt.size))
    logger.debug("lons_alt %s,lonssartiled %s ", lons_alt.shape, lonssartiled.shape)
    all_dists = haversine(lonssartiled, latssartiled, lons_alt.values, lats_alt.values)
    logger.debug("time  to get distances v2 : %1.2f sec", (time.time() - t0))
    return all_dists


WINDOW_HS_STATS = {
    "hs_alti_mean": ("mean", "mean of the altimeter Hs"),
    "hs_alti_median": ("median", "median of the altimeter Hs"),
    "hs_alti_std": (
        "std",
        "standard deviation (ddof=1, NaN if count < 2) of the altimeter Hs",
    ),
    "hs_alti_count": ("count", "number of finite altimeter Hs values"),
}


def window_hs_statistics(hs: np.ndarray) -> dict[str, float | int]:
    """Summarize the altimeter Hs of all the points of the colocation window.

    The points within ``delta_dist_km`` and ``delta_t_minutes`` of a WV
    imagette sample the sea state over a track segment (up to
    2 * ``delta_dist_km`` long) instead of the single closest point.

    Args:
        hs: Altimeter significant wave heights of the window (NaN ignored).

    Returns:
        Dictionary with the keys mean, median, std (ddof=1) and count.

    Examples:
        >>> s = window_hs_statistics(np.array([1.0, 2.0, 3.0, np.nan]))
        >>> s["mean"], s["median"], s["count"]
        (2.0, 2.0, 3)
        >>> np.isnan(window_hs_statistics(np.array([1.0]))["std"])
        True
    """
    values = np.asarray(hs, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    count = int(values.size)
    return {
        "mean": float(np.mean(values)) if count else np.nan,
        "median": float(np.median(values)) if count else np.nan,
        "std": float(np.std(values, ddof=1)) if count > 1 else np.nan,
        "count": count,
    }


def add_window_hs_statistics(
    subset_ok_match_alti: xr.Dataset,
    window_alti: xr.Dataset,
    swh_varname: str,
    delta_t_max_minutes: float,
    delta_dist_km: float | None = None,
) -> xr.Dataset:
    """Add the Hs statistics of the colocation window to the matchup.

    Args:
        subset_ok_match_alti: Matchup (closest point) dataset, 0-d.
        window_alti: All the altimeter points within the space/time window.
        swh_varname: Name of the Hs variable in ``window_alti``.
        delta_t_max_minutes: Time window in minutes (for the metadata).
        delta_dist_km: Space window in km (for the metadata), if known.

    Returns:
        The matchup dataset with hs_alti_mean / median / std / count.
    """
    stats = window_hs_statistics(window_alti[swh_varname].values)
    radius = f"{delta_dist_km:g} km and " if delta_dist_km is not None else ""
    window = f"within {radius}{delta_t_max_minutes:g} min of the WV imagette"
    units = window_alti[swh_varname].attrs.get("units", "m")
    for name, (key, txt) in WINDOW_HS_STATS.items():
        dtype = "int32" if key == "count" else "float32"
        subset_ok_match_alti[name] = xr.DataArray(
            np.array(stats[key], dtype=dtype), dims=()
        )
        subset_ok_match_alti[name].attrs = {
            "description": f"{txt} {window}",
            "original_variable_name": swh_varname,
        }
        if key != "count":
            subset_ok_match_alti[name].attrs["units"] = units
    return subset_ok_match_alti


def step_3_closer_temp_match(
    sar_dataset: xr.Dataset,
    subset_alti: xr.Dataset,
    delta_t_max_minutes: int,
    altidb: str,
    cpt: defaultdict,
    delta_dist_km: float | None = None,
) -> tuple[xr.Dataset | None, np.ndarray, defaultdict, xr.Dataset]:
    """Find altimeter points within the time window around SAR-WV acquisition.

    The matchup keeps the altimeter point closest in time (``hs_alti_closest``)
    and adds the mean, median, standard deviation and number of the Hs values
    of all the points within the space and time window
    (``hs_alti_mean``, ``hs_alti_median``, ``hs_alti_std``, ``hs_alti_count``).

    Args:
        sar_dataset: WV dataset.
        subset_alti: Subset of the initial altimeter dataset (space window).
        delta_t_max_minutes: Time window in minutes.
        altidb: 'cci' or 'cmems'.
        cpt: Counter.
        delta_dist_km: Space window in km, only written in the metadata.

    Returns:
        Tuple of matching altimeter dataset (closest point and window
        statistics), array of matching altimeter filenames, updated counter,
        and the dataset of all altimeter points within the time and space
        criteria.
    """
    swh_varname = VAR_NAMES[altidb]["swh_varname"]
    lon_varname = VAR_NAMES[altidb]["lon_varname"]
    lat_varname = VAR_NAMES[altidb]["lat_varname"]

    dates_alt_dt64 = subset_alti["time"].values
    if dates_alt_dt64.ndim == 0:
        dates_alt_dt64 = np.array([dates_alt_dt64])

    sar_dt64 = sar_dataset.time_sar.values
    diffs_times_seconds = np.abs((dates_alt_dt64 - sar_dt64) / np.timedelta64(1, "s"))
    mask_time_ok = diffs_times_seconds < delta_t_max_minutes * 60
    ind_closest_in_time = np.argmin(diffs_times_seconds)
    inds_ok_alti = np.flatnonzero(mask_time_ok)
    cpt["maximum_colocs_time_and_space"] += mask_time_ok.sum()
    subset_ok_match_alti_all_time_and_space = subset_alti.isel(time=inds_ok_alti)
    subset_ok_match_alti = None
    list_alti_files_timespace_match = []
    if len(inds_ok_alti) > 0:
        subset_ok_match_alti = subset_alti.isel(time=ind_closest_in_time)
        delta_d_closest_in_time = get_distances_v2(
            sar_dataset, subset_ok_match_alti, lon_varname, lat_varname
        )
        if isinstance(sar_dt64, list) or isinstance(sar_dt64, np.ndarray):
            sar_dt64 = sar_dt64[0]
        delta_t_closest_in_time = (subset_ok_match_alti["time"] - sar_dt64).astype(
            "timedelta64[s]"
        )
        subset_ok_match_alti["delta_t_closest"] = delta_t_closest_in_time
        subset_ok_match_alti["delta_t_closest"].attrs[
            "description"
        ] = "delta Time altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria"
        subset_ok_match_alti["delta_d_closest"] = xr.DataArray(
            delta_d_closest_in_time[0], dims=()
        )
        subset_ok_match_alti["delta_d_closest"].attrs[
            "description"
        ] = "delta distance altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria"
        subset_ok_match_alti["delta_d_closest"].attrs["units"] = "km"
        subset_ok_match_alti = add_window_hs_statistics(
            subset_ok_match_alti,
            subset_ok_match_alti_all_time_and_space,
            swh_varname,
            delta_t_max_minutes,
            delta_dist_km,
        )
        subset_ok_match_alti = subset_ok_match_alti.rename(
            {
                swh_varname: "hs_alti_closest",
                lon_varname: "lon_ALT",
                lat_varname: "lat_ALT",
            }
        )
        subset_ok_match_alti["hs_alti_closest"].attrs[
            "original_variable_name"
        ] = swh_varname
        subset_ok_match_alti["lon_ALT"].attrs["original_variable_name"] = lon_varname
        subset_ok_match_alti["lat_ALT"].attrs["original_variable_name"] = lat_varname
        list_alti_files_timespace_match = np.unique(subset_ok_match_alti["fname"])
    return (
        subset_ok_match_alti,
        list_alti_files_timespace_match,
        cpt,
        subset_ok_match_alti_all_time_and_space,
    )


def haversine(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Calculate the great circle distance between two points.

    Args:
        lon1: Longitude of first point in decimal degrees.
        lat1: Latitude of first point in decimal degrees.
        lon2: Longitude of second point in decimal degrees.
        lat2: Latitude of second point in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = EARTH_RADIUS_KM
    return c * r


def save_coloc_netcdf_file(ds_colocations: xr.Dataset, output_nc_file: str) -> bool:
    """Save colocation dataset to NetCDF file.

    Args:
        ds_colocations: Colocation dataset.
        output_nc_file: Output file path.

    Returns:
        True if file was written, False otherwise.
    """
    new_file_written = False
    if not os.path.exists(output_nc_file):
        if len(ds_colocations["oswLon"]) > 0:
            logger.info("start writing netCDF")
            new_attrs = {
                "colocation_institution": "Institut Français pour la Recherche et l Exploitation de la MER",
                "colocation_institution_abbreviation": " LOPS-IFREMER",
                "colocation_publisher_name": "ifremer/LOPS",
                "colocation_publisher_url": "https://www.umr-lops.fr/",
                "colocation_publisher_email": "lops-siam@listes.ifremer.fr",
                "colocation_product_description": "colocations between Sentinel-1 WV"
                " and altimeter coming from CCI sea state or CMEMS database",
                "colocation_library_version": unifiedwvalticolocs.__version__,
                "colocation_library_url": "https://github.com/umr-lops/unifiedwvalticolocs",
            }
            for kk, vv in new_attrs.items():
                ds_colocations.attrs[kk] = vv
            logger.info(output_nc_file)
            ds_colocations.to_netcdf(output_nc_file)
            new_file_written = True
        else:
            logger.info("no file to save")
    return new_file_written


def write_coloc_listing(
    outputlisting: str, coloc_listing_data: dict, redo: bool = False
) -> None:
    """Write colocation listing file.

    The listing contains fullpath SAR and basename alti files.
    A SAR file may appear multiple times if colocated with different alti files.

    Args:
        outputlisting: Output listing file path.
        coloc_listing_data: Dictionary mapping SAR full path to list of alti filenames.
        redo: If True, overwrite existing file.
    """
    if os.path.exists(outputlisting) and redo is False:
        logger.info("%s already exists", outputlisting)
    else:
        with open(outputlisting, "w") as fid:
            for sarfullpath in coloc_listing_data:
                for altifile_idx in range(len(coloc_listing_data[sarfullpath])):
                    sarfp = sarfullpath if sarfullpath is not None else "unknown"
                    fid.write(
                        sarfp
                        + ","
                        + coloc_listing_data[sarfullpath][altifile_idx]
                        + "\n"
                    )
        logger.info("output listing coloc : %s", outputlisting)


def preprocess_wv_s1_ocn(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess S1 WV OCN files for colocation.

    Args:
        ds: Input dataset.

    Returns:
        Preprocessed dataset with selected variables and time_sar dimension.
    """
    to_keep_vars = [
        "oswLon",
        "oswLat",
        "oswIncidenceAngle",
        "oswHeading",
        "oswPhs0",
        "oswWaveAge",
        "oswDepth",
        "oswTotalHs",
        "oswTotalHsStdev",
        "oswWindSpeed",
        "oswNrcs",
        "oswEcmwfWindSpeed",
        "oswNlWidth",
        "oswLandFlag",
        "oswLandCoverage",
        "oswQualityFlag",
        "oswAzSizeSLC",
    ]
    consolidated_lst_var_tokeep = []
    for vv in to_keep_vars:
        if vv in ds.variables:
            consolidated_lst_var_tokeep.append(vv)
        else:
            logger.debug("variable %s is not present in S1 WV OCN file", vv)

    ds = ds[consolidated_lst_var_tokeep]
    ds["time_sar"] = xr.DataArray(
        [
            datetime.datetime.strptime(
                os.path.basename(ds.encoding["source"]).split("-")[5], "%Y%m%dt%H%M%S"
            )
        ],
        dims=["time_sar"],
    )
    # src = os.path.basename(ds.encoding["source"])
    src = ds.encoding["source"]
    ds["source_file"] = xr.DataArray([src], dims=["time_sar"])
    ds = ds.squeeze(["oswRaSize", "oswAzSize"])

    for var in ds.data_vars:
        if ds[var].dims == ():
            ds[var] = ds[var].expand_dims(time_sar=ds.time_sar)

    return ds


def read_one_measurement_wv(measurement_file: str) -> xr.Dataset:
    """Open, preprocess and load one S1 WV OCN measurement file.

    Module-level function so that it can run in a worker process.

    Args:
        measurement_file: Path to a ``measurement/*.nc`` file of a WV SAFE.

    Returns:
        Preprocessed dataset loaded in memory (file closed).
    """
    with xr.open_dataset(measurement_file) as ds:
        return preprocess_wv_s1_ocn(ds).load()


def list_measurements_wv(safewv: str) -> list[str]:
    """List the measurement files of a WV SAFE.

    Args:
        safewv: Path to the SAFE.

    Returns:
        List of ``measurement/*.nc`` paths.
    """
    return glob.glob(os.path.join(safewv, "measurement", "*.nc"))


def concat_measurements_wv(safewv: str, measurements: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate the preprocessed measurements of one SAFE.

    Args:
        safewv: Path to the SAFE (for the log).
        measurements: Preprocessed measurement datasets.

    Returns:
        Dataset along ``time_sar``; an empty dataset if there is no measurement.
    """
    if measurements:
        return xr.concat(measurements, dim="time_sar").load()
    # A SAFE with no measurement/*.nc (corrupt/partial download) must not
    # crash xr.concat([]) (issue #52); skip it with an empty SAR dataset.
    logger.warning("no measurement/*.nc found in SAFE %s - skipping", safewv)
    return xr.Dataset({"empty": (["time_sar"], [])})


def load_sar_safes(lst_wv_safe: list[str], nproc: int = 1) -> dict[str, xr.Dataset]:
    """Read all the WV measurements of a list of SAFE, once.

    Reading the ~2000 small measurement files of a day is the bottleneck of the
    coloc and is latency-bound on the shared file systems: with ``nproc`` > 1
    the files are read by worker processes (not threads: HDF5 is not
    thread-safe), which gives a large speed-up even on a 1-CPU allocation.

    Args:
        lst_wv_safe: Paths to the SAFE.
        nproc: Number of reader processes (1 = sequential).

    Returns:
        Dictionary SAFE path -> dataset of its measurements along ``time_sar``.
    """
    t0 = time.time()
    files = {safe: list_measurements_wv(safe) for safe in lst_wv_safe}
    flat = [ff for safe in lst_wv_safe for ff in files[safe]]
    if nproc > 1 and len(flat) > 1:
        with ProcessPoolExecutor(
            max_workers=nproc, mp_context=get_context("spawn")
        ) as executor:
            datasets = list(executor.map(read_one_measurement_wv, flat, chunksize=16))
    else:
        datasets = [read_one_measurement_wv(ff) for ff in flat]
    out = {}
    start = 0
    for safe in lst_wv_safe:
        nb = len(files[safe])
        out[safe] = concat_measurements_wv(safe, datasets[start : start + nb])
        start += nb
    logger.info(
        "%i WV measurements of %i SAFE read in %1.1f s (nproc=%i)",
        len(flat),
        len(lst_wv_safe),
        time.time() - t0,
        nproc,
    )
    return out


def treat_one_measurement_wv(
    sards: xr.Dataset,
    date_sar_dt: datetime.datetime,
    ds_alti: xr.Dataset,
    tree_alti: cKDTree,
    altidb: str,
    coloc_listing: dict,
    cpt: defaultdict,
    conf: dict,
) -> tuple[xr.Dataset | None, dict, defaultdict]:
    """Associate a WV OCN measurement with altimeter observations.

    Args:
        sards: S1 OCN WV data, contains a unique WV image.
        date_sar_dt: WV starting measurement date.
        ds_alti: Altimeter data.
        tree_alti: KDTree for altimeter points.
        altidb: Altimeter database name ('cci' or 'cmems').
        coloc_listing: Dictionary to store filepath listing.
        cpt: Counter.
        conf: Configuration parameters.

    Returns:
        Tuple of (subset_ok_match_alti, coloc_listing, cpt).
    """
    subset_ok_match_alti = None
    cpt["nb_index_sar_browsed"] += 1
    # fullpath_l2_wv_ocn = sards.encoding["source"]
    fullpath_l2_wv_ocn = str(sards["source_file"].values)
    subset_alti = step_2_geographic_match(
        sards=sards,
        ds_alti=ds_alti,
        tree_alti=tree_alti,
        delta_dist_km=conf["delta_dist_km"],
    )
    if subset_alti is not None:
        cpt["coloc_in_space"] += len(subset_alti["time"])
        (
            subset_ok_match_alti,
            list_alti_files_timespace_mu,
            cpt,
            subset_ok_match_alti_all_time_and_space,
        ) = step_3_closer_temp_match(
            sar_dataset=sards,
            subset_alti=subset_alti,
            delta_t_max_minutes=conf["delta_t_minutes"],
            altidb=altidb,
            cpt=cpt,
            delta_dist_km=conf["delta_dist_km"],
        )
        if len(list_alti_files_timespace_mu) > 0:
            coloc_listing[fullpath_l2_wv_ocn] = list_alti_files_timespace_mu
            cpt["nb_coloc"] += 1
            subset_ok_match_alti["time_sar"] = date_sar_dt.replace(tzinfo=None)
    return subset_ok_match_alti, coloc_listing, cpt


def treat_one_safe_wv(
    safewv: str,
    ds_alti: xr.Dataset,
    tree_alti: cKDTree,
    altidb: str,
    coloc_listing: dict,
    cpt: defaultdict,
    conf: dict,
    dev: bool = False,
    progressbar: bool = True,
    sar_dataset_safe: xr.Dataset | None = None,
) -> tuple[xr.Dataset, dict, defaultdict]:
    """Colocate one SAFE OCN WV with altimeters.

    Args:
        safewv: Path to the SAFE OCN WV to process.
        ds_alti: Altimeter dataset.
        tree_alti: Spatial tree for altimeter data.
        altidb: Altimeter database name ('cci' or 'cmems').
        coloc_listing: Dictionary to store colocation listing.
        cpt: Counter.
        conf: Configuration dictionary.
        dev: If True, break after few matchups.
        progressbar: If True, show progress bar.
        sar_dataset_safe: Measurements of the SAFE already read (see
            ``load_sar_safes``); read from ``safewv`` if None.

    Returns:
        Tuple of (colocated_observations, coloc_listing, cpt).
    """
    logger.debug("SAR Sentinel-1 WV SAFE to process : %s ", safewv)
    colocated_observations = xr.Dataset({"empty": (["time_sar"], [])})
    cat_alti_mathcups_colocs_ds = []
    if sar_dataset_safe is None:
        measurement_wv_list = list_measurements_wv(safewv)
        logger.debug("Number of measurement in the SAFE : %d", len(measurement_wv_list))
        sar_dataset_safe = concat_measurements_wv(
            safewv, [read_one_measurement_wv(ff) for ff in measurement_wv_list]
        )
    else:
        # shared between the altimeters of a day: never modify it in place
        sar_dataset_safe = sar_dataset_safe.copy(deep=False)
    list_date_sar_dt = step_0_get_sar_dt(sards=sar_dataset_safe)

    if ds_alti:
        if progressbar:
            iterratotor = tqdm(range(len(list_date_sar_dt)), desc="WV measurement")
        else:
            iterratotor = range(len(list_date_sar_dt))
        for index_t_sar in iterratotor:
            alti_point_ds_match, coloc_listing, cpt = treat_one_measurement_wv(
                sar_dataset_safe.isel(time_sar=index_t_sar),
                date_sar_dt=list_date_sar_dt[index_t_sar],
                ds_alti=ds_alti,
                tree_alti=tree_alti,
                altidb=altidb,
                coloc_listing=coloc_listing,
                cpt=cpt,
                conf=conf,
            )
            if alti_point_ds_match is not None:
                cat_alti_mathcups_colocs_ds.append(alti_point_ds_match)
            if dev and cpt["nb_coloc"] > MAX_NB_MATCHUPS_DEV_MODE:
                logger.info("break loops over measurements after finding few matchups")
                break
        logger.debug("end of pair construction")
        if len(cat_alti_mathcups_colocs_ds) > 0:
            aggregated_alti_wv_matchups = xr.concat(
                cat_alti_mathcups_colocs_ds, dim="coloc_index"
            )
            colocated_observations = sar_dataset_safe.sel(
                time_sar=aggregated_alti_wv_matchups["time_sar"]
            )
            aggregated_alti_wv_matchups = aggregated_alti_wv_matchups.drop_vars(
                ["time_sar"]
            )
            aggregated_alti_wv_matchups = aggregated_alti_wv_matchups.rename_vars(
                {"time": "time_ALT"}
            )
            aggregated_alti_wv_matchups["time_ALT"] = aggregated_alti_wv_matchups[
                "time_ALT"
            ].astype("datetime64[s]")
            logger.debug("merge alti and SAR colocated values")
            logger.debug("associate SAR and alti information in the same dataset.")
            colocated_observations = xr.merge(
                [colocated_observations, aggregated_alti_wv_matchups], compat="override"
            )
            list_att = copy.copy(colocated_observations.attrs)
            for att in list_att:
                colocated_observations.attrs["sar_" + att] = (
                    colocated_observations.attrs[att]
                )
                del colocated_observations.attrs[att]
            for att in ds_alti.attrs:
                colocated_observations.attrs["alti_" + att] = ds_alti.attrs[att]
        else:
            cpt["nb_safe_without-matchup_alti"] += 1
            logger.debug("no matching alti for this SAFE.")
    else:
        logger.info("no altimeter files found in the time window around the SAR SAFE")
        cpt["nb_safe_without_alti_files"] += 1
    for aat in colocated_observations.attrs:
        logger.debug(
            "colocated_observations.attrs : %s = %s",
            aat,
            colocated_observations.attrs[aat],
        )
    return colocated_observations, coloc_listing, cpt


def get_path_alti(altidb: str, alt: str, conf: dict) -> tuple[str, str, str]:
    """Get path, acronym, and Hs variable name for a specific altimeter.

    Args:
        altidb: 'cci' or 'cmems'.
        alt: Altimeter name (e.g., 'cci_jason-3').
        conf: Configuration dictionary.

    Returns:
        Tuple of (path_altimeter, acronym_alti_path_ifr, swh_varname).

    Raises:
        ValueError: If altidb is not 'cci' or 'cmems'.
    """
    cmems_dir = conf["cmems_dir"]
    subset_alti_name_dir = conf["subset_alti_name_dir"]
    PATH_ALT = {
        "cmems": os.path.join(cmems_dir, subset_alti_name_dir),
        "cci": conf["cci_alti_dir"],
    }
    # Split on the FIRST underscore only: CCI catalog keys may contain
    # underscores themselves (e.g. "sentinel-3_a", "topex-poseidon_topex").
    mission = alt.partition("_")[2]
    if altidb == "cci":
        path_altimeter = os.path.join(PATH_ALT[altidb])
        swh_varname = "swh_denoised"
        acronym_alti_path_ifr = mission
    elif altidb == "cmems":
        path_altimeter = os.path.join(PATH_ALT[altidb] % POSSIBLES_CMEMS_ALTI[mission])
        swh_varname = "VAVH"
        acronym_alti_path_ifr = POSSIBLES_CMEMS_ALTI[mission]
        if acronym_alti_path_ifr == "swon":
            acronym_alti_path_ifr = "swot"
    else:
        raise ValueError(error_altidb % altidb)
    return path_altimeter, acronym_alti_path_ifr, swh_varname


def output_coloc_files(
    outputdir: str, sarunit: str, alt: str, day_analyzed: str, conf: dict[str, Any]
) -> tuple[str, str]:
    """Paths of the daily coloc NetCDF file and of its listing.

    Args:
        outputdir: Output directory.
        sarunit: SAR mission (S1A, S1B...).
        alt: Altimeter name (e.g., 'cmems_SARAL').
        day_analyzed: Date in YYYYMMDD format.
        conf: Configuration dictionary (delta_t_minutes, delta_dist_km).

    Returns:
        Tuple (NetCDF path, .lst path).
    """
    base = os.path.join(
        outputdir,
        sarunit + "_" + alt,
        day_analyzed[:4],
        "coloc_"
        + day_analyzed
        + "_"
        + sarunit
        + "_WV_"
        + alt
        + "_"
        + str(conf["delta_t_minutes"])
        + "_min_"
        + str(conf["delta_dist_km"])
        + "_km",
    )
    return base + ".nc", base + ".lst"


def list_sar_safes(day_analyzed: str, sarunit: str, conf: dict[str, Any]) -> list[str]:
    """List the S1 WV OCN SAFE of a day.

    Args:
        day_analyzed: Date in YYYYMMDD format.
        sarunit: SAR mission (S1A, S1B...).
        conf: Configuration dictionary (path_SAR).

    Returns:
        Sorted list of SAFE paths.
    """
    date = datetime.datetime.strptime(day_analyzed, "%Y%m%d")
    assert os.path.exists(conf["path_SAR"])
    pattern_sar = os.path.join(
        conf["path_SAR"],
        "sentinel-1" + sarunit[-1].lower(),
        "L2",
        "WV",
        sarunit + "_WV_OCN__2S",
        date.strftime("%Y"),
        date.strftime("%j"),
        "*.SAFE",
    )
    logger.info("SAR ESA Level-2 OCN SAFE pattern : %s", pattern_sar)
    lst_wv_safe_sorted = sorted(glob.glob(pattern_sar))
    logger.info("%s SAR WV SAFE found", len(lst_wv_safe_sorted))
    return lst_wv_safe_sorted


def coloc_one_alti(
    day_analyzed: str,
    alt: str,
    sarunit: str,
    outputdir: str,
    conf: dict[str, Any],
    lst_wv_safe_sorted: list[str],
    get_sar_safes: Callable[[], dict[str, xr.Dataset]] | None = None,
    dev: bool = False,
    redo: bool = False,
) -> defaultdict[str, int]:
    """Colocate the WV SAFE of a day with one altimeter and save the result.

    The caller checks the altimeter path and creates the output directory
    (see ``prepare_alti``).

    Args:
        day_analyzed: Date to analyze in YYYYMMDD format.
        alt: Altimeter name (e.g., 'cmems_SARAL').
        sarunit: SAR mission (S1A, S1B...).
        outputdir: Output directory.
        conf: Configuration dictionary.
        lst_wv_safe_sorted: SAFE of the day.
        get_sar_safes: Returns the measurements of the SAFE (read once per day,
            only called when the altimeter has data); if None, each SAFE is
            read sequentially by ``treat_one_safe_wv``.
        dev: Development mode flag.
        redo: If True, redo existing files.

    Returns:
        Counter with statistics.
    """
    cpt: defaultdict[str, int] = defaultdict(int)
    altidb = alt.split("_")[0]
    output_nc_file, output_lst_file = output_coloc_files(
        outputdir, sarunit, alt, day_analyzed, conf
    )
    path_altimeter, acronym_alti_path_ifr, _ = get_path_alti(altidb, alt, conf=conf)
    coloc_listing: dict[str, Any] = {}
    list_alti_in_raw_time_window = step_1_temp_match(
        date_sar_dt=datetime.datetime.strptime(day_analyzed, "%Y%m%d"),
        path_altimeters=path_altimeter,
        acro_alti=acronym_alti_path_ifr,
        altidb=altidb,
    )
    ds_alti = None
    tree_alti = None
    if len(lst_wv_safe_sorted) and len(list_alti_in_raw_time_window) > 0:
        ds_alti, tree_alti = read_all_alti_files(
            liste_altimeter_files=list_alti_in_raw_time_window,
            altidatabase=altidb,
            conf=conf,
        )
    else:
        logger.info("no altimeter files found in the time window around the SAR SAFE")
    if len(lst_wv_safe_sorted) and ds_alti is not None and len(ds_alti["time"]) > 0:
        sar_safes = get_sar_safes() if get_sar_safes is not None else {}
        all_safe_matchups = []
        pbar = tqdm(range(len(lst_wv_safe_sorted)), desc="WV SAFE")
        for ssi in pbar:
            string_counter = ";".join([f"{key, cpt[key]}" for key in cpt])
            pbar.set_description("WV SAFE : %s" % string_counter)
            safewv = lst_wv_safe_sorted[ssi]
            logger.debug("%i/%i", ssi + 1, len(lst_wv_safe_sorted))
            one_safe_colocs, coloc_listing, cpt = treat_one_safe_wv(
                safewv=safewv,
                ds_alti=ds_alti,
                tree_alti=tree_alti,
                altidb=altidb,
                coloc_listing=coloc_listing,
                cpt=cpt,
                dev=dev,
                progressbar=False,
                conf=conf,
                sar_dataset_safe=sar_safes.get(safewv),
            )
            if len(one_safe_colocs.time_sar) > 0:
                all_safe_matchups.append(one_safe_colocs)
            if dev and cpt["nb_coloc"] > MAX_NB_MATCHUPS_DEV_MODE:
                logger.info("break loops over SAFE after finding few matchups")
                break
        if len(all_safe_matchups) > 0:
            daily_colocated_observations = xr.concat(
                all_safe_matchups, dim="coloc_index"
            )
            if os.path.exists(output_nc_file) and redo:
                os.remove(output_nc_file)
            output_file_written = save_coloc_netcdf_file(
                daily_colocated_observations, output_nc_file
            )
            if output_file_written:
                logger.info("successful save output file: %s", output_nc_file)
            if len(daily_colocated_observations["oswLon"]) > 0:
                write_coloc_listing(output_lst_file, coloc_listing, redo=redo)
    else:
        logger.info(
            "no SAR WV data for %s or no altimeter data matching this date",
            day_analyzed,
        )
    return cpt


def make_sar_getter(
    lst_wv_safe_sorted: list[str], nproc: int
) -> Callable[[], dict[str, xr.Dataset]]:
    """Lazy, cached reader of the SAR measurements of a day.

    Args:
        lst_wv_safe_sorted: SAFE of the day.
        nproc: Number of reader processes.

    Returns:
        Function returning the measurements, read at the first call only.
    """
    cache: dict[str, dict[str, xr.Dataset]] = {}

    def get_sar_safes() -> dict[str, xr.Dataset]:
        if "sar" not in cache:
            cache["sar"] = load_sar_safes(lst_wv_safe_sorted, nproc=nproc)
        return cache["sar"]

    return get_sar_safes


def prepare_alti(
    alt: str, sarunit: str, outputdir: str, day_analyzed: str, conf: dict[str, Any]
) -> str:
    """Check the altimeter path and create the output directory.

    Args:
        alt: Altimeter name (e.g., 'cmems_SARAL').
        sarunit: SAR mission (S1A, S1B...).
        outputdir: Output directory.
        day_analyzed: Date in YYYYMMDD format.
        conf: Configuration dictionary.

    Returns:
        Path of the output NetCDF file.
    """
    path_altimeter, _, _ = get_path_alti(alt.split("_")[0], alt, conf=conf)
    logger.info("path_altimeter : %s", path_altimeter)
    assert os.path.exists(path_altimeter)
    output_nc_file, _ = output_coloc_files(outputdir, sarunit, alt, day_analyzed, conf)
    os.makedirs(os.path.dirname(output_nc_file), 0o0775, exist_ok=True)
    return output_nc_file


def core_coloc(
    day_analyzed: str,
    alt: str,
    sarunit: str,
    outputdir: str,
    conf: dict[str, Any],
    dev: bool = False,
    redo: bool = False,
    progressbar: bool = False,
    nproc: int = 1,
) -> defaultdict[str, int]:
    """Core colocation routine for one altimeter.

    Args:
        day_analyzed: Date to analyze in YYYYMMDD format.
        alt: Altimeter name (e.g., 'cmems_al').
        sarunit: SAR mission (S1A, S1B).
        outputdir: Output directory.
        conf: Configuration dictionary.
        dev: Development mode flag.
        redo: If True, redo existing files.
        progressbar: Kept for backward compatibility (the SAFE loop always
            shows a tqdm bar).
        nproc: Number of processes reading the SAR measurement files
            (1 = sequential, SAFE by SAFE).

    Returns:
        Counter with statistics.
    """
    path_altimeter, _, _ = get_path_alti(alt.split("_")[0], alt, conf=conf)
    logger.info("path_altimeter : %s", path_altimeter)
    assert os.path.exists(path_altimeter)
    lst_wv_safe_sorted = list_sar_safes(day_analyzed, sarunit, conf)
    output_nc_file, _ = output_coloc_files(outputdir, sarunit, alt, day_analyzed, conf)
    time.sleep(rng.integers(0, 10))
    os.makedirs(os.path.dirname(output_nc_file), 0o0775, exist_ok=True)
    if os.path.exists(output_nc_file) and redo is False:
        logger.info("output coloc S1-WV alti file already exists (redo is False)")
        sys.exit(0)
    return coloc_one_alti(
        day_analyzed,
        alt,
        sarunit,
        outputdir,
        conf,
        lst_wv_safe_sorted,
        make_sar_getter(lst_wv_safe_sorted, nproc) if nproc > 1 else None,
        dev=dev,
        redo=redo,
    )


def altimeters_of_day(day_analyzed: str, sarunit: str, alts: list[str]) -> list[str]:
    """Keep the altimeters whose acquisition window overlaps the SAR one that day.

    Args:
        day_analyzed: Date in YYYYMMDD format.
        sarunit: SAR mission (S1A, S1B...).
        alts: Candidate altimeter names; ``["all"]`` means every altimeter.

    Returns:
        Altimeter names operating on that day.

    Examples:
        >>> altimeters_of_day("20200115", "S1A", ["cci_saral", "cci_ers-1"])
        ['cci_saral']
    """
    if alts == ["all"]:
        alts = all_altimeters()
    day = datetime.datetime.strptime(day_analyzed, "%Y%m%d").date()
    out = []
    for alt in alts:
        window = date_window_for(alt, sarunit)
        if window is not None and window[0] <= day <= window[1]:
            out.append(alt)
    return out


def core_coloc_day(
    day_analyzed: str,
    alts: list[str],
    sarunit: str,
    outputdir: str,
    conf: dict[str, Any],
    dev: bool = False,
    redo: bool = False,
    nproc: int = 8,
) -> tuple[dict[str, defaultdict[str, int]], list[str]]:
    """Colocate the WV of one day with several altimeters, reading the SAR once.

    The WV measurement files (the bottleneck) are read once for all the
    altimeters, with ``nproc`` reader processes. An altimeter whose output
    already exists is skipped (unless ``redo``); a failure on one altimeter is
    logged and does not stop the others.

    Args:
        day_analyzed: Date to analyze in YYYYMMDD format.
        alts: Altimeter names (``["all"]`` = every altimeter operating that day).
        sarunit: SAR mission (S1A, S1B...).
        outputdir: Output directory.
        conf: Configuration dictionary.
        dev: Development mode flag.
        redo: If True, redo existing files.
        nproc: Number of processes reading the SAR measurement files.

    Returns:
        Tuple (counters per altimeter, altimeters that failed).
    """
    alts = altimeters_of_day(day_analyzed, sarunit, alts)
    todo = [
        alt
        for alt in alts
        if redo
        or not os.path.exists(
            output_coloc_files(outputdir, sarunit, alt, day_analyzed, conf)[0]
        )
    ]
    logger.info(
        "%s %s: %i altimeters operating, %i to process: %s",
        day_analyzed,
        sarunit,
        len(alts),
        len(todo),
        todo,
    )
    counters: dict[str, defaultdict[str, int]] = {}
    failed: list[str] = []
    if not todo:
        return counters, failed
    lst_wv_safe_sorted = list_sar_safes(day_analyzed, sarunit, conf)
    get_sar_safes = make_sar_getter(lst_wv_safe_sorted, nproc)
    for alt in todo:
        t0 = time.time()
        try:
            prepare_alti(alt, sarunit, outputdir, day_analyzed, conf)
            counters[alt] = coloc_one_alti(
                day_analyzed,
                alt,
                sarunit,
                outputdir,
                conf,
                lst_wv_safe_sorted,
                get_sar_safes,
                dev=dev,
                redo=redo,
            )
        except Exception:  # one altimeter must not stop the others
            logger.exception("coloc failed for %s %s %s", day_analyzed, sarunit, alt)
            failed.append(alt)
        logger.info("%s done in %1.1f s", alt, time.time() - t0)
    return counters, failed


def parse_alts(values: list[str]) -> list[str]:
    """Split and check the --alt values.

    Args:
        values: Values of --alt (space and/or comma separated).

    Returns:
        List of altimeter names, or ``["all"]``.

    Raises:
        ValueError: If an altimeter is unknown.

    Examples:
        >>> parse_alts(["cci_saral,cmems_SARAL"])
        ['cci_saral', 'cmems_SARAL']
        >>> parse_alts(["all"])
        ['all']
    """
    alts = [aa for vv in values for aa in vv.split(",") if aa]
    if "all" in alts:
        return ["all"]
    known = set(all_altimeters())
    unknown = [aa for aa in alts if aa not in known]
    if unknown:
        raise ValueError(f"unknown altimeter(s) {unknown}, expected {sorted(known)}")
    return alts


def entrypoint() -> None:
    """Entry point for the colocation script."""
    tinit = time.time()
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    parser = argparse.ArgumentParser(description="colocate S1 WV and altimeters")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir",
        help="folder where the co-location data (.nc) will be written",
        required=True,
    )
    parser.add_argument(
        "--startdate", required=True, help=" date to analyse: YYYYMMDD", type=str
    )
    parser.add_argument(
        "--sat", required=True, help="mission SAR: S1A or S1B...", type=str
    )
    parser.add_argument(
        "--alt",
        required=True,
        nargs="+",
        help="one or several altimeters (space or comma separated): "
        "cmems_SARAL cci_jason-3..., or 'all' (every altimeter operating that "
        "day). With several altimeters the WV files are read once for all.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="number of processes reading the WV measurement files (I/O bound: "
        "useful even on a 1-CPU job) [default=8]",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="redo existing files nc [optional, default=False->"
        " nothing done if file already exists]",
    )
    parser.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="display tdqm progress bar [optional, default=False]",
    )
    parser.add_argument(
        "--dev",
        help="quick run for dev/test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--config",
        help="path to config file (yml) with parameters for the script. ",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S"
        )
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S")
    logger.info(
        "Start of execution for script %s using "
        "WV Level-2 OCN and altimeters from "
        "CCI sea state L2P or CMEMS WAV L3",
        os.path.basename(__file__),
    )
    logger.info("development/test mode activated: %s", args.dev)
    config = get_conf_content(args.config)
    alts = parse_alts(args.alt)
    failed: list[str] = []
    if len(alts) == 1 and alts != ["all"]:
        cpt = core_coloc(
            sarunit=args.sat,
            alt=alts[0],
            outputdir=args.outputdir,
            dev=args.dev,
            day_analyzed=args.startdate,
            redo=args.redo,
            progressbar=args.progressbar,
            conf=config,
            nproc=args.nproc,
        )
        logger.info("counters: %s", cpt)
    else:
        counters, failed = core_coloc_day(
            day_analyzed=args.startdate,
            alts=alts,
            sarunit=args.sat,
            outputdir=args.outputdir,
            conf=config,
            dev=args.dev,
            redo=args.redo,
            nproc=args.nproc,
        )
        for alt, cpt in counters.items():
            logger.info("counters %s: %s", alt, dict(cpt))
    logger.info("memory in Mo: %s", getrusage(RUSAGE_SELF).ru_maxrss / 1000.0)
    logger.info("analysis done in %1.1f sec", time.time() - tinit)
    if failed:
        logger.error("coloc failed for: %s", failed)
        sys.exit(1)
    logger.info("end.")


if __name__ == "__main__":
    entrypoint()
