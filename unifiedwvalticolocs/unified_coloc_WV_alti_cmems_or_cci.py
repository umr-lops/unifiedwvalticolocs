"""Create NetCDF colocation datasets from altimeters and S1 WV OCN data.

Author: Antoine Grouazel

Colocates Sentinel-1 WV (wave) OCN Level-2 measurements with CMEMS WAVE L3
or CCI SeaState L2P altimeter observations, in space and time.
"""

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
from datetime import timezone
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


def step_3_closer_temp_match(
    sar_dataset: xr.Dataset,
    subset_alti: xr.Dataset,
    delta_t_max_minutes: int,
    altidb: str,
    cpt: defaultdict,
) -> tuple[xr.Dataset | None, np.ndarray, defaultdict, xr.Dataset]:
    """Find altimeter points within the time window around SAR-WV acquisition.

    Args:
        sar_dataset: WV dataset.
        subset_alti: Subset of the initial altimeter dataset.
        delta_t_max_minutes: Time window in minutes.
        altidb: 'cci' or 'cmems'.
        cpt: Counter.

    Returns:
        Tuple of matching altimeter dataset (closest point), array of matching
        altimeter filenames, updated counter, and the dataset of all altimeter
        points within the time and space criteria.
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

    Returns:
        Tuple of (colocated_observations, coloc_listing, cpt).
    """
    logger.debug("SAR Sentinel-1 WV SAFE to process : %s ", safewv)
    colocated_observations = xr.Dataset({"empty": (["time_sar"], [])})
    cat_alti_mathcups_colocs_ds = []
    measurement_wv_list = glob.glob(os.path.join(safewv, "measurement", "*.nc"))
    logger.debug("Number of measurement in the SAFE : %d", len(measurement_wv_list))
    tmpsarmeasu = []
    for iiwv in tqdm(range(len(measurement_wv_list)), disable=True):
        tmpsarmeasu.append(
            preprocess_wv_s1_ocn(xr.open_dataset(measurement_wv_list[iiwv]))
        )
    sar_dataset_safe = xr.concat(tmpsarmeasu, dim="time_sar").load()
    logger.debug("all SAR files loaded")
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
    if altidb == "cci":
        path_altimeter = os.path.join(PATH_ALT[altidb])
        swh_varname = "swh_denoised"
        acronym_alti_path_ifr = alt.split("_")[1]
    elif altidb == "cmems":
        path_altimeter = os.path.join(
            PATH_ALT[altidb] % POSSIBLES_CMEMS_ALTI[alt.split("_")[1]]
        )
        swh_varname = "VAVH"
        acronym_alti_path_ifr = POSSIBLES_CMEMS_ALTI[alt.split("_")[1]]
        if acronym_alti_path_ifr == "swon":
            acronym_alti_path_ifr = "swot"
    else:
        raise ValueError(error_altidb % altidb)
    return path_altimeter, acronym_alti_path_ifr, swh_varname


def core_coloc(
    day_analyzed: str,
    alt: str,
    sarunit: str,
    outputdir: str,
    conf: dict,
    dev: bool = False,
    redo: bool = False,
    progressbar: bool = False,
) -> defaultdict:
    """Core colocation routine.

    Args:
        day_analyzed: Date to analyze in YYYYMMDD format.
        alt: Altimeter name (e.g., 'cmems_al').
        sarunit: SAR mission (S1A, S1B).
        outputdir: Output directory.
        conf: Configuration dictionary.
        dev: Development mode flag.
        redo: If True, redo existing files.
        progressbar: If True, show progress bar.

    Returns:
        Counter with statistics.
    """
    date = datetime.datetime.strptime(day_analyzed, "%Y%m%d")
    cpt = defaultdict(int)
    Y = date.strftime("%Y")
    JY = date.strftime("%j")
    altidb = alt.split("_")[0]

    path_altimeter, acronym_alti_path_ifr, _ = get_path_alti(altidb, alt, conf=conf)
    logger.info("path_altimeter : %s", path_altimeter)
    assert os.path.exists(path_altimeter)
    assert os.path.exists(conf["path_SAR"])
    long_name_sar_unit = "sentinel-1" + sarunit[-1].lower()
    pattern_sar = os.path.join(
        conf["path_SAR"],
        long_name_sar_unit,
        "L2",
        "WV",
        sarunit + "_WV_OCN__2S",
        Y,
        JY,
        "*.SAFE",
    )
    logger.info("SAR ESA Level-2 OCN SAFE pattern : %s", pattern_sar)
    lst_wv_safe_sorted = sorted(glob.glob(pattern_sar))
    logger.info("%s SAR WV SAFE found", len(lst_wv_safe_sorted))
    output_nc_file = os.path.join(
        outputdir,
        sarunit + "_" + alt,
        date.strftime("%Y"),
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
        + "_km.nc",
    )
    time.sleep(rng.integers(0, 10))
    os.makedirs(os.path.dirname(output_nc_file), 0o0775, exist_ok=True)
    if os.path.exists(output_nc_file) and redo is False:
        logger.info("output coloc S1-WV alti file already exists (redo is False)")
        sys.exit(0)

    coloc_listing = {}
    list_alti_in_raw_time_window = step_1_temp_match(
        date_sar_dt=date,
        path_altimeters=path_altimeter,
        acro_alti=acronym_alti_path_ifr,
        altidb=altidb,
    )
    if len(list_alti_in_raw_time_window) > 0:
        ds_alti, tree_alti = read_all_alti_files(
            liste_altimeter_files=list_alti_in_raw_time_window,
            altidatabase=altidb,
            conf=conf,
        )
    else:
        ds_alti = None
        tree_alti = None
        logger.info("no altimeter files found in the time window around the SAR SAFE")
    if len(lst_wv_safe_sorted) and ds_alti is not None and len(ds_alti["time"]) > 0:
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
                progressbar=progressbar,
                conf=conf,
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
                output_lst_file = os.path.join(
                    outputdir,
                    sarunit + "_" + alt,
                    date.strftime("%Y"),
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
                    + "_km.lst",
                )
                write_coloc_listing(output_lst_file, coloc_listing, redo=redo)
    else:
        logger.info(
            "no SAR WV data for %s or no altimeter data matching this date",
            day_analyzed,
        )
    return cpt


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
        choices=["cmems_" + kk for kk in POSSIBLES_CMEMS_ALTI]
        + ["cci_" + kk for kk in POSSIBLES_CCI_ALTI],
        help="cmems_al,cmems_c2,cci_jason-3...",
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
    cpt = core_coloc(
        sarunit=args.sat,
        alt=args.alt,
        outputdir=args.outputdir,
        dev=args.dev,
        day_analyzed=args.startdate,
        redo=args.redo,
        progressbar=args.progressbar,
        conf=config,
    )
    logger.info("memory in Mo: %s", getrusage(RUSAGE_SELF).ru_maxrss / 1000.0)
    logger.info("counters: %s", cpt)
    logger.info("analysis done in %1.1f sec", time.time() - tinit)
    logger.info("end.")


if __name__ == "__main__":
    entrypoint()
