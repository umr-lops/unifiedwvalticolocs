"""
author: Antoine Grouazel
Script to create a NetCDF with colocation data's from ALT
 and WV OCN datasets

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

from unifiedwvalticolocs.utils import get_conf_content

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
t1 = time.time()
parser = argparse.ArgumentParser()

# CCI key:(subdir,beautiful sat name)
POSSIBLES_CCI_ALTI = {
    "cfosat": ("cfosat", "CFOSAT"),
    "envisat": ("envisat", "Envisat"),
    "ers-1": ("ers-1", "ERS-1"),
    "ers-2": ("ers-2", "ERS-2"),
    "gfo": ("gfo", "GFO"),
    "cryosat-2": ("cryosat-2", "CryoSat-2"),
    "jason-1": ("jason-1", "Jason-1"),
    "jason-2": ("jason-2", "Jason-2"),
    "jason-3": ("jason-3", "Jason-3"),
    "sentinel-3_a": ("sentinel-3_a", "Sentinel-3_A"),
    "sentinel-3_b": ("sentinel-3_b", "Sentinel-3_B"),
    # "sentinel-6": ("sentinel-6", "Sentinel-6_A"),
    "sentinel-6_a": ("sentinel-6_a", "Sentinel-6_A"),
    "topex-poseidon_poseidon": ("topex-poseidon_poseidon", "Topex-Poseidon"),
    "topex-poseidon_topex": ("topex-poseidon_topex", "Topex-Poseidon"),
    "saral": ("saral", "SARAL"),
    "swot": ("swot", "SWOT"),
}
POSSIBLES_CMEMS_ALTI = {
    "SARAL": "al",
    "cryosat-2": "c2",
    "CFOSAT": "cfo",
    "Jason-3": "j3",
    "Sentinel-3A": "s3a",
    "Sentinel-3B": "s3b",
    "HY2B": "h2b",
    "HY2C": "h2c",
    "Sentinel-6A": "s6a",
    "SWOT-Nadir": "swon",
}

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
# define the list of variables of interest to be copy/pasted from CCI altimeter files
CCI_VARS_OF_INTEREST = [
    "distance_to_coast",
    "bathymetry",
    "sea_ice_fraction",
    "era5_total_column_cloud_liquid_water",
    "era5_2m_air_temperature",
    "era5_sea_surface_temperature",
    "era5_wind_eastward",
    "era5_wind_northward",
    "era5_surface_pressure",
    "era5_windwave_swh",
    "era5wave_swh1",
    "era5wave_swh2",
    "era5wave_swh",
    "era5_swell_mean_period",
    "era5_mean_wave_period",
    "era5_windwave_period",
    "era5_peak_wave_period",
    "era5wave_mwd1",
    "era5wave_mwd2",
    "era5_mean_wave_direction",
    "era5_windwave_direction",
    "ww3_mean_wave_period_t0m1",
    "ww3_emb",
    "ww3_swh",
    "ww3_wavenumber_peakdness",
    "ww3_mean_wave_period",
    "ww3_wave_skewness",
    "ww3_mean_wave_direction",
    "ww3_peak_wave_period",
]


def from_npdt64_to_dt(dt64):
    # Convertir le numpy.datetime64 en timestamp (secondes depuis epoch)
    ref_date = np.datetime64("1970-01-01T00:00:00")
    ts = (dt64 - ref_date) / np.timedelta64(1, "s")
    # Créer un datetime "timezone-aware" en UTC (nouvelle méthode recommandée)
    dt = datetime.datetime.fromtimestamp(ts, datetime.UTC)

    return dt


def uf_from_npdt64_to_dt(a):
    return xr.apply_ufunc(from_npdt64_to_dt, a)


def step_0_get_sar_dt(sards):
    """
    :return:date_sar_dt: (datetime.datetime) return the datetime of
     the first measure of the SAR file
    """
    t0 = time.time()
    list_date_sar_dt = []
    logger.debug("step 0: get SAR dates")
    for xtimewv in range(len(sards["time_sar"])):  # loop to run alltime
        # log in the file
        date_sar = sards["time_sar"].values[xtimewv]
        dt = from_npdt64_to_dt(date_sar)
        list_date_sar_dt.append(dt)
    elapsed = time.time() - t0
    logger.debug("step0 done in %1.2f sec", elapsed)
    return list_date_sar_dt


def step_1_temp_match_cci(date_sar_dt, path_altimeters, acro_alti):
    """
    get all alti files in the range [D-1,D,D+1] with D the date of SAR acquisition.

    :param date_sar_dt:SAR acquisition time  ( datetime )
    :param path: Altimeter's dataset path (string)
    :param acro_alti str 2 letters

    :return: final_list_alti (list) each string is ALT's dataset path
    """

    final_list_alti = []
    sta = date_sar_dt - datetime.timedelta(
        days=1
    )  # I take a margin of 1 day to miss no files in the following rrule.rrule
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
            "ESACCI-SEASTATE-L2P-SWH-%s-%sT*-fv01.nc"
            % (POSSIBLES_CCI_ALTI[acro_alti][1], dd.strftime("%Y%m%d")),
        )
        logger.debug("pattern alti : %s", path_glob)
        final_list_alti += sorted(
            glob.glob(path_glob)
        )  # gather all ALT file within sta and sto range
    logger.debug("nb CCI files alti to read: %s", len(final_list_alti))
    logger.debug("output listing of alti: %s", final_list_alti)
    return final_list_alti


def step_1_temp_match(date_sar_dt, path_altimeters, acro_alti, altidb) -> str:
    """

    wrapper to handle both cmems and cci altimeter database

    :param date_sar_dt: datetime.datetime
    :param path_altimeters:  str
    :param acro_alti: str j2 or jason-3 or al ...
    :param altidb: str cci or cmems
    :return:
        final_list_alti (String array) each string is ALT's dataset path
    """
    if altidb == "cci":
        final_list_alti = step_1_temp_match_cci(date_sar_dt, path_altimeters, acro_alti)
    elif altidb == "cmems":
        final_list_alti = step_1_temp_match_cmems(
            date_sar_dt, path_altimeters, acro_alti
        )
    else:
        raise ValueError(error_altidb % altidb)
    return final_list_alti


def is_cmems_file_matching_in_time(
    one_nc_file_alti, lst_nc_files_alti_timematchup, groups_dates, sta, sto
):
    """
    Test whether an alti file is matching with a time window.
    If yes -> add the file to a list returned.

    Args:
        one_nc_file_alti (str): Path to the altimeter file.
        lst_nc_files_alti_timematchup (list): List of matching files.
        groups_dates (dict): Dictionary of dates.
        sta (datetime.datetime): Start time.
        sto (datetime.datetime): Stop time.

    Returns:
        tuple: A tuple containing:
            - lst_nc_files_alti_timematchup (list): Updated list.
            - groups_dates (dict): Updated dictionary.

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
    # if (
    #     (date_alt_sta >= start and date_alt_sto <= stop)
    #     or (start <= date_alt_sta <= stop)
    #     or (start <= date_alt_sto <= stop)
    #     or (start >= date_alt_sta and stop <= date_alt_sto)
    # ):
    if (  # consider all the files +/-1days (finer time sub-setting in step 2)
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
        ):  # remove duplicates
            lst_nc_files_alti_timematchup.append(one_nc_file_alti)
    return lst_nc_files_alti_timematchup, groups_dates


def step_1_temp_match_cmems(date_sar_dt, path_altimeters, acro_alti):
    """


    find the CMEMS L3 altimeters nc files in the range [D-1,D,D+1] with D the date of SAR acquisition.
    and pick up the lastest generated files in case of duplicates.


    :param date_sar_dt:SAR acquisition time  ( datetime )
    :param path: Altimeter's dataset path (string)
    :param acro_alti str 2 letters
    :return:
        lst_nc_files_alti_timematchup (list) each string
        is ALT's dataset path
    """
    ymdthms = "%Y%m%dT%H%M%S"
    ymdth = "%Y%m%dT%H"
    ymd = "%Y%m%d"
    lst_nc_files_alti_timematchup = []
    lst_nc_files_alti_sorted = []
    sta = date_sar_dt - datetime.timedelta(
        days=1
    )  # I take a margin of 1 day to miss no files in the following rrule.rrule
    sto = date_sar_dt + datetime.timedelta(days=1)

    # If sta and sto are naive, make them aware (assuming UTC)
    sta = sta.replace(tzinfo=timezone.utc)
    sto = sto.replace(tzinfo=timezone.utc)
    # logger.debug('path_altimeters : %s',path_altimeters)
    for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
        path_glob = os.path.join(
            path_altimeters,
            dd.strftime("%Y"),
            dd.strftime("%m"),
            f"global_vavh_l3_rt_{acro_alti}_{dd.strftime(ymd)}T*.nc",
        )
        lst_nc_files_alti_sorted += sorted(
            glob.glob(path_glob)
        )  # gather all ALT file within sta and sto range
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
    # browse all the files and pick up the latest generated files
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


def preproc_cmems_alti_files(ds):
    """
    add fname variables associated to each times to be able to have
      the filenames colocated

    :param ds: xr.Dataset
    :return:
        ds
    """
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    return ds


def preproc_cciseastate_alti_files(ds):
    """
    add fname variables associated to each times to be able
      to have the filenames colocated

    :param ds: xr.Dataset
    :return:
        ds
    """
    ds.load()
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    # keep only the data with good quality flag
    # swh_quality_level, flag_values = 0b, 1b, 2b, 3b ;flag_meanings = "undefined bad acceptable good" ;
    mask_good = ds["swh_quality_level"] == 3
    pct_good = 100.0 * mask_good.sum() / mask_good.size
    ds = ds.where(mask_good, drop=True)
    return ds, pct_good


def read_all_alti_files(liste_altimeter_files, altidatabase, conf):
    """
    read the altimeter files to get a xr.Dataset
    Apply preprocessing to add the fname variable associated to each times
    Apply a KDTree to be able to get the closest points in space
    Apply a filter to remove bad longitudes (lon>360) and convert lon>180 to lon-360
    Apply a filter to remove duplicates in time (keep the first one)
    Remove points with NaN in lon or lat

    Arguments:
        liste_altimeter_files (list): List of altimeter file paths.
        altidatabase (str): 'cci' or 'cmems'.
        conf (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - subset_alti1 (xarray.Dataset): Subset of the altimeter dataset.
            - tree_alti (scipy.spatial.KDTree): KDTree for spatial queries.

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
    # ds_alti = xr.open_mfdataset(
    #     liste_altimeter_files, combine="by_coords", preprocess=fctpreprocess,compat="override"
    # )
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
    # ds_alti = xr.open_mfdataset(
    #         liste_altimeter_files, preprocess=fctpreprocess
    #     )
    # add the version of the database CCI or cmems
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
    # points_alt = np.c_[subset_alti1[lat_varname], subset_alti1[lon_varname]]
    # tree_alti = KDTree(points_alt)

    points_alti_xyz = latlon_to_xyz(
        subset_alti1[lat_varname],
        subset_alti1[lon_varname],
    )

    tree_alti = cKDTree(points_alti_xyz)

    logger.debug("alti files loaded, number of points: %s", len(subset_alti1["time"]))
    logging.info("counter: %s", dict(counter))
    return subset_alti1, tree_alti


def latlon_to_xyz(lat, lon):
    """Convert latitude/longitude in degrees to unit-sphere XYZ."""
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


def step_2_geographic_match(sards, ds_alti, tree_alti, delta_dist_km):
    """
    Get altimeter points within a Haversine distance around SAR points.

    Parameters
    ----------
    sards : xarray.Dataset
        Dataset of a given WV image.
    ds_alti : xarray.Dataset
        Altimeter dataset.
    tree_alti : scipy.spatial.cKDTree
        KDTree built from altimeter coordinates converted to XYZ.
    delta_dist_km : float
        Maximum geographic distance in km.

    Returns
    -------
    xarray.Dataset or None
        Altimeter measurements within delta_dist_km of at least one SAR point.
    """

    points_sar_xyz = latlon_to_xyz(
        sards["oswLat"].values,
        sards["oswLon"].values,
    )

    # Convert Haversine radius to chord radius on unit sphere
    r_tree = 2 * np.sin(delta_dist_km / (2 * EARTH_RADIUS_KM))

    queryballpoint = tree_alti.query_ball_point(
        points_sar_xyz,
        r=r_tree,
    )

    # Merge matches from all SAR points
    queryballpoint = np.unique(np.concatenate(queryballpoint))

    if len(queryballpoint) > 0:
        return ds_alti.isel(time=queryballpoint)

    return None


def get_distances_v2(sar_dataset, subset_ok_match_alti, lon_varname, lat_varname):
    """
    Compute distances between SAR center and Alti points.

    Args:
        sar_dataset (xarray.Dataset): The SAR dataset.
        subset_ok_match_alti (xarray.Dataset): The Alti dataset subset.
        lon_varname (str): Variable name for Longitude in Alti ds.
        lat_varname (str): Variable name for Latitude in Alti ds.

    Returns:
        np.array: Array of distances in km.

    """
    t0 = time.time()
    lons_alt = subset_ok_match_alti[lon_varname]
    lats_alt = subset_ok_match_alti[lat_varname]
    # date_sar_dt = date_sar_dt.replace(tzinfo=None)
    # lonsar = sar_dataset.sel(time_sar=date_sar_dt)["oswLon"].values
    # latsar = sar_dataset.sel(time_sar=date_sar_dt)["oswLat"].values
    lonsar = sar_dataset["oswLon"].values
    latsar = sar_dataset["oswLat"].values
    lonssartiled = np.tile(lonsar, (lons_alt.size))
    latssartiled = np.tile(latsar, (lons_alt.size))
    logger.debug("lons_alt %s,lonssartiled %s ", lons_alt.shape, lonssartiled.shape)
    all_dists = haversine(lonssartiled, latssartiled, lons_alt.values, lats_alt.values)
    logger.debug("time  to get distances v2 : %1.2f sec", (time.time() - t0))
    return all_dists


def step_3_closer_temp_match(
    sar_dataset, subset_alti, delta_t_max_minutes, altidb, cpt
):
    """
    Find the altimeter points within the time window around SAR-WV acquisition.

    Args:
        sar_dataset (xarray.Dataset): WV dataset.
        subset_alti (xarray.Dataset): Subset of the initial ALTI dataset.
        delta_t_max_minutes (int): Time windows range (int in minute).
        altidb (str): 'cci' or 'cmems'.
        cpt (defaultdict): counter

    Returns:
        tuple: List of matching points, closest times, distances, etc.

    Raises:
        ValueError: If altidb is not 'cci' or 'cmems'.
    """

    swh_varname = VAR_NAMES[altidb]["swh_varname"]
    lon_varname = VAR_NAMES[altidb]["lon_varname"]
    lat_varname = VAR_NAMES[altidb]["lat_varname"]
    subset_ok_match_alti = None
    # delta_t_closest_in_space = np.nan
    # hs_alti_closest = np.nan
    # delta_d_closest_in_space = np.nan
    # closest_lon_alti = np.nan
    # closest_lat_alti = np.nan
    # closest_time = np.nan
    # lat_alti = []
    # lon_alti = []
    list_alti_files_timespace_match = []

    # 1. Get Alti Times as numpy datetime64 [ns]
    dates_alt_dt64 = subset_alti["time"].values
    if dates_alt_dt64.ndim == 0:
        dates_alt_dt64 = np.array([dates_alt_dt64])

    # 2. Convert SAR Date to numpy datetime64 [ns]
    # We strip timezone info to ensure compatibility with numpy's naive arithmetic
    # (assuming both are effectively UTC)
    # sar_dt64 = np.datetime64(date_sar_dt.replace(tzinfo=None))
    sar_dt64 = sar_dataset.time_sar.values

    # 3. Calculate absolute difference in seconds directly
    # This avoids the date2num epoch confusion entirely
    diffs_times_seconds = np.abs((dates_alt_dt64 - sar_dt64) / np.timedelta64(1, "s"))

    # 4. Filter
    mask_time_ok = diffs_times_seconds < delta_t_max_minutes * 60
    ind_closest_in_time = np.argmin(diffs_times_seconds)

    inds_ok_alti = np.flatnonzero(mask_time_ok)
    cpt[
        "maximum_colocs_time_and_space"
    ] += (
        mask_time_ok.sum()
    )  # this is bigger than the final number of colocation annotated in nc files, because we select only the closest in time

    if len(inds_ok_alti) > 0:
        # subset_ok_match_alti = subset_alti.isel(time=inds_ok_alti)
        # if we have points within the time window (typically 25min), then we only take the closest in time.
        subset_ok_match_alti = subset_alti.isel(time=ind_closest_in_time)
        # all_dists2 = get_distances_v2(
        #     sar_dataset, subset_ok_match_alti, lon_varname, lat_varname
        # )
        # ind_closest_in_dist = np.argmin(all_dists2)
        # delta_d_closest_in_space = all_dists2[ind_closest_in_dist]
        delta_d_closest_in_time = get_distances_v2(
            sar_dataset, subset_ok_match_alti, lon_varname, lat_varname
        )

        # closest_time = subset_ok_match_alti["time"]#.isel({'time':ind_closest_in_time}).values
        delta_t_closest_in_time = (subset_ok_match_alti["time"] - sar_dt64).astype(
            "timedelta64[s]"
        )
        subset_ok_match_alti["delta_t_closest"] = delta_t_closest_in_time
        subset_ok_match_alti["delta_t_closest"].attrs[
            "description"
        ] = "delta Time altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria"
        # subset_ok_match_alti['delta_t_closest'].attrs['units'] = 'seconds'
        subset_ok_match_alti["delta_d_closest"] = xr.DataArray(
            delta_d_closest_in_time[0], dims=()
        )
        subset_ok_match_alti["delta_d_closest"].attrs[
            "description"
        ] = "delta distance altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria"
        subset_ok_match_alti["delta_d_closest"].attrs["units"] = "km"
        # subset_ok_match_alti['delta_t_closest'] = xr.DataArray([delta_t_closest_in_time.values],
        #                                                        attrs={
        #                                                            'source':'coloc alti-wv',
        #                                                            "description":"delta Time altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria",
        #                                                            'units':'seconds'
        #                                                        })
        # subset_ok_match_alti['delta_d_closest'] = xr.DataArray([delta_d_closest_in_time.values],
        #                                                        attrs={
        #                                                            'source':'coloc alti-wv',
        #                                                            "description":"delta space altimeter-SAR for the closest altimeter point in time among the subset matching the coloc criteria",
        #                                                            'units':'km'
        #                                                        })

        # rename the SWH varname into a generic hs_alti_closest
        subset_ok_match_alti = subset_ok_match_alti.rename(
            {
                swh_varname: "hs_alti_closest",
                lon_varname: "lon_ALT",
                lat_varname: "lat_ALT",
            }
        )
        # Use simple indexing based on the subset we just created
        # hs_alti_closest = subset_ok_match_alti.isel(time=ind_closest_in_dist)[
        #     swh_varname
        # ].values

        # lat_alti = subset_ok_match_alti[lat_varname].values
        # lon_alti = subset_ok_match_alti[lon_varname].values

        # Note: No need to re-subtract 360 here if it was done in read_all_alti_files
        # But keeping it safe:
        # lon_alti[(lon_alti > 180)] -= 360.0

        # closest_lon_alti = lon_alti[ind_closest_in_dist]
        # closest_lat_alti = lat_alti[ind_closest_in_dist]
        #

        # delta_t_closest_in_space = (closest_time - sar_dt64).astype("timedelta64[s]")

        list_alti_files_timespace_match = np.unique(subset_alti["fname"])
    return subset_ok_match_alti, list_alti_files_timespace_match, cpt
    # return (
    #     list_alti_pts_matching_space_and_time,
    #     delta_t_closest_in_space,
    #     hs_alti_closest,
    #     lat_alti,
    #     lon_alti,
    #     delta_d_closest_in_space,
    #     closest_lon_alti,
    #     closest_lat_alti,
    #     closest_time,
    #     list_alti_files_timespace_match,
    #     subset_ok_match_alti,
    # )


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def save_coloc_netcdf_file(ds_colocations, output_nc_file):
    """

    :param ds_colocations: xarray dataset
    :param output_nc_file: str
    :return:
    """
    new_file_written = False
    if not os.path.exists(output_nc_file):
        if len(ds_colocations["oswLon"]) > 0:
            logger.info("start writting netCDF")

            # ds = xr.Dataset()
            # ds["lat_SAR"] = ds_colocations["oswLat"].assign_attrs(
            #     {
            #         "units": "degrees_north",
            #         "long_name": "SAR latitude",
            #         "standard_name": "latitude",
            #         "valid_min": -90.0,
            #         "valid_max": 90.0,
            #     }
            # )
            # ds["lon_SAR"] = ds_colocations["oswLon"].assign_attrs(
            #     {
            #         "units": "degrees_east",
            #         "long_name": "SAR longitude",
            #         "standard_name": "longitude",
            #         "valid_min": -180.0,
            #         "valid_max": 180.0,
            #     }
            # )

            # ds["time_ALTI"] = xr.DataArray(
            #     data=ds_colocations["liste_time_alt"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "description": "Alti date of the closest point in space",
            #         "standard_name": "time_ALT",
            #     },
            # )

            # ds["lat_ALT"] = xr.DataArray(
            #     data=ds_colocations["liste_lat_alt"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "degrees_north",
            #         "description": "Latitude",
            #         "standard_name": "Latitude",
            #         "vmin": "-90",
            #         "vmax": "90",
            #     },
            # )
            # ds["lon_ALT"] = xr.DataArray(
            #     data=ds_colocations["liste_lon_alt"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "degrees_east",
            #         "description": "Longitude",
            #         "standard_name": "Longitude",
            #         "vmin": "-180",
            #         "vmax": "180",
            #     },
            # )
            # ds["angle_of_incidence"] = ds_colocations["oswIncidenceAngle"].assign_attrs(
            #     {
            #         "units": "degrees",
            #         "long_name": "SAR incidence angle",
            #         "standard_name": "incidence_angle",
            #         "valid_min": -22.0,
            #         "valid_max": 38.0,
            #     }
            # )
            # ds["heading"] = ds_colocations["oswHeading"].assign_attrs(
            #     {
            #         "units": "degrees",
            #         "long_name": "SAR heading angle",
            #         "standard_name": "platform_heading",
            #         "valid_min": -180.0,
            #         "valid_max": 360.0,
            #     }
            # )

            # ds["oswTotalHs"] = ds_colocations["oswTotalHs"].assign_attrs(
            #     {
            #         "units": "m",
            #         "description": "SAR Sentinel-1 WV C-band significant wave height",
            #         "standard_name": "sea_surface_wave_significant_height",
            #         "vmax": "30",
            #         "vmin": "0",
            #         "coverage_content_type": "physicalMeasurement",
            #         "ancillary_variables": "oswTotalHsStdev",
            #         "band": "C",
            #         "algo": "Quach et al 2020",
            #         "info": "comes from ESA S-1 WV L2 OCN oswTotalHs variable,\
            #             and is comparable to variable swh of present product",
            #     }
            # )
            # ds["oswTotalHsStdev"] = ds_colocations["oswTotalHsStdev"]

            # source_altiwv = (
            #     "altimeter measurement gathered in Ifremer SAR-alti"
            #     " co-location product"
            # )
            # ds["hs_alti_mean"] = xr.DataArray(
            #     data=ds_colocations["liste_mean"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "altimeter mean of "
            #         "significant wave height co-located with SAR",
            #         "source": source_altiwv,
            #     },
            # )
            # ds["hs_alti_std"] = xr.DataArray(
            #     data=ds_colocations["liste_std"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "altimeter standard deviation"
            #         " of significant wave height co-located with SAR",
            #         "source": source_altiwv,
            #     },
            # )
            # ds["hs_alti_count"] = xr.DataArray(
            #     data=ds_colocations["liste_count"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "",
            #         "description": "number of altimeter SAR-co-located points",
            #         "source": source_altiwv,
            #     },
            # )
            # ds["hs_alti_closest"] = xr.DataArray(
            #     data=ds_colocations["liste_closest"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "m",
            #         "source": source_altiwv,
            #         "description": "significant wave"
            #         " height of the closest altimeter point in space",
            #     },
            # )
            # ds["delta_t_closest"] = xr.DataArray(
            #     data=ds_colocations["liste_DELTA_T_closer"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "source": source_altiwv,
            #         "description": "delta Time altimeter-SAR"
            #         " for the altimeter closest point in space",
            #     },
            # )
            # ds["delta_d_closest"] = xr.DataArray(
            #     data=ds_colocations["liste_DELTA_D_closer"],  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "km",
            #         "source": source_altiwv,
            #         "description": "delta space for the altimeter"
            #         " closest point in space",
            #     },
            # )

            # ds.attrs = {
            #     "colocation_institution": "Institut Français pour"
            #     " la Recherche et l Exploitation de la MER",
            #     "colocation_institution_abbreviation": " LOPS-IFREMER",
            #     "colocation_publisher_name": "ifremer/LOPS",
            #     "colocation_publisher_url": "https://www.umr-lops.fr/",
            #     "colocation_publisher_email": "lops-siam@listes.ifremer.fr",
            #     "colocation_product_description": "colocations between Sentinel-1 WV"
            #     " and altimeter coming from CCI sea state or CMEMS database",
            # }
            new_attrs = {
                "colocation_institution": "Institut Français pour la Recherche et l Exploitation de la MER",
                "colocation_institution_abbreviation": " LOPS-IFREMER",
                "colocation_publisher_name": "ifremer/LOPS",
                "colocation_publisher_url": "https://www.umr-lops.fr/",
                "colocation_publisher_email": "lops-siam@listes.ifremer.fr",
                "colocation_product_description": "colocations between Sentinel-1 WV"
                " and altimeter coming from CCI sea state or CMEMS database",
            }
            for kk in new_attrs:
                ds_colocations.attrs[kk] = new_attrs[kk]
            # for attr in ds_colocations.attrs:
            #     ds.attrs[attr] = ds_colocations.attrs[attr]

            logger.info(output_nc_file)
            ds_colocations.to_netcdf(output_nc_file)
            new_file_written = True
        else:
            logger.info("no file to save")
    return new_file_written


# def add_oswtotalhs_to_sar_dataset(sar_wv_ds, sar_unit):
#     """

#     :param sar_wv_ds: xarray.Dataset CCI sea state IFR WV product (orbit file)
#     :param sar_unit: str S1A or ...
#     :return:
#     """
#     all_oswtotalhs = []
#     all_oswtotalhsstdev = []
#     for tt in sar_wv_ds["time"].values:
#         logger.debug("tt : %s", tt)
#         dt = from_npdt64_to_dt(tt)
#         fp_ocn = get_full_path_ocn_wv_from_approximate_date(dt, sar_unit, level="L2")
#         toths = np.nan
#         tothsstdev = np.nan
#         if fp_ocn and os.path.exists(fp_ocn):
#             tmpocn = xr.open_dataset(fp_ocn)
#             if "oswTotalHs" in tmpocn:
#                 toths = tmpocn["oswTotalHs"].values[0][0]
#             if "oswTotalHsStdev" in tmpocn:
#                 tothsstdev = tmpocn["oswTotalHsStdev"].values[0][0]
#         all_oswtotalhs.append(toths)
#         all_oswtotalhsstdev.append(tothsstdev)
#     sar_wv_ds["oswTotalHs"] = xr.DataArray(
#         all_oswtotalhs,
#         dims=["time"],
#         attrs={
#             "description": "values annotated in "
#             "S-1 WV L2 OCN oswTotalHs variable since 2022-06-07 ",
#             "unit": "m",
#             "algo": "Quach et al 2020",
#         },
#     )
#     sar_wv_ds["oswTotalHsStdev"] = xr.DataArray(
#         all_oswtotalhsstdev,
#         dims=["time"],
#         attrs={
#             "description": "values annotated in S-1"
#             " WV L2 OCN oswTotalHsStdev variable since 2022-06-07 ",
#             "unit": "m",
#             "algo": "Quach et al 2020",
#         },
#     )
#     return sar_wv_ds


# def get_original_wv_slc(date_sar, sar_unit):
#     """

#     :param date_sar: adtetime.datetime
#     :param sar_unit: str S1A or S1B or ...
#     :return: str or None
#     """
#     pot_sar_measu = get_full_path_ocn_wv_from_approximate_date(
#         date_sar, sar_unit, level="L1"
#     )
#     return pot_sar_measu


def write_coloc_listing(outputlisting, coloc_listing_data, redo=False):
    """
    the listing will contain fullpathsar, basename alti
    it can contain many times the same SAR file
    (since a single WV can be colocated with different alti files)
    :param outputlisting:
    :param coloc_listing_data:
    :param redo:
    :return:
    """
    if os.path.exists(outputlisting) and redo is False:
        logger.info("%s already exists", outputlisting)
    else:
        fid = open(outputlisting, "w")
        for sarfullpath in coloc_listing_data.keys():
            for altifile_idx in range(len(coloc_listing_data[sarfullpath])):
                if sarfullpath is None:
                    sarfp = "unknown"
                else:
                    sarfp = sarfullpath
                fid.write(
                    sarfp + "," + coloc_listing_data[sarfullpath][altifile_idx] + "\n"
                )
        fid.close()
        logger.info("output listing coloc : %s", outputlisting)


def preprocess_wv_s1_ocn(ds):
    """
    preprocess function to be used in xarray open_mfdataset for S1 WV OCN files
    :param ds:
    :return:
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
    ds = ds.squeeze(["oswRaSize", "oswAzSize"])
    for var in ds.data_vars:
        if ds[var].dims == ():
            ds[var] = ds[var].expand_dims(time_sar=ds.time_sar)

    return ds


def treat_one_measurement_wv(
    sards,
    date_sar_dt,
    sarunit,
    ds_alti,
    tree_alti,
    altidb,
    coloc_listing,
    cpt,
    conf,
):
    """

    Associate a WV OCN measurement with altimeter observation.

    Args:
        sards (xr.Dataset): S1 OCN WV data, contains a unique WV image.
        date_sar_dt (datetime): Contains the WV starting measurement date.
        sarunit (str): S1A or S1B or ...
        ds_alti (xr.Dataset): Altimeter data.
        tree_alti (KDTree): KDTree for altimeter points.
        altidb (str): Altimeter database name (e.g., 'cci' or '
        coloc_listing (dict): To store filepath (meta-coloc or pre-coloc).
        cpt (collection.defaultdict): Counter.
        conf (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing (subset_ok_match_alti, coloc_listing).

    """
    subset_ok_match_alti = None
    cpt["nb_index_sar_browsed"] += 1
    # date_sar_dt = list_date_sar_dt[index_t_sar]
    # fillpath_l1_wv_slc = get_original_wv_slc(date_sar_dt, sar_unit=sarunit)
    fullpath_l2_wv_ocn = sards.encoding["source"]
    # coloc_listing[fullpath_l2_wv_ocn] = []
    subset_alti = step_2_geographic_match(
        sards=sards,
        ds_alti=ds_alti,
        tree_alti=tree_alti,
        delta_dist_km=conf["delta_dist_km"],
    )

    if subset_alti is not None:
        cpt["coloc_in_space"] += len(subset_alti["time"])
        # if subset_alti["time"].values.size > 0:
        # (
        #     list_alti_pts_matching_space_and_time,
        #     delta_t_closest,
        #     hs_alti_closest,
        #     lat_alti,
        #     lon_alti,
        #     delta_d_closer,
        #     closest_lon,
        #     closest_lat,
        #     closest_time,
        #     list_alti_files_timespace_mu,
        #     subset_ok_match_alti,
        # ) = step_3_closer_temp_match(
        #     sar_dataset=sards,
        #     subset_alti=subset_alti,
        #     delta_t_max_minutes=conf["delta_t_minutes"],
        #     altidb=altidb,
        # )
        subset_ok_match_alti, list_alti_files_timespace_mu, cpt = (
            step_3_closer_temp_match(
                sar_dataset=sards,
                subset_alti=subset_alti,
                delta_t_max_minutes=conf["delta_t_minutes"],
                altidb=altidb,
                cpt=cpt,
            )
        )
        # if subset_ok_match_alti is not None:
        #     swh = subset_ok_match_alti[swh_varname].values
        #     swh_count = len(swh)
        # else:
        #     swh = np.array([])
        #     swh_count = 0

        # if swh_count > 0:
        #     # Use a context manager to locally ignore the expected RuntimeWarnings
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings(
        #             "ignore", message="Degrees of freedom <= 0 for slice"
        #         )
        #         warnings.filterwarnings("ignore", message="Mean of empty slice")

        #         swh_mean = np.nanmean(swh)
        #         swh_std = np.nanstd(swh)
        # else:
        #     swh_mean = np.nan
        #     swh_std = np.nan
        if len(list_alti_files_timespace_mu) > 0:
            coloc_listing[fullpath_l2_wv_ocn] = list_alti_files_timespace_mu
            cpt["nb_coloc"] += 1
            subset_ok_match_alti["time_sar"] = date_sar_dt.replace(tzinfo=None)
            #
            # dict4colocs["liste_lat_alt"].append(closest_lat)
            # dict4colocs["liste_lon_alt"].append(closest_lon)
            # dict4colocs["liste_time_alt"].append(closest_time)
            # dict4colocs["times_SAR"].append(date_sar_dt.replace(tzinfo=None))
            # dict4colocs["liste_count"].append(swh_count)
            # dict4colocs["liste_mean"].append(swh_mean)
            # dict4colocs["liste_std"].append(swh_std)
            # dict4colocs["liste_closest"].append(hs_alti_closest)
            # dict4colocs["liste_DELTA_T_closer"].append(delta_t_closest)
            # dict4colocs["liste_DELTA_D_closer"].append(delta_d_closer)
    return subset_ok_match_alti, coloc_listing, cpt


def treat_one_safe_wv(
    safewv,
    ds_alti,
    tree_alti,
    altidb,
    coloc_listing,
    cpt,
    conf,
    dev=False,
    progressbar=True,
):
    """

    Colocate one SAFE OCN WV with altimeters

    :param safewv: path to the SAFE OCN WV to process
    :param ds_alti: xarray Dataset containing the altimeter data
    :param tree_alti: spatial tree for altimeter data
    :param altidb: name of the altimeter database to use (cci or cmems)
    :param coloc_listing: dictionary to store the listing of colocations (key: SAR file, value: list of altimeter files)
    :param cpt: counter to keep track of various statistics (e.g., number of SAR indices with matching altimeter, number of SAR indices without corresponding altimeter files, etc.)
    :param conf: configuration dictionary containing parameters like delta_t_sat, delta_t_sat_short, etc.
    :param dev: True -> break after finding few matchups
    :param progressbar: True to show progressbar, False to disable it
    :return:
    tuple: (colocated_observations, coloc_listing, cpt)
        colocated_observations: xarray.Dataset containing the colocated SAR and altimeter observations
        coloc_listing: updated dictionary with the listing of colocations
        cpt: updated counter with statistics about the colocation process


    """
    logger.debug("SAR Sentinel-1 WV SAFE to process : %s ", safewv)
    colocated_observations = xr.Dataset({"empty": (["time_sar"], [])})
    # dict4colocs = {}
    # dict4colocs["times_SAR"] = []  # list of SAR Datetime
    # dict4colocs["liste_count"] = []  # list of wave
    # dict4colocs["liste_mean"] = []  # list of mean wave
    # dict4colocs["liste_std"] = []  # list of std wave
    # dict4colocs["liste_lat_alt"] = []  # list of lat alt
    # dict4colocs["liste_lon_alt"] = []  # list of lon alt
    # dict4colocs["liste_time_alt"] = []  # list of time alt
    # dict4colocs["liste_closest"] = []  # list closest wave
    # dict4colocs["liste_DELTA_T_closer"] = []
    # dict4colocs["liste_DELTA_D_closer"] = []
    cat_alti_mathcups_colocs_ds = []
    sarunit = os.path.basename(safewv)[0:3]
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

    # get all the altimeter files that are in the raw time window (delta_t_sat_long) around the SAR SAFE
    # this step is done at SAFE level to avoid reading at each measurement the same alti files.

    if ds_alti:
        # cpt["nb_safe_with_alti_files"] += 1
        if progressbar:
            iterratotor = tqdm(range(len(list_date_sar_dt)), desc="WV measurement")
        else:
            iterratotor = range(len(list_date_sar_dt))
        for index_t_sar in iterratotor:  # loop over WV measurements
            # treat a measurement wv here

            alti_point_ds_match, coloc_listing, cpt = treat_one_measurement_wv(
                sar_dataset_safe.isel(time_sar=index_t_sar),
                date_sar_dt=list_date_sar_dt[index_t_sar],
                sarunit=sarunit,
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
            # drop time_sar variable from aggregated_alti_wv_matchups since it was only used to subset the sar dataset
            aggregated_alti_wv_matchups = aggregated_alti_wv_matchups.drop_vars(
                ["time_sar"]
            )
            aggregated_alti_wv_matchups = aggregated_alti_wv_matchups.rename_vars(
                {"time": "time_ALT"}
            )
            aggregated_alti_wv_matchups["time_ALT"] = aggregated_alti_wv_matchups[
                "time_ALT"
            ].astype("datetime64[s]")
            # rename the dim time_alti in the alti dataset to prepare merge and concat

            # aggregated_alti_wv_matchups = aggregated_alti_wv_matchups.rename_dims({'time_alti':'coloc_index'})
            # alti_colocated_ds = xr.Dataset()
            # for vv in dict4colocs:
            #     if vv == "liste_time_alt":
            #         valval = np.array(dict4colocs[vv]).astype("M8[ns]")
            #     elif vv == "liste_DELTA_T_closer":
            #         valval = np.array(dict4colocs[vv]).astype("m8[ns]")
            #     else:
            #         valval = np.array(dict4colocs[vv])
            #     alti_colocated_ds[vv] = xr.DataArray(
            #         valval,
            #         dims=["time_sar"],
            #         coords={"time_sar": colocated_observations["time_sar"].values},
            #     )
            logger.debug("merge alti and SAR colocated values")
            logger.debug("associate SAR and alti information in the same dataset.")
            colocated_observations = xr.merge(
                [colocated_observations, aggregated_alti_wv_matchups], compat="override"
            )
            # attributes added
            list_att = copy.copy(colocated_observations.attrs)
            for att in list_att:
                # rename the existing attributes SAR with a sar_ prefix
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


def get_path_alti(altidb, alt, conf):
    """
    get the path, acronym and Hs variable name of a specific
      altimeter for a given database (altidb)


    """
    cmems_dir = conf["cmems_dir"]
    subset_alti_name_dir = conf["subset_alti_name_dir"]
    PATH_ALT = {
        "cmems": os.path.join(cmems_dir, subset_alti_name_dir),
        "cci": conf["cci_alti_dir"],
        # v4 followed by v4/data/satellite/altimeter/l2p/
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
        if acronym_alti_path_ifr == "swon":  # particular case for SWOT
            acronym_alti_path_ifr = "swot"
    else:
        raise ValueError(error_altidb % altidb)

    return path_altimeter, acronym_alti_path_ifr, swh_varname


def core_coloc(
    day_analyzed,
    alt,
    sarunit,
    outputdir,
    conf,
    dev=False,
    redo=False,
    progressbar=False,
):
    """

    :param day_analyzed: str
    :param alt: str
    :param sarunit: str S1A ,S1B ...
    :param outputdir: str
    :param dev: bool
    :param redo: bool
    :return:
    """
    date = datetime.datetime.strptime(day_analyzed, "%Y%m%d")
    cpt = defaultdict(int)
    Y = date.strftime("%Y")
    JY = date.strftime("%j")
    altidb = alt.split("_")[0]

    path_altimeter, acronym_alti_path_ifr, swh_varname = get_path_alti(
        altidb, alt, conf=conf
    )

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

    # get the alti dataset only once:
    list_alti_in_raw_time_window = step_1_temp_match(
        date_sar_dt=date,
        path_altimeters=path_altimeter,
        acro_alti=acronym_alti_path_ifr,
        altidb=altidb,
    )
    if len(list_alti_in_raw_time_window) > 0:
        # this step is done only once because all the SAR obs
        #  from a day will be associated to the same alti ds
        ds_alti, tree_alti = read_all_alti_files(
            liste_altimeter_files=list_alti_in_raw_time_window,
            altidatabase=altidb,
            conf=conf,
        )

    if len(lst_wv_safe_sorted):
        all_safe_matchups = []
        # for ssi,safewv in enumerate(lst_wv_safe_sorted):
        pbar = tqdm(range(len(lst_wv_safe_sorted)), desc="WV SAFE")
        for ssi in pbar:
            # format the counter cpt in string "key1=value1; key2=value2; ..."
            string_counter = ";".join([f"{key, cpt[key]}" for key in cpt])
            pbar.set_description("WV SAFE : %s" % string_counter)
            safewv = lst_wv_safe_sorted[ssi]
            logger.debug("%i/%i", ssi + 1, len(lst_wv_safe_sorted))
            # treat one safe here
            one_safe_colocs, coloc_listing, cpt = treat_one_safe_wv(
                safewv=safewv,
                ds_alti=ds_alti,
                tree_alti=tree_alti,
                altidb=altidb,
                # acronym_alti_path_ifr=acronym_alti_path_ifr,
                coloc_listing=coloc_listing,
                cpt=cpt,
                dev=dev,
                progressbar=progressbar,
                conf=conf,
            )
            # if len(one_safe_colocs.time)>0 and len(one_safe_colocs.time_sar)>0:
            if len(one_safe_colocs.time_sar) > 0:
                all_safe_matchups.append(one_safe_colocs)
            if dev and cpt["nb_coloc"] > MAX_NB_MATCHUPS_DEV_MODE:
                logger.info("break loops over SAFE after finding few matchups")
                break
        if len(all_safe_matchups) > 0:
            daily_colocated_observations = xr.concat(
                all_safe_matchups, dim="coloc_index"
            )
            # end of the loop over SAR SAFE
            if os.path.exists(output_nc_file) and redo:
                os.remove(output_nc_file)
            output_file_written = save_coloc_netcdf_file(
                daily_colocated_observations, output_nc_file
            )
            if output_file_written:
                logger.info("successfull save output file: %s", output_nc_file)

            if len(daily_colocated_observations["oswLon"]) > 0:
                # write listing coloc
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
        logger.info("no SAR WV data for %s", day_analyzed)
    return cpt


def entrypoint():
    tinit = time.time()
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    parser = argparse.ArgumentParser(description="colocate S1 WV and altimeters")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir",
        # default=DIR_OUTPUT,
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
