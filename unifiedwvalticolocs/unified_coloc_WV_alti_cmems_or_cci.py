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
from s1ifr.get_full_path_from_measurement import (
    get_full_path_ocn_wv_from_approximate_date,
)
from scipy.spatial import KDTree
from tqdm import tqdm

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
# from find_closest_l2anad_in_time.py import find_all_l2anad_between_
# start_and_stop_date
# sys.path.append("/home1/datahome/satwave/sources_en_exploitation2/
# cfosat-calval-exe/")
# Input = '/home/datawork-cersat-public/project/cci-seastate/sandbox/data/sar/
# v3.0/S1A_wv1/2021/001/S1A_wv1_20210101_level2_LOPS_SWH_SAR_v3.0.nc'
# path_SAR = '/home/datawork-cersat-public/project/cci-seastate/sandbox/
# data/sar/v3.0/'
# path_SAR = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/
# analysis/s1_data_analysis/hs_nn/cci_orbit_files/v3.2"
path_SAR = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/"
# path = '/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_
# OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-j3/'
# path_alt = '/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_
# OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-%s'
subset_alti_name_dir = "cmems_obs-wave_glo_phy-swh_nrt_%s-l3_PT1S"
cmems_dir = "/home/ref-cmems-public/tac/wave/WAVE_GLO_PHY_SWH_L3_NRT_014_001/"
PATH_ALT = {
    "cmems": os.path.join(cmems_dir, subset_alti_name_dir),
    # "cci": "/home/datawork-cersat-public/provider/cci_seastate/products/v3/"
    "cci": "/home/ref-cersat-public/ocean-waves/cci-seastate/v4/",
    # v4 followed by v4/data/satellite/altimeter/l2p/
}

DIR_OUT_ROOT = "/home/datawork-cersat-public/cache/project/mpc-sentinel1"
DIR_OUT_SUBDIRS = "analysis/s1_data_analysis/hs_nn/unified_colocs_wv_alti"
DIR_OUTPUT = os.path.join(DIR_OUT_ROOT, DIR_OUT_SUBDIRS)
delta_t_sat = 3  # hours
delta_t_sat_short = 3 * 3600  # in seconds
DELTA_DIST = 2  # degree
MAX_NB_MATCHUPS_DEV_MODE = 3
error_altidb = "altidb %s not handled"
t1 = time.time()
parser = argparse.ArgumentParser()

# CCI key:(subdir,beautiful sat name)
POSSIBLES_CCI_ALTI = {
    "cryosat-2": ("cryosat-2", "CryoSat-2"),
    "jason-2": ("jason-2", "Jason-2"),
    "jason-3": ("jason-3", "Jason-3"),
    "sentinel-3a": ("sentinel-3_a", "Sentinel-3_A"),
    "sentinel-3b": ("sentinel-3_b", "Sentinel-3_B"),
    "sentinel-6": ("sentinel-6", "Sentinel-6_A"),
    "saral": ("saral", "SARAL"),
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
    logging.debug("step 0: get SAR dates")
    for xtimewv in range(len(sards["time_sar"])):  # loop to run alltime
        # log in the file
        date_sar = sards["time_sar"].values[xtimewv]
        dt = from_npdt64_to_dt(date_sar)
        list_date_sar_dt.append(dt)
    elapsed = time.time() - t0
    logging.debug("step0 done in %1.2f sec", elapsed)
    return list_date_sar_dt


def step_1_temp_match_cci(date_sar_dt, delta_t_sat, path_altimeters, acro_alti):
    """
    get all alti files for a given day

    :param date_sar_dt:SAR acquisition time  ( datetime )
    :param delta_t_sat:acquisition Range (int in hour)
    :param path: Alt's dataset path (string)
    :param acro_alti str 2 letters

    :return: final_list_alti (String array) each string is ALT's dataset path
    """

    final_list_alti = []
    start = date_sar_dt - datetime.timedelta(hours=delta_t_sat)
    stop = date_sar_dt + datetime.timedelta(hours=delta_t_sat)
    sta = start - datetime.timedelta(
        days=1
    )  # I take a margin of 1 day to miss no files in the following rrule.rrule
    sto = stop + datetime.timedelta(days=1)
    for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
        path_glob = os.path.join(
            path_altimeters,
            "data",
            "satellite",
            "altimeter",
            "l2p",
            POSSIBLES_CCI_ALTI[acro_alti][0],
            dd.strftime("%Y"),
            dd.strftime("%j"),
            "ESACCI-SEASTATE-L2P-SWH-%s-%sT*-fv01.nc"
            % (POSSIBLES_CCI_ALTI[acro_alti][1], dd.strftime("%Y%m%d")),
        )
        logging.debug("pattern alti : %s", path_glob)
        final_list_alti += sorted(
            glob.glob(path_glob)
        )  # gather all ALT file within sta and sto range
    logging.info("nb CCI files alti to read: %s", len(final_list_alti))
    logging.debug("output listing of alti: %s", final_list_alti)
    return final_list_alti


def step_1_temp_match(
    date_sar_dt, delta_t_sat, path_altimeters, acro_alti, altidb
) -> str:
    """

    wrapper to handle both cmems and cci altimeter database

    :param date_sar_dt: datetime.datetime
    :param delta_t_sat: int
    :param path_altimeters:  str
    :param acro_alti: str j2 or jason-3 or al ...
    :param altidb: str cci or cmems
    :return:
        final_list_alti (String array) each string is ALT's dataset path
    """
    if altidb == "cci":
        final_list_alti = step_1_temp_match_cci(
            date_sar_dt, delta_t_sat, path_altimeters, acro_alti
        )
    elif altidb == "cmems":
        final_list_alti = step_1_temp_match_cmems(
            date_sar_dt, delta_t_sat, path_altimeters, acro_alti
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


def step_1_temp_match_cmems(date_sar_dt, delta_t_sat, path_altimeters, acro_alti):
    """
    :param date_sar_dt:SAR acquisition time  ( datetime )
    :param delta_t_sat:acquisition Range (int in hour)
    :param path: Alt's dataset path (string)
    :param acro_alti str 2 letters
    :return:
        lst_nc_files_alti_timematchup (String array) each string
        is ALT's dataset path
    """
    ymdthms = "%Y%m%dT%H%M%S"
    ymdth = "%Y%m%dT%H"
    ymd = "%Y%m%d"
    lst_nc_files_alti_timematchup = []
    lst_nc_files_alti_sorted = []
    start = date_sar_dt - datetime.timedelta(hours=delta_t_sat)
    stop = date_sar_dt + datetime.timedelta(hours=delta_t_sat)
    sta = start - datetime.timedelta(
        days=1
    )  # I take a margin of 1 day to miss no files in the following rrule.rrule
    sto = stop + datetime.timedelta(days=1)

    # If sta and sto are naive, make them aware (assuming UTC)
    sta = sta.replace(tzinfo=timezone.utc)
    sto = sto.replace(tzinfo=timezone.utc)
    # logging.debug('path_altimeters : %s',path_altimeters)
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
    logging.debug(
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
    logging.debug("output listing of alti: %s", final_list_alti)
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
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    return ds


def read_all_alti_files(liste_altimeter_files, altidatabase):
    """
    read the altimeter files to get a xr.Dataset

    """
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
    ds_alti = xr.open_mfdataset(
        liste_altimeter_files, combine="by_coords", preprocess=fctpreprocess
    )
    tmp_lons = copy.copy(ds_alti[lon_varname].values)
    mask_bad_lon = tmp_lons > 180

    tmp_lons[mask_bad_lon] -= 360.0
    super_bad = tmp_lons > 360
    tmp_lons[super_bad] = np.nan
    logging.debug("tmp_lons : %s %s", np.nanmax(tmp_lons), np.nanmin(tmp_lons))
    ds_alti[lon_varname] = xr.DataArray(
        tmp_lons, dims=["time"], coords={"time": ds_alti["time"].values}
    )
    subset_alti1 = ds_alti.where(np.isfinite(ds_alti[lon_varname]), drop=True)
    points_alt = np.c_[ds_alti[lat_varname], ds_alti[lon_varname]]
    tree_alti = KDTree(points_alt)
    logging.debug("alti files loaded, number of points: %s", len(subset_alti1["time"]))
    return subset_alti1, tree_alti


def step_2_geographic_match(sards, ds_alti, tree_alti):
    """

    get altimeter points that are within a radius around a set of WV images

    :param sards: xarray.Dataset of a given image WV
    :param ds_alti: xarray.core.Dataset altimeter data
    :param altidatabase (str): cci or cmems
    :return:liste_time : (numpy dt64 Array) time measure for each matching ALT
    """
    subset_alti2 = None
    points_sar = np.c_[sards["oswLat"].values, sards["oswLon"].values]

    queryballpoint = tree_alti.query_ball_point(points_sar, r=DELTA_DIST)
    queryballpoint = np.array(queryballpoint[0])
    if len(queryballpoint) > 0:
        subset_alti2 = ds_alti.isel(time=queryballpoint)  # ['time'].values
    return subset_alti2


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
    lons_alt = subset_ok_match_alti[lon_varname].values
    lats_alt = subset_ok_match_alti[lat_varname].values
    # date_sar_dt = date_sar_dt.replace(tzinfo=None)
    # lonsar = sar_dataset.sel(time_sar=date_sar_dt)["oswLon"].values
    # latsar = sar_dataset.sel(time_sar=date_sar_dt)["oswLat"].values
    lonsar = sar_dataset["oswLon"].values
    latsar = sar_dataset["oswLat"].values
    lonssartiled = np.tile(lonsar, (len(lons_alt)))
    latssartiled = np.tile(latsar, (len(lons_alt)))
    logging.debug("lons_alt %s,lonssartiled %s ", lons_alt.shape, lonssartiled.shape)
    all_dists = haversine(lonssartiled, latssartiled, lons_alt, lats_alt)
    logging.debug("time  to get distances v2 : %1.2f sec", (time.time() - t0))
    return all_dists


def step_3_closer_temp_match(sar_dataset, subset_alti, delta_t_sat_short, altidb):
    """
    Find the altimeter points within the time window around SAR-WV acquisition.

    Args:
        sar_dataset (xarray.Dataset): WV dataset.
        subset_alti (xarray.Dataset): Subset of the initial ALTI dataset.
        delta_t_sat_short (int): Time windows range (int in hour).
        altidb (str): 'cci' or 'cmems'.

    Returns:
        tuple: List of matching points, closest times, distances, etc.

    Raises:
        ValueError: If altidb is not 'cci' or 'cmems'.
    """

    list_alti_pts_matching_space_and_time = []

    if altidb == "cci":
        swh_varname = "swh_denoised"
        lon_varname = "lon"
        lat_varname = "lat"
    elif altidb == "cmems":
        swh_varname = "VAVH"
        lon_varname = "longitude"
        lat_varname = "latitude"
    else:
        raise ValueError(error_altidb % altidb)

    delta_t_closest_in_space = np.nan
    hs_alti_closest = np.nan
    delta_d_closest_in_space = np.nan
    closest_lon_alti = np.nan
    closest_lat_alti = np.nan
    closest_time = np.nan
    lat_alti = []
    lon_alti = []
    list_alti_files_timespace_match = []

    # --- FIX STARTS HERE ---

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
    mask_time_ok = diffs_times_seconds < delta_t_sat_short
    list_alti_pts_matching_space_and_time = dates_alt_dt64[mask_time_ok]

    # --- FIX ENDS HERE ---

    inds_ok_alti = np.flatnonzero(mask_time_ok)

    if len(inds_ok_alti) > 0:
        subset_ok_match_alti = subset_alti.isel(time=inds_ok_alti)
        all_dists2 = get_distances_v2(
            sar_dataset, subset_ok_match_alti, lon_varname, lat_varname
        )
        ind_closest_in_dist = np.argmin(all_dists2)
        delta_d_closest_in_space = all_dists2[ind_closest_in_dist]

        # Use simple indexing based on the subset we just created
        hs_alti_closest = subset_ok_match_alti.isel(time=ind_closest_in_dist)[
            swh_varname
        ].values

        lat_alti = subset_ok_match_alti[lat_varname].values
        lon_alti = subset_ok_match_alti[lon_varname].values

        # Note: No need to re-subtract 360 here if it was done in read_all_alti_files
        # But keeping it safe:
        lon_alti[(lon_alti > 180)] -= 360.0

        closest_lon_alti = lon_alti[ind_closest_in_dist]
        closest_lat_alti = lat_alti[ind_closest_in_dist]
        closest_time = subset_ok_match_alti["time"].values[ind_closest_in_dist]

        delta_t_closest_in_space = (closest_time - sar_dt64).astype("timedelta64[s]")

        list_alti_files_timespace_match = np.unique(subset_alti["fname"])

    return (
        list_alti_pts_matching_space_and_time,
        delta_t_closest_in_space,
        hs_alti_closest,
        lat_alti,
        lon_alti,
        delta_d_closest_in_space,
        closest_lon_alti,
        closest_lat_alti,
        closest_time,
        list_alti_files_timespace_match,
    )


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
            logging.info("start writting netCDF")
            ds = xr.Dataset()
            ds["lat_SAR"] = ds_colocations["oswLat"].assign_attrs(
                {
                    "units": "degrees_north",
                    "long_name": "SAR latitude",
                    "standard_name": "latitude",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                }
            )
            ds["lon_SAR"] = ds_colocations["oswLon"].assign_attrs(
                {
                    "units": "degrees_east",
                    "long_name": "SAR longitude",
                    "standard_name": "longitude",
                    "valid_min": -180.0,
                    "valid_max": 180.0,
                }
            )

            ds["time_ALTI"] = xr.DataArray(
                data=ds_colocations["liste_time_alt"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "description": "Alti date of the closest point in space",
                    "standard_name": "time_ALT",
                },
            )

            ds["lat_ALT"] = xr.DataArray(
                data=ds_colocations["liste_lat_alt"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "degrees_north",
                    "description": "Latitude",
                    "standard_name": "Latitude",
                    "vmin": "-90",
                    "vmax": "90",
                },
            )
            ds["lon_ALT"] = xr.DataArray(
                data=ds_colocations["liste_lon_alt"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "degrees_east",
                    "description": "Longitude",
                    "standard_name": "Longitude",
                    "vmin": "-180",
                    "vmax": "180",
                },
            )
            ds["angle_of_incidence"] = ds_colocations["oswIncidenceAngle"].assign_attrs(
                {
                    "units": "degrees",
                    "long_name": "SAR incidence angle",
                    "standard_name": "incidence_angle",
                    "valid_min": -22.0,
                    "valid_max": 38.0,
                }
            )
            ds["heading"] = ds_colocations["oswHeading"].assign_attrs(
                {
                    "units": "degrees",
                    "long_name": "SAR heading angle",
                    "standard_name": "platform_heading",
                    "valid_min": -180.0,
                    "valid_max": 360.0,
                }
            )

            ds["oswTotalHs"] = ds_colocations["oswTotalHs"].assign_attrs(
                {
                    "units": "m",
                    "description": "SAR Sentinel-1 WV C-band significant wave height",
                    "standard_name": "sea_surface_wave_significant_height",
                    "vmax": "30",
                    "vmin": "0",
                    "coverage_content_type": "physicalMeasurement",
                    "ancillary_variables": "oswTotalHsStdev",
                    "band": "C",
                    "algo": "Quach et al 2020",
                    "info": "comes from ESA S-1 WV L2 OCN oswTotalHs variable,\
                        and is comparable to variable swh of present product",
                }
            )
            ds["oswTotalHsStdev"] = ds_colocations["oswTotalHsStdev"]

            source_altiwv = (
                "altimeter measurement gathered in Ifremer SAR-alti"
                " co-location product"
            )
            ds["hs_alti_mean"] = xr.DataArray(
                data=ds_colocations["liste_mean"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "description": "altimeter mean of "
                    "significant wave height co-located with SAR",
                    "source": source_altiwv,
                },
            )
            ds["hs_alti_std"] = xr.DataArray(
                data=ds_colocations["liste_std"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "description": "altimeter standard deviation"
                    " of significant wave height co-located with SAR",
                    "source": source_altiwv,
                },
            )
            ds["hs_alti_count"] = xr.DataArray(
                data=ds_colocations["liste_count"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "",
                    "description": "number of altimeter SAR-co-located points",
                    "source": source_altiwv,
                },
            )
            ds["hs_alti_closest"] = xr.DataArray(
                data=ds_colocations["liste_closest"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "source": source_altiwv,
                    "description": "significant wave"
                    " height of the closest altimeter point in space",
                },
            )
            ds["delta_t_closest"] = xr.DataArray(
                data=ds_colocations["liste_DELTA_T_closer"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "source": source_altiwv,
                    "description": "delta Time altimeter-SAR"
                    " for the altimeter closest point in space",
                },
            )
            ds["delta_d_closest"] = xr.DataArray(
                data=ds_colocations["liste_DELTA_D_closer"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "km",
                    "source": source_altiwv,
                    "description": "delta space for the altimeter"
                    " closest point in space",
                },
            )

            ds.attrs = {
                "institution": "Institut Français pour"
                " la Recherche et l Exploitation de la MER",
                "institution_abbreviation": " LOPS-IFREMER",
                "publisher_name": "ifremer/LOPS",
                "publisher_url": "https://www.umr-lops.fr/",
                "publisher_email": "lops-siam@listes.ifremer.fr",
                "product_description": "colocations between WV"
                " and altimeter coming from CCi sea state or CMEMS database",
            }

            logging.info(output_nc_file)
            ds.to_netcdf(output_nc_file)
            new_file_written = True
        else:
            logging.info("no file to save")
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
#         logging.debug("tt : %s", tt)
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


def get_original_wv_slc(date_sar, sar_unit):
    """

    :param date_sar: adtetime.datetime
    :param sar_unit: str S1A or S1B or ...
    :return: str or None
    """
    pot_sar_measu = get_full_path_ocn_wv_from_approximate_date(
        date_sar, sar_unit, level="L1"
    )
    return pot_sar_measu


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
        logging.info("%s already exists", outputlisting)
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
        logging.info("output listing coloc : %s", outputlisting)


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
            logging.debug("variable %s is not present in S1 WV OCN file", vv)

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
    list_date_sar_dt,
    sarunit,
    index_t_sar,
    altidb,
    coloc_listing,
    dict4colocs,
    cpt,
    path_altimeter,
    acronym_alti_path_ifr,
    swh_varname,
):
    """

    Associate a WV OCN measurement with altimeter observation.

    Args:
        sards (xr.Dataset): S1 OCN WV data, contains a unique WV image.
        list_date_sar_dt (list): Contains the WV starting measurement dates.
        sarunit (str): S1A or S1B or ...
        index_t_sar (int): Index of SAR WV in the sards or list_date_sar_dt.
        altidb (str): cmems or cci.
        coloc_listing (dict): To store filepath (meta-coloc or pre-coloc).
        dict4colocs (dict): Contain the altimeters values.
        cpt (collection.defaultdict): Counter.
        path_altimeter (str): Directory where altimeter files are stored.
        acronym_alti_path_ifr (str): Acronym for folder path.
        swh_varname (str): Variable name for altimeter SWH.

    Returns:
        tuple: A tuple containing (dict4colocs, coloc_listing).

    """
    ds_alti = None
    cpt["nb_index_sar_browsed"] += 1
    date_sar_dt = list_date_sar_dt[index_t_sar]
    fillpath_l1_wv_slc = get_original_wv_slc(date_sar_dt, sar_unit=sarunit)
    coloc_listing[fillpath_l1_wv_slc] = []
    liste_step1 = step_1_temp_match(
        date_sar_dt,
        delta_t_sat,
        path_altimeters=path_altimeter,
        acro_alti=acronym_alti_path_ifr,
        altidb=altidb,
    )
    if len(liste_step1) > 0:
        # this step is done only once because all the SAR obs
        #  from a day will be associated to the same alti ds
        ds_alti, tree_alti = read_all_alti_files(
            liste_altimeter_files=liste_step1, altidatabase=altidb
        )

    if ds_alti:
        subset_alti = step_2_geographic_match(
            sards=sards,
            ds_alti=ds_alti,
            tree_alti=tree_alti,
        )
        if subset_alti is not None:
            # if subset_alti["time"].values.size > 0:
            (
                list_alti_pts_matching_space_and_time,
                delta_t_closest,
                hs_alti_closest,
                lat_alti,
                lon_alti,
                delta_d_closer,
                closest_lon,
                closest_lat,
                closest_time,
                list_alti_files_timespace_mu,
            ) = step_3_closer_temp_match(
                sar_dataset=sards,
                subset_alti=subset_alti,
                delta_t_sat_short=delta_t_sat_short,
                altidb=altidb,
            )
            swh = subset_alti.sel(time=list_alti_pts_matching_space_and_time)[
                swh_varname
            ].values
            swh_count = len(swh)

            if swh_count > 0:
                # Use nanmean/nanstd to safely handle NaNs if present in the data
                swh_mean = np.nanmean(swh)
                swh_std = np.nanstd(swh)
            else:
                swh_mean = np.nan
                swh_std = np.nan
            if len(list_alti_pts_matching_space_and_time) > 0:
                coloc_listing[fillpath_l1_wv_slc] = list_alti_files_timespace_mu
                cpt["nb_index_sar_with_matching_alti"] += 1
                dict4colocs["liste_lat_alt"].append(closest_lat)
                dict4colocs["liste_lon_alt"].append(closest_lon)
                dict4colocs["liste_time_alt"].append(closest_time)
                dict4colocs["times_SAR"].append(date_sar_dt.replace(tzinfo=None))
                dict4colocs["liste_count"].append(swh_count)
                dict4colocs["liste_mean"].append(swh_mean)
                dict4colocs["liste_std"].append(swh_std)
                dict4colocs["liste_closest"].append(hs_alti_closest)
                dict4colocs["liste_DELTA_T_closer"].append(delta_t_closest)
                dict4colocs["liste_DELTA_D_closer"].append(delta_d_closer)

    else:
        cpt["nb_index_sar_without_alti_file_corresponding"] += 1
        logging.debug("no files found")
    return dict4colocs, coloc_listing, cpt


def treat_one_safe_wv(
    safewv,
    path_altimeter,
    altidb,
    acronym_alti_path_ifr,
    swh_varname,
    coloc_listing,
    cpt,
    dev=False,
    progressbar=True,
):
    """

    Colocate one SAFE OCN WV with altimeters

    :param safewv: Description
    :param path_altimeter: Description
    :param altidb: Description
    :param acronym_alti_path_ifr: Description
    :param swh_varname: Description
    :param coloc_listing: Description
    :param cpt: Description
    :param dev: True -> break after finding few matchups
    :param progressbar: Description


    """
    logging.debug("SAR Sentinel-1 WV SAFE to process : %s ", safewv)
    dict4colocs = {}
    dict4colocs["times_SAR"] = []  # list of SAR Datetime
    dict4colocs["liste_count"] = []  # list of wave
    dict4colocs["liste_mean"] = []  # list of mean wave
    dict4colocs["liste_std"] = []  # list of std wave
    dict4colocs["liste_lat_alt"] = []  # list of lat alt
    dict4colocs["liste_lon_alt"] = []  # list of lon alt
    dict4colocs["liste_time_alt"] = []  # list of time alt
    dict4colocs["liste_closest"] = []  # list closest wave
    dict4colocs["liste_DELTA_T_closer"] = []
    dict4colocs["liste_DELTA_D_closer"] = []
    sarunit = os.path.basename(safewv)[0:3]
    measurement_wv_list = glob.glob(os.path.join(safewv, "measurement", "*.nc"))
    logging.debug("Number of measurement in the SAFE : %d", len(measurement_wv_list))
    tmpsarmeasu = []
    for iiwv in tqdm(range(len(measurement_wv_list)), disable=True):
        tmpsarmeasu.append(
            preprocess_wv_s1_ocn(xr.open_dataset(measurement_wv_list[iiwv]))
        )
    sar_dataset_safe = xr.concat(tmpsarmeasu, dim="time_sar").load()
    logging.debug("all SAR files loaded")
    list_date_sar_dt = step_0_get_sar_dt(sards=sar_dataset_safe)
    if progressbar:
        iterratotor = tqdm(range(len(list_date_sar_dt)), desc="WV measurement")
    else:
        iterratotor = range(len(list_date_sar_dt))
    for index_t_sar in iterratotor:  # loop over WV measurements
        # treat a measurement wv here
        dict4colocs, coloc_listing, cpt = treat_one_measurement_wv(
            sar_dataset_safe.isel(time_sar=index_t_sar),
            list_date_sar_dt,
            sarunit,
            index_t_sar=index_t_sar,
            altidb=altidb,
            coloc_listing=coloc_listing,
            dict4colocs=dict4colocs,
            cpt=cpt,
            path_altimeter=path_altimeter,
            acronym_alti_path_ifr=acronym_alti_path_ifr,
            swh_varname=swh_varname,
        )
        if dev and cpt["nb_index_sar_with_matching_alti"] > MAX_NB_MATCHUPS_DEV_MODE:
            logging.info("break loops over measurements after finding few matchups")
            break
    logging.debug("end of pair construction")
    colocated_observations = sar_dataset_safe.sel(time_sar=dict4colocs["times_SAR"])

    alti_colocated_ds = xr.Dataset()
    for vv in dict4colocs:
        if vv == "liste_time_alt":
            valval = np.array(dict4colocs[vv]).astype("M8[ns]")
        elif vv == "liste_DELTA_T_closer":
            valval = np.array(dict4colocs[vv]).astype("m8[ns]")
        else:
            valval = np.array(dict4colocs[vv])
        alti_colocated_ds[vv] = xr.DataArray(
            valval,
            dims=["time_sar"],
            coords={"time_sar": colocated_observations["time_sar"].values},
        )
    logging.debug("merge alti and SAR colocated values")
    colocated_observations = xr.merge([colocated_observations, alti_colocated_ds])
    return colocated_observations, coloc_listing, cpt


def get_path_alti(altidb, alt):
    """
    get the path, acronym and Hs variable name of a specific
      altimeter for a given database (altidb)


    """
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
    startdate, alt, sarunit, outputdir, dev=False, redo=False, progressbar=False
):
    """

    :param startdate:datetime.datetime
    :param alt: str
    :param sarunit: str S1A ,S1B ...
    :param outputdir: str
    :param dev: bool
    :param redo: bool
    :return:
    """
    date = datetime.datetime.strptime(startdate, "%Y%m%d")
    cpt = defaultdict(int)
    Y = date.strftime("%Y")
    JY = date.strftime("%j")
    altidb = alt.split("_")[0]

    path_altimeter, acronym_alti_path_ifr, swh_varname = get_path_alti(altidb, alt)

    logging.info("path_altimeter : %s", path_altimeter)
    assert os.path.exists(path_altimeter)
    assert os.path.exists(path_SAR)
    long_name_sar_unit = "sentinel-1" + sarunit[-1].lower()
    pattern_sar = os.path.join(
        path_SAR,
        long_name_sar_unit,
        "L2",
        "WV",
        sarunit + "_WV_OCN__2S",
        Y,
        JY,
        "*.SAFE",
    )
    logging.info("SAR ESA CCI Sea state Ifr pattern : %s", pattern_sar)
    lst_wv_safe_sorted = sorted(glob.glob(pattern_sar))
    logging.info("%s SAR WV SAFE found", len(lst_wv_safe_sorted))
    output_nc_file = os.path.join(
        outputdir,
        sarunit + "_" + alt,
        date.strftime("%Y"),
        "coloc_"
        + startdate
        + "_"
        + sarunit
        + "_WV_"
        + alt
        + "_"
        + str(delta_t_sat)
        + "_hours_"
        + str(DELTA_DIST)
        + "_degree.nc",
    )
    time.sleep(rng.integers(0, 10))
    os.makedirs(os.path.dirname(output_nc_file), 0o0775, exist_ok=True)
    if os.path.exists(output_nc_file) and redo is False:
        logging.info("output coloc S1-WV alti file already exists (redo is False)")
        sys.exit(0)

    coloc_listing = {}

    if len(lst_wv_safe_sorted):
        all_safe_matchups = []
        # for ssi,safewv in enumerate(lst_wv_safe_sorted):
        pbar = tqdm(range(len(lst_wv_safe_sorted)), desc="WV SAFE")
        for ssi in pbar:
            pbar.set_description(
                "WV SAFE : nb colocs %i" % cpt["nb_index_sar_with_matching_alti"]
            )
            safewv = lst_wv_safe_sorted[ssi]
            logging.debug("%i/%i", ssi + 1, len(lst_wv_safe_sorted))
            # treat one safe here
            one_safe_colocs, coloc_listing, cpt = treat_one_safe_wv(
                safewv,
                path_altimeter,
                altidb,
                acronym_alti_path_ifr,
                swh_varname,
                coloc_listing,
                cpt=cpt,
                dev=dev,
                progressbar=progressbar,
            )
            # if len(one_safe_colocs.time)>0 and len(one_safe_colocs.time_sar)>0:
            if len(one_safe_colocs.time_sar) > 0:
                all_safe_matchups.append(one_safe_colocs)
            if (
                dev
                and cpt["nb_index_sar_with_matching_alti"] > MAX_NB_MATCHUPS_DEV_MODE
            ):
                logging.info("break loops over SAFE after finding few matchups")
                break
        if len(all_safe_matchups) > 0:
            daily_colocated_observations = xr.concat(all_safe_matchups, dim="time_sar")
            # end of the loop over SAR SAFE
            if os.path.exists(output_nc_file) and redo:
                os.remove(output_nc_file)
            output_file_written = save_coloc_netcdf_file(
                daily_colocated_observations, output_nc_file
            )
            if output_file_written:
                logging.info("successfull save output file: %s", output_nc_file)

            if len(daily_colocated_observations["oswLon"]) > 0:
                # write listing coloc
                output_lst_file = os.path.join(
                    outputdir,
                    sarunit + "_" + alt,
                    date.strftime("%Y"),
                    "coloc_"
                    + startdate
                    + "_"
                    + sarunit
                    + "_WV_"
                    + alt
                    + "_"
                    + str(delta_t_sat)
                    + "_hours_"
                    + str(DELTA_DIST)
                    + "_degree.lst",
                )
                write_coloc_listing(output_lst_file, coloc_listing, redo=redo)
    else:
        logging.info("no SAR WV data for %s", startdate)
    return cpt


def entrypoint():
    tinit = time.time()
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    parser = argparse.ArgumentParser(description="example main")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir",
        default=DIR_OUTPUT,
        help="folder where the data will be written [optional]",
        required=False,
    )
    parser.add_argument("--startdate", required=True, help="YYYYMMDD", type=str)
    parser.add_argument("--sat", required=True, help="S1A or S1B...", type=str)
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
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S"
        )
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S")
    logging.info(
        "Start of execution for script %s using "
        "WV Level-2 OCN and altimeters from "
        "CCI sea state L2P or CMEMS WAV L3",
        os.path.basename(__file__),
    )
    logging.info("development/test mode activated: %s", args.dev)
    cpt = core_coloc(
        sarunit=args.sat,
        alt=args.alt,
        outputdir=args.outputdir,
        dev=args.dev,
        startdate=args.startdate,
        redo=args.redo,
        progressbar=args.progressbar,
    )
    logging.info("memory in Mo: %s", getrusage(RUSAGE_SELF).ru_maxrss / 1000.0)
    logging.info("counters: %s", cpt)
    logging.info("analysis done in %1.1f sec", time.time() - tinit)
    logging.info("end.")


if __name__ == "__main__":
    entrypoint()
