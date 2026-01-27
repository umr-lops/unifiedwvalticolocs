"""
Antoine Grouazel
12 July 2021
inspired from Gabriel Morand work and tuned for CCI WV v3.2 fot
Script to create a NetCDF with colocation data's from Jason-3 ALT and CCI SAR v3.0 dataset
"""
import netCDF4
import os
import sys
import copy
import numpy as np
import xarray as xr
from tqdm import tqdm
import glob
import pandas as pd
import datetime
from datetime import timezone
import time
import logging
from collections import defaultdict
import argparse
from scipy.spatial import KDTree
from dateutil import rrule
from resource import getrusage, RUSAGE_SELF
from s1ifr.get_full_path_from_measurement import (
    get_full_path_ocn_wv_from_approximate_date,
)
import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning
)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(
    action="ignore", message="invalid value encountered in true_divide"
)
warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice")
# from find_closest_l2anad_in_time.py import find_all_l2anad_between_start_and_stop_date
# sys.path.append("/home1/datahome/satwave/sources_en_exploitation2/cfosat-calval-exe/")
# Input = '/home/datawork-cersat-public/project/cci-seastate/sandbox/data/sar/v3.0/S1A_wv1/2021/001/S1A_wv1_20210101_level2_LOPS_SWH_SAR_v3.0.nc'
# path_SAR = '/home/datawork-cersat-public/project/cci-seastate/sandbox/data/sar/v3.0/'
#path_SAR = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/cci_orbit_files/v3.2"
path_SAR = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/"
# path = '/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-j3/'
# path_alt = '/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-%s'
PATH_ALT = {
    "cmems": "/home/ref-cmems-public/tac/wave/WAVE_GLO_PHY_SWH_L3_NRT_014_001/cmems_obs-wave_glo_phy-swh_nrt_%s-l3_PT1S",
    # "cci": "/home/datawork-cersat-public/provider/cci_seastate/products/v3/", # v3
    "cci": "/home/ref-cersat-public/ocean-waves/cci-seastate/v4/", #v4 followed by v4/data/satellite/altimeter/l2p/
}
# DIR_OUTPUT = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/cci_orbit_files/v3.2_colocations_unified_v1/"  # oct 2023
DIR_OUTPUT = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/unified_colocs_wv_alti"
DELTA_T_SAT = 3  # hours
DELTA_T_SAT_SHORT = 3 * 3600  # in seconds
DELTA_DIST = 2  # degree

t1 = time.time()
parser = argparse.ArgumentParser()
# CCI key:(subdir,beautiful sat name)
POSSIBLES_CCI_ALTI = {
    "cryosat-2": ("cryosat-2", "CryoSat-2"),
    #'envisat':'ENVISATe',
    #'jason-1':'Jason-1',
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
    "SWOT-Nadir": "swon"
}


# def from_npdt64_to_dt(dt64):
#     ref_date = np.datetime64(datetime.datetime(1970, 1, 1))
#     # np.datetime64('1970-01-01T00:00:00Z') # DeprecationWarning: parsing timezone aware datetimes is deprecated
#     ts = (dt64 - ref_date) / np.timedelta64(1, "s")
#     dt = datetime.datetime.utcfromtimestamp(ts)

#     return dt

def from_npdt64_to_dt(dt64):
    # Convertir le numpy.datetime64 en timestamp (secondes depuis epoch)
    ref_date = np.datetime64("1970-01-01T00:00:00")
    ts = (dt64 - ref_date) / np.timedelta64(1, "s")
    # Créer un datetime "timezone-aware" en UTC (nouvelle méthode recommandée)
    dt = datetime.datetime.fromtimestamp(ts, datetime.UTC)

    return dt


def uf_from_npdt64_to_dt(a):
    return xr.apply_ufunc(from_npdt64_to_dt, a)


def Step_0_get_SAR_dt(sards):
    """
    :return:date_SAR_dt: (datetime.datetime) return the datetime of the first mesure of the SAR file
    """
    t0 = time.time()
    liste_date_SAR_dt = []
    logging.debug("step 0: get SAR dates")
    for xtimeWV in range(len(sards["time_sar"])):  # loop to run alltime log in the file
        date_SAR = sards["time_sar"].values[xtimeWV]
        dt = from_npdt64_to_dt(date_SAR)
        liste_date_SAR_dt.append(dt)
    # liste_date_SAR_dt = uf_from_npdt64_to_dt(sards["time"].values)
    # liste_date_SAR_dt = sards["time"].apply_ufunc(from_npdt64_to_dt)
    elapsed = time.time() - t0
    logging.debug("step0 done in %1.2f sec", elapsed)
    return liste_date_SAR_dt  # return list of all time mesruement in the SAR file


def Step_1_temp_match_cci(date_SAR_dt, DELTA_T_SAT, path_altimeters, acro_alti):
    """
    get all alti files for a given day
    :param date_SAR_dt:SAR acquisition time  ( datetime )
    :param DELTA_T_SAT:acquisition Range (int in hour)
    :param path: Alt's dataset path (string)
    :param acro_alti str 2 letters
    :return: final_list_alti (String array) each string is ALT's dataset path
    """
    # logging.debug('Step 1: get list of alti files matching +/- 1 day around SAR dates')

    final_list_alti = []
    start = date_SAR_dt - datetime.timedelta(hours=DELTA_T_SAT)
    stop = date_SAR_dt + datetime.timedelta(hours=DELTA_T_SAT)
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


def Step_1_temp_match(date_SAR_dt, DELTA_T_SAT, path_altimeters, acro_alti, altidb)->str:
    """

    wrapper to handle both cmems and cci altimeter database

    :param date_SAR_dt: datetime.datetime
    :param DELTA_T_SAT: int
    :param path_altimeters:  str
    :param acro_alti: str j2 or jason-3 or al ...
    :param altidb: str cci or cmems
    :return:
        final_list_alti (String array) each string is ALT's dataset path
    """
    if altidb == "cci":
        final_list_alti = Step_1_temp_match_cci(
            date_SAR_dt, DELTA_T_SAT, path_altimeters, acro_alti
        )
    elif altidb == "cmems":
        final_list_alti = Step_1_temp_match_cmems(
            date_SAR_dt, DELTA_T_SAT, path_altimeters, acro_alti
        )
    else:
        raise Exception("altidb %s not handled" % altidb)
    return final_list_alti


def Step_1_temp_match_cmems(date_SAR_dt, DELTA_T_SAT, path_altimeters, acro_alti):
    """
    :param date_SAR_dt:SAR acquisition time  ( datetime )
    :param DELTA_T_SAT:acquisition Range (int in hour)
    :param path: Alt's dataset path (string)
    :param acro_alti str 2 letters
    :return: liste (String array) each string is ALT's dataset path
    """
    # logging.debug('Step 1: get list of alti files matching +/- 1 day around SAR dates')

    liste = []
    ALT_DATA = []
    start = date_SAR_dt - datetime.timedelta(hours=DELTA_T_SAT)
    stop = date_SAR_dt + datetime.timedelta(hours=DELTA_T_SAT)
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
            "global_vavh_l3_rt_%s_%sT*.nc" % (acro_alti, dd.strftime("%Y%m%d")),
        )
        ALT_DATA += sorted(
            glob.glob(path_glob)
        )  # gather all ALT file within sta and sto range
    groups_dates = {}
    for gg in ALT_DATA:
        date_ALT_sta = datetime.datetime.strptime(
            os.path.basename(gg).split("_")[5], "%Y%m%dT%H%M%S"
        )
        date_ALT_sto = datetime.datetime.strptime(
            os.path.basename(gg).split("_")[6], "%Y%m%dT%H%M%S"
        )
        generation_date_ALT_sto = datetime.datetime.strptime(
            os.path.basename(gg).split("_")[7].replace(".nc", ""), "%Y%m%dT%H%M%S"
        )
        if date_ALT_sta.strftime("%Y%m%dT%H") not in groups_dates:
            groups_dates[date_ALT_sta.strftime("%Y%m%dT%H")] = [generation_date_ALT_sto]
        else:
            groups_dates[date_ALT_sta.strftime("%Y%m%dT%H")].append(
                generation_date_ALT_sto
            )
        date_ALT_sta = date_ALT_sta.replace(tzinfo=timezone.utc)
        date_ALT_sto = date_ALT_sto.replace(tzinfo=timezone.utc)
        # if (
        #     (date_ALT_sta >= start and date_ALT_sto <= stop)
        #     or (start <= date_ALT_sta <= stop)
        #     or (start <= date_ALT_sto <= stop)
        #     or (start >= date_ALT_sta and stop <= date_ALT_sto)
        # ):
        if (  # consider all the files +/-1days (finer time sub-setting if done in step 2)
            (date_ALT_sta >= sta and date_ALT_sto <= sto)
            or (sta <= date_ALT_sta <= sto)
            or (sta <= date_ALT_sto <= sto)
            or (sta >= date_ALT_sta and sto <= date_ALT_sto)
        ):
            if (
                datetime.datetime.strptime(
                    os.path.basename(gg).split("_")[5], "%Y%m%dT%H%M%S"
                )
                not in liste
            ):  # remove duplicates
                liste.append(gg)
    logging.debug("liste : %s", len(liste))
    # browse all the files and pick up the latest generated files
    final_list_alti = []
    for uu in liste:
        date_ALT_sta = datetime.datetime.strptime(
            os.path.basename(uu).split("_")[5], "%Y%m%dT%H%M%S"
        )
        max_group = np.amax(np.array(groups_dates[date_ALT_sta.strftime("%Y%m%dT%H")]))
        generation_date_ALT_sto = datetime.datetime.strptime(
            os.path.basename(uu).split("_")[7].replace(".nc", ""), "%Y%m%dT%H%M%S"
        )
        if max_group == generation_date_ALT_sto:
            final_list_alti.append(uu)
        else:
            pass
            # logging.debug('pas trouved : %s',max_group)
    logging.debug("output listing of alti: %s", final_list_alti)
    return final_list_alti


def preproc_cmems_alti_files(ds):
    """
    add fname variables associated to each times to be able to have the filenames colocated
    :param ds:
    :return:
    """
    filee = ds.encoding["source"]
    tmpfname = np.empty(ds["time"].shape, dtype="O")
    tmpfname[:] = os.path.basename(filee)
    ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    return ds


def preproc_cciseastate_alti_files(ds):
    """
    add fname variables associated to each times to be able to have the filenames colocated
    :param ds:
    :return:
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
        raise Exception("altidb %s not handled" % altidatabase)
    # ds_alti = xr.open_mfdataset(liste_altimeter_files, combine='by_coords')
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
    points_ALT = np.c_[ds_alti[lat_varname], ds_alti[lon_varname]]
    tree_ALT = KDTree(points_ALT)
    logging.debug('alti files loaded, number of points: %s', len(subset_alti1['time']))
    return subset_alti1, tree_ALT


def step_2_geographic_match(sards, ds_alti, tree_ALT, time_sar_index, altidatabase):
    """

    :param ds_alti: xarray.core.Dataset altimeter data
    :param altidatabase (str): cci or cmems
    :param time_sar_index (int)
    :return:liste_time : (numpy dt64 Array) time measure for each matching ALT
    """
    if altidatabase == "cci":
        lon_varname = "lon"
        lat_varname = "lat"
    elif altidatabase == "cmems":
        lon_varname = "longitude"
        lat_varname = "latitude"
    else:
        raise Exception("altidb %s not handled" % altidatabase)
    subset_alti2 = None
    points_SAR = np.c_[
        sards["oswLat"].values[time_sar_index], sards["oswLon"].values[time_sar_index]
    ]

    queryballpoint = tree_ALT.query_ball_point(points_SAR, r=DELTA_DIST)
    queryballpoint = np.array(queryballpoint[0])
    if len(queryballpoint) > 0:
        subset_alti2 = ds_alti.isel(time=queryballpoint)  # ['time'].values
    return subset_alti2


# def get_distances_v1(inds_ok_alti,SARdataset,subset_ok_match_alti,lon_varname,lat_varname,date_SAR_dt):
#     t0 = time.time()
#     all_dists = []
#     for ind_selected_alti_pts in range(len(inds_ok_alti)):
#         Dist = haversine(
#             SARdataset.sel(time=date_SAR_dt)["lon"].values,
#             SARdataset.sel(time=date_SAR_dt)["lat"].values,
#             subset_ok_match_alti.isel(time=ind_selected_alti_pts)[lon_varname].values,
#             subset_ok_match_alti.isel(time=ind_selected_alti_pts)[lat_varname].values,
#         )
#         all_dists.append(Dist)
#     all_dists = np.array(all_dists)
#     logging.info("time get_distances_v1 : %1.2f sec", (time.time() - t0))
#     return all_dists


def get_distances_v2(
    SARdataset, subset_ok_match_alti, date_SAR_dt, lon_varname, lat_varname
):
    t0 = time.time()
    lons_alt = subset_ok_match_alti[lon_varname].values
    lats_alt = subset_ok_match_alti[lat_varname].values
    date_SAR_dt = date_SAR_dt.replace(tzinfo=None)
    lonsar = SARdataset.sel(time_sar=date_SAR_dt)["oswLon"].values
    latsar = SARdataset.sel(time_sar=date_SAR_dt)["oswLat"].values
    lonssartiled = np.tile(lonsar, (len(lons_alt)))
    latssartiled = np.tile(latsar, (len(lons_alt)))
    logging.debug("lons_alt %s,lonssartiled %s ", lons_alt.shape, lonssartiled.shape)
    all_dists = haversine(lonssartiled, latssartiled, lons_alt, lats_alt)
    logging.debug("time get_distances_v2 : %1.2f sec", (time.time() - t0))
    return all_dists


def step_3_closer_temp_match(
    SARdataset, subset_alti, date_SAR_dt, DELTA_T_SAT_SHORT, altidb
):
    """
    find the altimeter points that are within the time window around SAr acquisition.

    Args:
        SARdataset (xarray.Dataset):  WV 
        subset_alti (xarray.Dataset):  subset of the initial ALTI dataset (only points selected at geographic match)
        date_SAR_dt (datetime):  SAR acquisition time  ( utf datetime )
        DELTA_T_SAT_SHORT (int): time windows range (int in hour), e.g. co-locations are within +/-DELTA_T_SAT_SHORT
        altidb (str): 'cci' or 'cmems
    Returns:
        list_alti_pts_matching_space_and_time (String array) each string is a measurement TIME from an ALTI between +1 and -1 hour from the date_SAR_dt
        DELTA_T_closer (String array) Closest matching measurement in time
        HS_closer (Float) Closest VAVH measurement
        DELTA_D_closer (Float) Closest matching measurement in Space
        lon_alt (Array) Array of matching Lon
        lat_alt (Array) Array of matching Lat
        closest_lon_alti: float
        closest_lat_alti: float
        closest_time (np.datetime64) : time of alti for which we can find the closest distance in space wrt to WV
        list_alti_files_timespace_match: list of str basename of altimeter paths CMEMS or CCI sea state that match in time and space with WV
    """
    list_alti_pts_matching_space_and_time = []
    # ds_alt = xr.open_mfdataset(liste, combine='by_coords')
    # logging.info(liste_alt)
    # DELTA_D_closest = 20000 #arbitrary value set to have a value minimum for closest distance
    if altidb == "cci":
        swh_varname = "swh_denoised"
        lon_varname = "lon"
        lat_varname = "lat"

    elif altidb == "cmems":
        swh_varname = "VAVH"
        lon_varname = "longitude"
        lat_varname = "latitude"
    else:
        raise Exception("altidb %s not handled" % altidb)

    DELTA_T_closest = np.timedelta64(DELTA_T_SAT_SHORT, "s")
    HS_closest = np.nan
    DELTA_D_closest = np.nan
    closest_lon_alti = np.nan
    closest_lat_alti = np.nan
    closest_time = np.nan
    lat_alti = []
    lon_alti = []
    list_alti_files_timespace_match = []
    UNITS_TIME = "seconds since 2010-01-01"
    dates_ALT_dt64 = subset_alti["time"].values.squeeze()
    if dates_ALT_dt64.size == 1:
        dates_ALT_dt64 = np.array([dates_ALT_dt64])
    dates_ALT_dt = np.array([pd.Timestamp(jj) for jj in dates_ALT_dt64])
    sar_date_num = netCDF4.date2num(date_SAR_dt, UNITS_TIME, calendar="standard")
    # dates_ALT_dt = np.apply_along_axis(pd.Timestamp,0,dates_ALT_dt64)
    dates_ALT_num = netCDF4.date2num(dates_ALT_dt, UNITS_TIME, calendar="standard")
    diffs_times = abs(dates_ALT_num - sar_date_num)
    list_alti_pts_matching_space_and_time = subset_alti["time"].values[
        (diffs_times < DELTA_T_SAT_SHORT)
    ]
    inds_ok_alti = np.where(diffs_times < DELTA_T_SAT_SHORT)[0]
    if len(inds_ok_alti) > 0:
        subset_ok_match_alti = subset_alti.isel(time=inds_ok_alti)
        # all_dists = get_distances_v1(
        #     inds_ok_alti,
        #     SARdataset,
        #     subset_ok_match_alti,
        #     lon_varname,
        #     lat_varname,
        #     date_SAR_dt,
        # )
        all_dists2 = get_distances_v2(
            SARdataset, subset_ok_match_alti, date_SAR_dt, lon_varname, lat_varname
        )
        ind_closest_in_dist = np.argmin(all_dists2)
        DELTA_D_closest = all_dists2[ind_closest_in_dist]
        HS_closest = subset_ok_match_alti.isel(time=ind_closest_in_dist)[
            swh_varname
        ].values
        lat_alti = subset_alti.sel(time=list_alti_pts_matching_space_and_time)[
            lat_varname
        ].values
        lon_alti = subset_alti.sel(time=list_alti_pts_matching_space_and_time)[
            lon_varname
        ].values
        lon_alti[
            (lon_alti > 180)
        ] -= 360.0  # because CMEMS data is between 0 and 360. deg
        closest_lon_alti = lon_alti[ind_closest_in_dist]
        closest_lat_alti = lat_alti[ind_closest_in_dist]
        closest_time = list_alti_pts_matching_space_and_time[ind_closest_in_dist]
        date_SAR_dt_naive = date_SAR_dt.replace(tzinfo=None)
        DELTA_T_closest = (
            closest_time - np.datetime64(date_SAR_dt_naive)
        ).astype("timedelta64[s]")
        # DELTA_T_closest = (closest_time - np.datetime64(date_SAR_dt)).astype("<m8[s]") # drop silently the tz -> warning raised
        list_alti_files_timespace_match = np.unique(subset_alti["fname"])
    return (
        list_alti_pts_matching_space_and_time,
        DELTA_T_closest,
        HS_closest,
        lat_alti,
        lon_alti,
        DELTA_D_closest,
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
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # c = 2 * asin(sqrt(a))
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def save_coloc_netCDF_file(
    ds_colocations, output_nc_file):
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
            ds["lat_SAR"] = ds_colocations["oswLat"].assign_attrs({
                    "units": "degrees_north",
                    "long_name": "SAR latitude",
                    "standard_name": "latitude",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                })
            # ds["lat_SAR"] = xr.DataArray(
            #     data=ds_colocations["oswLat"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time_sar"].values},
            #     attrs={
            #         "units": "degrees_north",
            #         "description": "latitude",
            #         "standard_name": "latitude",
            #         "vmin": "-90",
            #         "vmax": "90",
            #     },
            # )
            ds["lon_SAR"] = ds_colocations["oswLon"].assign_attrs({
                "units": "degrees_east",
                "long_name": "SAR longitude",
                "standard_name": "longitude",
                "valid_min": -180.0,
                "valid_max": 180.0,
            })
            # ds["lon_SAR"] = xr.DataArray(
            #     data=ds_colocations["oswLon"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "degrees_east",
            #         "description": "Longitude",
            #         "standard_name": "Longitude",
            #         "vmin": "-180",
            #         "vmax": "180",
            #     },
            # )

            ds["time_ALTI"] = xr.DataArray(
                data=ds_colocations["liste_time_alt"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    #'units': 'time',
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
            ds["angle_of_incidence"] = ds_colocations["oswIncidenceAngle"].assign_attrs({
                "units": "degrees",
                "long_name": "SAR incidence angle",
                "standard_name": "incidence_angle",
                "valid_min": -22.0,
                "valid_max": 38.0,
            })
            # ds["angle_of_incidence"] = xr.DataArray(
            #     data=ds_colocations["oswIncidenceAngle"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "degrees",
            #         "description": "incidence angle",
            #         "standard_name": "incidence_angle",
            #         "vmin": "22",
            #         "vmax": "38",
            #     },
            # )
            ds["heading"] = ds_colocations["oswHeading"].assign_attrs({
                "units": "degrees",
                "long_name": "SAR heading angle",
                "standard_name": "platform_heading",
                "valid_min": -180.0,
                "valid_max": 360.0,
            })
            # ds["heading"] = xr.DataArray(
            #     data=ds_colocations["oswHeading"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "degrees",
            #         "description": "heading angle",
            #         "standard_name": "platform heading",
            #         "vmin": "360",
            #         "vmax": "180",
            #     },
            # )
            # ds["swh"] = xr.DataArray(
            #     data=ds_colocations["swh"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "C band significant wave height",
            #         "standard_name": "sea_surface_wave_significant_height",
            #         "vmax": "30",
            #         "vmin": "0",
            #         "coverage_content_type": "physicalMeasurement",
            #         "ancillary_variables": "swh_quality swh_rejection_flags",
            #         "band": "C",
            #         "source": "CCI Sea state IFREMER SAR S1 WV dataset",
            #     },
            # )
            ds["oswTotalHs"] = ds_colocations["oswTotalHs"].assign_attrs({
                "units": "m",
                "description": "SAR Sentinel-1 WV C-band significant wave height",
                "standard_name": "sea_surface_wave_significant_height",
                "vmax": "30",
                "vmin": "0",
                "coverage_content_type": "physicalMeasurement",
                "ancillary_variables": "oswTotalHsStdev",
                "band": "C",
                "algo": "Quach et al 2020",
                "info": "comes from ESA S-1 WV L2 OCN oswTotalHs variable, and is comparable to variable swh of present product",
            })
            # ds["oswTotalHs"] = xr.DataArray(
            #     data=ds_colocations["oswTotalHs"].values,
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "C band significant wave height",
            #         "standard_name": "sea_surface_wave_significant_height",
            #         "vmax": "30",
            #         "vmin": "0",
            #         "coverage_content_type": "physicalMeasurement",
            #         "ancillary_variables": "swh_quality swh_rejection_flags",
            #         "band": "C",
            #         "info": "comes from ESA S-1 WV L2 OCN oswTotalHs variable, and is comparable to variable swh of present product",
            #     },
            # )
            # ds["oswTotalHsStdev"] = xr.DataArray(
            #     ds_colocations["oswTotalHsStdev"].values,
            #     dims=["time_sar"],
            #     attrs=ds_colocations["oswTotalHs"].attrs,
            # )
            ds["oswTotalHsStdev"] = ds_colocations["oswTotalHsStdev"]
            # ds["swh_uncertainty"] = xr.DataArray(
            #     data=ds_colocations["swh_uncertainty"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "standard deviation associated to hs : level of confidence of the NN model",
            #         "standard_name": "swh_uncertainty",
            #         "vmin": "0",
            #         "vmax": "6",
            #         "source": "CCI Sea state IFREMER SAR S1 WV dataset",
            #     },
            # )
            # ds["swh_quality"] = xr.DataArray(
            #     data=ds_colocations["swh_quality"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "units": "m",
            #         "description": "quality of C band significant wave height measurement",
            #         "standard_name": "swh_quality",
            #         "flag_values": "0L, 1L, 2L, 3L",
            #         "flag_meanings": "undefined bad acceptable good",
            #         "coverage_content_type": "qualityInformation",
            #         "band": "C",
            #         "source": "CCI Sea state IFREMER SAR S1 WV dataset",
            #     },
            # )
            # ds["swh_rejection_flags"] = xr.DataArray(
            #     data=ds_colocations["swh_rejection_flags"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "unit": "none",
            #         "description": "consolidated instrument and ice flags",
            #         "standard_name": "swh_rejection_flags",
            #         "flag_masks": "1L, 2L, 4L, 8L, 16L, 32L, 64L, 128L",
            #         "flag_meanings": "nb_of_valid_swh_too_low swh_validity not_water sea_ice sigma_validity waveform_validity swh_rms_outlier swh_outlier",
            #         "coverage_content_type": "qualityInformation",
            #         "band": "C",
            #         "source": "CCI Sea state IFREMER SAR S1 WV dataset",
            #     },
            # )
            # ds["distance_to_coast"] = xr.DataArray(
            #     data=ds_colocations["distance_to_coast"].values,  # enter data here
            #     dims=["time_sar"],
            #     coords={"time_sar": ds_colocations["time"].values},
            #     attrs={
            #         "unit": "km",
            #         "description": "distance to coast for WV image center using hybrid method raster/polygons openstreemap",
            #         "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
            #         "standard_name": "distance_to_coast",
            #         "vmin": "0",
            #         "vmax": "4000",
            #     },
            # )
            ds["hs_alti_mean"] = xr.DataArray(
                data=ds_colocations["liste_mean"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "description": "altimeter mean of significant wave height co-located with SAR",
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                },
            )
            ds["hs_alti_std"] = xr.DataArray(
                data=ds_colocations["liste_std"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "description": "altimeter standard deviation of significant wave height co-located with SAR",
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                },
            )
            ds["hs_alti_count"] = xr.DataArray(
                data=ds_colocations["liste_count"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "",
                    "description": "number of altimeter SAR-co-located points",
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                },
            )
            ds["hs_alti_closest"] = xr.DataArray(
                data=ds_colocations["liste_closest"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "m",
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                    "description": "significant wave height of the closest altimeter point in space",
                },
            )
            ds["delta_t_closest"] = xr.DataArray(
                data=ds_colocations["liste_DELTA_T_closer"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    #'units': 'hours',
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                    "description": "delta Time altimeter-SAR for the altimeter closest point in space",
                },
            )
            ds["delta_d_closest"] = xr.DataArray(
                data=ds_colocations["liste_DELTA_D_closer"],  # enter data here
                dims=["time_sar"],
                coords={"time_sar": ds_colocations["time_sar"].values},
                attrs={
                    "units": "km",
                    "source": "altimeter measurement gathered in Ifremer SAR-alti co-location product",
                    "description": "delta space for the altimeter closest point in space",
                },
            )

            ds.attrs = {
                "institution": "Institut Français pour la Recherche et l Exploitation de la MER",
                "institution_abbreviation": " LOPS-IFREMER",
                "publisher_name": "ifremer/LOPS",
                "publisher_url": "https://www.umr-lops.fr/",
                "publisher_email": "lops-siam@listes.ifremer.fr",
                "product_description": "colocations between WV and altimeter coming from CCi sea state or CMEMS database",
                # attributs giving the names of the files used is useless since a .lst file is also generated
                # "file": "SAR :"
                # + " ".join([os.path.basename(kk) for kk in input_sar_listing])
                # + "   ALT :"
                # + str(" ".join(list_alti_files_basename))
                # + "",
            }

            logging.info(output_nc_file)
            ds.to_netcdf(output_nc_file)
            new_file_written = True
        else:
            logging.info("no file to save")
    return new_file_written


def add_oswTotalHs_to_SAR_dataset(sar_wv_ds, sar_unit):
    """

    :param sar_wv_ds: xarray.Dataset CCI sea state IFR WV product (orbit file)
    :param sar_unit: str S1A or ...
    :return:
    """
    all_oswTotalHs = []
    all_oswTotalHsStdev = []
    for tt in sar_wv_ds["time"].values:
        logging.debug("tt : %s", tt)
        # ts = (tt - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        # # dt = datetime.datetime.utcfromtimestamp(ts)
        # dt = datetime.datetime.fromtimestamp(ts, timezone.utc)
        dt = from_npdt64_to_dt(tt)
        fp_ocn = get_full_path_ocn_wv_from_approximate_date(dt, sar_unit, level="L2")
        toths = np.nan
        tothsstdev = np.nan
        if fp_ocn:
            if os.path.exists(fp_ocn):
                tmpocn = xr.open_dataset(fp_ocn)
                if "oswTotalHs" in tmpocn:
                    toths = tmpocn["oswTotalHs"].values[0][0]
                if "oswTotalHsStdev" in tmpocn:
                    tothsstdev = tmpocn["oswTotalHsStdev"].values[0][0]
        all_oswTotalHs.append(toths)
        all_oswTotalHsStdev.append(tothsstdev)
    sar_wv_ds["oswTotalHs"] = xr.DataArray(
        all_oswTotalHs,
        dims=["time"],
        attrs={
            "description": "values annotated in S-1 WV L2 OCN oswTotalHs variable since 2022-06-07 ",
            "unit": "m",
            "algo": "Quach et al 2020",
        },
    )
    sar_wv_ds["oswTotalHsStdev"] = xr.DataArray(
        all_oswTotalHsStdev,
        dims=["time"],
        attrs={
            "description": "values annotated in S-1 WV L2 OCN all_oswTotalHsStdev variable since 2022-06-07 ",
            "unit": "m",
            "algo": "Quach et al 2020",
        },
    )
    return sar_wv_ds


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
    it can contain many times the same SAR file (since a single WV can be colocated with different alti files)
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
    # filee = ds.encoding["source"]
    # tmpfname = np.empty(ds["time"].shape, dtype="O")
    # tmpfname[:] = os.path.basename(filee)
    # ds["fname"] = xr.DataArray(tmpfname, dims=["time"])
    # all_vars_owi = [var for var in ds.variables if 'owi' in var]
    # all_vars_rvl = [var for var in ds.variables if 'rvl' in var]
    # osw_heavy_vars = ['oswCartSpecRe','oswCartSpecIm','oswPolSpec',
    #                   'oswPolSpecNV','oswPartitions','oswQualityCrossSpectraRe','oswQualityCrossSpectraIm',
    #                   "oswK",'oswPhi','oswSpecRes',]
    to_keep_vars = ['oswLon','oswLat','oswIncidenceAngle','oswHeading','oswPhs0','oswWaveAge','oswDepth',
                    'oswTotalHs','oswTotalHsStdev','oswWindSpeed','oswNrcs','oswEcmwfWindSpeed','oswNlWidth','oswLandFlag',
                    'oswLandCoverage','oswQualityFlag','oswAzSizeSLC',]
    # ds = ds.drop_vars(all_vars_owi + all_vars_rvl + osw_heavy_vars, errors='ignore')
    ds = ds[to_keep_vars]
    ds['time_sar'] = xr.DataArray([datetime.datetime.strptime(os.path.basename(ds.encoding["source"]).split('-')[5],'%Y%m%dt%H%M%S')],
                                  dims=['time_sar'])
    # ds = ds.expand_dims('time')
    # ds = ds.expand_dims({"time": ds.time})
    ds = ds.squeeze(['oswRaSize','oswAzSize'])
    for var in ds.data_vars:
        if ds[var].dims == ():
            ds[var] = ds[var].expand_dims(time_sar=ds.time_sar)
    
    return ds

def treat_one_safe_wv(safewv,path_altimeter,altidb,acronym_alti_path_ifr,swh_varname,
                      coloc_listing,cpt,dev=False,progressbar=True):
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
    dict4colocs["liste_DELTA_T_closer"] = []  # list of closest alt mesure in time
    dict4colocs["liste_DELTA_D_closer"] = []  # list of closest alt mesure in space
    sarunit = os.path.basename(safewv)[0:3]
    measurement_wv_list = glob.glob(os.path.join(safewv, "measurement", "*.nc"))
    # if dev:
    #     logging.info('dev mode reduce, number of measurement')
    #     measurement_wv_list = measurement_wv_list[0:3]
    logging.debug('Number of measurement in the SAFE : %d',len(measurement_wv_list))
    # SAR = xr.open_mfdataset(measurement_wv_list,
    #                 combine="nested",compat='no_conflicts',join='outer',preprocess=preprocess_wv_s1_ocn).compute()
    tmpsarmeasu = []
    for iiwv in tqdm(range(len(measurement_wv_list)),disable=True):
        tmpsarmeasu.append(preprocess_wv_s1_ocn(xr.open_dataset(measurement_wv_list[iiwv])))
    SAR = xr.concat(tmpsarmeasu,dim='time_sar').load()
    logging.debug("all SAR files loaded")
    # SAR = add_oswTotalHs_to_SAR_dataset(sar_wv_ds=SAR, sar_unit=args.sat)
    liste_match = []
    liste_step1 = []

    liste_date_SAR_dt = Step_0_get_SAR_dt(sards=SAR)
    # pbar = tqdm(range(len(liste_date_SAR_dt)), desc='start')
    # for index_t_sar in range(len(liste_date_SAR_dt)):

    ds_alti = None
    tree_ALT = None
    if progressbar:
        iterratotor = tqdm(range(len(liste_date_SAR_dt)),desc='WV measurement')
    else:
        iterratotor = range(len(liste_date_SAR_dt))
    for index_t_sar in iterratotor: # loop over WV measurements
        # for index_t_sar in pbar:
        # dede = "nb sar measurement progression %s/%s nb matchups %s" % (
        #     index_t_sar,
        #     len(liste_date_SAR_dt),
        #     len(liste_match),
        # )
        # pbar.set_description(dede)

        # if dev and len(dict4colocs["times_SAR"]) > 3:
        #     logging.info("break after finding few matchups dev")
        #     break
        cpt["nb_index_sar_browsed"] += 1
        date_SAR_dt = liste_date_SAR_dt[index_t_sar]
        fullpathL1WVSLC = get_original_wv_slc(date_SAR_dt, sar_unit=sarunit)
        coloc_listing[fullpathL1WVSLC] = []
        if ds_alti is None:
            liste_step1 = Step_1_temp_match(
                date_SAR_dt,
                DELTA_T_SAT,
                path_altimeters=path_altimeter,
                acro_alti=acronym_alti_path_ifr,
                altidb=altidb,
            )
            if len(liste_step1) > 0:
                # this step is done only once because all the SAR obs from a day will be associated to the same alti ds
                ds_alti, tree_ALT = read_all_alti_files(
                    liste_altimeter_files=liste_step1, altidatabase=altidb
                )

        if ds_alti:
            subset_alti = step_2_geographic_match(
                sards=SAR,
                ds_alti=ds_alti,
                tree_ALT=tree_ALT,
                time_sar_index=index_t_sar,
                altidatabase=altidb,
            )
            if subset_alti is not None:
                # if subset_alti["time"].values.size > 0:
                (
                    list_alti_pts_matching_space_and_time,
                    delta_t_closest,
                    HS_closer,
                    lat_alti,
                    lon_alti,
                    DELTA_D_closer,
                    closest_lon,
                    closest_lat,
                    closest_time,
                    list_alti_files_timespace_match,
                ) = step_3_closer_temp_match(
                    SAR, subset_alti, date_SAR_dt, DELTA_T_SAT_SHORT, altidb=altidb
                )
                swh = subset_alti.sel(time=list_alti_pts_matching_space_and_time)[
                    swh_varname
                ].values
                # swh = step_4_SW_match(list_alti_pts_matching_space_and_time, liste_step1)
                swh_count = len(swh)
                swh_mean = np.mean(swh, 0)
                swh_std = np.std(swh, 0)
                if len(list_alti_pts_matching_space_and_time) > 0:
                    coloc_listing[fullpathL1WVSLC] = list_alti_files_timespace_match
                    cpt["nb_index_sar_with_matching_alti"] += 1
                    if dev and cpt["nb_index_sar_with_matching_alti"]>3:
                        logging.info('break loops after finding few matchups')
                        break
                    # graphic_display(liste_step1,liste_geo,list_alti_pts_matching_space_and_time,DELTA_DIST)
                    # lon_alti = min(lon_alti) # TODO fix
                    # lat_alti = min(lat_alti)
                    # time_alti = min(list_alti_pts_matching_space_and_time)
                    dict4colocs["liste_lat_alt"].append(closest_lat)
                    dict4colocs["liste_lon_alt"].append(closest_lon)
                    dict4colocs["liste_time_alt"].append(closest_time)
                    liste_match.append(index_t_sar)
                    dict4colocs["times_SAR"].append(date_SAR_dt.replace(tzinfo=None))
                    dict4colocs["liste_count"].append(swh_count)
                    dict4colocs["liste_mean"].append(swh_mean)
                    dict4colocs["liste_std"].append(swh_std)
                    dict4colocs["liste_closest"].append(HS_closer)
                    dict4colocs["liste_DELTA_T_closer"].append(delta_t_closest)
                    dict4colocs["liste_DELTA_D_closer"].append(DELTA_D_closer)
                # if time_index == 50 or 250 or 500 or 750 :

        else:
            cpt["nb_index_sar_without_alti_file_corresponding"] += 1
            logging.debug("no files found")
        # if index_t_sar % 20 == 0 :
        #     logging.info('nb sar measurement progresion %s/%s nb matchups %s',index_t_sar,len(liste_date_SAR_dt),len(liste_match))
    logging.debug("end of pair construction")
    colocated_observations = SAR.sel(time_sar=dict4colocs["times_SAR"])
    # colocated_observations = add_oswTotalHs_to_SAR_dataset(
    #     sar_wv_ds=colocated_observations, sar_unit=sarunit
    # )
    # logging.info("oswTotalHs added to SAR dataset")
    alti_colocated_ds = xr.Dataset()
    for vv in dict4colocs:
        if vv == "liste_time_alt":
            valval = np.array(dict4colocs[vv]).astype("M8[ns]")
        elif vv == "liste_DELTA_T_closer":
            valval = np.array(dict4colocs[vv]).astype("m8[ns]")
        else:
            valval = np.array(dict4colocs[vv])
        # alti_colocated_ds[vv] = xr.DataArray(np.array(dict4colocs[vv]),dims=['time'],coords={'time':colocated_observations['time'].values})
        alti_colocated_ds[vv] = xr.DataArray(
            valval,
            dims=["time_sar"],
            coords={"time_sar": colocated_observations["time_sar"].values},
        )
    logging.debug("merge alti and SAR colocated values")
    
    # list_alti_files += [os.path.basename(ggh) for ggh in liste_step1]
    colocated_observations = xr.merge([colocated_observations, alti_colocated_ds])
    return colocated_observations,coloc_listing,cpt

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
    M = date.month
    D = date.day
    JY = date.strftime("%j")
    altidb = alt.split("_")[0]
    if altidb == "cci":
        path_altimeter = os.path.join(PATH_ALT[altidb])
    elif altidb == "cmems":
        path_altimeter = os.path.join(
            PATH_ALT[altidb] % POSSIBLES_CMEMS_ALTI[alt.split("_")[1]]
        )
    else:
        raise Exception("altidb %s not handled" % altidb)

    logging.info("path_altimeter : %s", path_altimeter)
    long_name_sar_unit  = 'sentinel-1'+sarunit[-1].lower()
    pattern_sar = os.path.join(
                path_SAR,
                long_name_sar_unit,
                'L2',
                'WV',sarunit+"_WV_OCN__2S",
                Y,
                JY,
                "*.SAFE"
            )
    logging.info("SAR ESA CCI Sea state Ifr pattern : %s", pattern_sar)
    Input_SAR = sorted(
        glob.glob(pattern_sar)
    )
    logging.info("%s SAR WV SAFE found", len(Input_SAR))
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
        + str(DELTA_T_SAT)
        + "_hours_"
        + str(DELTA_DIST)
        + "_degree.nc",
    )
    time.sleep(np.random.randint(0, 10, 1)[0])
    os.makedirs(os.path.dirname(output_nc_file), 0o0775,exist_ok=True)
    if os.path.exists(output_nc_file) and redo is False:
        logging.info("output coloc S1-WV alti file already exists (redo is False)")
        sys.exit(0)

    coloc_listing = {}
    # list_alti_files = []
    
    if len(Input_SAR):
        # if dev is True:
            # logging.info("development mode activated")
            # Input_SAR = Input_SAR[0:3]
        if altidb == "cci":
            swh_varname = "swh_denoised"
            acronym_alti_path_ifr = alt.split("_")[1]
            # acronym_alti_path_ifr = POSSIBLES_CCI_ALTI[alt.split("_")[1]][0]
        elif altidb == "cmems":
            swh_varname = "VAVH"
            acronym_alti_path_ifr = POSSIBLES_CMEMS_ALTI[alt.split("_")[1]]
            if acronym_alti_path_ifr=="swon": #particular case for SWOT
                acronym_alti_path_ifr="swot"
        else:
            raise Exception("altidb %s not handled" % altidb)
        all_safe_matchups = []
        # for ssi,safewv in enumerate(Input_SAR):
        pbar = tqdm(range(len(Input_SAR)),desc='WV SAFE')
        for ssi in pbar:
            pbar.set_description('WV SAFE : nb colocs %i'%cpt["nb_index_sar_with_matching_alti"])
            safewv = Input_SAR[ssi]
            logging.debug('%i/%i',ssi+1,len(Input_SAR))
            # treat one safe here
            one_safe_colocs,coloc_listing,cpt = treat_one_safe_wv(safewv,path_altimeter,altidb,acronym_alti_path_ifr,swh_varname,
                      coloc_listing,cpt=cpt,dev=dev,progressbar=progressbar,)
            # if len(one_safe_colocs.time)>0 and len(one_safe_colocs.time_sar)>0:
            if len(one_safe_colocs.time_sar)>0:
                all_safe_matchups.append(one_safe_colocs)
            if dev and cpt["nb_index_sar_with_matching_alti"]>3:
                logging.info('break loops after finding few matchups')
                break
        if len(all_safe_matchups)>0:
            daily_colocated_observations = xr.concat(all_safe_matchups,dim='time_sar')
        # end of the loop over SAR SAFE
        if os.path.exists(output_nc_file) and redo:
            os.remove(output_nc_file)

       
            # colocated_observations = xr.concat([colocated_observations,alti_colocated_ds],dim='time')
        output_file_written = save_coloc_netCDF_file(
            daily_colocated_observations, output_nc_file)
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
                + str(DELTA_T_SAT)
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
        help="redo existing files nc [optional, default=False-> nothing done if file already exists]",
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
    logging.info('Start of execution for script %s using\
                  WV Level-2 OCN and altimeters from CCI sea state L2P or CMEMS WAV L3',os.path.basename(__file__))
    logging.info('development/test mode activated: %s',args.dev)
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
    logging.info("analysis done in %1.1f sec", time.time() - t1)
    logging.info("end.")
if __name__ == "__main__":
    entrypoint()

