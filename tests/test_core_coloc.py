"""Unit tests for core_coloc in unified_coloc_WV_alti_cmems_or_cci."""

import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import core_coloc


@pytest.fixture
def mock_conf():
    """Minimal configuration dict matching what get_conf_content returns."""
    return {
        "path_SAR": "/data/sar",
        "cmems_dir": "/data/cmems",
        "subset_alti_name_dir": "cmems_obs-wave_glo_phy-swh_nrt_%s-l3_PT1S",
        "cci_alti_dir": "/data/cci",
        # New keys used by core_coloc for output filename and colocation criteria
        "delta_t_minutes": 25,
        "delta_dist_km": 50.0,
    }


@pytest.fixture
def mock_params(mock_conf):
    """Standard input parameters for core_coloc."""
    return {
        "day_analyzed": "20260112",  # renamed from date_analyzed
        "alt": "cmems_Jason-3",
        "sarunit": "S1A",
        "outputdir": os.path.join(tempfile.gettempdir(), "output"),
        "conf": mock_conf,
    }


@pytest.fixture
def mock_coloc_ds():
    """A small xarray Dataset standing in for a colocated SAR/alti dataset."""
    return xr.Dataset(
        {"oswLon": (("time_sar",), [10.0, 20.0])},
        coords={
            "time_sar": [
                np.datetime64("2026-01-12T12:00:00"),
                np.datetime64("2026-01-12T12:01:00"),
            ]
        },
    )


# ---------------------------------------------------------------------------
# Case 1: no SAR data in the input directory
# ---------------------------------------------------------------------------
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.time.sleep")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_no_sar_data(
    mock_makedirs,
    mock_exists,
    mock_glob,
    mock_get_path,
    mock_sleep,
    mock_params,
):
    """No SAR SAFE found -> counter remains empty, no output written."""
    mock_get_path.return_value = ("/path/alt", "j3", "VAVH")
    # 1) path_altimeter exists, 2) path_SAR exists, 3) output_nc_file absent
    mock_exists.side_effect = [True, True, False]
    mock_glob.return_value = []  # no SAR files, and no alti files either

    cpt = core_coloc(**mock_params)

    # defaultdict(int) is empty when the SAR loop is never entered
    assert cpt == {}
    # 1 glob for SAR + 3 globs inside step_1_temp_match_cmems (D-1, D, D+1)
    assert mock_glob.call_count == 4
    mock_makedirs.assert_called_once()
    mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Case 2: full success path with one SAR SAFE and one colocation
# ---------------------------------------------------------------------------
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.write_coloc_listing")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.save_coloc_netcdf_file")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.concat")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_safe_wv")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.read_all_alti_files")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.time.sleep")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_success_flow(
    mock_makedirs,
    mock_exists,
    mock_glob,
    mock_get_path,
    mock_sleep,
    mock_step1,
    mock_read_alti,
    mock_treat,
    mock_concat,
    mock_save_nc,
    mock_write_lst,
    mock_params,
    mock_coloc_ds,
):
    """One SAR SAFE + one alti file -> output .nc and .lst are written."""
    mock_get_path.return_value = ("/path/alt", "j3", "VAVH")
    # path_altimeter, path_SAR exist; output .nc and .lst do not
    mock_exists.side_effect = [True, True, False, False]

    mock_glob.return_value = [
        "/data/S1A_WV_OCN__2SSV_20260112T120000_20260112T120000_" "056789_000000.SAFE"
    ]

    # step_1_temp_match returns a non-empty alti file list
    mock_step1.return_value = ["/path/alt/alti_file.nc"]
    # read_all_alti_files returns (ds_alti, tree_alti); ds_alti["time"] must be
    # non-empty for core_coloc to enter the matchup loop.
    ds_alti_mock = MagicMock()
    ds_alti_mock.__getitem__.return_value = np.array(["2026-01-12T12:00:00"])
    mock_read_alti.return_value = (ds_alti_mock, MagicMock())

    # treat_one_safe_wv returns (dataset, listing, counter) - 3-tuple now
    new_cpt = defaultdict(int, {"nb_coloc": 1})
    mock_treat.return_value = (mock_coloc_ds, {"listing": "data"}, new_cpt)

    # xr.concat is mocked; return our fake colocated dataset
    mock_concat.return_value = mock_coloc_ds
    mock_save_nc.return_value = True

    result_cpt = core_coloc(**mock_params)

    # New counter name is "nb_coloc" (was nb_index_sar_with_matching_alti)
    assert result_cpt["nb_coloc"] == 1

    mock_step1.assert_called_once()
    mock_read_alti.assert_called_once()
    mock_treat.assert_called_once()

    # conf must be forwarded to treat_one_safe_wv
    _, kwargs = mock_treat.call_args
    assert "conf" in kwargs
    assert kwargs["conf"] is mock_params["conf"]

    mock_concat.assert_called_once()
    mock_save_nc.assert_called_once()
    mock_write_lst.assert_called_once()
    mock_makedirs.assert_called_once()
    mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Case 3: alti files found but treat_one_safe_wv returns empty datasets
# ---------------------------------------------------------------------------
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.save_coloc_netcdf_file")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.concat")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_safe_wv")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.read_all_alti_files")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.time.sleep")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_sar_but_no_matchups(
    mock_makedirs,
    mock_exists,
    mock_glob,
    mock_get_path,
    mock_sleep,
    mock_step1,
    mock_read_alti,
    mock_treat,
    mock_concat,
    mock_save_nc,
    mock_params,
):
    """SAR file present but no colocation found -> no .nc written."""
    mock_get_path.return_value = ("/path/alt", "j3", "VAVH")
    # mock_exists.side_effect = [True, True, False]
    mock_exists.side_effect = [True, True, False, False]
    mock_glob.return_value = ["/data/foo.SAFE"]
    mock_step1.return_value = ["/path/alt/alti_file.nc"]
    ds_alti_mock = MagicMock()
    ds_alti_mock.__getitem__.return_value = np.array(["2026-01-12T12:00:00"])
    mock_read_alti.return_value = (ds_alti_mock, MagicMock())

    # treat_one_safe_wv returns an EMPTY dataset (len(time_sar) == 0)
    empty_ds = xr.Dataset(
        {"oswLon": (("time_sar",), [])},
        coords={"time_sar": np.array([], dtype="datetime64[ns]")},
    )
    mock_treat.return_value = (empty_ds, {}, defaultdict(int))

    cpt = core_coloc(**mock_params)

    # Loop ran, but no matchup was collected, so concat/save are not called
    mock_treat.assert_called_once()
    mock_concat.assert_not_called()
    mock_save_nc.assert_not_called()
    assert cpt.get("nb_coloc", 0) == 0
