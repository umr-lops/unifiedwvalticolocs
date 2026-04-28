import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import core_coloc


@pytest.fixture
def mock_conf():
    """Provides a minimal configuration dictionary matching what get_conf_content returns."""
    return {
        "path_SAR": "/data/sar",
        "cmems_dir": "/data/cmems",
        "subset_alti_name_dir": "cmems_obs-wave_glo_phy-swh_nrt_%s-l3_PT1S",
        "cci_alti_dir": "/data/cci",
        "delta_t_sat": 3,
        "delta_t_sat_short": 3 * 3600,
        "DELTA_DIST": 2,
    }


@pytest.fixture
def mock_params(mock_conf):
    """Provides standard input parameters for core_coloc."""
    return {
        "startdate": "20260112",
        "alt": "cmems_Jason-3",
        "sarunit": "S1A",
        "outputdir": os.path.join(tempfile.gettempdir(), "output"),
        "conf": mock_conf,
    }


@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_no_sar_data(
    mock_makedirs, mock_exists, mock_glob, mock_get_path, mock_params
):
    """Test: Ensure it handles cases where no SAR files are found in the directory."""
    mock_get_path.return_value = ("/path/alt", "j3", "VAVH")
    # path_altimeter exists, path_SAR exists, output_nc_file does not exist yet
    mock_exists.side_effect = [True, True, False]
    mock_glob.return_value = []  # No SAR SAFE files found

    cpt = core_coloc(**mock_params)

    assert cpt == {}  # defaultdict(int) remains empty when no SAR data found
    mock_glob.assert_called_once()


@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_safe_wv")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.concat")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.save_coloc_netcdf_file")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.write_coloc_listing")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_success_flow(
    mock_makedirs,
    mock_write_lst,
    mock_save_nc,
    mock_concat,
    mock_treat,
    mock_exists,
    mock_glob,
    mock_get_path,
    mock_params,
):
    """Test: Full success path with one SAR file and one colocation."""

    mock_get_path.return_value = ("/path/alt", "j3", "VAVH")
    # path_altimeter, path_SAR exist; output_nc_file does not
    mock_exists.side_effect = [True, True, False, False]
    mock_glob.return_value = [
        "/data/S1A_WV_OCN__2SSV_20260112T120000_20260112T120000_056789_000000.SAFE"
    ]

    # Mock treat_one_safe_wv return value
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.time_sar = [1]
    mock_ds.__len__.return_value = 1
    mock_ds.__getitem__.return_value = [10, 20]  # simulate oswLon check in save logic

    new_cpt = defaultdict(int, {"nb_index_sar_with_matching_alti": 1})
    mock_treat.return_value = (mock_ds, {"listing": "data"}, new_cpt)

    mock_concat.return_value = mock_ds
    mock_save_nc.return_value = True

    result_cpt = core_coloc(**mock_params)

    assert result_cpt["nb_index_sar_with_matching_alti"] == 1
    mock_treat.assert_called_once()
    # Verify conf was forwarded to treat_one_safe_wv
    _, kwargs = mock_treat.call_args
    assert "conf" in kwargs
    mock_save_nc.assert_called_once()
    mock_write_lst.assert_called_once()
