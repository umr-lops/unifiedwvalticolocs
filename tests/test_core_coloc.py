import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import core_coloc


@pytest.fixture
def mock_params():
    """Provides standard input parameters for the function."""
    return {
        "startdate": "20260112",
        "alt": "SRAL_C",
        "sarunit": "S1A",
        "outputdir": os.path.join(tempfile.gettempdir(), "output"),
    }


# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.sys.exit')
# def test_core_coloc_exit_if_exists_no_redo(mock_exit, mock_exists, mock_get_path, mock_params):
#     """Test 1: Script should exit if output file exists and redo is False."""
#     # Setup mocks
#     mock_get_path.return_value = ("/path/alt", "acronym", "swh_var")
#     # First two calls for path_altimeter and path_SAR (True), third for output_nc_file (True)
#     mock_exists.side_effect = [True, True, True]

#     core_coloc(**mock_params, redo=False)

#     mock_exit.assert_called_once_with(0)


@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists")
@patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs")
def test_core_coloc_no_sar_data(
    mock_makedirs, mock_exists, mock_glob, mock_get_path, mock_params
):
    """Test 2: Ensure it handles cases where no SAR files are found in the directory."""
    mock_get_path.return_value = ("/path/alt", "acronym", "swh_var")
    mock_exists.side_effect = [True, True, False]  # path_alt, path_SAR, output_nc_file
    mock_glob.return_value = []  # No files found

    cpt = core_coloc(**mock_params)

    assert cpt == {}  # Defaultdict(int) will be empty
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
    """Test 3: Full success path with one SAR file and one colocation."""

    # 1. Setup Mocks
    mock_get_path.return_value = ("/path/alt", "acronym", "swh_var")
    mock_exists.return_value = True  # Paths exist, but we mock save_nc logic anyway
    mock_exists.side_effect = [True, True, False, False]  # alt, sar, out_nc, redo_check
    mock_glob.return_value = ["/data/S1A_WV_OCN__2S_2026.SAFE"]

    # Mock the return of treat_one_safe_wv
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.time_sar = [1]
    mock_ds.__len__.return_value = 1
    # Specific mock for the listing dictionary check
    mock_ds.__getitem__.return_value = [10, 20]  # for oswLon check

    new_cpt = defaultdict(int, {"nb_index_sar_with_matching_alti": 1})
    mock_treat.return_value = (mock_ds, {"listing": "data"}, new_cpt)

    mock_concat.return_value = mock_ds
    mock_save_nc.return_value = True

    # 2. Execute
    result_cpt = core_coloc(**mock_params)

    # 3. Assertions
    assert result_cpt["nb_index_sar_with_matching_alti"] == 1
    mock_treat.assert_called_once()
    mock_save_nc.assert_called_once()
    mock_write_lst.assert_called_once()


# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_path_alti')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.path.exists')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_safe_wv')
# @patch('unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.os.makedirs')
# def test_core_coloc_dev_mode_break(mock_makedirs, mock_treat, mock_exists, mock_glob, mock_get_path, mock_params):
#     """Test 4: Dev mode should break loop after MAX_NB_MATCHUPS_DEV_MODE is reached."""

#     mock_get_path.return_value = ("/path/alt", "acronym", "swh_var")
#     mock_exists.side_effect = [True, True, False]
#     # Provide 5 files, but we want it to break after the first if it meets criteria
#     mock_glob.return_value = ["f1.SAFE", "f2.SAFE", "f3.SAFE"]

#     # Simulate finding many matchups in the first file
#     from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import MAX_NB_MATCHUPS_DEV_MODE
#     high_cpt = defaultdict(int, {"nb_index_sar_with_matching_alti": MAX_NB_MATCHUPS_DEV_MODE + 1})

#     mock_ds = MagicMock(spec=xr.Dataset)
#     mock_ds.time_sar = [1]
#     mock_treat.return_value = (mock_ds, {}, high_cpt)

#     core_coloc(**mock_params, dev=True)

#     # treat_one_safe_wv should only be called once because of the break
#     assert mock_treat.call_count == 1
