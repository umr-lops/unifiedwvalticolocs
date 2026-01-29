import datetime
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# Adjust import
from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    treat_one_measurement_wv,
)


class TestTreatMeasurement:

    @pytest.fixture
    def common_inputs(self):
        """Standard arguments for the function."""
        # Create a dummy SAR dataset (not strictly read if mocks work well, but good for safety)
        ds_sar = xr.Dataset(
            {"oswTotalHs": (("time_sar"), [2.0])},
            coords={"time_sar": [np.datetime64("2022-01-01T12:00:00")]},
        )

        return {
            "sards": ds_sar,
            "list_date_sar_dt": [
                datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            ],
            "sarunit": "S1A",
            "index_t_sar": 0,
            "altidb": "cmems",
            "coloc_listing": {},
            "dict4colocs": defaultdict(list),
            "cpt": defaultdict(int),
            "path_altimeter": "/tmp/alti",
            "acronym_alti_path_ifr": "j3",
            "swh_varname": "VAVH",
        }

    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.read_all_alti_files")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_original_wv_slc")
    def test_treat_measurement_match_success(
        self,
        mock_get_slc,
        mock_step1,
        mock_read_alti,
        mock_step2,
        mock_step3,
        common_inputs,
    ):
        """
        Scenario: Everything works.
        Files found -> Geo Match -> Time Match -> Stats calculated.
        """
        # 1. Mock file finding (Step 1)
        mock_get_slc.return_value = "/path/to/slc"
        mock_step1.return_value = ["file1.nc"]  # Found 1 file

        # 2. Mock reading files
        mock_ds_alti = MagicMock()  # Represents the Alti Xarray Dataset
        mock_tree = MagicMock()
        mock_read_alti.return_value = (mock_ds_alti, mock_tree)

        # 3. Mock Geo Match (Step 2)
        # It returns a subset dataset. We'll use the same mock for simplicity.
        mock_step2.return_value = mock_ds_alti

        # 4. Mock Time Match (Step 3)
        # This function returns a BIG tuple. We need to match the unpacking signature.
        # Signature: list_pts, dt_close, hs_close, lat, lon, dd_close, c_lon, c_lat, c_time, files

        fake_time = np.datetime64("2022-01-01T12:00:10")
        list_pts = [fake_time]  # Must be list-like

        mock_step3.return_value = (
            list_pts,  # list_alti_pts_matching_space_and_time
            np.timedelta64(10, "s"),  # delta_t_closest
            2.5,  # hs_alti_closest
            [45.0],  # lat_alti
            [-10.0],  # lon_alti
            5.0,  # delta_d_closer
            -10.0,  # closest_lon
            45.0,  # closest_lat
            fake_time,  # closest_time
            ["file1.nc"],  # list_alti_files_timespace_match
        )

        # 5. Mock the Data Selection for Stats
        # The code calls: subset_alti.sel(time=...)[swh_varname].values
        # We need to ensure .values returns a numpy array of SWH to calculate mean/std
        mock_sel = mock_ds_alti.sel.return_value
        mock_var = mock_sel.__getitem__.return_value
        mock_var.values = np.array([2.0, 3.0])  # Mean should be 2.5

        # --- EXECUTE ---
        dict_res, list_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        # --- ASSERTIONS ---

        # 1. Counters
        assert cpt_res["nb_index_sar_browsed"] == 1
        assert cpt_res["nb_index_sar_with_matching_alti"] == 1

        # 2. Dictionary content
        assert len(dict_res["times_SAR"]) == 1
        assert dict_res["liste_count"][0] == 2  # len([2.0, 3.0])
        assert dict_res["liste_mean"][0] == 2.5  # mean(2.0, 3.0)
        assert dict_res["liste_std"][0] == 0.5  # std(2.0, 3.0)
        assert dict_res["liste_closest"][0] == 2.5

        # 3. Listing
        assert "/path/to/slc" in list_res
        assert list_res["/path/to/slc"] == ["file1.nc"]

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_original_wv_slc")
    def test_treat_measurement_no_files(self, mock_get_slc, mock_step1, common_inputs):
        """
        Scenario: Step 1 finds no matching altimeter files for this day.
        """
        # Step 1 returns empty list
        mock_step1.return_value = []
        mock_get_slc.return_value = "/path/to/slc"

        # Note: We need to define ds_alti in the scope of the function if liste_step1 is empty.
        # Assuming the source code handles the `if ds_alti:` check correctly
        # (e.g. ds_alti = None implicitly or defined before).
        # Based on your snippet, there might be an UnboundLocalError potential if ds_alti isn't init to None.
        # But assuming it works:

        # --- EXECUTE ---
        # We wrap in try/except just in case the source code has the UnboundLocalError bug
        # If it doesn't crash, we assert the logic.
        try:
            dict_res, list_res, cpt_res = treat_one_measurement_wv(**common_inputs)
        except UnboundLocalError:
            pytest.fail(
                "Source code has UnboundLocalError: ds_alti is not defined when step 1 is empty"
            )

        # --- ASSERTIONS ---
        assert cpt_res["nb_index_sar_browsed"] == 1
        # No match found counter should increment
        assert cpt_res["nb_index_sar_without_alti_file_corresponding"] == 1
        assert len(dict_res["times_SAR"]) == 0
        assert len(list_res["/path/to/slc"]) == 0

    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.read_all_alti_files")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_original_wv_slc")
    def test_treat_measurement_geo_match_but_no_time_match(
        self, mock_get_slc, mock_step1, mock_read, mock_step2, mock_step3, common_inputs
    ):
        """
        Scenario: Files found, Geo match OK, but Step 3 returns 0 matches in time window.
        """
        mock_step1.return_value = ["f.nc"]
        mock_read.return_value = (MagicMock(), MagicMock())
        mock_step2.return_value = MagicMock()  # Geo match found

        # Step 3 returns empty list of points
        mock_step3.return_value = (
            [],  # list_pts (Empty)
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        dict_res, list_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        # Nothing should be added to dictionary
        assert len(dict_res["times_SAR"]) == 0
        # Counter should NOT increment for match
        assert cpt_res["nb_index_sar_with_matching_alti"] == 0
