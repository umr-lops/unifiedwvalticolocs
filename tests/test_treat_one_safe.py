import os
import tempfile
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Adjust import based on your actual package structure
from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import treat_one_safe_wv


class TestTreatSafe:

    @pytest.fixture
    def mock_inputs(self):
        """Standard inputs for the function."""
        return {
            "safewv": os.path.join(tempfile.gettempdir(), "S1A_TEST.SAFE"),
            "path_altimeter": os.path.join(tempfile.gettempdir(), "alti"),
            "altidb": "cmems",
            "acronym_alti_path_ifr": "j3",
            "swh_varname": "VAVH",
            "coloc_listing": {},
            "cpt": defaultdict(int),
            "dev": False,
        }

    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.preprocess_wv_s1_ocn"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.open_dataset")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_0_get_sar_dt")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_measurement_wv"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_treat_one_safe_wv_success(
        self,
        mock_glob,
        mock_treat_meas,
        mock_step0,
        mock_open_ds,
        mock_preprocess,
        mock_inputs,
    ):
        """
        Test the successful processing of a SAFE folder with 2 measurements.
        Scenario:
          - Measurement 0: Finds a colocation (Match).
          - Measurement 1: No colocation found (No Match).
        Result:
          - The final dataset should contain only Measurement 0 data, merged with Alti data.
        """
        # ---------------------------------------------------------
        # 1. SETUP MOCKS
        # ---------------------------------------------------------

        # Simulate finding 2 measurement files
        mock_glob.return_value = ["meas_1.nc", "meas_2.nc"]

        # Create dummy SAR data (what xr.open_dataset/preprocess returns)
        # We need time_sar to enable concatenation and selection
        t0 = np.datetime64("2022-01-01T10:00:00")
        t1 = np.datetime64("2022-01-01T10:01:00")

        # Mocking what preprocess returns (usually a dataset with 1 time step)
        ds1 = xr.Dataset(
            {"oswTotalHs": (("time_sar"), [1.5])}, coords={"time_sar": [t0]}
        )
        ds2 = xr.Dataset(
            {"oswTotalHs": (("time_sar"), [2.0])}, coords={"time_sar": [t1]}
        )

        # We need mock_preprocess to return these when called in the loop
        mock_preprocess.side_effect = [ds1, ds2]

        # Mock step_0 to return datetime objects corresponding to the dataset
        mock_step0.return_value = [pd.to_datetime(t0), pd.to_datetime(t1)]

        # ---------------------------------------------------------
        # 2. DEFINE LOGIC FOR INNER PROCESSING (treat_one_measurement_wv)
        # ---------------------------------------------------------

        def side_effect_treat(sards, list_dates, unit, index_t_sar, *args, **kwargs):
            """
            Simulate treat_one_measurement_wv logic.
            It modifies dict4colocs in place.
            """
            d = kwargs["dict4colocs"]

            # Simulate MATCH for the first measurement (index 0)
            if index_t_sar == 0:
                d["times_SAR"].append(list_dates[index_t_sar])
                d["liste_count"].append(10)
                d["liste_mean"].append(2.5)
                d["liste_std"].append(0.1)
                d["liste_lat_alt"].append(45.0)
                d["liste_lon_alt"].append(-10.0)
                d["liste_time_alt"].append(
                    list_dates[index_t_sar]
                )  # Same time for simplicity
                d["liste_closest"].append(2.4)
                d["liste_DELTA_T_closer"].append(np.timedelta64(10, "s"))
                d["liste_DELTA_D_closer"].append(5.0)

                # Update counters
                kwargs["cpt"]["nb_index_sar_with_matching_alti"] += 1

            # Simulate NO MATCH for the second measurement (index 1)
            # We simply don't append anything to dict4colocs

            return d, kwargs["coloc_listing"], kwargs["cpt"]

        mock_treat_meas.side_effect = side_effect_treat

        # ---------------------------------------------------------
        # 3. EXECUTE FUNCTION
        # ---------------------------------------------------------
        result_ds, result_listing, result_cpt = treat_one_safe_wv(**mock_inputs)

        # ---------------------------------------------------------
        # 4. ASSERTIONS
        # ---------------------------------------------------------

        # Check counters
        assert result_cpt["nb_index_sar_with_matching_alti"] == 1

        # Check Final Dataset Structure
        # It should have length 1 (only the matching one kept)
        assert len(result_ds["time_sar"]) == 1
        assert result_ds["time_sar"].values[0] == t0

        # Check that SAR variables are present
        assert "oswTotalHs" in result_ds
        assert result_ds["oswTotalHs"].values[0] == 1.5

        # Check that Alti variables (from dict4colocs) are merged in
        assert "liste_mean" in result_ds
        assert result_ds["liste_mean"].values[0] == 2.5

        # Check data types of specific converted columns
        assert result_ds["liste_time_alt"].dtype.kind == "M"  # datetime64
        assert result_ds["liste_DELTA_T_closer"].dtype.kind == "m"  # timedelta64

        # Verify calls
        assert mock_glob.called
        assert mock_treat_meas.call_count == 2  # Called for both measurements

    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.preprocess_wv_s1_ocn"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.open_dataset")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_0_get_sar_dt")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_measurement_wv"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_treat_one_safe_wv_dev_break(
        self,
        mock_glob,
        mock_treat_meas,
        mock_step0,
        mock_open_ds,
        mock_preprocess,
        mock_inputs,
    ):
        """
        Test that dev mode breaks the loop early.
        """
        mock_inputs["dev"] = True

        # We pretend there are 10 files
        n_files = 10
        mock_glob.return_value = [f"meas_{i}.nc" for i in range(n_files)]

        # Generate 10 unique timestamps
        t_start = np.datetime64("2022-01-01T10:00:00")
        times = [t_start + np.timedelta64(i, "m") for i in range(n_files)]

        # FIX: Create a unique dataset for each file so xr.concat produces a unique index
        datasets = []
        for t in times:
            ds = xr.Dataset({"v": (("time_sar"), [1])}, coords={"time_sar": [t]})
            datasets.append(ds)

        # Use side_effect to return a different dataset each time it's called
        mock_preprocess.side_effect = datasets

        # Mock step0 to return the list of corresponding python datetimes
        mock_step0.return_value = [pd.to_datetime(t) for t in times]

        # Logic: Increment counter aggressively to trigger break
        def side_effect_break(sards, list_dates, *args, **kwargs):
            # Artificially inflate match count > 3 (MAX_NB_MATCHUPS_DEV_MODE)
            kwargs["cpt"]["nb_index_sar_with_matching_alti"] = 5
            return kwargs["dict4colocs"], kwargs["coloc_listing"], kwargs["cpt"]

        mock_treat_meas.side_effect = side_effect_break

        # Execute
        treat_one_safe_wv(**mock_inputs)

        # Assertion:
        # We have 10 files.
        # The loop processes the first file (index 0).
        # It calls treat_one_measurement_wv, which sets count=5.
        # It checks "if dev and count > 3". This is True.
        # It breaks.
        # So treat_one_measurement_wv should have been called EXACTLY once.
        # (Or maybe twice depending on how you initialize, but definitely not 10 times)
        assert mock_treat_meas.call_count == 1
