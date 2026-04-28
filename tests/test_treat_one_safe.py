import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import treat_one_safe_wv


class TestTreatSafe:
    @pytest.fixture
    def mock_conf(self):
        return {
            "delta_t_sat": 3,
            "delta_t_sat_short": 3 * 3600,
            "DELTA_DIST": 2,
        }

    @pytest.fixture
    def mock_inputs(self, mock_conf):
        # Realistic S1 SAFE filename — index 5 (split on "_") is the start date.
        safe_name = (
            "S1A_WV_OCN__2SSV_20220101T100000_20220101T100020_000000_000000_0000.SAFE"
        )
        return {
            "safewv": os.path.join(tempfile.gettempdir(), safe_name),
            "path_altimeter": tempfile.gettempdir(),
            "altidb": "cmems",
            "acronym_alti_path_ifr": "j3",
            "swh_varname": "VAVH",
            "coloc_listing": {},
            "cpt": defaultdict(int),
            "dev": False,
            "conf": mock_conf,
        }

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.read_all_alti_files")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
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
        mock_treat,
        mock_step0,
        mock_open,
        mock_pre,
        mock_step1,
        mock_read,
        mock_inputs,
    ):
        """Happy path: alti files found, one measurement processed, dataset returned."""
        mock_glob.return_value = ["meas.nc"]
        mock_step1.return_value = ["alti.nc"]
        mock_read.return_value = (MagicMock(), MagicMock())

        t0 = np.datetime64("2022-01-01T10:00:00")
        ds = xr.Dataset(
            {"oswTotalHs": (("time_sar",), [1.5])}, coords={"time_sar": [t0]}
        )
        mock_pre.return_value = ds
        mock_step0.return_value = [pd.to_datetime(t0)]

        mock_treat.return_value = ({"times_SAR": [t0]}, {}, defaultdict(int))

        result_ds, _, _ = treat_one_safe_wv(**mock_inputs)

        assert "oswTotalHs" in result_ds

        # delta_t_sat is passed as the second positional arg to step_1_temp_match
        step1_args, _ = mock_step1.call_args
        assert step1_args[1] == mock_inputs["conf"]["delta_t_sat"]

        # treat_one_measurement_wv is called without conf (conf is consumed
        # directly inside treat_one_safe_wv for step_1_temp_match, not forwarded)
        mock_treat.assert_called_once()

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_1_temp_match")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.preprocess_wv_s1_ocn"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.open_dataset")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_0_get_sar_dt")
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_treat_one_safe_wv_no_alti_files(
        self,
        mock_glob,
        mock_step0,
        mock_open,
        mock_pre,
        mock_step1,
        mock_inputs,
    ):
        """When step_1 finds no altimeter files the safe counter is incremented
        and an empty dataset is returned."""
        mock_glob.return_value = ["meas.nc"]
        mock_step1.return_value = []  # no alti files in time window

        t0 = np.datetime64("2022-01-01T10:00:00")
        ds = xr.Dataset(
            {"oswTotalHs": (("time_sar",), [1.5])}, coords={"time_sar": [t0]}
        )
        mock_pre.return_value = ds
        mock_step0.return_value = [pd.to_datetime(t0)]

        result_ds, _, cpt_res = treat_one_safe_wv(**mock_inputs)

        assert cpt_res["nb_safe_without_alti_files"] == 1
        assert cpt_res["nb_safe_with_alti_files"] == 0
        # Returned dataset should be empty (no time_sar matchups)
        assert len(result_ds.time_sar) == 0
