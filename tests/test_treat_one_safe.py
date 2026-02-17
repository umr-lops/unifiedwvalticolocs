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
    def mock_inputs(self):
        # Realistic S1 SAFE filename: S1A_WV_OCN__2SSV_20220101T100000_...
        # Index 5 is the start date.
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
        mock_glob.return_value = ["meas.nc"]
        mock_step1.return_value = ["alti.nc"]
        mock_read.return_value = (MagicMock(), MagicMock())

        t0 = np.datetime64("2022-01-01T10:00:00")
        ds = xr.Dataset(
            {"oswTotalHs": (("time_sar"), [1.5])}, coords={"time_sar": [t0]}
        )
        mock_pre.return_value = ds
        mock_step0.return_value = [pd.to_datetime(t0)]

        # Mock treat_one_measurement_wv to return required 3 values
        mock_treat.return_value = ({"times_SAR": [t0]}, {}, defaultdict(int))

        result_ds, _, _ = treat_one_safe_wv(**mock_inputs)
        assert "oswTotalHs" in result_ds
