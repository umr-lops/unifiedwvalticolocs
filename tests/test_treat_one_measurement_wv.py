import datetime
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    treat_one_measurement_wv,
)


class TestTreatMeasurement:
    @pytest.fixture
    def common_inputs(self):
        ds_sar = xr.Dataset(
            {"oswTotalHs": (("time_sar"), [2.0])},
            coords={"time_sar": [np.datetime64("2022-01-01T12:00:00")]},
        )
        # Create dummy Alti objects to pass as arguments
        mock_ds_alti = MagicMock(spec=xr.Dataset)
        mock_tree = MagicMock()

        return {
            "sards": ds_sar,
            "list_date_sar_dt": [
                datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            ],
            "sarunit": "S1A",
            "index_t_sar": 0,
            "ds_alti": mock_ds_alti,  # Added
            "tree_alti": mock_tree,  # Added
            "altidb": "cmems",
            "coloc_listing": {},
            "dict4colocs": defaultdict(list),
            "cpt": defaultdict(int),
            "swh_varname": "VAVH",
        }

    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.get_original_wv_slc")
    def test_treat_measurement_match_success(
        self, mock_get_slc, mock_step2, mock_step3, common_inputs
    ):
        mock_get_slc.return_value = "/path/to/slc"
        mock_step2.return_value = MagicMock()

        fake_time = np.datetime64("2022-01-01T12:00:10")

        # Create a mock subset dataset for the 11th return value
        mock_subset = MagicMock()
        mock_subset.__getitem__.return_value.values = np.array([2.0, 3.0])

        mock_step3.return_value = (
            [fake_time],
            np.timedelta64(10, "s"),
            2.5,
            [45.0],
            [-10.0],
            5.0,
            -10.0,
            45.0,
            fake_time,
            ["file1.nc"],
            mock_subset,  # 11th value
        )

        dict_res, list_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        assert cpt_res["nb_index_sar_with_matching_alti"] == 1
        assert dict_res["liste_mean"][0] == 2.5
