"""Unit tests for treat_one_measurement_wv in unified_coloc_WV_alti_cmems_or_cci."""

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
    """Tests for treat_one_measurement_wv."""

    @pytest.fixture
    def mock_conf(self):
        return {
            "delta_dist_km": 50.0,
            "delta_t_minutes": 25,
        }

    @pytest.fixture
    def common_inputs(self, mock_conf):
        ds_sar = xr.Dataset(
            {"oswTotalHs": (("time_sar",), [2.0])},
            coords={"time_sar": [np.datetime64("2022-01-01T12:00:00")]},
        )
        ds_sar.encoding["source"] = "/path/to/wv_ocn.nc"
        mock_ds_alti = MagicMock(spec=xr.Dataset)
        mock_tree = MagicMock()

        return {
            "sards": ds_sar,
            "date_sar_dt": datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            "ds_alti": mock_ds_alti,
            "tree_alti": mock_tree,
            "altidb": "cmems",
            "coloc_listing": {},
            "cpt": defaultdict(int),
            "conf": mock_conf,
        }

    # ------------------------------------------------------------------ success
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    def test_treat_measurement_match_success(
        self, mock_step2, mock_step3, common_inputs
    ):
        """step_2 finds points, step_3 finds time+space match -> nb_coloc++."""
        mock_step2.return_value = MagicMock()  # non-None subset

        # subset_ok_match_alti must accept item assignment (['time_sar'] = ...)
        mock_subset = xr.Dataset(
            {"hs_alti_closest": ((), 2.5)},
        )
        mock_step3.return_value = (
            mock_subset,
            np.array(["alti_file.nc"]),
            common_inputs["cpt"],  # cpt is mutated in-place in the source
        )

        subset_res, listing_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        assert cpt_res["nb_index_sar_browsed"] == 1
        assert cpt_res["nb_coloc"] == 1
        # fullpath_l2_wv_ocn comes from sards.encoding["source"]
        assert listing_res["/path/to/wv_ocn.nc"].tolist() == ["alti_file.nc"]
        assert subset_res is mock_subset
        # time_sar was attached as a scalar datetime
        assert "time_sar" in subset_res

    # ------------------------------------------------------------------ no geo
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    def test_treat_measurement_no_geographic_match(self, mock_step2, common_inputs):
        """step_2 returns None -> no colocation recorded."""
        mock_step2.return_value = None

        subset_res, listing_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        assert cpt_res["nb_index_sar_browsed"] == 1
        assert cpt_res["nb_coloc"] == 0
        assert subset_res is None
        assert listing_res == {}

    # ------------------------------------------------------------------ no time
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    def test_treat_measurement_no_time_match(
        self, mock_step2, mock_step3, common_inputs
    ):
        """step_2 finds points but step_3 finds none in the time window."""
        mock_step2.return_value = MagicMock()
        # empty fname list -> nothing recorded, subset_ok is None
        mock_step3.return_value = (
            None,
            np.array([]),
            common_inputs["cpt"],
        )

        subset_res, listing_res, cpt_res = treat_one_measurement_wv(**common_inputs)

        assert cpt_res["nb_index_sar_browsed"] == 1
        assert cpt_res["nb_coloc"] == 0
        assert subset_res is None
        assert listing_res == {}

    # ------------------------------------------------------------------ conf
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_3_closer_temp_match"
    )
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_2_geographic_match"
    )
    def test_treat_measurement_conf_forwarded(
        self, mock_step2, mock_step3, common_inputs
    ):
        """conf['delta_dist_km'] and conf['delta_t_minutes'] must be forwarded."""
        mock_step2.return_value = MagicMock()
        mock_step3.return_value = (None, np.array([]), common_inputs["cpt"])

        treat_one_measurement_wv(**common_inputs)

        # step_2 uses delta_dist_km
        _, step2_kwargs = mock_step2.call_args
        assert step2_kwargs["delta_dist_km"] == common_inputs["conf"]["delta_dist_km"]

        # step_3 uses delta_t_max_minutes (renamed from delta_t_sat_short)
        _, step3_kwargs = mock_step3.call_args
        assert (
            step3_kwargs["delta_t_max_minutes"]
            == common_inputs["conf"]["delta_t_minutes"]
        )
        assert step3_kwargs["altidb"] == common_inputs["altidb"]
        assert step3_kwargs["cpt"] is common_inputs["cpt"]
