"""Unit tests for treat_one_safe_wv in unified_coloc_WV_alti_cmems_or_cci."""

import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import treat_one_safe_wv


class TestTreatSafe:
    @pytest.fixture
    def mock_conf(self):
        return {
            "delta_dist_km": 50.0,
            "delta_t_minutes": 25,
        }

    @pytest.fixture
    def mock_alti_ds(self):
        """Small CMEMS-like altimeter dataset with one track point."""
        return xr.Dataset(
            {
                "latitude": (("time",), [10.0]),
                "longitude": (("time",), [10.0]),
                "VAVH": (("time",), [2.5]),
                "fname": (("time",), ["alti.nc"]),
            },
            coords={"time": [np.datetime64("2022-01-01T10:00:00")]},
        )

    @pytest.fixture
    def mock_inputs(self, mock_conf, mock_alti_ds):
        # Realistic S1 SAFE path; measurement/*.nc is globbed inside the SAFE.
        safe_name = (
            "S1A_WV_OCN__2SSV_20220101T100000_20220101T100020_000000_000000_0000.SAFE"
        )
        return {
            "safewv": os.path.join(tempfile.gettempdir(), safe_name),
            "ds_alti": mock_alti_ds,
            "tree_alti": MagicMock(),
            "altidb": "cmems",
            "coloc_listing": {},
            "cpt": defaultdict(int),
            "dev": False,
            "conf": mock_conf,
        }

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_0_get_sar_dt")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_measurement_wv"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.open_dataset")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.preprocess_wv_s1_ocn"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_treat_one_safe_wv_success(
        self,
        mock_glob,
        mock_pre,
        mock_open,
        mock_treat,
        mock_step0,
        mock_inputs,
    ):
        """Happy path: one WV measurement matched -> merged dataset returned."""
        t0 = np.datetime64("2022-01-01T10:00:00")
        mock_glob.return_value = ["meas.nc"]
        mock_open.return_value = MagicMock()
        ds_sar = xr.Dataset(
            {"oswTotalHs": (("time_sar",), [1.5])},
            coords={"time_sar": [t0]},
        )
        mock_pre.return_value = ds_sar
        mock_step0.return_value = [np.datetime64("2022-01-01T10:00:00")]

        # The real step_3 output carries the closest point plus a 'time' coord;
        # mirror that so the time->time_ALT rename/astype in treat_one_safe_wv runs.
        match_ds = xr.Dataset(
            {
                "hs_alti_closest": ((), 2.5),
                "time": ((), np.datetime64("2022-01-01T10:00:00")),
            },
            coords={"time_sar": t0},
        )
        mock_treat.return_value = (match_ds, {}, defaultdict(int))

        result_ds, listing, cpt_res = treat_one_safe_wv(**mock_inputs)

        assert "oswTotalHs" in result_ds
        assert "hs_alti_closest" in result_ds
        assert "time_ALT" in result_ds
        mock_treat.assert_called_once()
        # conf is forwarded to treat_one_measurement_wv
        _, treat_kwargs = mock_treat.call_args
        assert treat_kwargs["conf"] is mock_inputs["conf"]
        assert treat_kwargs["altidb"] == "cmems"

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.step_0_get_sar_dt")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.treat_one_measurement_wv"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.xr.open_dataset")
    @patch(
        "unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.preprocess_wv_s1_ocn"
    )
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_treat_one_safe_wv_no_alti_files(
        self,
        mock_glob,
        mock_pre,
        mock_open,
        mock_treat,
        mock_step0,
        mock_inputs,
    ):
        """When no altimeter data was loaded, the safe counter is incremented
        and an empty dataset is returned."""
        t0 = np.datetime64("2022-01-01T10:00:00")
        mock_glob.return_value = ["meas.nc"]
        mock_open.return_value = MagicMock()
        ds_sar = xr.Dataset(
            {"oswTotalHs": (("time_sar",), [1.5])},
            coords={"time_sar": [t0]},
        )
        mock_pre.return_value = ds_sar
        mock_step0.return_value = [np.datetime64("2022-01-01T10:00:00")]

        # Empty altimeter dataset -> falsy -> no matchup loop.
        mock_inputs["ds_alti"] = xr.Dataset()

        result_ds, _, cpt_res = treat_one_safe_wv(**mock_inputs)

        assert cpt_res["nb_safe_without_alti_files"] == 1
        assert "oswTotalHs" not in result_ds
        mock_treat.assert_not_called()
