import datetime
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from scipy.spatial import KDTree

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    from_npdt64_to_dt,
    haversine,
    is_cmems_file_matching_in_time,
    step_0_get_sar_dt,
    step_1_temp_match,
    step_2_geographic_match,
    step_3_closer_temp_match,
)


class TestHelpers:
    def test_haversine(self):
        """Test calculation of distance between two points."""
        # Distance between (0,0) and (0,1) deg lat is approx 111.19 km
        dist = haversine(0, 0, 0, 1)
        assert np.isclose(dist, 111.19, atol=0.1)

        # Distance between same point is 0
        dist_zero = haversine(10, 10, 10, 10)
        assert dist_zero == 0.0

    def test_from_npdt64_to_dt(self):
        """Test conversion from numpy datetime64 to python datetime."""
        np_dt = np.datetime64("2022-01-01T12:00:00")
        py_dt = from_npdt64_to_dt(np_dt)

        assert isinstance(py_dt, datetime.datetime)
        assert py_dt.year == 2022
        assert py_dt.hour == 12
        # Check timezone awareness (UTC)
        assert py_dt.tzinfo == datetime.UTC

    def test_is_cmems_file_matching_in_time(self):
        """Test parsing CMEMS filenames and checking time windows."""
        # Pattern: global_vavh_l3_rt_XX_START_STOP_GEN.nc
        # Dates are YYYYMMDDTHHMMSS
        fname = (
            "global_vavh_l3_rt_j3_20220101T000000_20220101T235959_20220102T030000.nc"
        )

        # Window covers the file
        sta = datetime.datetime(2021, 12, 31, tzinfo=datetime.UTC)
        sto = datetime.datetime(2022, 1, 2, tzinfo=datetime.UTC)

        lst = []
        groups = {}

        lst_out, groups_out = is_cmems_file_matching_in_time(
            fname, lst, groups, sta, sto
        )

        assert fname in lst_out
        assert "20220101T00" in groups_out

        # Window outside file
        sta_out = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
        sto_out = datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC)

        lst_2 = []
        lst_out_2, _ = is_cmems_file_matching_in_time(
            fname, lst_2, groups, sta_out, sto_out
        )
        assert len(lst_out_2) == 0


class TestSteps:
    @pytest.fixture
    def mock_sar_ds(self):
        """Create a dummy SAR dataset."""
        times = [np.datetime64("2022-01-01T12:00:00")]
        lats = [10.0]
        lons = [10.0]

        ds = xr.Dataset(
            {
                "oswLat": (("time_sar",), lats),
                "oswLon": (("time_sar",), lons),
            },
            coords={"time_sar": times},
        )
        return ds

    @pytest.fixture
    def mock_alti_ds(self):
        """Create a dummy Alti dataset."""
        # Alti time is 1 hour after SAR time
        times = [np.datetime64("2022-01-01T13:00:00")]
        lats = [10.0]
        lons = [10.0]
        swh = [2.5]
        fname = ["dummy_alti.nc"]

        ds = xr.Dataset(
            {
                "latitude": (("time",), lats),
                "longitude": (("time",), lons),
                "VAVH": (("time",), swh),
                "fname": (("time",), fname),
            },
            coords={"time": times},
        )
        return ds

    def test_step_0_get_sar_dt(self, mock_sar_ds):
        """Test extraction of SAR datetimes."""
        dt_list = step_0_get_sar_dt(mock_sar_ds)
        assert len(dt_list) == 1
        assert isinstance(dt_list[0], datetime.datetime)
        assert dt_list[0].hour == 12

    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_step_1_temp_match_cci(self, mock_glob):
        """Test file finding logic for CCI altimeter database."""
        mock_glob.return_value = ["file1.nc", "file2.nc"]

        date_sar = datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

        files = step_1_temp_match(
            date_sar_dt=date_sar,
            delta_t_sat=3,
            path_altimeters=tempfile.gettempdir(),
            acro_alti="jason-3",
            altidb="cci",
        )

        # The loop runs for 3 days (Dec 31, Jan 1, Jan 2) and the mock returns
        # 2 files per iteration, so we expect 6 files total.
        assert len(files) == 6
        assert "file1.nc" in files
        assert mock_glob.called

    def test_step_1_temp_match_invalid_altidb(self):
        """Test that an invalid altidb raises a ValueError."""
        date_sar = datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
        with pytest.raises(ValueError):
            step_1_temp_match(
                date_sar_dt=date_sar,
                delta_t_sat=3,
                path_altimeters=tempfile.gettempdir(),
                acro_alti="jason-3",
                altidb="unknown_db",
            )

    def test_step_2_geographic_match(self, mock_sar_ds, mock_alti_ds):
        """Test KDTree spatial matching."""
        # Create tree from alti points
        points_alt = np.c_[mock_alti_ds["latitude"], mock_alti_ds["longitude"]]
        tree = KDTree(points_alt)

        # Run step 2 with a generous spatial window so the co-located points match
        subset = step_2_geographic_match(
            mock_sar_ds, mock_alti_ds, tree, delta_dist_degree=2.0
        )

        # Should match because SAR and alti coords are identical (10, 10)
        assert subset is not None
        assert len(subset["time"]) == 1

    def test_step_2_geographic_match_no_match(self, mock_sar_ds, mock_alti_ds):
        """Test that a SAR point far from all alti points returns None."""
        points_alt = np.c_[mock_alti_ds["latitude"], mock_alti_ds["longitude"]]
        tree = KDTree(points_alt)

        # Move SAR far away so no alti point falls within the search radius
        mock_sar_ds["oswLat"] = (("time_sar",), [50.0])
        subset_fail = step_2_geographic_match(
            mock_sar_ds, mock_alti_ds, tree, delta_dist_degree=2.0
        )

        assert subset_fail is None

    def test_step_3_closer_temp_match_success(self, mock_sar_ds, mock_alti_ds):
        """Test exact time and space matching logic: alti 1 h after SAR."""
        delta_t_short = 3 * 3600  # 3-hour window

        result = step_3_closer_temp_match(
            sar_dataset=mock_sar_ds,
            subset_alti=mock_alti_ds,
            delta_t_sat_short=delta_t_short,
            altidb="cmems",
        )

        list_alti_pts, delta_t, hs_closest, _, _, _, _, _, _, _, _ = result

        assert len(list_alti_pts) == 1
        assert np.isclose(hs_closest, 2.5)
        # Alti is 1 h *after* SAR, so delta_t = +3600 s
        assert delta_t == np.timedelta64(3600, "s")

    def test_step_3_closer_temp_match_fail_time(self, mock_sar_ds, mock_alti_ds):
        """Test filtering out points outside the time window (30-min window)."""
        delta_t_short = 1800  # 30-minute window

        # Alti is at 13:00, SAR at 12:00 → difference is 1 h → outside 30-min window
        result = step_3_closer_temp_match(
            sar_dataset=mock_sar_ds,
            subset_alti=mock_alti_ds,
            delta_t_sat_short=delta_t_short,
            altidb="cmems",
        )

        list_alti_pts = result[0]
        assert len(list_alti_pts) == 0
        assert np.isnan(result[2])  # hs_closest should be NaN

    def test_step_3_date_arithmetic_fix(self):
        """
        Verify that time-difference arithmetic works correctly when SAR and
        alti times are identical (expected delta_t == 0 s).
        """
        times = [np.datetime64("2022-01-01T12:00:00")]

        ds_alti = xr.Dataset(
            {
                "latitude": (("time",), [10.0]),
                "longitude": (("time",), [10.0]),
                "VAVH": (("time",), [2.0]),
                "fname": (("time",), ["f.nc"]),
            },
            coords={"time": times},
        )

        ds_sar = xr.Dataset(
            {
                "oswLat": (("time_sar",), [10.0]),
                "oswLon": (("time_sar",), [10.0]),
            },
            coords={"time_sar": times},
        )

        result = step_3_closer_temp_match(
            sar_dataset=ds_sar,
            subset_alti=ds_alti,
            delta_t_sat_short=3600,
            altidb="cmems",
        )

        delta_t = result[1]
        assert delta_t == np.timedelta64(0, "s")
