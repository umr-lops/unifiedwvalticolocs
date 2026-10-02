"""Unit tests for unified_coloc_WV_alti_cmems_or_cci."""

import datetime
import tempfile
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from scipy.spatial import cKDTree

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    from_npdt64_to_dt,
    haversine,
    is_cmems_file_matching_in_time,
    latlon_to_xyz,
    step_0_get_sar_dt,
    step_1_temp_match,
    step_2_geographic_match,
    step_3_closer_temp_match,
)


class TestHelpers:
    """Tests for pure helper functions."""

    def test_haversine(self):
        """Distance between two points ~111 km per degree of latitude."""
        dist = haversine(0, 0, 0, 1)
        assert np.isclose(dist, 111.19, atol=0.1)

        dist_zero = haversine(10, 10, 10, 10)
        assert dist_zero == 0.0

    def test_from_npdt64_to_dt(self):
        """numpy.datetime64 is converted to an aware UTC datetime."""
        np_dt = np.datetime64("2022-01-01T12:00:00")
        py_dt = from_npdt64_to_dt(np_dt)

        assert isinstance(py_dt, datetime.datetime)
        assert py_dt.year == 2022
        assert py_dt.hour == 12
        assert py_dt.tzinfo == datetime.UTC

    def test_latlon_to_xyz_unit_sphere(self):
        """latlon_to_xyz produces points on the unit sphere."""
        xyz = latlon_to_xyz(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
        # Equator/greenwich -> (1, 0, 0); North pole -> (0, 0, 1)
        assert np.allclose(xyz[0], [1.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(xyz[1], [0.0, 0.0, 1.0], atol=1e-9)
        # Norm must be 1
        assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0)

    def test_is_cmems_file_matching_in_time(self):
        """Parse CMEMS filename and test overlap with a time window."""
        # Pattern: global_vavh_l3_rt_XX_START_STOP_GEN.nc
        fname = (
            "global_vavh_l3_rt_j3_" "20220101T000000_20220101T235959_20220102T030000.nc"
        )

        sta = datetime.datetime(2021, 12, 31, tzinfo=datetime.UTC)
        sto = datetime.datetime(2022, 1, 2, tzinfo=datetime.UTC)

        lst_out, groups_out = is_cmems_file_matching_in_time(fname, [], {}, sta, sto)

        assert fname in lst_out
        assert "20220101T00" in groups_out

        # Window outside the file
        sta_out = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
        sto_out = datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC)

        lst_out_2, _ = is_cmems_file_matching_in_time(
            fname, [], groups_out, sta_out, sto_out
        )
        assert len(lst_out_2) == 0


class TestSteps:
    """Tests for the numbered pipeline steps."""

    @pytest.fixture
    def mock_sar_ds(self) -> xr.Dataset:
        """Create a dummy SAR dataset (one WV measurement)."""
        times = [np.datetime64("2022-01-01T12:00:00")]
        lats = [10.0]
        lons = [10.0]
        return xr.Dataset(
            {
                "oswLat": (("time_sar",), lats),
                "oswLon": (("time_sar",), lons),
            },
            coords={"time_sar": times},
        )

    @pytest.fixture
    def mock_alti_ds(self) -> xr.Dataset:
        """Create a dummy CMEMS-like alti dataset (1 point, 1 h after SAR)."""
        times = [np.datetime64("2022-01-01T13:00:00")]
        return xr.Dataset(
            {
                "latitude": (("time",), [10.0]),
                "longitude": (("time",), [10.0]),
                "VAVH": (("time",), [2.5]),
                "fname": (("time",), ["dummy_alti.nc"]),
            },
            coords={"time": times},
        )

    # ------------------------------------------------------------------ step 0
    def test_step_0_get_sar_dt(self, mock_sar_ds):
        """Extract SAR datetimes from the SAR dataset."""
        dt_list = step_0_get_sar_dt(mock_sar_ds)
        assert len(dt_list) == 1
        assert isinstance(dt_list[0], datetime.datetime)
        assert dt_list[0].hour == 12

    # ------------------------------------------------------------------ step 1
    @patch("unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci.glob.glob")
    def test_step_1_temp_match_cci(self, mock_glob):
        """CCI: glob is called for each of 3 days (D-1, D, D+1)."""
        mock_glob.return_value = ["file1.nc", "file2.nc"]

        date_sar = datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

        files = step_1_temp_match(
            date_sar_dt=date_sar,
            path_altimeters=tempfile.gettempdir(),
            acro_alti="jason-3",
            altidb="cci",
        )

        # 3 days x 2 files returned by the mock = 6
        assert len(files) == 6
        assert "file1.nc" in files
        assert mock_glob.called

    def test_step_1_temp_match_invalid_altidb(self):
        """Unknown altidb raises a ValueError."""
        date_sar = datetime.datetime(2022, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
        with pytest.raises(ValueError):
            step_1_temp_match(
                date_sar_dt=date_sar,
                path_altimeters=tempfile.gettempdir(),
                acro_alti="jason-3",
                altidb="unknown_db",
            )

    # ------------------------------------------------------------------ step 2
    def test_step_2_geographic_match(self, mock_sar_ds, mock_alti_ds):
        """Match when SAR and alti points are identical (distance 0 km)."""
        points_alt_xyz = latlon_to_xyz(
            mock_alti_ds["latitude"].values,
            mock_alti_ds["longitude"].values,
        )
        tree = cKDTree(points_alt_xyz)

        subset = step_2_geographic_match(
            mock_sar_ds,
            mock_alti_ds,
            tree,
            delta_dist_km=2.0,
        )

        assert subset is not None
        assert len(subset["time"]) == 1

    def test_step_2_geographic_match_no_match(self, mock_sar_ds, mock_alti_ds):
        """SAR far from all alti points -> None."""
        points_alt_xyz = latlon_to_xyz(
            mock_alti_ds["latitude"].values,
            mock_alti_ds["longitude"].values,
        )
        tree = cKDTree(points_alt_xyz)

        # Move SAR 40 deg north: ~4400 km away, well outside a 2 km radius
        mock_sar_ds["oswLat"] = (("time_sar",), [50.0])

        subset_fail = step_2_geographic_match(
            mock_sar_ds,
            mock_alti_ds,
            tree,
            delta_dist_km=2.0,
        )
        assert subset_fail is None

    # ------------------------------------------------------------------ step 3
    def test_step_3_closer_temp_match_success(self, mock_sar_ds, mock_alti_ds):
        """Alti 1 h after SAR is kept when the window is 3 h."""
        cpt = defaultdict(int)

        subset_ok, files_list, cpt_out = step_3_closer_temp_match(
            sar_dataset=mock_sar_ds,
            subset_alti=mock_alti_ds,
            delta_t_max_minutes=180,  # 3 h
            altidb="cmems",
            cpt=cpt,
        )

        assert subset_ok is not None
        assert len(files_list) == 1
        assert "dummy_alti.nc" in files_list
        assert np.isclose(subset_ok["hs_alti_closest"].values, 2.5)
        # Alti is 1 h after SAR -> +3600 s
        assert np.all(subset_ok["delta_t_closest"].values == np.timedelta64(3600, "s"))
        # Same location -> distance 0
        assert np.isclose(subset_ok["delta_d_closest"].values, 0.0)
        assert cpt_out["maximum_colocs_time_and_space"] == 1

    def test_step_3_closer_temp_match_fail_time(self, mock_sar_ds, mock_alti_ds):
        """Alti 1 h after SAR is rejected when the window is 30 min."""
        cpt = defaultdict(int)

        subset_ok, files_list, _ = step_3_closer_temp_match(
            sar_dataset=mock_sar_ds,
            subset_alti=mock_alti_ds,
            delta_t_max_minutes=30,
            altidb="cmems",
            cpt=cpt,
        )

        assert subset_ok is None
        assert len(files_list) == 0

    def test_step_3_renames_variables(self, mock_sar_ds, mock_alti_ds):
        """On success, variables are renamed to generic names."""
        cpt = defaultdict(int)

        subset_ok, _, _ = step_3_closer_temp_match(
            sar_dataset=mock_sar_ds,
            subset_alti=mock_alti_ds,
            delta_t_max_minutes=180,
            altidb="cmems",
            cpt=cpt,
        )

        assert subset_ok is not None
        # Generic names present, original ones gone
        assert "hs_alti_closest" in subset_ok
        assert "lon_ALT" in subset_ok
        assert "lat_ALT" in subset_ok
        assert "VAVH" not in subset_ok
        assert "longitude" not in subset_ok
        assert "latitude" not in subset_ok

    def test_step_3_date_arithmetic_zero(self):
        """When SAR and alti times coincide, delta_t == 0 s."""
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

        cpt = defaultdict(int)
        subset_ok, _, _ = step_3_closer_temp_match(
            sar_dataset=ds_sar,
            subset_alti=ds_alti,
            delta_t_max_minutes=60,
            altidb="cmems",
            cpt=cpt,
        )

        assert subset_ok is not None
        assert np.all(subset_ok["delta_t_closest"].values == np.timedelta64(0, "s"))

    def test_step_3_invalid_altidb(self, mock_sar_ds, mock_alti_ds):
        """Unknown altidb raises a KeyError (VAR_NAMES lookup)."""
        cpt = defaultdict(int)
        with pytest.raises(KeyError):
            step_3_closer_temp_match(
                sar_dataset=mock_sar_ds,
                subset_alti=mock_alti_ds,
                delta_t_max_minutes=180,
                altidb="unknown_db",
                cpt=cpt,
            )
