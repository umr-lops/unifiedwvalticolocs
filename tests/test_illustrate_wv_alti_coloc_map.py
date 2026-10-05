"""Unit tests for illustrate_wv_alti_coloc_map."""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from shapely.geometry import Point, Polygon

import unifiedwvalticolocs.illustrate_wv_alti_coloc_map as illust
from unifiedwvalticolocs.illustrate_wv_alti_coloc_map import (
    _alti_tree,
    _find_ns,
    _format_closest_alti_line,
    _ns_map,
    _swap_polygon_axes,
    _valid_alti,
    as_lonlat_polygon,
    haversine_circle,
    imagette_polygon_from_ds,
    manifest_footprint_to_polygon,
    parse_manifest_footprints,
    parse_ocn_filename,
    plot_coloc_sequence,
    plot_safe_and_imagette,
    safe_basename_from_manifest,
)

MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Manifest_File xmlns:safe="http://www.esa.int/safe/sentinel-1/1.0"
               xmlns:gml="http://www.opengis.net/gml/3.2">
  <DataObjectSection>
    <safe:footPrint>
      <gml:coordinates>10.0,20.0 10.0,21.0 11.0,21.0 11.0,20.0</gml:coordinates>
    </safe:footPrint>
    <safe:footPrint>
      <gml:coordinates>11.0,20.0 11.0,21.0 12.0,21.0 12.0,20.0</gml:coordinates>
    </safe:footPrint>
    <safe:footPrint/>
  </DataObjectSection>
</Manifest_File>
"""

OCN_FILENAME = "s1a-wv2-ocn-vv-20260112t045817-20260112t045820-062729-07ddb1-036.nc"


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Write a minimal two-frame manifest.safe and return its path."""
    safe_dir = tmp_path / "S1A_WV_OCN__2SSV_20260112T045817_062729_07ddb1_036.SAFE"
    safe_dir.mkdir()
    path = safe_dir / "manifest.safe"
    path.write_text(MANIFEST_XML)
    return path


@pytest.fixture
def dssar() -> xr.Dataset:
    """Single-measurement WV OCN dataset with a valid OCN filename source."""
    ds = xr.Dataset(
        {
            "oswLon": (("time_sar",), [10.0]),
            "oswLat": (("time_sar",), [10.0]),
        },
        coords={"time_sar": [np.datetime64("2026-01-12T04:58:17")]},
    )
    ds.encoding["source"] = f"/data/{OCN_FILENAME}"
    return ds


@pytest.fixture
def dsalti() -> xr.Dataset:
    """Small CCI-like altimeter track: two points near the SAR, one far away."""
    return xr.Dataset(
        {
            "swh_denoised": (("time",), [1.0, 1.1, 1.2]),
            "lon": (("time",), [10.0, 10.01, 30.0]),
            "lat": (("time",), [10.0, 10.01, 30.0]),
            "fname": (("time",), ["a.nc", "a.nc", "b.nc"]),
        },
        coords={
            "time": [
                np.datetime64("2026-01-12T04:58:20"),
                np.datetime64("2026-01-12T04:59:00"),
                np.datetime64("2026-01-12T05:10:00"),
            ]
        },
    )


class FakeGoogleTiles:
    """GoogleTiles stand-in exposing a real CRS without any network access."""

    crs = PlateCarree()

    def __init__(self, *args, **kwargs):
        pass


class TestManifestParsing:
    """Tests for manifest.safe footPrint parsing."""

    def test_parse_manifest_footprints(self, manifest_path: Path):
        """Two valid footPrints are parsed, the empty one is skipped."""
        fps = parse_manifest_footprints(manifest_path)
        assert len(fps) == 2
        assert fps[0].frame_id == 0
        assert fps[1].frame_id == 1
        assert fps[0].corners[0] == (10.0, 20.0)
        assert fps[1].corners[-1] == (12.0, 20.0)

    def test_ns_map_and_find_ns(self, manifest_path: Path):
        """Namespace map contains both URIs and prefixes resolve."""
        ns = _ns_map(manifest_path)
        assert ns["safe"] == "http://www.esa.int/safe/sentinel-1/1.0"
        assert ns["gml"] == "http://www.opengis.net/gml/3.2"
        assert _find_ns(ns, "safe", "safe/sentinel-1") == ns["safe"]
        assert _find_ns(ns, "nope", "opengis.net/gml") == ns["gml"]

    def test_find_ns_missing_raises(self, manifest_path: Path):
        """Unknown prefix and URI substring raise KeyError."""
        ns = _ns_map(manifest_path)
        with pytest.raises(KeyError):
            _find_ns(ns, "nope", "does-not-exist")

    def test_safe_basename_from_manifest(self, manifest_path: Path):
        """manifest.safe resolves to its parent SAFE directory name."""
        assert safe_basename_from_manifest(manifest_path) == (
            "S1A_WV_OCN__2SSV_20260112T045817_062729_07ddb1_036.SAFE"
        )
        assert safe_basename_from_manifest("/data/foo.nc") == "foo.nc"


class TestFootPrintGeometry:
    """Tests for footprint polygon conversion and axis-order helpers."""

    def test_manifest_footprint_to_polygon(self, manifest_path: Path):
        """Corners (lat, lon) become a closed (lon, lat) polygon."""
        fp = parse_manifest_footprints(manifest_path)[0]
        poly = manifest_footprint_to_polygon(fp)
        assert poly.is_valid
        # (lat, lon) = (10.0, 20.0) -> (lon, lat) = (20.0, 10.0)
        assert poly.exterior.coords[0] == (20.0, 10.0)
        assert poly.exterior.coords[0] == poly.exterior.coords[-1]
        assert poly.area == pytest.approx(1.0)

    def test_as_lonlat_polygon_already_lonlat(self):
        """x range beyond 90 -> unambiguously (lon, lat), returned as-is."""
        poly = Polygon([(100.0, 10.0), (170.0, 10.0), (170.0, 50.0), (100.0, 50.0)])
        assert as_lonlat_polygon(poly) is poly

    def test_as_lonlat_polygon_swaps_latlon(self):
        """y range beyond 90 -> unambiguously (lat, lon), gets swapped."""
        poly = Polygon([(10.0, 100.0), (10.0, 170.0), (50.0, 170.0), (50.0, 100.0)])
        out = as_lonlat_polygon(poly)
        assert out.exterior.coords[0] == (100.0, 10.0)

    def test_as_lonlat_polygon_ambiguous_no_reference(self):
        """Both axes in [-90, 90] without reference -> assumed (lon, lat)."""
        poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)])
        assert as_lonlat_polygon(poly) is poly

    def test_as_lonlat_polygon_ambiguous_with_reference(self):
        """Reference point closer to the swapped centroid triggers a swap."""
        # Stored as (x=lat 5..7, y=lon 8..10): reference (9, 6) is the
        # swapped (lon, lat) centroid.
        poly = Polygon([(5.0, 8.0), (5.0, 10.0), (7.0, 10.0), (7.0, 8.0)])
        out = as_lonlat_polygon(poly, ref_lon=9.0, ref_lat=6.0)
        assert out.exterior.coords[0] == (8.0, 5.0)

    def test_as_lonlat_polygon_invalid_bounds(self):
        """Coordinates outside WGS84 bounds raise ValueError."""
        poly = Polygon([(200.0, 10.0), (210.0, 10.0), (210.0, 20.0), (200.0, 20.0)])
        with pytest.raises(ValueError, match="WGS84"):
            as_lonlat_polygon(poly)

    def test_swap_polygon_axes(self):
        """Exterior and interior rings are swapped coordinate-wise."""
        poly = Polygon(
            [(0.0, 1.0), (2.0, 1.0), (2.0, 3.0), (0.0, 3.0)],
            [[(0.5, 1.5), (1.5, 1.5), (1.5, 2.5), (0.5, 2.5)]],
        )
        out = _swap_polygon_axes(poly)
        assert out.exterior.coords[0] == (1.0, 0.0)
        assert out.interiors[0].coords[0] == (1.5, 0.5)


class TestOcnFilename:
    """Tests for the WV OCN filename parser."""

    def test_parse_ocn_filename_valid(self):
        """A standard OCN nc filename is fully parsed."""
        info = parse_ocn_filename(OCN_FILENAME)
        assert info is not None
        assert info.sat == "S1A"
        assert info.wv_number == 2
        assert info.pol == "VV"
        assert info.t_start == datetime.datetime(
            2026, 1, 12, 4, 58, 17, tzinfo=datetime.UTC
        )
        assert info.t_stop == datetime.datetime(
            2026, 1, 12, 4, 58, 20, tzinfo=datetime.UTC
        )
        assert info.orbit == "062729"
        assert info.abs_orbit == "07DDB1"
        assert info.imagette_number == 36

    def test_parse_ocn_filename_with_path(self):
        """Only the basename is matched, the directory is ignored."""
        assert parse_ocn_filename(Path("/data") / OCN_FILENAME) is not None

    def test_parse_ocn_filename_invalid(self):
        """A non-matching filename returns None."""
        assert parse_ocn_filename("s1a-iw-full-res.nc") is None


class TestImagettePolygon:
    """Tests for the imagette polygon builder."""

    def test_single_point_wv(self):
        """A (1, 1) WV becomes a tiny buffered point polygon."""
        ds = xr.Dataset(
            {"oswLon": (("time_sar",), [10.0]), "oswLat": (("time_sar",), [10.0])},
            coords={"time_sar": [np.datetime64("2026-01-12T04:58:17")]},
        )
        poly = imagette_polygon_from_ds(ds)
        assert poly is not None
        assert poly.is_valid
        assert poly.centroid.x == pytest.approx(10.0, abs=1e-3)
        assert poly.centroid.y == pytest.approx(10.0, abs=1e-3)

    def test_grid_wv(self):
        """A 2D grid becomes the perimeter polygon of the grid."""
        lon = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
        lat = np.array([[20.0, 20.0, 20.0], [21.0, 21.0, 21.0], [22.0, 22.0, 22.0]])
        ds = xr.Dataset(
            {"oswLon": (("y", "x"), lon), "oswLat": (("y", "x"), lat)},
            coords={"y": [0, 1, 2], "x": [0, 1, 2]},
        )
        poly = imagette_polygon_from_ds(ds)
        assert poly is not None
        assert poly.is_valid
        assert poly.centroid.x == pytest.approx(11.0, abs=1e-6)
        assert poly.centroid.y == pytest.approx(21.0, abs=1e-6)

    def test_no_finite_coordinates(self):
        """All-NaN coordinates return None."""
        ds = xr.Dataset(
            {"oswLon": (("time_sar",), [np.nan]), "oswLat": (("time_sar",), [np.nan])},
            coords={"time_sar": [np.datetime64("2026-01-12T04:58:17")]},
        )
        assert imagette_polygon_from_ds(ds) is None


class TestHaversineCircle:
    """Tests for the geodesic circle helper."""

    def test_circle_shape_and_closure(self):
        """Returns n_points per axis and a closed loop."""
        lon, lat = haversine_circle(10.0, 10.0, 30.0, n_points=73)
        assert lon.shape == (73,)
        assert lat.shape == (73,)
        assert lon[0] == pytest.approx(lon[-1])
        assert lat[0] == pytest.approx(lat[-1])

    def test_north_point(self):
        """The first point (bearing 0) is due north of the center."""
        lon, lat = haversine_circle(10.0, 10.0, 30.0)
        assert lon[0] == pytest.approx(10.0, abs=1e-9)
        # ~111.19 km per degree of latitude
        assert lat[0] == pytest.approx(10.0 + 30.0 / 111.19, abs=1e-3)

    def test_zero_radius(self):
        """Zero radius collapses the circle to the center."""
        lon, lat = haversine_circle(10.0, 10.0, 0.0)
        assert np.allclose(lon, 10.0)
        assert np.allclose(lat, 10.0)


class TestAltiHelpers:
    """Tests for the internal altimeter dataset helpers."""

    def test_valid_alti_drops_nans_lonlat(self):
        """Non-finite lon/lat records are dropped (lon/lat naming)."""
        ds = xr.Dataset(
            {
                "lon": (("time",), [10.0, np.nan, 12.0]),
                "lat": (("time",), [10.0, 10.0, 12.0]),
            },
            coords={"time": [1, 2, 3]},
        )
        out = _valid_alti(ds)
        assert len(out["time"]) == 2

    def test_valid_alti_drops_nans_longlat(self):
        """Non-finite longitude/latitude records are dropped (CMEMS naming)."""
        ds = xr.Dataset(
            {
                "longitude": (("time",), [10.0, np.nan]),
                "latitude": (("time",), [10.0, 10.0]),
            },
            coords={"time": [1, 2]},
        )
        out = _valid_alti(ds)
        assert len(out["time"]) == 1

    def test_alti_tree(self, dsalti: xr.Dataset):
        """The tree holds one node per valid record."""
        tree = _alti_tree(_valid_alti(dsalti))
        assert len(tree.data) == 3

    @pytest.mark.parametrize("with_delta", [True, False])
    def test_format_closest_alti_line(self, with_delta: bool):
        """Formats filename, time and optional delta-t."""
        ds = xr.Dataset(
            {"VAVH": ((), 1.0)},
            coords={
                "time_ALT": np.datetime64("2026-01-12T04:58:20"),
            },
        )
        ds["fname"] = ((), "global_vavh_l3_rt_j3.nc")
        if with_delta:
            ds["delta_t_closest"] = ((), np.timedelta64(3, "s"))

        line = _format_closest_alti_line(ds)
        assert line is not None
        assert "global_vavh_l3_rt_j3.nc" in line
        assert "2026-01-12 04:58:20" in line
        if with_delta:
            assert "Δt=+3 s" in line

    def test_format_closest_alti_line_none(self):
        """None input returns None."""
        assert _format_closest_alti_line(None) is None

    def test_format_closest_alti_line_no_time(self):
        """Without any time variable only the filename is shown."""
        ds = xr.Dataset({"VAVH": ((), 1.0)})
        ds["fname"] = ((), "a.nc")
        assert _format_closest_alti_line(ds) == "closest alti: a.nc"


@pytest.fixture
def no_tiles(monkeypatch):
    """Patch GoogleTiles and GeoAxes.add_image to avoid any network access."""
    monkeypatch.setattr(illust, "GoogleTiles", lambda *a, **k: FakeGoogleTiles())
    monkeypatch.setattr(GeoAxes, "add_image", lambda *a, **k: None)


class TestPlotting:
    """Tests for the plotting routines (Agg backend, no network)."""

    def test_plot_safe_and_imagette(
        self,
        no_tiles,
        manifest_path: Path,
        dssar: xr.Dataset,
        dsalti: xr.Dataset,
        tmp_path: Path,
    ):
        """Full render returns fig/ax and the coloc sub-datasets."""
        out_png = tmp_path / "plot.png"
        fig, ax, extras = plot_safe_and_imagette(
            manifest_path,
            dssar,
            dsalti=dsalti,
            altidb="cci",
            output_png=out_png,
            return_datasets=True,
        )
        try:
            assert isinstance(fig, plt.Figure)
            assert out_png.exists()
            # 2 of 3 alti points are within 30 km of the SAR point
            assert len(extras["alti_in_radius"]["time"]) == 2
            assert extras["closest_in_time"] is not None
            assert extras["closest_in_time"]["hs_alti_closest"].item() == pytest.approx(
                1.0
            )
            assert len(extras["alti_in_time_and_space"]["time"]) == 2
        finally:
            plt.close(fig)

    def test_plot_safe_and_imagette_no_alti(
        self,
        no_tiles,
        manifest_path: Path,
        dssar: xr.Dataset,
    ):
        """Without altimeter data the sub-datasets stay None."""
        fig, ax, extras = plot_safe_and_imagette(
            manifest_path, dssar, return_datasets=True
        )
        try:
            assert extras["alti_in_radius"] is None
            assert extras["closest_in_time"] is None
        finally:
            plt.close(fig)

    def test_plot_safe_and_imagette_zoom_on_imagette(
        self,
        no_tiles,
        manifest_path: Path,
        dssar: xr.Dataset,
    ):
        """zoom_on_imagette overrides the default SAFE bounding box."""
        fig, ax = plot_safe_and_imagette(
            manifest_path, dssar, zoom_on_imagette=True, imagette_zoom_span_deg=1.5
        )
        try:
            left, right, bottom, top = ax.get_extent()
            assert left == pytest.approx(10.0 - 1.5)
            assert right == pytest.approx(10.0 + 1.5)
        finally:
            plt.close(fig)

    def test_plot_coloc_sequence(
        self,
        no_tiles,
        manifest_path: Path,
        dssar: xr.Dataset,
        dsalti: xr.Dataset,
        tmp_path: Path,
    ):
        """The 7-step sequence saves one png per step and closes the figures."""
        fps = parse_manifest_footprints(manifest_path)
        results = plot_coloc_sequence(
            manifest_path,
            dssar,
            dsalti,
            fps[0],
            altidb="cci",
            outputdir=tmp_path,
            close_after_save=True,
        )
        assert results == []
        expected = [
            "step_1_safe_footprints.png",
            "step_2_safe_and_imagette.png",
            "step_3_with_alti_track.png",
            "step_4_with_radius_circle.png",
            "step_5_geographic_matchups.png",
            "step_6_time_and_space_matchups.png",
            "step_7_closest_in_time.png",
        ]
        for name in expected:
            assert (tmp_path / name).exists()

    def test_plot_coloc_sequence_polygon_footprint(
        self,
        no_tiles,
        manifest_path: Path,
        dssar: xr.Dataset,
        dsalti: xr.Dataset,
        tmp_path: Path,
    ):
        """A shapely Polygon footprint is accepted and figures stay open."""
        poly = Point(10.0, 10.0).buffer(0.5)
        results = plot_coloc_sequence(
            manifest_path,
            dssar,
            dsalti,
            poly,
            altidb="cci",
            outputdir=tmp_path,
            close_after_save=False,
            show=True,
        )
        try:
            assert len(results) == 7
            assert all(fig is not None for fig, _ in results)
        finally:
            plt.close("all")
