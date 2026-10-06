"""Plot SAFE footprints + one WV imagette + altimeter track over Google tiles.

Illustration helper for the unified WV/altimeter colocation product.

Requires the optional ``viz`` extra (``cartopy``, ``matplotlib``, ``shapely``)::

    pip install unifiedwvalticolocs[viz]
"""

from typing import cast

import datetime as dt
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    EARTH_RADIUS_KM,
    latlon_to_xyz,
    step_2_geographic_match,
    step_3_closer_temp_match,
)

logger = logging.getLogger(__name__)

# A single step of the colocation illustration sequence: a display name plus
# the keyword arguments forwarded to :func:`plot_safe_and_imagette`.
Step = dict[str, object]


# ---------------------------------------------------------------------------
# 1) manifest.safe parsing
# ---------------------------------------------------------------------------
@dataclass
class FootPrint:
    """A single SAR frame footprint."""

    frame_id: int
    corners: list[tuple[float, float]]  # (lat, lon)


def _ns_map(manifest_path: str | Path) -> dict[str, str]:
    """Return the XML namespace prefix -> URI map of a manifest.safe file.

    Args:
        manifest_path: Path to the ``manifest.safe`` file.

    Returns:
        Dictionary mapping namespace prefixes to URIs.
    """
    # manifest.safe is a local ESA-generated file, not untrusted input.
    ns_map = {}
    for _, (prefix, uri) in ET.iterparse(
        manifest_path, events=["start-ns"]
    ):  # nosec: S314
        ns_map[prefix] = uri
    return ns_map


def _find_ns(ns: dict[str, str], prefix: str, uri_substring: str) -> str:
    """Resolve a namespace URI by preferred prefix, else URI substring.

    Args:
        ns: Namespace prefix -> URI map (see :func:`_ns_map`).
        prefix: Preferred namespace prefix.
        uri_substring: Substring to match against URIs if the prefix is absent.

    Returns:
        The matching namespace URI.

    Raises:
        KeyError: If neither the prefix nor a matching URI is found.
    """
    if prefix in ns:
        return ns[prefix]
    for uri in ns.values():
        if uri_substring in uri:
            return uri
    raise KeyError(f"{prefix} not found in {ns}")


def parse_manifest_footprints(manifest_path: str | Path) -> list[FootPrint]:
    """Extract all ``<safe:footPrint>`` elements as (lat, lon) polygons.

    Args:
        manifest_path: Path to the SAFE's ``manifest.safe`` file.

    Returns:
        List of :class:`FootPrint`, one per ``<safe:footPrint>`` element.
    """
    ns = _ns_map(manifest_path)
    safe_ns = _find_ns(ns, "safe", "safe/sentinel-1")
    gml_ns = _find_ns(ns, "gml", "opengis.net/gml")

    root = ET.parse(manifest_path).getroot()  # nosec: S314
    footprints: list[FootPrint] = []
    for i, fp in enumerate(root.iter(f"{{{safe_ns}}}footPrint")):
        coord = fp.find(f"{{{gml_ns}}}coordinates")
        if coord is None or not coord.text:
            continue
        corners = [
            (float(part[0]), float(part[1]))
            for part in (tok.split(",") for tok in coord.text.split())
        ]
        footprints.append(FootPrint(frame_id=i, corners=corners))
    return footprints


def manifest_footprint_to_polygon(fp: FootPrint) -> Polygon:
    """Convert a manifest FootPrint (lat, lon) to a shapely Polygon (lon, lat).

    Args:
        fp: A ``FootPrint`` dataclass as returned by
            ``parse_manifest_footprints`` (corners in EPSG:4326 lat/lon order).

    Returns:
        A closed ``shapely.geometry.Polygon`` in (lon, lat) axis order,
        ready to be plotted with Cartopy's ``PlateCarree``.
    """
    coords = [(lon, lat) for (lat, lon) in fp.corners]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return Polygon(coords)


def as_lonlat_polygon(
    poly: Polygon,
    ref_lon: float | None = None,
    ref_lat: float | None = None,
) -> Polygon:
    """Return a shapely Polygon guaranteed to be in (lon, lat) axis order.

    Shapely is axis-agnostic; ``Polygon([(a, b), ...])`` treats ``a`` as x and
    ``b`` as y. This helper decides whether a swap is needed so the result can
    be plotted with Cartopy as ``(x=lon, y=lat)``.

    The decision uses, in order:

    1. Range checks: if one axis exceeds the latitude bound (abs(value) > 90)
       it must be longitude, so the orientation is unambiguous.
    2. Reference point: when both axes fit in [-90, 90] (near-equator /
       Greenwich ambiguity), and ``ref_lon`` / ``ref_lat`` are given (e.g. an
       ``oswLon`` / ``oswLat`` pair), the orientation whose centroid is closer
       to the reference wins.

    Args:
        poly: Input polygon, in either axis order.
        ref_lon: Optional reference longitude (typically ``oswLon``).
        ref_lat: Optional reference latitude (typically ``oswLat``).

    Returns:
        A Polygon in (lon, lat) axis order. If the input was already correct,
        the same object is returned.

    Raises:
        ValueError: If neither orientation fits a valid WGS84 lon/lat range.
    """
    coords = np.asarray(poly.exterior.coords, dtype=float)
    xs, ys = coords[:, 0], coords[:, 1]

    x_is_lat = np.all(np.abs(xs) <= 90.0)
    x_is_lon = np.all(np.abs(xs) <= 180.0)
    y_is_lat = np.all(np.abs(ys) <= 90.0)
    y_is_lon = np.all(np.abs(ys) <= 180.0)

    # Case 1: unambiguous — one axis clearly out of latitude bounds
    if x_is_lon and y_is_lat and not x_is_lat:
        return poly  # x exceeds 90 -> x is lon, already (lon, lat)
    if x_is_lat and y_is_lon and not y_is_lat:
        logger.debug("Polygon detected as (lat, lon) — swapping to (lon, lat).")
        return _swap_polygon_axes(poly)

    # Case 2: ambiguous — use the reference point if provided
    if not (x_is_lon and y_is_lat):
        # Neither orientation fits a valid lon/lat range at all
        raise ValueError(
            f"Polygon coordinates do not fit WGS84 bounds. "
            f"x range = [{xs.min()}, {xs.max()}], "
            f"y range = [{ys.min()}, {ys.max()}]."
        )

    if ref_lon is None or ref_lat is None:
        # Both orientations plausible and no reference -> assume (lon, lat)
        logger.debug(
            "Polygon axis order ambiguous (both in [-90, 90]); "
            "assuming (lon, lat). Pass ref_lon/ref_lat to disambiguate."
        )
        return poly

    cx, cy = poly.centroid.x, poly.centroid.y
    d_native = (cx - ref_lon) ** 2 + (cy - ref_lat) ** 2
    d_swapped = (cy - ref_lon) ** 2 + (cx - ref_lat) ** 2

    if d_swapped < d_native:
        logger.debug("Polygon axis order resolved via reference point — swapping.")
        return _swap_polygon_axes(poly)
    return poly


def _swap_polygon_axes(poly: Polygon) -> Polygon:
    """Return a new Polygon with (x, y) swapped on every exterior/interior ring.

    Args:
        poly: Input polygon.

    Returns:
        A Polygon with all exterior/interior ring coordinates swapped.
    """
    exterior = [(y, x) for (x, y) in poly.exterior.coords]
    interiors = [[(y, x) for (x, y) in ring.coords] for ring in poly.interiors]
    return Polygon(exterior, interiors)


# ---------------------------------------------------------------------------
# 2) OCN filename parsing
# ---------------------------------------------------------------------------
_OCN_RE = re.compile(
    r"^(?P<sat>s1[abcd])-wv(?P<wvnum>\d)-ocn-(?P<pol>vv|vh)-"
    r"(?P<tstart>\d{8}t\d{6})-(?P<tstop>\d{8}t\d{6})-"
    r"(?P<orbit>\d{6})-(?P<absorbit>[0-9a-f]+)-"
    r"(?P<imagette>\d+)\.nc$",
    re.IGNORECASE,
)


@dataclass
class OcnInfo:
    """Metadata parsed from a WV OCN nc filename."""

    sat: str
    wv_number: int
    pol: str
    t_start: dt.datetime
    t_stop: dt.datetime
    orbit: str
    abs_orbit: str
    imagette_number: int


def parse_ocn_filename(nc_path: str | Path) -> OcnInfo | None:
    """Parse a WV OCN nc filename. Returns None if the pattern doesn't match.

    Args:
        nc_path: Path (or bare filename) of a WV OCN ``.nc`` file.

    Returns:
        An :class:`OcnInfo` on success, or None if the name doesn't match.
    """
    name = Path(nc_path).name
    m = _OCN_RE.match(name)
    if m is None:
        return None

    def _to_dt(s: str) -> dt.datetime:
        return dt.datetime.strptime(s.upper(), "%Y%m%dT%H%M%S").replace(tzinfo=dt.UTC)

    return OcnInfo(
        sat=m.group("sat").upper(),
        wv_number=int(m.group("wvnum")),
        pol=m.group("pol").upper(),
        t_start=_to_dt(m.group("tstart")),
        t_stop=_to_dt(m.group("tstop")),
        orbit=m.group("orbit"),
        abs_orbit=m.group("absorbit").upper(),
        imagette_number=int(m.group("imagette")),
    )


def safe_basename_from_manifest(manifest_path: str | Path) -> str:
    """Return the SAFE directory basename.

    Args:
        manifest_path: Path to a ``manifest.safe`` file (or any file).

    Returns:
        The parent directory name when the file is named ``manifest.safe``,
        else the file name itself.
    """
    p = Path(manifest_path)
    return p.parent.name if p.name.lower() == "manifest.safe" else p.name


# ---------------------------------------------------------------------------
# 3) Imagette polygon from OCN dataset
# ---------------------------------------------------------------------------
def imagette_polygon_from_ds(dssar: xr.Dataset) -> Polygon | None:
    """Build a shapely Polygon (lon, lat) from the OCN oswLon/oswLat grid.

    Handles the standard WV case (1, 1) by returning a small circular Polygon
    (buffered Point) so the caller always gets a valid, closed geometry.

    Args:
        dssar: WV OCN dataset containing ``oswLon`` and ``oswLat``.

    Returns:
        A ``shapely.geometry.Polygon`` in (lon, lat) axis order, or None if
        the dataset has no valid coordinates.
    """
    lon = np.atleast_2d(np.asarray(dssar["oswLon"].values).squeeze())
    lat = np.atleast_2d(np.asarray(dssar["oswLat"].values).squeeze())

    mask = np.isfinite(lon) & np.isfinite(lat)
    if not mask.any():
        return None

    if lon.shape == (1, 1):
        # Single-point WV -> tiny circular polygon so we still have an
        # exterior for plotting and a centroid that matches oswLon/oswLat.
        return Point(float(lon.ravel()[0]), float(lat.ravel()[0])).buffer(1e-3)

    perim_lon = np.concatenate(
        [lon[0, :], lon[1:, -1], lon[-1, -2::-1], lon[-2:0:-1, 0], lon[0, :1]]
    )
    perim_lat = np.concatenate(
        [lat[0, :], lat[1:, -1], lat[-1, -2::-1], lat[-2:0:-1, 0], lat[0, :1]]
    )
    coords = list(zip(perim_lon.tolist(), perim_lat.tolist()))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return Polygon(coords)


# ---------------------------------------------------------------------------
# 4) Geodesic circle helper
# ---------------------------------------------------------------------------
def haversine_circle(
    lon0: float, lat0: float, radius_km: float, n_points: int = 181
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (lon, lat) points forming a geodesic circle of given radius.

    Args:
        lon0: Center longitude in degrees.
        lat0: Center latitude in degrees.
        radius_km: Circle radius in kilometers.
        n_points: Number of vertices (>= 4).

    Returns:
        Tuple of longitude and latitude arrays, closed loop.
    """
    angular = radius_km / EARTH_RADIUS_KM
    bearings = np.deg2rad(np.linspace(0.0, 360.0, n_points))

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    lat_rad = np.arcsin(
        np.sin(lat0_rad) * np.cos(angular)
        + np.cos(lat0_rad) * np.sin(angular) * np.cos(bearings)
    )
    lon_rad = lon0_rad + np.arctan2(
        np.sin(bearings) * np.sin(angular) * np.cos(lat0_rad),
        np.cos(angular) - np.sin(lat0_rad) * np.sin(lat_rad),
    )
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi

    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


# ---------------------------------------------------------------------------
# 5) Internal alti helpers
# ---------------------------------------------------------------------------
def _alti_lonlat(dsalti: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) float arrays from an altimeter dataset.

    Args:
        dsalti: Altimeter dataset with either ``lon``/``lat`` or
            ``longitude``/``latitude`` variables.

    Returns:
        Tuple of (longitude, latitude) float arrays.
    """
    if "lon" in dsalti and "lat" in dsalti:
        lon = np.asarray(dsalti["lon"].values, dtype=float)
        lat = np.asarray(dsalti["lat"].values, dtype=float)
    else:
        lon = np.asarray(dsalti["longitude"].values, dtype=float)
        lat = np.asarray(dsalti["latitude"].values, dtype=float)
    return lon, lat


def _valid_alti(dsalti: xr.Dataset) -> xr.Dataset:
    """Drop altimeter records with non-finite lon or lat.

    Args:
        dsalti: Altimeter dataset with a ``time`` dimension.

    Returns:
        The dataset subset to records with finite coordinates.
    """
    lon, lat = _alti_lonlat(dsalti)
    mask = np.isfinite(lon) & np.isfinite(lat)
    return dsalti.isel(time=np.flatnonzero(mask))


def _alti_tree(dsalti_valid: xr.Dataset) -> cKDTree:
    """Build a cKDTree over the (lat, lon) of a valid altimeter dataset.

    Args:
        dsalti_valid: Altimeter dataset with finite coordinates.

    Returns:
        A ``scipy.spatial.cKDTree`` over the unit-sphere XYZ coordinates.
    """
    lon, lat = _alti_lonlat(dsalti_valid)
    xyz = latlon_to_xyz(lat, lon)
    return cKDTree(xyz)


def _format_closest_alti_line(ds_closest: xr.Dataset | None) -> str | None:
    """Build the alti info line for the figure suptitle.

    Returns something like:
        closest alti: global_vavh_l3_rt_j3_20250612T...nc  —  time_ALT 2025-06-12 14:11:32 UTC  (Δt=+15 s)

    Or None if the dataset lacks the required fields.

    Args:
        ds_closest: Altimeter dataset for the closest-in-time matchup, or None.

    Returns:
        The formatted suptitle line, or None if ``ds_closest`` is None.
    """
    if ds_closest is None:
        return None

    # --- filename --------------------------------------------------------
    fname = "?"
    if "fname" in ds_closest:
        raw = np.asarray(ds_closest["fname"].values).ravel()
        if raw.size > 0:
            fname = str(raw[0])

    # --- time (prefer time_ALT if it exists, else time) ------------------
    t_alti = None
    for cand in ("time_ALT", "time"):
        if cand in ds_closest.coords or cand in ds_closest:
            t_alti = ds_closest[cand].values
            break
    if t_alti is None:
        return f"closest alti: {fname}"
    # handle shape () and (1,) uniformly
    t64 = np.asarray(t_alti).ravel()[0].astype("datetime64[s]")
    # pretty-print as "YYYY-MM-DD HH:MM:SS"
    t_str = t64.astype("M8[s]").item().strftime("%Y-%m-%d %H:%M:%S")

    # --- Δt if available -------------------------------------------------
    dt_suffix = ""
    if "delta_t_closest" in ds_closest:
        dt_val = np.asarray(ds_closest["delta_t_closest"].values).ravel()[0]
        dt_sec = float(
            np.timedelta64(dt_val, "s").astype("timedelta64[s]")
            / np.timedelta64(1, "s")
        )
        dt_suffix = f"  (Δt={dt_sec:+.0f} s)"

    return f"closest alti: {fname}  —  time_ALT {t_str} UTC{dt_suffix}"


# ---------------------------------------------------------------------------
# 6) Main plotting routine
# ---------------------------------------------------------------------------
def plot_safe_and_imagette(
    manifest_path: str | Path,
    dssar: xr.Dataset,
    dsalti: xr.Dataset | None = None,
    tree_alti: cKDTree | None = None,
    imagette_footprint: Polygon | None = None,
    zoom: int = 7,
    pad_deg: float = 0.5,
    output_png: str | Path | None = None,
    annotate: bool = True,
    # --- alti track styling ---
    alti_color: str = "cyan",
    alti_marker_size: float = 6.0,
    highlight_closest_alti: bool = True,
    # --- radius circle + alti subsets ---
    show_radius_circle: bool = True,
    radius_km: float = 30.0,
    show_alti_in_radius: bool = True,
    show_alti_time_and_space_match: bool = True,
    show_closest_in_time: bool = True,
    delta_t_minutes: int = 25,
    altidb: str = "cmems",
    # --- layout ---
    hide_top_xticks: bool = True,
    title_pad: float = 20.0,
    zoom_on_imagette: bool = False,
    imagette_zoom_span_deg: float = 1.5,
    # --- comparison display ---
    show_centroid_comparison: bool = True,
    # --- returns ---
    return_datasets: bool = False,
    extras: dict[str, xr.Dataset | None] | None = None,
    show_imagette: bool = True,
    extent: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (11, 10),
    dpi: int = 100,
) -> tuple[Figure, Axes] | tuple[Figure, Axes, dict[str, xr.Dataset | None]]:
    """Render SAFE footprints + WV imagette + alti track over satellite tiles.

    Args:
        manifest_path: Path to the SAFE's ``manifest.safe``.
        dssar: WV OCN dataset containing ``oswLat``/``oswLon``.
        dsalti: Optional full altimeter track (``lon``, ``lat``, ``time``).
        tree_alti: Optional prebuilt cKDTree over ``dsalti``. Built internally
            from ``dsalti`` if None.
        imagette_footprint: Optional ``shapely.geometry.Polygon`` in
            **(lon, lat)** axis order. If None, it is derived from
            ``dssar['oswLon']`` / ``dssar['oswLat']``.
        zoom: Tile zoom level (5–6 = whole SAFE, 11–12 = one vignette).
        pad_deg: Padding around the SAFE bounding box, in degrees.
        output_png: Optional output file path.
        annotate: If True, annotate the imagette with date/time + number.
        alti_color: Matplotlib color for the full altimeter track.
        alti_marker_size: Marker size for the full-track scatter.
        highlight_closest_alti: If True, ring the alti point closest in time
            to the imagette acquisition (global argmin over the full track).
        show_radius_circle: If True, draw the ``radius_km`` haversine circle
            around the imagette centroid.
        radius_km: Radius of the search circle, in kilometers.
        show_alti_in_radius: If True, plot alti points within ``radius_km``
            (via ``step_2_geographic_match``) in **orange**.
        show_alti_time_and_space_match: If True, plot alti points matching
            the (radius, ``delta_t_minutes``) criteria (4th return of
            ``step_3_closer_temp_match``) in **green**.
        show_closest_in_time: If True, ring the point returned as the first
            return of ``step_3_closer_temp_match`` (single best colocation).
        delta_t_minutes: Time window passed to ``step_3_closer_temp_match``.
        altidb: ``"cci"`` or ``"cmems"`` — selects the SWH var name inside
            ``step_3_closer_temp_match``.
        hide_top_xticks: If True, suppress longitude labels on the top edge.
        title_pad: Padding between the map and the axes title (points).
        zoom_on_imagette: If True, override extent to focus on the imagette
            (needed to actually see the radius circle).
        imagette_zoom_span_deg: Half-span (degrees) used when
            ``zoom_on_imagette=True``.
        show_centroid_comparison: If True, show both the polygon centroid
            and the oswLon/oswLat values in a small-font box.
        return_datasets: If True, also return a dict with extracted alti
            sub-datasets. See **Returns**.
        extras: Precomputed alti sub-datasets (see **Returns**). If None they
            are computed from ``dsalti`` when needed and populated in place.
        show_imagette: If True, draw the WV imagette footprint.
        extent: Optional fixed map extent
            ``(left, right, bottom, top)`` in PlateCarree degrees.
        figsize: Figure size in inches.
        dpi: Resolution used when saving ``output_png``.

    Returns:
        If ``return_datasets`` is False: ``(fig, ax)``.
        If True: ``(fig, ax, extras)`` with ``extras`` containing:

            - ``"alti_in_radius"``: ``xr.Dataset | None``
            - ``"alti_in_time_and_space"``: ``xr.Dataset | None``
            - ``"closest_in_time"``: ``xr.Dataset | None``
    """
    # ------------------------------------------------------------------ SAFE
    safe_fps = parse_manifest_footprints(manifest_path)
    safe_name = safe_basename_from_manifest(manifest_path)

    lats_all = [c[0] for fp in safe_fps for c in fp.corners]
    lons_all = [c[1] for fp in safe_fps for c in fp.corners]
    lat_min, lat_max = min(lats_all), max(lats_all)
    lon_min, lon_max = min(lons_all), max(lons_all)

    # ------------------------------------------------------------- imagette
    if show_imagette:
        if imagette_footprint is None:
            imagette_footprint = imagette_polygon_from_ds(dssar)
    else:
        imagette_footprint = None

    if imagette_footprint is not None:
        # exterior.xy returns (lons, lats) since the Polygon is (lon, lat)
        fp_lon, fp_lat = imagette_footprint.exterior.xy
        fp_lon = np.asarray(fp_lon)
        fp_lat = np.asarray(fp_lat)
    else:
        fp_lon = fp_lat = np.array([])

    # Centroid comes straight from shapely
    centroid: tuple[float, float] | None = None
    if imagette_footprint is not None:
        cx_arr, cy_arr = imagette_footprint.centroid.xy
        centroid = (float(cx_arr[0]), float(cy_arr[0]))

    # --------------------------------------------- oswLon/oswLat ground truth
    osw_lon_val = float(np.asarray(dssar["oswLon"].values).squeeze())
    osw_lat_val = float(np.asarray(dssar["oswLat"].values).squeeze())

    # --------------------------------------------------------- OCN filename
    nc_path = dssar.encoding.get("source")
    logger.debug("nc_path: %s", nc_path)
    ocn_info = parse_ocn_filename(nc_path) if nc_path else None

    # ------------------------------------- alti subsets from colocation module
    if dsalti is not None:
        dsalti_valid = _valid_alti(dsalti)
        logger.debug("alti valid dataset: %d records", len(dsalti_valid["time"]))
    else:
        dsalti_valid = None
    if extras is None:
        extras = {
            "alti_in_radius": None,
            "alti_in_time_and_space": None,
            "closest_in_time": None,
        }
        need_radius = show_alti_in_radius or return_datasets
        need_step3 = (
            show_alti_time_and_space_match or show_closest_in_time or return_datasets
        )
        logger.debug("need_step3: %s", need_step3)
        logger.debug("need_radius: %s", need_radius)

        if dsalti_valid is not None and (need_radius or need_step3):
            if tree_alti is None:
                tree_alti = _alti_tree(dsalti_valid)
            extras["alti_in_radius"] = step_2_geographic_match(
                sards=dssar,
                ds_alti=dsalti_valid,
                tree_alti=tree_alti,
                delta_dist_km=radius_km,
            )
            logger.debug(
                "nb points ok in radius: %s",
                (
                    0
                    if extras["alti_in_radius"] is None
                    else len(extras["alti_in_radius"]["time"])
                ),
            )

            if need_step3 and extras["alti_in_radius"] is not None:
                cpt: defaultdict[str, int] = defaultdict(int)
                (
                    closest_in_time,
                    _files_list,
                    _cpt,
                    all_ts_match,
                ) = step_3_closer_temp_match(
                    sar_dataset=dssar,
                    subset_alti=extras["alti_in_radius"],
                    delta_t_max_minutes=delta_t_minutes,
                    altidb=altidb,
                    cpt=cpt,
                )
                extras["closest_in_time"] = closest_in_time
                extras["alti_in_time_and_space"] = all_ts_match
            else:
                logger.debug("no step3 because no alti in radius or not needed")
        else:
            logger.debug("no step3 because no alti valid or not needed")

    # ---------------------------------------------------- Cartopy + tiles
    esri_url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    google = GoogleTiles(url=esri_url, style="satellite", cache=True)
    fig = plt.figure(figsize=figsize)
    ax: GeoAxes = plt.axes(projection=google.crs)
    if extent is not None:
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    elif zoom_on_imagette and centroid is not None:
        cx, cy = centroid
        ax.set_extent(
            [
                cx - imagette_zoom_span_deg,
                cx + imagette_zoom_span_deg,
                cy - imagette_zoom_span_deg,
                cy + imagette_zoom_span_deg,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_extent(
            [
                lon_min - pad_deg,
                lon_max + pad_deg,
                lat_min - pad_deg,
                lat_max + pad_deg,
            ],
            crs=ccrs.PlateCarree(),
        )

    ax.add_image(google, zoom)

    # ------------------------------------------------------ full alti track
    if dsalti_valid is not None:
        alt_lon, alt_lat = _alti_lonlat(dsalti_valid)

        if alt_lon.size > 0:
            ax.plot(
                alt_lon,
                alt_lat,
                color=alti_color,
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
                transform=ccrs.PlateCarree(),
                label="Altimeter track",
            )
            ax.scatter(
                alt_lon,
                alt_lat,
                s=alti_marker_size,
                c=alti_color,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.85,
                zorder=3,
                transform=ccrs.PlateCarree(),
            )

        if highlight_closest_alti and ocn_info is not None:
            alt_time = dsalti_valid["time"].values
            t_ref = np.datetime64(ocn_info.t_start.replace(tzinfo=None))
            dt_sec = np.abs((alt_time - t_ref) / np.timedelta64(1, "s")).astype(float)
            k = int(np.argmin(dt_sec))
            ax.plot(
                alt_lon[k],
                alt_lat[k],
                marker="o",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor=alti_color,
                markeredgewidth=2.5,
                linestyle="none",
                zorder=5,
                transform=ccrs.PlateCarree(),
                label=f"Track closest (Δt={dt_sec[k]:.0f} s)",
            )

    # ------------------------------------------ radius + alti subsets plots
    if centroid is not None and show_radius_circle and radius_km > 0:
        circ_lon, circ_lat = haversine_circle(centroid[0], centroid[1], radius_km)
        ax.plot(
            circ_lon,
            circ_lat,
            color="yellow",
            linewidth=1.8,
            linestyle="--",
            zorder=4,
            transform=ccrs.PlateCarree(),
            label=f"{radius_km:.0f} km circle",
        )

    if extras["alti_in_radius"] is not None and show_alti_in_radius:
        logger.debug("add alti in radius")
        ds_r = extras["alti_in_radius"]
        r_lon, r_lat = _alti_lonlat(ds_r)
        ax.scatter(
            r_lon,
            r_lat,
            s=alti_marker_size * 5.0,
            marker="s",
            c="orange",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            transform=ccrs.PlateCarree(),
            label=f"Alti within {radius_km:.0f} km",
        )

    if extras["alti_in_time_and_space"] is not None and show_alti_time_and_space_match:
        logger.debug("add alti in time and space")
        ds_ts = extras["alti_in_time_and_space"]
        ts_lon, ts_lat = _alti_lonlat(ds_ts)
        ax.scatter(
            ts_lon,
            ts_lat,
            s=alti_marker_size * 2.5,
            c="green",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            transform=ccrs.PlateCarree(),
            label=f"Alti ≤{radius_km:.0f} km & ≤{delta_t_minutes} min",
        )

    if extras["closest_in_time"] is not None and show_closest_in_time:
        logger.debug("add the closest in time in the coloc subset")
        ds_c = extras["closest_in_time"]
        ax.scatter(
            ds_c["lon_ALT"].values,
            ds_c["lat_ALT"].values,
            s=alti_marker_size * 4.0,
            marker="o",
            facecolors="none",
            edgecolors="magenta",
            linewidths=2.5,
            zorder=7,
            transform=ccrs.PlateCarree(),
            label="Closest in time (coloc)",
        )

    # ---------------------------------------------------- SAFE frame lines
    for fp in safe_fps:
        lats = [c[0] for c in fp.corners] + [fp.corners[0][0]]
        lons = [c[1] for c in fp.corners] + [fp.corners[0][1]]
        ax.plot(
            lons,
            lats,
            color="white",
            linewidth=1.0,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            label="SAFE frame" if fp.frame_id == 0 else None,
        )

    # -------------------------------------------------- highlighted imagette
    if imagette_footprint is not None:
        ax.plot(
            fp_lon,
            fp_lat,
            color="lime",
            linewidth=2.5,
            zorder=8,
            transform=ccrs.PlateCarree(),
            label="Selected WV imagette",
        )
        ax.fill(
            fp_lon,
            fp_lat,
            color="lime",
            alpha=0.30,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
        if centroid is not None:
            ax.plot(
                centroid[0],
                centroid[1],
                marker="*",
                markersize=14,
                linestyle="none",
                color="lime",
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=8,
                transform=ccrs.PlateCarree(),
            )

    # ------------------------------------------------------------- annotate
    if annotate and ocn_info is not None and centroid is not None:
        cx, cy = centroid
        label = (
            f"Imagette #{ocn_info.imagette_number:03d}  "
            f"(WV{ocn_info.wv_number} {ocn_info.pol})\n"
            f"{ocn_info.t_start.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"→ {ocn_info.t_stop.strftime('%H:%M:%S')} UTC"
        )
        ax.annotate(
            label,
            xy=(cx, cy),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(60, 60),
            textcoords="offset points",
            color="black",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="white",
                ec="lime",
                lw=1.5,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="lime",
                lw=1.8,
                shrinkA=0,
                shrinkB=2,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=9,
        )

    # ------------------------------------------------ centroid comparison
    if show_centroid_comparison and centroid is not None:
        cx, cy = centroid
        dlon = abs(cx - osw_lon_val)
        dlat = abs(cy - osw_lat_val)
        comparison = (
            f"polygon.centroid : ({cx:+.5f}, {cy:+.5f})\n"
            f"oswLon / oswLat  : ({osw_lon_val:+.5f}, {osw_lat_val:+.5f})\n"
            f"Δ                : ({dlon:.6f}, {dlat:.6f})°"
        )
        ax.annotate(
            comparison,
            xy=(cx, cy),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(-60, -80),
            textcoords="offset points",
            fontsize=7,
            family="monospace",
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="lightyellow",
                ec="gray",
                lw=0.5,
                alpha=0.9,
            ),
            zorder=9,
        )

    # ------------------------------------------------------------ cosmetics
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="white",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = not hide_top_xticks
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_title(safe_name, fontsize=12, pad=title_pad)
    logger.debug("ocn_info: %s", ocn_info)
    if ocn_info is not None:
        lines = [
            f"Imagette {ocn_info.imagette_number:03d} — "
            f"{ocn_info.t_start.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ]

        # Only show the alti info when a track is actually drawn
        # (avoids polluting step 1 & 2 of the sequence with alti metadata)
        if dsalti is not None and extras.get("closest_in_time") is not None:
            alti_line = _format_closest_alti_line(extras["closest_in_time"])
            if alti_line is not None:
                lines.append(alti_line)

        n_lines = len(lines)
        # Nudge the y-position up when there are 2 lines to avoid crowding
        y_pos = 0.905 if n_lines == 1 else 0.918
        fig.suptitle(
            "\n".join(lines),
            fontsize=10,
            y=y_pos,
        )

    if output_png:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    if return_datasets:
        return fig, ax, extras
    return fig, ax


def plot_coloc_sequence(
    manifest_path: str | Path,
    dssar: xr.Dataset,
    dsalti: xr.Dataset,
    imagette_footprint: Polygon | FootPrint,
    radius_km: float = 30.0,
    delta_t_minutes: int = 25,
    altidb: str = "cmems",
    zoom: int = 11,
    span_deg: float = 2.0,
    alti_marker_size: float = 25.0,
    outputdir: str | Path | None = None,
    fmt: str = "png",
    dpi: int = 150,
    figsize: tuple[float, float] = (11, 10),
    close_after_save: bool = True,
    show: bool = False,
) -> list[tuple[Figure | None, Axes | None]]:
    """Generate the 7-step colocation illustration sequence.

    Steps:
        1. SAR footprints only
        2. + selected WV imagette
        3. + full altimeter track
        4. + radius circle
        5. + alti points within radius (geographic matchups)
        6. + alti points within radius AND time window
        7. + closest-in-time point among the time+space matchups

    Args:
        manifest_path: Path to the SAFE's ``manifest.safe``.
        dssar: WV OCN dataset (single measurement).
        dsalti: Full altimeter track dataset.
        imagette_footprint: Polygon (lon, lat) or FootPrint from the manifest.
        radius_km: Radius of the geographic search circle.
        delta_t_minutes: Time window for the colocation.
        altidb: ``"cci"`` or ``"cmems"``.
        zoom: Tile zoom level (11–12 to actually see the 30 km circle).
        span_deg: Half-span (degrees) for the fixed extent. 2° ≈ 220 km at
            mid-latitudes — a good compromise to show both circle and track.
        alti_marker_size: Marker size for the alti track (default large).
        outputdir: If provided, save each step as ``step_1.png``,
            ``step_2.png``...
        fmt: Image format when saving.
        dpi: Figure dpi.
        figsize: Figure size in inches.
        close_after_save: If True and ``outputdir`` is set, close each figure
            after saving (prevents keeping 7 figures in memory).
        show: If True, call ``plt.show()`` after each step (interactive use).

    Returns:
        List of ``(fig, ax)`` tuples, one per step. Empty list if
        ``close_after_save=True`` and figures were closed.
    """
    # --- Normalize the footprint to a Polygon (lon, lat) -----------------
    if isinstance(imagette_footprint, FootPrint):
        imagette_polygon = manifest_footprint_to_polygon(imagette_footprint)
    else:
        ref_lon = float(np.asarray(dssar["oswLon"].values).squeeze())
        ref_lat = float(np.asarray(dssar["oswLat"].values).squeeze())
        imagette_polygon = as_lonlat_polygon(
            imagette_footprint, ref_lon=ref_lon, ref_lat=ref_lat
        )

    # --- Fixed extent from imagette centroid -----------------------------
    cx, cy = imagette_polygon.centroid.xy
    cx, cy = float(cx[0]), float(cy[0])
    fixed_extent = (
        cx - span_deg,
        cx + span_deg,
        cy - span_deg,
        cy + span_deg,
    )

    # --- Precompute the colocation subsets once --------------------------
    dsalti_valid = _valid_alti(dsalti)
    tree_alti = _alti_tree(dsalti_valid)

    extras: dict[str, xr.Dataset | None] = {
        "alti_in_radius": None,
        "alti_in_time_and_space": None,
        "closest_in_time": None,
    }
    extras["alti_in_radius"] = step_2_geographic_match(
        sards=dssar,
        ds_alti=dsalti_valid,
        tree_alti=tree_alti,
        delta_dist_km=radius_km,
    )
    if extras["alti_in_radius"] is not None:
        cpt: defaultdict[str, int] = defaultdict(int)
        (
            closest_in_time,
            _files,
            _cpt,
            all_ts_match,
        ) = step_3_closer_temp_match(
            sar_dataset=dssar,
            subset_alti=extras["alti_in_radius"],
            delta_t_max_minutes=delta_t_minutes,
            altidb=altidb,
            cpt=cpt,
        )
        extras["closest_in_time"] = closest_in_time
        extras["alti_in_time_and_space"] = all_ts_match

    # --- Common kwargs for all steps -------------------------------------
    common: dict[str, object] = dict(
        manifest_path=manifest_path,
        dssar=dssar,
        imagette_footprint=imagette_polygon,
        altidb=altidb,
        zoom=zoom,
        radius_km=radius_km,
        delta_t_minutes=delta_t_minutes,
        alti_marker_size=alti_marker_size,
        extent=fixed_extent,
        figsize=figsize,
        dpi=dpi,
        highlight_closest_alti=False,  # avoid double ring with "closest_in_time"
        annotate=False,  # keep steps clean
        show_centroid_comparison=False,  # keep steps clean
        title_pad=30,
    )

    steps: list[Step] = [
        # 1) SAR footprints only
        dict(
            name="step_1_safe_footprints",
            kwargs=dict(
                **common,
                dsalti=None,
                show_imagette=False,
                show_radius_circle=False,
                show_alti_in_radius=False,
                show_alti_time_and_space_match=False,
                show_closest_in_time=False,
                extras=extras,  # pass precomputed; unused here because dsalti=None
            ),
        ),
        # 2) + imagette
        dict(
            name="step_2_safe_and_imagette",
            kwargs=dict(
                **common,
                dsalti=None,
                show_imagette=True,
                show_radius_circle=False,
                show_alti_in_radius=False,
                show_alti_time_and_space_match=False,
                show_closest_in_time=False,
                extras=extras,
            ),
        ),
        # 3) + alti track
        dict(
            name="step_3_with_alti_track",
            kwargs=dict(
                **common,
                dsalti=dsalti,
                show_imagette=True,
                show_radius_circle=False,
                show_alti_in_radius=False,
                show_alti_time_and_space_match=False,
                show_closest_in_time=False,
                extras=extras,
            ),
        ),
        # 4) + radius circle
        dict(
            name="step_4_with_radius_circle",
            kwargs=dict(
                **common,
                dsalti=dsalti,
                show_imagette=True,
                show_radius_circle=True,
                show_alti_in_radius=False,
                show_alti_time_and_space_match=False,
                show_closest_in_time=False,
                extras=extras,
            ),
        ),
        # 5) + geographic matchups (step_2)
        dict(
            name="step_5_geographic_matchups",
            kwargs=dict(
                **common,
                dsalti=dsalti,
                show_imagette=True,
                show_radius_circle=True,
                show_alti_in_radius=True,
                show_alti_time_and_space_match=False,
                show_closest_in_time=False,
                extras=extras,
            ),
        ),
        # 6) + time-and-space matchups (step_3)
        dict(
            name="step_6_time_and_space_matchups",
            kwargs=dict(
                **common,
                dsalti=dsalti,
                show_imagette=True,
                show_radius_circle=True,
                show_alti_in_radius=True,
                show_alti_time_and_space_match=True,
                show_closest_in_time=False,
                extras=extras,
            ),
        ),
        # 7) + closest in time
        dict(
            name="step_7_closest_in_time",
            kwargs=dict(
                **common,
                dsalti=dsalti,
                show_imagette=True,
                show_radius_circle=True,
                show_alti_in_radius=True,
                show_alti_time_and_space_match=True,
                show_closest_in_time=True,
                extras=extras,
            ),
        ),
    ]

    results: list[tuple[Figure | None, Axes | None]] = []
    for i, step in enumerate(steps, start=1):
        logger.debug("plotting step %d: %s", i, step["name"])
        step_kwargs: dict[str, object] = cast(dict[str, object], step["kwargs"])
        if outputdir is not None:
            step_kwargs["output_png"] = Path(outputdir) / f"{step['name']}.{fmt}"
        result = plot_safe_and_imagette(**step_kwargs)  # type: ignore[arg-type]
        fig, ax = cast(tuple[Figure, Axes], result)
        if show:
            plt.show()
        results.append((fig, ax))
        if outputdir is not None and close_after_save:
            plt.close(fig)
            results[-1] = (None, None)  # keep index alignment
            logger.info("saved %s", step["name"])

    if close_after_save:
        return []
    return results
