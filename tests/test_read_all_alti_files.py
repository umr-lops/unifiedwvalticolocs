"""Tests of read_all_alti_files on CCI files with heterogeneous coordinates."""

import numpy as np
import pandas as pd
import xarray as xr

from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import (
    drop_extra_coords,
    read_all_alti_files,
)


def _cci_file(path, t0, with_z):
    """Write a minimal CCI sea state L2P file, optionally with a ``z`` coordinate."""
    time = pd.date_range(t0, periods=5, freq="1s")
    coords = {
        "time": time,
        "lon": ("time", np.linspace(10.0, 10.4, 5)),
        "lat": ("time", np.linspace(40.0, 40.4, 5)),
    }
    if with_z:
        coords["z"] = ("time", np.full(5, 9.8, dtype="float32"))
    ds = xr.Dataset(
        {
            "swh_adjusted": ("time", np.full(5, 2.0, dtype="float32")),
            "swh_quality_level": ("time", np.full(5, 3, dtype="int8")),
        },
        coords=coords,
    )
    ds.to_netcdf(path)
    return str(path)


def test_read_all_alti_files_mixed_z_coordinate(tmp_path):
    """Files with and without ``z`` (CCI v5 SWOT) are concatenated."""
    files = [
        _cci_file(tmp_path / "a.nc", "2023-03-11T00:00:00", with_z=True),
        _cci_file(tmp_path / "b.nc", "2023-03-11T01:00:00", with_z=False),
    ]
    conf = {"cci_alti_dir": "/data/cci/v5/"}
    ds, tree = read_all_alti_files(files, "cci", conf)
    assert ds.sizes["time"] == 10
    assert "z" not in ds.variables
    assert tree.n == 10
    assert ds.attrs["altimeter_version_database"] == "v5"


def test_drop_extra_coords_keeps_requested():
    """Only the coordinates listed in ``keep`` survive; data variables stay."""
    ds = xr.Dataset(
        {"swh": ("time", [1.0])},
        coords={"time": [0], "lon": ("time", [1.0]), "z": ("time", [2.0])},
    )
    out = drop_extra_coords(ds, ("time", "lon"))
    assert sorted(out.coords) == ["lon", "time"]
    assert "swh" in out
