"""Tests of the one-task-per-day mode (WV files read once for all altimeters)."""

import argparse
import os
import shutil
from collections import defaultdict
from unittest.mock import patch

import pytest
import xarray as xr

import unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci as U
from unifiedwvalticolocs.create_listing_jobarray import (
    DEFAULT_SIF,
    all_altis,
    create_listing_jobarray,
)

FIXTURE = os.path.join(
    os.path.dirname(U.__file__),
    "data4tests",
    "s1a-wv2-ocn-vv-20260112t045817-20260112t045820-062729-07ddb1-036.nc",
)


@pytest.fixture
def two_safes(tmp_path):
    """Two SAFE with 2 and 1 measurements, and one SAFE without measurement."""
    safes = []
    for isafe, nmeas in enumerate([2, 1, 0]):
        safe = tmp_path / f"S1A_WV_OCN__2SSV_{isafe}.SAFE"
        (safe / "measurement").mkdir(parents=True)
        for imeas in range(nmeas):
            name = os.path.basename(FIXTURE).replace("-036.nc", f"-{imeas:03d}.nc")
            shutil.copy(FIXTURE, safe / "measurement" / name)
        safes.append(str(safe))
    return safes


def test_load_sar_safes_parallel_equals_sequential(two_safes):
    seq = U.load_sar_safes(two_safes, nproc=1)
    par = U.load_sar_safes(two_safes, nproc=2)
    assert list(seq) == list(par) == two_safes
    assert len(seq[two_safes[0]]["time_sar"]) == 2
    assert len(seq[two_safes[1]]["time_sar"]) == 1
    assert "empty" in seq[two_safes[2]]
    for safe in two_safes[:2]:
        xr.testing.assert_identical(seq[safe], par[safe])


def test_make_sar_getter_reads_once():
    with patch.object(U, "load_sar_safes", return_value={"a": 1}) as mock_load:
        get = U.make_sar_getter(["a"], nproc=4)
        assert get() == {"a": 1} and get() == {"a": 1}
    mock_load.assert_called_once_with(["a"], nproc=4)


def test_parse_alts_unknown():
    with pytest.raises(ValueError):
        U.parse_alts(["cci_saral,cci_nope"])


def test_altimeters_of_day_all():
    alts = U.altimeters_of_day("20200115", "S1A", ["all"])
    assert "cci_saral" in alts and "cmems_SARAL" in alts
    assert "cci_ers-1" not in alts
    # S1B failed on 2021-12-23
    assert U.altimeters_of_day("20220601", "S1B", ["all"]) == []


def test_core_coloc_day_skips_existing_and_isolates_failures(tmp_path):
    conf = {"delta_t_minutes": 25, "delta_dist_km": 30}
    alts = ["cci_saral", "cmems_SARAL", "cci_jason-3"]
    done = U.output_coloc_files(str(tmp_path), "S1A", "cci_jason-3", "20200115", conf)
    os.makedirs(os.path.dirname(done[0]))
    open(done[0], "w").close()

    def fake_coloc(day, alt, *args, **kwargs):
        if alt == "cmems_SARAL":
            raise RuntimeError("boom")
        return defaultdict(int, {"nb_coloc": 3})

    with (
        patch.object(U, "list_sar_safes", return_value=["s.SAFE"]),
        patch.object(U, "prepare_alti"),
        patch.object(U, "coloc_one_alti", side_effect=fake_coloc) as mock_coloc,
    ):
        counters, failed = U.core_coloc_day(
            "20200115", alts, "S1A", str(tmp_path), conf, nproc=2
        )
    assert failed == ["cmems_SARAL"]
    assert list(counters) == ["cci_saral"]
    assert counters["cci_saral"]["nb_coloc"] == 3
    # jason-3 output exists -> not processed; the SAR getter is shared
    called = [c.args[1] for c in mock_coloc.call_args_list]
    assert called == ["cci_saral", "cmems_SARAL"]
    getters = {id(c.args[6]) for c in mock_coloc.call_args_list}
    assert len(getters) == 1


def test_listing_per_day(tmp_path):
    listing = tmp_path / "listing.txt"
    args = argparse.Namespace(
        verbose=False,
        overwrite=False,
        start="20240101",
        stop="20240103",
        infra="hpc",
        outputdir=str(tmp_path / "out"),
        image=DEFAULT_SIF,
        alt=all_altis,
        config="conf.yml",
        sar_units=["S1A", "S1C"],
        outputpath_csv=str(listing),
        output_type="txt",
        per_day=True,
        nproc=8,
    )
    _, cpt = create_listing_jobarray(args)
    lines = listing.read_text().splitlines()
    assert cpt == len(lines) == 2 * 3
    assert all("--alt all " in ll and ll.endswith("--nproc 8") for ll in lines)
