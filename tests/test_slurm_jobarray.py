"""Tests for the SLURM task script, listing defaults and date windows."""

import argparse
import datetime
import os
import stat
from pathlib import Path

import pytest

import unifiedwvalticolocs
from unifiedwvalticolocs.create_listing_jobarray import create_listing_jobarray
from unifiedwvalticolocs.utils import (
    all_altimeters,
    date_window_for,
)

PACKAGE_DIR = Path(unifiedwvalticolocs.__file__).parent
TASK_SCRIPT = PACKAGE_DIR / "unified_coloc_WV_alti_cmems_or_cci_slurm.bash"


# ---------------------------------------------------------------------------
# date windows
# ---------------------------------------------------------------------------
class TestDateWindows:
    def test_active_cci_mission_stops_today(self):
        """An operational CCI mission window ends today."""
        window = date_window_for("cci_cryosat-2", "S1A")
        assert window == (datetime.date(2014, 4, 4), datetime.date.today())

    def test_decommissioned_before_sar_is_empty(self):
        """Envisat stopped (2012) before S1A started (2014) -> no overlap."""
        assert date_window_for("cci_envisat", "S1A") is None

    def test_decommissioned_overlap(self):
        """Decommissioned mission overlapping S1B (2016+) is kept."""
        window = date_window_for("cci_jason-2", "S1B")
        assert window == (datetime.date(2016, 4, 25), datetime.date(2019, 10, 1))

    def test_cmems_starts_2019(self):
        """Every CMEMS NRT 014 mission starts 2019-10-01 and runs to today."""
        window = date_window_for("cmems_Jason-3", "S1A")
        assert window == (datetime.date(2019, 10, 1), datetime.date.today())

    def test_empty_intersection(self):
        """No overlap (ERS-1 vs S1A) -> None."""
        assert date_window_for("cci_ers-1", "S1A") is None

    def test_unknown_altidb_raises(self):
        with pytest.raises(KeyError, match="not handled"):
            date_window_for("esa_jason-3", "S1A")

    def test_unknown_sar_raises(self):
        with pytest.raises(KeyError):
            date_window_for("cci_cryosat-2", "S9X")

    def test_all_altimeters(self):
        """26 alts: 16 CCI + 10 CMEMS, all prefixed."""
        alts = all_altimeters()
        assert len(alts) == 26
        assert alts == sorted(alts)
        assert all(a.startswith(("cci_", "cmems_")) for a in alts)


# ---------------------------------------------------------------------------
# create_listing_jobarray default dates
# ---------------------------------------------------------------------------
def _args(tmp_path, **overrides):
    base = dict(
        verbose=False,
        overwrite=False,
        start=None,
        stop=None,
        infra="ice",
        outputdir=None,
        image="/fake/image.sif",
        alt=["cmems_Jason-3", "cci_ers-1"],
        config="/fake/config.yml",
        sar_units=["S1A"],
        output_type="csv",
        outputpath_csv=str(tmp_path / "listing.csv"),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestDefaultDates:
    def test_per_pair_default_window(self, tmp_path):
        """cmems_Jason-3 starts 2019-10-01; cci_ers-1 (empty) is skipped."""
        args = _args(tmp_path)
        listing, cpt = create_listing_jobarray(args)

        import pandas as pd

        df = pd.read_csv(listing)
        assert list(df["alt"]) == ["cmems_Jason-3"] * cpt
        assert df["startdate"].iloc[0] == 20191001
        # cci_ers-1 vs S1A has no overlap -> contributes 0 rows.
        assert "cci_ers-1" not in set(df["alt"])
        # last row is today (inclusive window end).
        assert int(df["startdate"].iloc[-1]) == int(
            datetime.date.today().strftime("%Y%m%d")
        )

    def test_explicit_start_stop_overrides_defaults(self, tmp_path):
        """--start/--stop apply to every pair, including empty ones."""
        args = _args(tmp_path, start="20240101", stop="20240103")
        listing, cpt = create_listing_jobarray(args)

        import pandas as pd

        df = pd.read_csv(listing)
        # 2 alts x 3 days: --start/--stop are both inclusive (01, 02, 03).
        assert cpt == 6
        assert set(df["startdate"]) == {20240101, 20240102, 20240103}
        assert set(df["alt"]) == {"cmems_Jason-3", "cci_ers-1"}


# ---------------------------------------------------------------------------
# SLURM task script (fake apptainer on PATH)
#
# Job arrays are submitted with turboblast, which runs
# ``bash _slurm.bash <line>`` for each line of the txt listing.
# ---------------------------------------------------------------------------
def _write_fake(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_task_script_single_shot(tmp_path, monkeypatch):
    """One coloc run: the args drive the apptainer call."""
    import subprocess

    calls = tmp_path / "apptainer_calls.txt"
    _write_fake(tmp_path / "apptainer", 'echo "$@" >> "%s"\n' % calls)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    result = subprocess.run(
        [
            "bash",
            str(TASK_SCRIPT),
            "--startdate",
            "20240101",
            "--sat",
            "S1A",
            "--alt",
            "cmems_Jason-3",
            "--outputdir",
            "/out",
            "--image",
            "/img.sif",
            "--config",
            "/conf.yml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    called = calls.read_text().splitlines()[0].split()
    assert "procunifiedwvalticolocs" in called
    assert "--startdate" in called and "20240101" in called
    assert "--sat" in called and "S1A" in called
    assert "--alt" in called and "cmems_Jason-3" in called


def test_task_script_missing_args(tmp_path):
    """Without the required arguments the script refuses to run."""
    import subprocess

    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), "--startdate", "20240101"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Missing required arguments" in result.stderr


def test_txt_listing_line_drives_task_script(tmp_path, monkeypatch):
    """Each txt listing line is a valid task-script invocation (turboblast contract)."""
    import shlex
    import subprocess

    listing = tmp_path / "listing.txt"
    args = _args(tmp_path, output_type="txt", outputpath_csv=str(listing))
    create_listing_jobarray(args)
    line = listing.read_text().splitlines()[0]

    calls = tmp_path / "apptainer_calls.txt"
    _write_fake(tmp_path / "apptainer", 'echo "$@" >> "%s"\n' % calls)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), *shlex.split(line)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    called = calls.read_text().splitlines()[0]
    assert "--startdate 20191001" in called
    assert "--sat S1A" in called
    assert "--alt cmems_Jason-3" in called
