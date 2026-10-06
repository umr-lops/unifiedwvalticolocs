"""Tests for the SLURM job-array helpers and default date windows."""

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
TASK_SCRIPT = PACKAGE_DIR / "unified_coloc_WV_alti_cmems_or_cci_slurm_array.bash"
SUBMIT_SCRIPT = PACKAGE_DIR / "submit_slurm_jobarray.sh"


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
# SLURM bash scripts (fake apptainer / sbatch on PATH)
# ---------------------------------------------------------------------------
def _write_fake(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_task_script_runs_row(tmp_path, monkeypatch):
    """Task N reads CSV row N+2 and runs apptainer with those fields."""
    csv = tmp_path / "listing.csv"
    csv.write_text(
        "startdate,sat,alt,outputdir,image,config\n"
        "20240101,S1A,cmems_Jason-3,/out,/img.sif,/conf.yml\n"
        "20240102,S1B,cci_cryosat-2,/out,/img.sif,/conf.yml\n"
    )
    calls = tmp_path / "apptainer_calls.txt"
    fake_apptainer = _write_fake(
        tmp_path / "apptainer",
        'echo "$@" >> "%s"\n' % calls,
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")

    import subprocess

    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), "--listing", str(csv)],
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "1"},
    )
    assert result.returncode == 0, result.stderr
    called = calls.read_text().splitlines()[0].split()
    assert "procunifiedwvalticolocs" in called
    assert "--startdate" in called and "20240102" in called
    assert "--sat" in called and "S1B" in called
    assert "--alt" in called and "cci_cryosat-2" in called
    assert fake_apptainer.exists()


def test_task_script_missing_task_id(tmp_path, monkeypatch):
    """Without SLURM_ARRAY_TASK_ID the script refuses to run."""
    csv = tmp_path / "listing.csv"
    csv.write_text("startdate,sat,alt,outputdir,image,config\n")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    import subprocess

    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), "--listing", str(csv)],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"},
    )
    assert result.returncode == 1
    assert "SLURM_ARRAY_TASK_ID" in result.stderr


def test_task_script_out_of_range(tmp_path, monkeypatch):
    """A task ID beyond the CSV rows fails cleanly."""
    csv = tmp_path / "listing.csv"
    csv.write_text(
        "startdate,sat,alt,outputdir,image,config\n"
        "20240101,S1A,cmems_Jason-3,/out,/img.sif,/conf.yml\n"
    )
    import subprocess

    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), "--listing", str(csv)],
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "42"},
    )
    assert result.returncode == 1
    assert "out of range" in result.stderr


def test_submitter_builds_and_submits_array(tmp_path, monkeypatch):
    """Submitter runs the listing CLI then sbatch --array=0-(N-1)."""
    listing = tmp_path / "listing.csv"
    n_rows = 5
    # Records its arguments and parses --outputpath-csv (like the real CLI).
    cli_args = tmp_path / "cli_args.txt"
    _write_fake(
        tmp_path / "create-unified-wv-alti-job-array-listing",
        'printf "%s\\n" "$@" >> "' + str(cli_args) + '"\n'
        'out=""; prev=""\n'
        'for a in "$@"; do\n'
        '  if [[ "$prev" == "--outputpath-csv" ]]; then out="$a"; fi\n'
        '  prev="$a"\n'
        "done\n"
        'echo "startdate,sat,alt,outputdir,image,config" > "$out"\n'
        f"for i in $(seq 1 {n_rows}); do\n"
        '  echo "20240101,S1A,cmems_Jason-3,/out,/img.sif,/conf.yml"\n'
        'done >> "$out"\n',
    )
    sbatch_calls = tmp_path / "sbatch_calls.txt"
    _write_fake(
        tmp_path / "sbatch",
        'echo "$@" >> "%s"\n' % sbatch_calls,
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    import subprocess

    result = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--outputpath-csv",
            str(listing),
            "--sar-units",
            "S1A",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert listing.exists()
    # The listing CLI is always called with --infra hpc (SLURM infra).
    cli = cli_args.read_text().splitlines()
    assert "--infra" in cli and "hpc" in cli
    assert "--sar-units" in cli and "S1A" in cli
    sbatch_args = sbatch_calls.read_text().splitlines()[0].split()
    assert "--array=0-4" in sbatch_args
    assert str(TASK_SCRIPT) in sbatch_args
    assert "--listing" in sbatch_args and str(listing) in sbatch_args
