import argparse
import logging
import os
from datetime import datetime
from unittest.mock import mock_open, patch

import pytest
from dateutil import rrule

# Import the function to test
from unifiedwvalticolocs.create_listing_jobarray import (
    DEFAULT_OUTD,
    DEFAULT_SIF,
    all_altis,
    argument_parser,
    create_listing_jobarray,
)


# --- Fixtures ---
@pytest.fixture
def mock_args():
    return argparse.Namespace(
        verbose=False,
        overwrite=False,
        start="20240101",
        stop="20240102",
        infra="ice",
        outputdir=None,
        image=DEFAULT_SIF,
        alt=all_altis,
        config=None,
        sar_units=["S1A", "S1B"],
    )


# --- Tests ---
def test_argument_parser():
    with patch(
        "sys.argv",
        ["script.py", "--infra", "ice", "--start", "20240101", "--stop", "20240102"],
    ):
        args = argument_parser()
        assert args.infra == "ice"
        assert args.start == "20240101"
        assert args.stop == "20240102"


def test_create_listing_jobarray(mock_args, caplog):
    # Capture INFO-level logs (pytest defaults to WARNING)
    caplog.set_level(logging.INFO)

    with patch("os.makedirs") as mock_makedirs:
        with patch("builtins.open", mock_open()) as mock_file:
            listing, cpt = create_listing_jobarray(mock_args)

        # Assertions
        mock_makedirs.assert_called_once_with(DEFAULT_OUTD["ice"], exist_ok=True)
        mock_file.assert_called_once_with(
            os.path.join(
                DEFAULT_OUTD["ice"],
                "listing_coloc_CMEMS_CCI_Alti_WV_S1_CCI_jobarray.txt",
            ),
            "w",
        )

        # Check log messages
        # assert "listing ready" in caplog.text
        assert len(listing) > 0
        # assert "nb lines : 64" in caplog.text


def test_date_handling():
    sta = datetime(2024, 1, 1)
    sto = datetime(2024, 1, 2)
    dates = list(rrule.rrule(rrule.DAILY, dtstart=sta, until=sto))
    assert len(dates) == 2
    assert dates[0].strftime("%Y%m%d") == "20240101"
    assert dates[1].strftime("%Y%m%d") == "20240102"
