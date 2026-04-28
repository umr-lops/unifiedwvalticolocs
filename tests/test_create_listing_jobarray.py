import argparse
import logging
import os
from datetime import datetime
from unittest.mock import patch

import pytest
from dateutil import rrule

# Import the function to test
from unifiedwvalticolocs.create_listing_jobarray import (
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
        outputpath_csv="/path/to/listing.csv",
        output_type="csv",
    )


# --- Tests ---
def test_argument_parser():
    with patch(
        "sys.argv",
        [
            "script.py",
            "--infra",
            "ice",
            "--start",
            "20240101",
            "--stop",
            "20240102",
            "--outputpath-csv",
            "/path/to/listing.csv",
            "--output-type",
            "csv",
        ],
    ):
        args = argument_parser()
        assert args.infra == "ice"
        assert args.start == "20240101"
        assert args.stop == "20240102"


def test_create_listing_jobarray(mock_args, caplog):
    caplog.set_level(logging.INFO)

    with patch("os.makedirs") as mock_makedirs:
        # Mock pandas DataFrame.to_csv to avoid real file I/O
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            # Call the function
            listing, cpt = create_listing_jobarray(mock_args)

        # Assertions
        mock_makedirs.assert_called_once_with(
            os.path.dirname(mock_args.outputpath_csv), exist_ok=True
        )

        # Verify to_csv was called with expected arguments
        mock_to_csv.assert_called_once()
        # Optional: check the path argument if needed
        # assert mock_to_csv.call_args[0][0].endswith("listing.csv")
        assert mock_to_csv.call_args[1].get("index") is False

        # Check output
        assert len(listing) > 0
        # assert cpt == 2 * 16 * 2  # 2 SAR units * 16 altimeters * 2 days


def test_date_handling():
    sta = datetime(2024, 1, 1)
    sto = datetime(2024, 1, 2)
    dates = list(rrule.rrule(rrule.DAILY, dtstart=sta, until=sto))
    assert len(dates) == 2
    assert dates[0].strftime("%Y%m%d") == "20240101"
    assert dates[1].strftime("%Y%m%d") == "20240102"
