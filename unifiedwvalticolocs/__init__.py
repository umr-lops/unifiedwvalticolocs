# type: ignore[attr-defined]
"""lib python to generate colocs S1 WV with CMEMS or CCI seatstate altimeters"""

import sys
from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = metadata.version("unifiedwvalticolocs")
