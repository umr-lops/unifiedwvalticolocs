"""lib python to generate colocs S1 WV with CMEMS or CCI seatstate altimeters"""

from importlib import metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

version = __version__
