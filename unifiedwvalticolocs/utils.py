"""Shared helpers: config loading, altimeter/SAR mission catalogs and date windows."""

import datetime
import logging

from yaml import CLoader as Loader
from yaml import load

logger = logging.getLogger("unifiedwvalticolocs.utils")
logger.addHandler(logging.NullHandler())


def get_conf_content(conf_path: str) -> dict:
    """Load the YAML configuration file from the specified path.

    Args:
        conf_path: The file path to the YAML configuration file.

    Returns:
        The content of the YAML configuration file as a dictionary.
    """
    with open(conf_path, encoding="utf-8") as stream:
        conf = load(stream, Loader=Loader)
    return conf


# ---------------------------------------------------------------------------
# Altimeter mission catalogs
# ---------------------------------------------------------------------------
# CCI SeaState L2P: key -> (subdir, satellite name as it appears in filenames)
POSSIBLES_CCI_ALTI: dict[str, tuple[str, str]] = {
    "cfosat": ("cfosat", "CFOSAT"),
    "cryosat-2": ("cryosat-2", "CryoSat-2"),
    "envisat": ("envisat", "Envisat"),
    "ers-1": ("ers-1", "ERS-1"),
    "ers-2": ("ers-2", "ERS-2"),
    "gfo": ("gfo", "GFO"),
    "jason-1": ("jason-1", "Jason-1"),
    "jason-2": ("jason-2", "Jason-2"),
    "jason-3": ("jason-3", "Jason-3"),
    "saral": ("saral", "SARAL"),
    "sentinel-3_a": ("sentinel-3_a", "Sentinel-3_A"),
    "sentinel-3_b": ("sentinel-3_b", "Sentinel-3_B"),
    "sentinel-6_a": ("sentinel-6_a", "Sentinel-6_A"),
    "swot": ("swot", "SWOT"),
    "topex-poseidon_poseidon": ("topex-poseidon_poseidon", "Topex-Poseidon"),
    "topex-poseidon_topex": ("topex-poseidon_topex", "Topex-Poseidon"),
}

# CMEMS WAVE L3 NRT: key -> product subdir acronym (subset_alti_name_dir % acronym)
POSSIBLES_CMEMS_ALTI: dict[str, str] = {
    "CFOSAT": "cfo",
    "HY2B": "h2b",
    "HY2C": "h2c",
    "Jason-3": "j3",
    "SARAL": "al",
    "Sentinel-3A": "s3a",
    "Sentinel-3B": "s3b",
    "Sentinel-6A": "s6a",
    "SWOT-Nadir": "swon",
    "cryosat-2": "c2",
}


def all_altimeters() -> list[str]:
    """Return every supported altimeter name as ``<db>_<mission>``.

    Returns:
        Sorted list, e.g. ``["cci_cryosat-2", ..., "cmems_Jason-3"]``.
    """
    names = [f"cci_{k}" for k in POSSIBLES_CCI_ALTI]
    names += [f"cmems_{k}" for k in POSSIBLES_CMEMS_ALTI]
    return sorted(names)


# ---------------------------------------------------------------------------
# Mission date windows (defaults for the job-array listing)
# ---------------------------------------------------------------------------
def _d(year: int, month: int, day: int) -> datetime.date:
    return datetime.date(year, month, day)


# Sentinel-1 WV OCN Level-2: commissioning date per mission. Stop = today
# (the missions are all operational).
SAR_MISSION_DATES: dict[str, tuple[datetime.date, datetime.date | None]] = {
    "S1A": (_d(2014, 4, 4), None),
    "S1B": (_d(2016, 4, 25), None),
    "S1C": (_d(2025, 5, 16), None),
    "S1D": (_d(2026, 1, 7), None),
}

# CCI SeaState L2P v5: first/last acquisition per mission, from the v5 catalog
# (generated 2026-10-02). stop=None means the mission is still operational:
# the window is capped at "today" by date_window_for().
CCI_MISSION_DATES: dict[str, tuple[datetime.date, datetime.date | None]] = {
    "cfosat": (_d(2019, 1, 1), None),
    "cryosat-2": (_d(2010, 7, 16), None),
    "envisat": (_d(2002, 5, 14), _d(2012, 4, 8)),
    "ers-1": (_d(1991, 8, 3), _d(1996, 6, 2)),
    "ers-2": (_d(1995, 5, 14), _d(2003, 7, 2)),
    "gfo": (_d(2000, 1, 7), _d(2008, 9, 17)),
    "jason-1": (_d(2002, 1, 15), _d(2013, 6, 21)),
    "jason-2": (_d(2008, 7, 4), _d(2019, 10, 1)),
    "jason-3": (_d(2016, 2, 12), _d(2025, 12, 1)),
    "saral": (_d(2013, 3, 14), None),
    "sentinel-3_a": (_d(2016, 5, 5), None),
    "sentinel-3_b": (_d(2018, 6, 7), None),
    "sentinel-6_a": (_d(2020, 12, 17), None),
    "swot": (_d(2023, 1, 16), None),
    "topex-poseidon_poseidon": (_d(1996, 6, 12), _d(2001, 1, 23)),
    "topex-poseidon_topex": (_d(1992, 10, 13), _d(2005, 10, 4)),
}

# CMEMS WAVE L3 NRT (product WAVE_GLO_PHY_SWH_L3_NRT_014_001): every mission
# starts 2019-10-01 and updates daily up to now.
CMEMS_MISSION_DATES: dict[str, tuple[datetime.date, datetime.date | None]] = {
    "SARAL": (_d(2019, 10, 1), None),
    "CFOSAT": (_d(2019, 10, 1), None),
    "HY2B": (_d(2019, 10, 1), None),
    "HY2C": (_d(2019, 10, 1), None),
    "Jason-3": (_d(2019, 10, 1), None),
    "Sentinel-3A": (_d(2019, 10, 1), None),
    "Sentinel-3B": (_d(2019, 10, 1), None),
    "Sentinel-6A": (_d(2019, 10, 1), None),
    "SWOT-Nadir": (_d(2019, 10, 1), None),
    "cryosat-2": (_d(2019, 10, 1), None),
}


def date_window_for(
    alt: str, sar_unit: str
) -> tuple[datetime.date, datetime.date] | None:
    """Default (start, stop) date window of a SAR x altimeter pair.

    The window is the intersection of the SAR mission window and the altimeter
    mission window. ``stop=None`` (mission still operational) resolves to
    today.

    Args:
        alt: Altimeter name, e.g. ``"cci_cryosat-2"`` or ``"cmems_Jason-3"``.
        sar_unit: SAR mission, e.g. ``"S1A"`` (case-insensitive).

    Returns:
        Inclusive ``(start, stop)`` date pair, or None if the windows do not
        overlap.

    Raises:
        KeyError: If the altimeter or the SAR unit is not in the catalogs.
    """
    altidb, _, mission = alt.partition("_")
    if altidb == "cci":
        alt_window = CCI_MISSION_DATES[mission]
    elif altidb == "cmems":
        alt_window = CMEMS_MISSION_DATES[mission]
    else:
        raise KeyError(f"altidb {altidb!r} not handled (expected 'cci' or 'cmems')")

    sar_window = SAR_MISSION_DATES[sar_unit.upper()]

    start = max(alt_window[0], sar_window[0])
    stops = [w[1] or datetime.date.today() for w in (alt_window, sar_window)]
    stop = min(stops)

    if start > stop:
        return None
    return start, stop
