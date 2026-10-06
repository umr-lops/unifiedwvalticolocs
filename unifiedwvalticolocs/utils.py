"""Shared helpers: config loading and the altimeter mission catalogs."""

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
