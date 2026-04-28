#!/scale/project/lops-siam-airflow/envs_exploit/micromamba/py27/bin/python2.7

import logging
import os
import subprocess

import numpy as np

from unifiedwvalticolocs.create_listing_jobarray import (
    create_listing_jobarray,
    parse_arguments,
)

prunexe = "/appli/prun/bin/prun"
DEFAULT_OUTD = "/home1/scratch/satwave/unified_WV_alti_colocs"
DEFAULT_SIF = "/scale/project/lops-siam-airflow/envs_exploit/apptainer/unifiedwvalticolocs_2026.1.30.post4.sif"
DEFAULT_CONFIG = "/scale/project/lops-siam-airflow/configs_exploit/unifiedwvalticolocs/ice_prod_config.yml"

all_altis = [
    "cmems_SARAL",
    "cmems_cryosat-2",
    "cmems_CFOSAT",
    "cmems_Jason-3",
    "cmems_Sentinel-3A",
    "cmems_Sentinel-3B",
    "cmems_HY2B",
    "cmems_HY2C",
    "cmems_Sentinel-6A",
    "cmems_SWOT-Nadir",
    "cci_cryosat-2",
    "cci_jason-2",
    "cci_jason-3",
    "cci_sentinel-3a",
    "cci_sentinel-3b",
    "cci_saral",
    "cci_sentinel-6",
]

if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s %(levelname)-5s %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s"
        )
    listing, cpt = create_listing_jobarray(args)

    # 1. Calculate the split value
    split_lines = str(int(np.ceil(cpt / 9900.0)))

    # 2. Get the absolute path of the pbs script
    pbs = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "unified_coloc_WV_alti_cmems_or_cci.pbs"
        )
    )
    assert os.path.exists(pbs)
    # 3. Construct the command as a clean list
    cmd = [
        prunexe,
        "--split-max-lines=" + split_lines,
        "--background",
        "-e",
        pbs,
        listing,
    ]
    logging.info("cmd to cast = %s", " ".join(cmd))
    # 4. Execute
    try:
        st = subprocess.check_call(cmd, shell=False)
        logging.info("status cmd = %s", st)
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with return code %s", e.returncode)
        raise
