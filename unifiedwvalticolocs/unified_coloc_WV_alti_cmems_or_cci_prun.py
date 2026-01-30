#!/scale/project/lops-siam-airflow/envs_exploit/micromamba/py27/bin/python2.7

import datetime
import logging
import os
import subprocess

import numpy as np
from dateutil import rrule

DEFAULT_OUTD = "/home1/scratch/satwave/unified_WV_alti_colocs"
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
    "cci_cryosat-2",  # 2018 - 2023
    "cci_jason-2",  # 2009 - 2018
    "cci_jason-3",  # 2018 - 2023
    "cci_sentinel-3a",  # 2018 - 2023
    "cci_sentinel-3b",  # 2018 - 2023
    "cci_saral",  # 2013 - 2024
    "cci_sentinel-6",  # 2020 - 2023
]
if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="redo existing files [default is False]",
    )
    parser.add_argument(
        "--listing",
        required=False,
        help="listing for PRUN [optional, if None -> it is written on the fly,"
        " otherwise the listing should be already filled] ",
        default=None,
    )
    parser.add_argument(
        "--start", required=False, help="YYYYMMDD [optional] ", default=None
    )
    parser.add_argument(
        "--stop", required=False, help="YYYYMMDD [optional] ", default=None
    )
    parser.add_argument(
        "--outputdir",
        required=False,
        default=DEFAULT_OUTD,
        help="path where to store output coloc (.nc) [default=%s]" % DEFAULT_OUTD,
    )
    parser.add_argument(
        "--alt",
        required=False,
        type=str,
        help="which altimeter you want to specificaly treat, example: "
        "'cmems_al' or 'cmems_c2' [default=all alti available]",
        default=all_altis,
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    if not isinstance(args.alt, list):
        alti_chosen = [args.alt]
    else:
        alti_chosen = args.alt
    logging.info("alti chosen: %s", args.alt)
    prunexe = "/appli/prun/bin/prun"
    # listing below is written on the fly
    if args.listing:
        listing = args.listing  # added for daily coloc with satwave/prun
        cpt = len(open(listing).readlines())
    else:
        listing = (
            "/home1/scratch/satwave/listing_coloc_CMEMS_CCI_Alti_WV_S1_CCI_prun.txt"
        )
        if args.start:
            sta = datetime.datetime.strptime(args.start, "%Y%m%d")
        else:
            # sta = datetime.datetime(2019,10,1)
            # sta = datetime.datetime(2019, 7, 16)
            # dataset-wav-alti-l3-swh-rt-global-cfo start before
            sta = datetime.datetime(2014, 4, 1)
        if args.stop:
            sto = datetime.datetime.strptime(args.stop, "%Y%m%d")
        else:
            sto = datetime.datetime.today()
        fid = open(listing, "w")
        cpt = 0

        for sarunit in ["S1A", "S1B", "S1C", "S1D"]:
            for satalti in alti_chosen:
                for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
                    if args.overwrite:
                        fid.write(
                            "{} {} {} {} --redo\n".format(
                                dd.strftime("%Y%m%d"), sarunit, satalti, args.outputdir
                            )
                        )
                    else:
                        fid.write(
                            "%s %s %s %s\n"
                            % (dd.strftime("%Y%m%d"), sarunit, satalti, args.outputdir)
                        )
                    cpt += 1
        fid.close()
    logging.info("listing ready : %s nb lines : %s", listing, cpt)
    # call prun
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
    # Every space-separated argument must be its own item in the list
    cmd = [
        prunexe,
        "--split-max-lines=" + split_lines,
        "--background",
        "-e",
        pbs,
        listing,
    ]

    logging.info("cmd to cast = %s", " ".join(cmd))

    # 4. Execute with shell=False (Bandit compliant)
    try:
        st = subprocess.check_call(cmd, shell=False)
        logging.info("status cmd = %s", st)
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with return code %s", e.returncode)
        raise
