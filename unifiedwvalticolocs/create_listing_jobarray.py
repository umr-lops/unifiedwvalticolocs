#!/scale/project/lops-siam-airflow/envs_exploit/micromamba/py27/bin/python2.7

import argparse
import datetime
import getpass
import logging
import os

from dateutil import rrule

username = getpass.getuser()

DEFAULT_OUTD = {
    "ice": os.path.join("/home1/scratch/", username, "unified_WV_alti_colocs"),
    "hpc": os.path.join("/scratch/", username, "unified_WV_alti_colocs"),
}
DEFAULT_SIF = "/scale/project/lops-siam-airflow/envs_exploit/apptainer/unifiedwvalticolocs_2026.1.30.post4.sif"
DEFAULT_CONFIG = {
    "ice": "/scale/project/lops-siam-airflow/configs_exploit/unifiedwvalticolocs/ice_prod_config.yml",
    "hpc": "/scale/project/lops-siam-airflow/configs_exploit/unifiedwvalticolocs/hpc_prod_config.yml",
}

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


def argument_parser():
    parser = argparse.ArgumentParser(
        description="create listing for job array PBS or SLURM"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="redo existing files [default is False]",
    )
    parser.add_argument(
        "--start", required=False, help="YYYYMMDD [optional] ", default=None
    )
    parser.add_argument(
        "--stop", required=False, help="YYYYMMDD [optional] ", default=None
    )
    parser.add_argument(
        "--infra",
        required=True,
        choices=["ice", "hpc"],
        help="Infrastructure to run on: 'ice' or 'hpc' [required]",
    )
    parser.add_argument(
        "--outputdir",
        required=False,
        help="path where to store output coloc (.nc) [optional, set based on infrastructure] ",
    )
    parser.add_argument(
        "--image",
        required=False,
        default=DEFAULT_SIF,
        help="Path to the .sif apptainer image  [default=%s]" % DEFAULT_SIF,
    )
    parser.add_argument(
        "--alt",
        required=False,
        type=str,
        help="which altimeter you want to specifically treat [default=all]",
        default=all_altis,
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to the YAML config file for unifiedwvalticolocs [optional, set based on infrastructure]",
    )
    parser.add_argument(
        "--sar-units",
        required=False,
        nargs="+",
        default=["S1A", "S1B", "S1C", "S1D"],
        help="List of SAR units to process (e.g., --sar-units S1A S1B) [default is all S1 units]",
    )
    args = parser.parse_args()
    return args


def entrypoint():
    args = argument_parser()
    create_listing_jobarray(args)


def create_listing_jobarray(args):

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s %(levelname)-5s %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s"
        )

    if not isinstance(args.alt, list):
        alti_chosen = [args.alt]
    else:
        alti_chosen = args.alt

    if args.infra == "ice":
        default_outputdir = DEFAULT_OUTD["ice"]
        default_config = DEFAULT_CONFIG["ice"]
    elif args.infra == "hpc":
        default_outputdir = DEFAULT_OUTD["hpc"]
        default_config = DEFAULT_CONFIG["hpc"]
    else:
        raise ValueError("Invalid infrastructure choice. Use 'ice' or 'hpc'.")

    if args.outputdir:
        outputdir = args.outputdir
    else:
        outputdir = default_outputdir

    if args.config:
        config_path = args.config
    else:
        config_path = default_config

    logging.info("alti chosen: %s", args.alt)
    logging.info("image chosen: %s", args.image)
    logging.info("infra chosen: %s", args.infra)
    logging.info("default output dir: %s", outputdir)
    logging.info("default config: %s", config_path)
    logging.info("SAR units chosen: %s", args.sar_units)

    os.makedirs(default_outputdir, exist_ok=True)
    listing = os.path.join(
        default_outputdir,
        "listing_coloc_CMEMS_CCI_Alti_WV_S1_CCI_jobarray.txt",
    )

    if args.start:
        sta = datetime.datetime.strptime(args.start, "%Y%m%d")
    else:
        sta = datetime.datetime(2014, 4, 1)

    if args.stop:
        sto = datetime.datetime.strptime(args.stop, "%Y%m%d")
    else:
        sto = datetime.datetime.today()

    fid = open(listing, "w")
    cpt = 0

    for sarunit in args.sar_units:
        for satalti in alti_chosen:
            for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
                cmd_line = (
                    "--startdate %s --sat %s --alt %s --outputdir %s --image %s --config %s"
                    % (
                        dd.strftime("%Y%m%d"),
                        sarunit,
                        satalti,
                        outputdir,
                        args.image,
                        config_path,
                    )
                )
                if args.overwrite:
                    cmd_line += " --redo"

                fid.write(cmd_line + "\n")
                cpt += 1
    fid.close()

    logging.info("listing ready : %s nb lines : %s", listing, cpt)
    return listing, cpt


if __name__ == "__main__":
    entrypoint()
