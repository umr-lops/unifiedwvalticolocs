#!/usr/bin/env bash
#SBATCH --time=19:40:00
#SBATCH --mem=5G
#SBATCH --job-name=unifiedcolocAltiWV
#SBATCH --mail-type=NONE

# Configuration
optssimg="exec -B /scale/reference/ -B /legacy/project/cersat/public -B /scratch -B /ontap"

# Default value for image
IMAGNAME="/scale/project/lops-siam-airflow/envs_exploit/apptainer/unifiedwvalticolocs_2026.1.30.post4.sif"

# Default values
STARTDATE=""
SAT=""
ALT=""
OUTPUTDIR=""
REDO=""
DEV=""
CONFIG=""

# Function to display help
usage() {
    echo "Usage: $(basename "$0") [options]"
    echo ""
    echo "Required Arguments:"
    echo "  -d, --startdate DATE   The start date for processing (e.g., 20240101)"
    echo "  -s, --sat SATELLITE    The satellite name"
    echo "  -a, --alt ALTIMETER    The altimeter name"
    echo "  -o, --outputdir DIR    The directory where results will be saved"
    echo "  -c, --config FILE      Path to the YAML config file"
    echo ""
    echo "Options:"
    echo "  -i, --image SIF_FILE   Path to the SIF image (default: $IMAGNAME)"
    echo "  --redo                 Enable overwrite mode (forces reprocessing)"
    echo "  --dev                  Enable developer mode"
    echo "  -h, --help             Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  sbatch $(basename "$0") --startdate 20240101 --sat S1A --alt cmems_Jason-3 --outputdir ./results --config ./conf.yml"
    echo ""
    echo "Note: SLURM directives (#SBATCH) at the top of the script can be"
    echo "  overridden at submission time, e.g.:"
    echo "  sbatch --time=02:00:00 --mem=8G $(basename "$0") ..."
}

# Parse Command Line Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--startdate) STARTDATE="$2"; shift 2 ;;
        -s|--sat)       SAT="$2";       shift 2 ;;
        -a|--alt)       ALT="$2";       shift 2 ;;
        -o|--outputdir) OUTPUTDIR="$2"; shift 2 ;;
        -c|--config)    CONFIG="$2";    shift 2 ;;
        -i|--image)     IMAGNAME="$2";  shift 2 ;;
        --redo)         REDO="--redo";  shift 1 ;;
        --dev)          DEV="--dev";    shift 1 ;;
        -h|--help)      usage;          exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Validation: Check if required arguments are provided
if [[ -z "$STARTDATE" || -z "$SAT" || -z "$ALT" || -z "$OUTPUTDIR" || -z "$CONFIG" ]]; then
    echo "Error: Missing required arguments."
    usage
    exit 1
fi

echo "--- Run Configuration ---"
echo "Start Date: $STARTDATE"
echo "Satellite:  $SAT"
echo "Altimeter:  $ALT"
echo "Output Dir: $OUTPUTDIR"
echo "Config:     $CONFIG"
echo "Image SIF:  $IMAGNAME"
[[ -n "$REDO" ]] && echo "Redo:       ON" || echo "Redo:       OFF"
[[ -n "$DEV" ]]  && echo "Dev Mode:   ON" || echo "Dev Mode:   OFF"
echo "SLURM Job ID:   ${SLURM_JOB_ID}"
echo "SLURM Node:     ${SLURM_NODELIST}"
echo "-------------------------"

# Execution
apptainer $optssimg "$IMAGNAME" procunifiedwvalticolocs \
    --outputdir "$OUTPUTDIR" \
    --startdate "$STARTDATE" \
    --sat "$SAT" \
    --alt "$ALT" \
    --config "$CONFIG" \
    $REDO \
    $DEV

echo "end of python script exe"
