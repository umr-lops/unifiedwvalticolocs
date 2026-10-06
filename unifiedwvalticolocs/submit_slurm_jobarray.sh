#!/usr/bin/env bash
# Build the listing CSV (via create-unified-wv-alti-job-array-listing) and
# submit it as a SLURM job array, one task per CSV row.
#
# SLURM-only: the listing is always built with --infra hpc (ice runs on PBS).
#
#   ./submit_slurm_jobarray.sh --outputpath-csv /scratch/u/listing.csv \
#       --sar-units S1A --alt cmems_Jason-3
set -euo pipefail

# SBATCH defaults (override with --sbatch-time / --sbatch-mem).
SBATCH_TIME="19:40:00"
SBATCH_MEM="5G"
JOBNAME="unifiedcolocAltiWV"

OUTPUTPATH_CSV=""
REDO=""
DEV=""

usage() {
    echo "Usage: $(basename "$0") --outputpath-csv CSV [listing options] [submit options]"
    echo ""
    echo "Always builds the listing with --infra hpc (SLURM; ice runs on PBS)."
    echo ""
    echo "Listing options (forwarded to create-unified-wv-alti-job-array-listing, which"
    echo "runs with --output-type csv):"
    echo "  --start YYYYMMDD      Override default start date [optional]"
    echo "  --stop YYYYMMDD       Override default stop date [optional]"
    echo "  --sar-units S1A S1B   SAR units [default: all]"
    echo "  --alt NAME            One altimeter [default: all]"
    echo "  --image SIF           Apptainer image [default: hpc package default]"
    echo "  --config YAML         Colocation config file [default: hpc default]"
    echo "  --outputdir DIR       Coloc output dir [default: hpc default]"
    echo ""
    echo "Submit options:"
    echo "  --outputpath-csv CSV  Listing file to build and submit [required]"
    echo "  --sbatch-time TIME    Wall time per task [default $SBATCH_TIME]"
    echo "  --sbatch-mem MEM      Memory per task [default $SBATCH_MEM]"
    echo "  --redo                Force reprocessing of existing output"
    echo "  --dev                 Developer mode"
    echo "  -h, --help            Show this help message"
}

listing_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outputpath-csv) OUTPUTPATH_CSV="$2"; shift 2 ;;
        --sbatch-time) SBATCH_TIME="$2"; shift 2 ;;
        --sbatch-mem) SBATCH_MEM="$2"; shift 2 ;;
        --redo) REDO="--redo"; shift 1 ;;
        --dev) DEV="--dev"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) listing_args+=("$1"); shift ;;
    esac
done

if [[ -z "$OUTPUTPATH_CSV" ]]; then
    echo "Error: --outputpath-csv is required." >&2
    usage
    exit 1
fi

# 1) Build the listing CSV (per-pair default date windows are applied by the
#    listing CLI; explicit --start/--stop override them). --infra is always
#    hpc: SLURM job arrays only run on the hpc (ice uses PBS).
create-unified-wv-alti-job-array-listing \
    ${listing_args[@]+"${listing_args[@]}"} \
    --infra hpc \
    --outputpath-csv "$OUTPUTPATH_CSV" \
    --output-type csv

# 2) Number of data rows (skip the header).
NROWS=$(($(wc -l < "$OUTPUTPATH_CSV") - 1))
if [[ "$NROWS" -le 0 ]]; then
    echo "Error: listing $OUTPUTPATH_CSV has no data rows." >&2
    exit 1
fi

# 3) Per-task script sits next to this one (installed with the package).
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SCRIPT="$SELF_DIR/unified_coloc_WV_alti_cmems_or_cci_slurm_array.bash"
if [[ ! -f "$TASK_SCRIPT" ]]; then
    echo "Error: per-task script not found: $TASK_SCRIPT" >&2
    exit 1
fi

echo "--- Submitting SLURM array: $NROWS tasks ---"
sbatch \
    --array="0-$(( NROWS - 1 ))" \
    --job-name="$JOBNAME" \
    --time="$SBATCH_TIME" \
    --mem="$SBATCH_MEM" \
    --mail-type=NONE \
    "$TASK_SCRIPT" \
    --listing "$OUTPUTPATH_CSV" \
    $REDO \
    $DEV

echo "done. Check the array with: squeue -u $(id -un)"
