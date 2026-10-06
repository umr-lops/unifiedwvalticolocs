#!/usr/bin/env bash
# Per-task script for a SLURM job array: one task = one row of the listing CSV.
#
# Submitted by submit_slurm_jobarray.sh, which sets SLURM_ARRAY_TASK_ID and
# passes the absolute listing path. Do not run this script directly outside a
# SLURM array.
set -euo pipefail

LISTING=""
REDO=""
DEV=""

usage() {
    echo "Usage: $(basename "$0") --listing LISTING_CSV [--redo] [--dev]"
    echo ""
    echo "One SLURM array task per listing CSV row (header-aware). The CSV"
    echo "columns are: startdate,sat,alt,outputdir,image,config"
    echo ""
    echo "  --listing CSV   Absolute path to the listing CSV (required)"
    echo "  --redo          Force reprocessing of existing output"
    echo "  --dev           Developer mode"
    echo "  -h, --help      Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --listing) LISTING="$2"; shift 2 ;;
        --redo)    REDO="--redo"; shift 1 ;;
        --dev)     DEV="--dev"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$LISTING" ]]; then
    echo "Error: --listing is required." >&2
    usage
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: must run as a SLURM array task (SLURM_ARRAY_TASK_ID unset)." >&2
    exit 1
fi

if [[ ! -f "$LISTING" ]]; then
    echo "Error: listing not found: $LISTING" >&2
    exit 1
fi

# Row = task ID + 2 (1-based row number, +1 for the CSV header).
LINE=$(( SLURM_ARRAY_TASK_ID + 2 ))
ROW=$(sed -n "${LINE}p" "$LISTING")

if [[ -z "$ROW" || "$ROW" == "startdate,"* ]]; then
    echo "Error: task ID $SLURM_ARRAY_TASK_ID out of range for listing $LISTING." >&2
    exit 1
fi

IFS=',' read -r STARTDATE SAT ALT OUTPUTDIR IMAGE CONFIG <<< "$ROW"

if [[ -z "$STARTDATE" || -z "$SAT" || -z "$ALT" || -z "$OUTPUTDIR" || -z "$IMAGE" || -z "$CONFIG" ]]; then
    echo "Error: malformed listing row $LINE: $ROW" >&2
    exit 1
fi

echo "--- SLURM task $SLURM_ARRAY_TASK_ID : $STARTDATE $SAT $ALT ---"
echo "Output dir: $OUTPUTDIR"
echo "Image SIF:  $IMAGE"
echo "Config:     $CONFIG"

optssimg="exec -B /scale/reference/ -B /legacy/project/cersat/public -B /scratch -B /ontap"
apptainer $optssimg "$IMAGE" procunifiedwvalticolocs \
    --outputdir "$OUTPUTDIR" \
    --startdate "$STARTDATE" \
    --sat "$SAT" \
    --alt "$ALT" \
    --config "$CONFIG" \
    $REDO \
    $DEV

echo "end of task $SLURM_ARRAY_TASK_ID"
