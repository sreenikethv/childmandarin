#!/bin/bash
set -euo pipefail

CONFIG="${1:?Usage: $0 CONFIG.cfg}"
[[ -f "$CONFIG" ]] || {
    echo "ERROR: CONFIG must be a file: $CONFIG" >&2
    exit 1
}

source "$CONFIG"

ROOT="/projects/assigned/ChildLang"
FT="$ROOT/finetune_fairseq"

MANIFEST="$FT/manifest/$EXPERIMENT_NAME"
OUTPUT="$FT/outputs/$EXPERIMENT_NAME"
MODEL="$PRETRAINED_MODEL"

START_TIME=$(date +%s)

elapsed() {
    local seconds=$(( $(date +%s) - START_TIME ))
    printf '%02dh:%02dm:%02ds' \
        $((seconds / 3600)) \
        $(((seconds % 3600) / 60)) \
        $((seconds % 60))
}

run_step() {
    local name="$1"
    shift

    local step_start
    step_start=$(date +%s)

    echo
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name"
    echo "Elapsed total: $(elapsed)"
    echo "============================================================"

    "$@"

    local step_seconds=$(( $(date +%s) - step_start ))

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $name"
    printf 'Task time: %02dh:%02dm:%02ds\n' \
        $((step_seconds / 3600)) \
        $(((step_seconds % 3600) / 60)) \
        $((step_seconds % 60))
    echo "Total elapsed: $(elapsed)"
}

echo
echo "============================================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config:     $CONFIG"
echo "============================================================"

run_step \
    "Running make_manifest.sh" \
    "$FT/scripts/make_manifest.sh" "$CONFIG"

if [[ "$TRAIN_MODEL" == "true" ]]; then

    run_step \
        "Running train_manifest.sh" \
        "$FT/scripts/train_manifest.sh" "$CONFIG"

    MODEL="$OUTPUT/checkpoint_best.pt"

else
    echo
    echo "Training disabled."
fi

run_step \
    "Running test_manifest.sh" \
    "$FT/scripts/test_manifest.sh" \
    "$MANIFEST" \
    "$OUTPUT" \
    "$EXPERIMENT_NAME"

echo
echo "============================================================"
echo "Experiment complete: $EXPERIMENT_NAME"
echo "Total time: $(elapsed)"
echo "============================================================"