#!/bin/bash

set -euo pipefail

CONFIG="${1:?Usage: make_manifest.sh CONFIG}"

source "$CONFIG"

PROJECT_ROOT="/projects/assigned/ChildLang"
FINETUNE_ROOT="$PROJECT_ROOT/finetune_fairseq"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is missing from $CONFIG}"
: "${DATASET:?DATASET is missing from $CONFIG}"
: "${ROLE:?ROLE is missing from $CONFIG}"
: "${DIALECTS:?DIALECTS is missing from $CONFIG}"
: "${TEST_DIALECTS:?TEST_DIALECTS is missing from $CONFIG}"

MANIFEST_DIR="$FINETUNE_ROOT/manifest/$EXPERIMENT_NAME"
TONE_DIR="$MANIFEST_DIR/tone_labels"

mkdir -p "$MANIFEST_DIR"
mkdir -p "$TONE_DIR"

# ============================================================
# Tone labels
# ============================================================

echo "Creating tone labels..."
echo "  Output: $TONE_DIR"

python \
    "$FINETUNE_ROOT/scripts/make_tone_labels.py" \
    --childtalk "$DATASET" \
    --output "$TONE_DIR"

# ============================================================
# Manifest configuration
# ============================================================

echo "Creating manifest..."
echo "  Experiment:       $EXPERIMENT_NAME"
echo "  Train dialects:   $DIALECTS"
echo "  Test dialects:    $TEST_DIALECTS"
echo "  Train role:       $ROLE"
echo "  Train age:        ${AGE_MIN:-ALL}-${AGE_MAX:-ALL}"
echo "  Test role:        ${TEST_ROLE:-$ROLE}"
echo "  Test age:         ${TEST_AGE_MIN:-ALL}-${TEST_AGE_MAX:-ALL}"
echo "  Curriculum:       ${CURRICULUM:-false}"

MANIFEST_ARGS=(
    --childtalk "$DATASET"
    --tone-dir "$TONE_DIR"
    --output "$MANIFEST_DIR"
    --dialects "$DIALECTS"
    --test-dialects "$TEST_DIALECTS"
    --role "$ROLE"
)

# ============================================================
# Optional training filters
# ============================================================

if [[ -n "${AGE_MIN:-}" ]]; then
    MANIFEST_ARGS+=(
        --age-min "$AGE_MIN"
    )
fi

if [[ -n "${AGE_MAX:-}" ]]; then
    MANIFEST_ARGS+=(
        --age-max "$AGE_MAX"
    )
fi

# ============================================================
# Optional test filters
# ============================================================

if [[ -n "${TEST_ROLE:-}" ]]; then
    MANIFEST_ARGS+=(
        --test-role "$TEST_ROLE"
    )
fi

if [[ -n "${TEST_AGE_MIN:-}" ]]; then
    MANIFEST_ARGS+=(
        --test-age-min "$TEST_AGE_MIN"
    )
fi

if [[ -n "${TEST_AGE_MAX:-}" ]]; then
    MANIFEST_ARGS+=(
        --test-age-max "$TEST_AGE_MAX"
    )
fi

# ============================================================
# Curriculum
#
# CURRICULUM=true:
#   ROLE=adult
#       stage_01 = all selected adults
#
#   ROLE=child
#       stage_01 = oldest selected child age
#       stage_02 = oldest + next age
#       ...
#
#   ROLE=both
#       stage_01 = all selected adults
#       stage_02 = adults + oldest children
#       stage_03 = adults + oldest + next-oldest
#       ...
#
# Dialects are pooled within each age/role group.
# ============================================================

if [[ "${CURRICULUM:-false}" == "true" ]]; then
    MANIFEST_ARGS+=(
        --curriculum
    )
fi

# ============================================================
# Generate manifest
# ============================================================

python \
    "$FINETUNE_ROOT/scripts/make_manifest.py" \
    "${MANIFEST_ARGS[@]}"

echo
echo "Manifest created:"
echo "  $MANIFEST_DIR"