#!/bin/bash

set -euo pipefail

MANIFEST="$1"
OUT="$2"
NAME="$3"

PROJECT_ROOT="/projects/assigned/ChildLang"
RESULT_DIR="$PROJECT_ROOT/finetune_fairseq/results/$NAME"

mkdir -p "$RESULT_DIR"

# ------------------------------------------------------------
# Find model to test
#
# Curriculum:
#   use checkpoint_best.pt from the highest stage
#
# Non-curriculum:
#   use checkpoint_best.pt directly in OUT
# ------------------------------------------------------------

STAGES=( "$OUT"/stage_* )

if [[ -d "${STAGES[0]}" ]]; then

    HIGHEST_STAGE="$(
        printf '%s\n' "${STAGES[@]}" |
        sort -V |
        tail -n 1
    )"

    MODEL="$HIGHEST_STAGE/checkpoint_best.pt"

    echo "Curriculum experiment detected."
    echo "Highest stage: $(basename "$HIGHEST_STAGE")"

else

    MODEL="$OUT/checkpoint_best.pt"

    echo "Non-curriculum experiment."

fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model does not exist:"
    echo "  $MODEL"
    exit 1
fi

echo "Testing model:"
echo "  $MODEL"

source /usr/local/python/venv/childlang/bin/activate

python \
    "$PROJECT_ROOT/fairseq/fairseq_cli/hydra_validate.py" \
    --config-dir "$PROJECT_ROOT/finetune_fairseq" \
    --config-name finetune \
    task.labels=ltr \
    task.data="$MANIFEST" \
    dataset.valid_subset=test \
    dataset.max_tokens=1000000 \
    dataset.max_tokens_valid=1000000 \
    common_eval.path="$MODEL" \
    > "$RESULT_DIR/results.txt"

deactivate

echo "Results written to:"
echo "  $RESULT_DIR/results.txt"