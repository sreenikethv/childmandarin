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
DATA="$FT/manifest/$EXPERIMENT_NAME"
OUT="$FT/outputs/$EXPERIMENT_NAME"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is missing}"
: "${PRETRAINED_MODEL:?PRETRAINED_MODEL is missing}"
: "${TRAIN_MODEL:?TRAIN_MODEL is missing}"
: "${CURRICULUM:?CURRICULUM is missing}"

[[ -d "$DATA" ]] || {
    echo "ERROR: manifest not found: $DATA" >&2
    exit 1
}

[[ -f "$PRETRAINED_MODEL" ]] || {
    echo "ERROR: model not found: $PRETRAINED_MODEL" >&2
    exit 1
}

mkdir -p "$OUT"

# ------------------------------------------------------------
# No training
# ------------------------------------------------------------

if [[ "$TRAIN_MODEL" != "true" ]]; then
    cp "$PRETRAINED_MODEL" "$OUT/checkpoint_best.pt"

    echo "Training disabled."
    echo "Model: $OUT/checkpoint_best.pt"
    exit 0
fi

source /usr/local/python/venv/childlang/bin/activate

MODEL="$PRETRAINED_MODEL"

# This is the original Wav2Vec 2.0 model used to construct
# Wav2VecCtc. Fine-tuned checkpoints are restored separately.
BASE_MODEL="$ROOT/models/libri960_big.pt"

# ------------------------------------------------------------
# Common Hydra training arguments
# ------------------------------------------------------------

train() {
    local data="$1"
    local model="$2"
    local save_dir="$3"

    mkdir -p "$save_dir"

    python "$ROOT/fairseq/fairseq_cli/hydra_train.py" \
        distributed_training.distributed_port=0 \
        task.data="$data" \
        task.labels=ltr \
        dataset.valid_subset=dev \
        model.w2v_path="$BASE_MODEL" \
        checkpoint.restore_file="$model" \
        checkpoint.reset_dataloader=true \
        checkpoint.reset_optimizer=true \
        checkpoint.reset_lr_scheduler=true \
        checkpoint.save_dir="$save_dir" \
        checkpoint.best_checkpoint_metric=uer \
        checkpoint.maximize_best_checkpoint_metric=false \
        --config-dir "$FT" \
        --config-name finetune
}

# ------------------------------------------------------------
# Curriculum
# ------------------------------------------------------------

if [[ "$CURRICULUM" == "true" ]]; then

    mapfile -t STAGES < <(
        find "$DATA" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name 'stage_*' \
            -printf '%f\n' |
        sort -V
    )

    [[ "${#STAGES[@]}" -gt 0 ]] || {
        echo "ERROR: no curriculum stages found in $DATA" >&2
        exit 1
    }

    for STAGE in "${STAGES[@]}"; do
        STAGE_DATA="$DATA/$STAGE"
        STAGE_OUT="$OUT/$STAGE"

        echo
        echo "Training $STAGE"
        echo "  Data:  $STAGE_DATA"
        echo "  Model: $MODEL"

        train "$STAGE_DATA" "$MODEL" "$STAGE_OUT"

        if [[ ! -f "$STAGE_OUT/checkpoint_best.pt" ]]; then
            echo "ERROR: checkpoint_best.pt was not produced for $STAGE" >&2
            exit 1
        fi

        MODEL="$STAGE_OUT/checkpoint_best.pt"
    done

else

    echo "Regular training"
    echo "  Data:  $DATA"
    echo "  Model: $MODEL"

    train "$DATA" "$MODEL" "$OUT"

    if [[ ! -f "$OUT/checkpoint_best.pt" ]]; then
        echo "ERROR: checkpoint_best.pt was not produced." >&2
        exit 1
    fi

fi

deactivate

echo
echo "Training complete."
echo "Best model: $MODEL"