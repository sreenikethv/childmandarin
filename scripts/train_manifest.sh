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

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

# ------------------------------------------------------------
# No training
# ------------------------------------------------------------

if [[ "$TRAIN_MODEL" != "true" ]]; then
    cp "$PRETRAINED_MODEL" "$OUT/checkpoint_best.pt"

    log "Training disabled."
    log "Model: $OUT/checkpoint_best.pt"
    exit 0
fi

source /usr/local/python/venv/childlang/bin/activate

MODEL="$PRETRAINED_MODEL"

# Original wav2vec 2.0 pretraining checkpoint.
# This is used to construct Wav2VecCtc.
BASE_MODEL="$ROOT/models/libri960_big.pt"

# ------------------------------------------------------------
# Proof-of-concept training parameters
#
# These intentionally do NOT modify the .cfg file.
# They only limit this smoke test.
# ------------------------------------------------------------

MAX_UPDATE=200

# Do not periodically create numbered checkpoints.
# checkpoint_best.pt / checkpoint_last.pt are sufficient.
SAVE_INTERVAL_UPDATES=1000000000

# ------------------------------------------------------------
# Train one stage
# ------------------------------------------------------------

train() {
    local data="$1"
    local model="$2"
    local save_dir="$3"
    local stage="$4"

    mkdir -p "$save_dir"

    log "========================================"
    log "Starting $stage"
    log "Data:       $data"
    log "Input model: $model"
    log "Output:     $save_dir"
    log "Max update: $MAX_UPDATE"
    log "========================================"

    local start_time
    start_time=$(date +%s)

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
        checkpoint.best_checkpoint_metric=uer \
        checkpoint.maximize_best_checkpoint_metric=false \
        checkpoint.save_dir="$save_dir" \
        --config-dir "$FT" \
        --config-name finetune

# REMOVED:
#        model.freeze_finetune_updates=0 \
#        dataset.validate_after_updates=0 \
#        dataset.validate_interval=10 \
#        optimization.max_update="$MAX_UPDATE" \
#        checkpoint.no_epoch_checkpoints=true \
#        checkpoint.no_last_checkpoints=false \
#        checkpoint.save_interval=1000000000 \
#        checkpoint.save_interval_updates="$SAVE_INTERVAL_UPDATES" \
#        checkpoint.keep_interval_updates=0 \
#        checkpoint.keep_last_epochs=0 \


    local end_time
    end_time=$(date +%s)

    log "$stage training process exited successfully."
    log "Elapsed: $((end_time - start_time)) seconds"

    # --------------------------------------------------------
    # Critical progression check
    # --------------------------------------------------------

    if [[ ! -f "$save_dir/checkpoint_best.pt" ]]; then
        log "ERROR: $stage did not produce checkpoint_best.pt"
        log "Stage progression halted."
        exit 1
    fi

    log "$stage checkpoint verified:"
    log "  $save_dir/checkpoint_best.pt"

    # Do not allow an incomplete temporary checkpoint to
    # accidentally become the model for the next stage.
    if [[ -f "$save_dir/checkpoint8.pt.tmp" ]]; then
        rm -f "$save_dir/checkpoint8.pt.tmp"
    fi

    log "$stage complete."
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

    log "Found ${#STAGES[@]} curriculum stages."

    for STAGE in "${STAGES[@]}"; do

        STAGE_DATA="$DATA/$STAGE"
        STAGE_OUT="$OUT/$STAGE"

        log ""
        log "Preparing $STAGE"
        log "Previous checkpoint: $MODEL"

        train \
            "$STAGE_DATA" \
            "$MODEL" \
            "$STAGE_OUT" \
            "$STAGE"

        # ----------------------------------------------------
        # Explicitly advance to the verified best checkpoint.
        # ----------------------------------------------------

        MODEL="$STAGE_OUT/checkpoint_best.pt"

        [[ -f "$MODEL" ]] || {
            log "ERROR: cannot advance to $STAGE"
            exit 1
        }

        log "ADVANCING TO NEXT STAGE"
        log "Next input model:"
        log "  $MODEL"
    done

else

    log "Regular training"

    train \
        "$DATA" \
        "$MODEL" \
        "$OUT" \
        "training"

    MODEL="$OUT/checkpoint_best.pt"
fi

deactivate

log ""
log "========================================"
log "TRAINING COMPLETE"
log "Best model:"
log "$MODEL"
log "========================================"