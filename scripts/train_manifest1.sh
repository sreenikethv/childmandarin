#!/bin/bash
set -euo pipefail

CONFIG="${1:?Usage: $0 CONFIG.cfg}"
[[ -f "$CONFIG" ]] || { echo "ERROR: CONFIG must be a file: $CONFIG" >&2; exit 1; }

source "$CONFIG"

ROOT="/projects/assigned/ChildLang"
FT="$ROOT/finetune_fairseq"
DATA="$FT/manifest/$EXPERIMENT_NAME"
STAGE="$DATA/stage_01"
OUT="$FT/outputs/$EXPERIMENT_NAME"

[[ -d "$STAGE" ]] || { echo "ERROR: curriculum stage not found: $STAGE" >&2; exit 1; }
[[ -f "$PRETRAINED_MODEL" ]] || { echo "ERROR: model not found: $PRETRAINED_MODEL" >&2; exit 1; }

source /usr/local/python/venv/childlang/bin/activate

python "$ROOT/fairseq/fairseq_cli/hydra_train.py" \
    distributed_training.distributed_port=0 \
    task.data="$STAGE" \
    task.labels=ltr \
    dataset.valid_subset=dev \
    model.w2v_path="$ROOT/models/libri960_big.pt" \
    checkpoint.restore_file="$PRETRAINED_MODEL" \
    checkpoint.save_dir="$OUT" \
#    checkpoint.reset_dataloader=true \
    --config-dir "$FT" \
    --config-name finetune

deactivate