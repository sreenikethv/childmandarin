#!/bin/bash
set -euo pipefail

ROOT="/projects/assigned/ChildLang/finetune_fairseq"
CONFIG_DIR="$ROOT/configs"
LOG_DIR="$ROOT/logs"
CONFIG_LIST="$CONFIG_DIR/config_list.txt"

: > "$CONFIG_LIST"

for f in "$CONFIG_DIR"/*.cfg; do
    [[ -f "$f" ]] || continue

    base="$(basename "$f")"
    name="${base%.cfg}"

    echo "$name $base" >> "$CONFIG_LIST"
    mkdir -p "$LOG_DIR/$name"
done