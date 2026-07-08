#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 electricity}"
FEATURES="${FEATURES:-M}"
PRED_LEN="${PRED_LEN:-24}"
SEQ_LEN="${SEQ_LEN:-96}"
MODEL="${MODEL:-all}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/run_train_base_models}"
LOG_DIR="${LOG_ROOT}/${TIMESTAMP}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"

is_dry_run() {
    case "${DRY_RUN,,}" in
        1|true|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

run_dataset() {
    local dataset="$1"
    local log_file="${LOG_DIR}/${dataset}.log"
    local cmd=(
        "$PYTHON_BIN"
        "1_train_base_models.py"
        "--dataset" "$dataset"
        "--features" "$FEATURES"
        "--pred_len" "$PRED_LEN"
        "--seq_len" "$SEQ_LEN"
        "--model" "$MODEL"
        "--train_epochs" "$TRAIN_EPOCHS"
        "--batch_size" "$BATCH_SIZE"
        "--num_workers" "$NUM_WORKERS"
    )

    printf '\n[%s] Running dataset=%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$dataset"
    printf '[%s] Command:' "$dataset"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if is_dry_run; then
        return 0
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" -ne 0 ]]; then
        printf '[%s] Failed with exit code %s. See %s\n' "$dataset" "$status" "$log_file" >&2
        return "$status"
    fi
}

for dataset in $DATASETS; do
    run_dataset "$dataset"
done

printf '\nAll requested runs finished. Console logs: %s\n' "$LOG_DIR"
