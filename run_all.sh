#!/bin/bash

# ./run_all.sh "bigscience/mt0-large" "T2" 200 "detox_classification"

# Проверка аргументов
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_path> <base_output_dir> <top_n> <losses>"
    echo "losses options: detox, detox_classification, detox_contrastive, all"
    exit 1
fi

# Аргументы
MODEL_PATH=$1
BASE_OUTPUT_DIR=$2
TOP_N=$3
LOSSES=$4

# Обновим OUTPUT_DIR с учетом лоссов
OUTPUT_DIR="${BASE_OUTPUT_DIR}_${LOSSES}"
mkdir -p "$OUTPUT_DIR"

# Устанавливаем лог-файл только после создания директории
LOG_FILE="$OUTPUT_DIR/full_log.txt"

# Оборачиваем весь основной блок в перенаправление
{
    echo "==> Script started at $(date)"

    WEIGHTS=(1.0)

    # Шаг 1: Обучение
    for WEIGHT in "${WEIGHTS[@]}"; do
        echo "==> Training with weight: $WEIGHT and losses: $LOSSES"
        python three_losses.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR/weight_$WEIGHT" \
            --weight "$WEIGHT" \
            --top_n "$TOP_N" \
            --losses "$LOSSES"
    done

    # Шаг 2: Предсказания
    CHECKPOINT_DIR="$OUTPUT_DIR/weight_1.0"
    CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found in $CHECKPOINT_DIR"
        exit 1
    fi

    echo "==> Running predictions from checkpoint: $CHECKPOINT"

    python predict_from_checkpoint.py \
        --checkpoint_path="$CHECKPOINT" \
        --model_name="$MODEL_PATH" \
        --top_n="$TOP_N"

    # Шаг 3: Оценка
    echo "==> Evaluating results in $CHECKPOINT_DIR"

    python evaluate_j.py \
        --data_path="$CHECKPOINT_DIR"

    echo "==> Script finished at $(date)"
} > "$LOG_FILE" 2>&1