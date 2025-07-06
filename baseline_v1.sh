#!/bin/bash

# ./baseline_v1.sh "bigscience/mt0-large" "base_res_v2" -1

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_path> <output_dir> <top_n>"
    exit 1
fi

# Assign arguments to variables
MODEL_PATH=$1
OUTPUT_DIR=$2
TOP_N=$3

# Define an array of weights to use
mkdir -p "$OUTPUT_DIR"

echo "Running baseline:"
LOG_FILE="$OUTPUT_DIR/logs.txt"
# python baseline_detoxifier.py --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR" --top_n "$TOP_N"
python predict_from_checkpoint.py --checkpoint_path="base_res_v2/checkpoint-84" --model_name="$MODEL_PATH" --top_n=-1
cp "$CURRENT_OUTPUT_DIR/toxic.tsv" .
zip res.zip "toxic.tsv"
