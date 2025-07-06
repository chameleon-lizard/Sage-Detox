#!/bin/bash

# ./script_v1.sh "bigscience/mt0-xl" "tl_script_res_v6_3L" -1

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
WEIGHTS=(1.0)

# Large: 1e-3 best
# Define an array of learning rates to use
LEARNING_RATES=(0.00005)

# Create the base output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each weight
for WEIGHT in "${WEIGHTS[@]}"; do
    # Loop through each learning rate
    for LR in "${LEARNING_RATES[@]}"; do
        echo "Running with weight: $WEIGHT and learning rate: $LR"

        # Create a subdirectory for the current weight and learning rate combination
        CURRENT_OUTPUT_DIR="$OUTPUT_DIR/weight_${WEIGHT}_lr_${LR//./_}" # Replace dot with underscore for directory name
        mkdir -p "$CURRENT_OUTPUT_DIR"

        # Define the log file path for the current run
        LOG_FILE="$CURRENT_OUTPUT_DIR/logs.txt"

        # Run the Python script with the specified parameters
        # Redirect standard output and standard error to the log file
        # CUDA_VISIBLE_DEVICES=1 python two_losses.py \
        #     --model_path "$MODEL_PATH" \
        #     --output_dir "$CURRENT_OUTPUT_DIR" \
        #     --weight "$WEIGHT" \
        #     --top_n "$TOP_N" \
        #     --lr "$LR" # \
            # > "$LOG_FILE" 2>&1

        CUDA_VISIBLE_DEVICES=0 python predict_from_checkpoint.py --checkpoint_path="$CURRENT_OUTPUT_DIR/checkpoint-1500" --model_name="$MODEL_PATH" --top_n=-1
        # python evaluate_j.py --data_path="$CURRENT_OUTPUT_DIR"
        cp "$CURRENT_OUTPUT_DIR/toxic.tsv" .
        zip res.zip "toxic.tsv"

        echo "Finished run for weight: $WEIGHT and learning rate: $LR. Logs saved to $LOG_FILE"
    done
done

echo "All runs completed."



