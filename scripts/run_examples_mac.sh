#!/bin/bash
set -e  # Exit immediately if any command fails

# Script for running Step1X-Edit on M-Series Macs (M1/M2/M3)
# This script uses MPS (Metal Performance Shaders) for GPU acceleration

# IMPORTANT: Update the model_path below to point to your Step1X-Edit model directory
MODEL_PATH="/path/to/step1x-edit/"  # CHANGE THIS to your actual model path

# Validate that model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist."
    echo "Please update MODEL_PATH in this script to point to your Step1X-Edit model directory."
    exit 1
fi

python inference.py --input_dir ./examples \
    --model_path "$MODEL_PATH" \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_en_mac \
    --device mps \
    --seed 1234 --size_level 1024 --version v1.1

python inference.py --input_dir ./examples \
    --model_path "$MODEL_PATH" \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn_mac \
    --device mps \
    --seed 1234 --size_level 1024 --version v1.1
