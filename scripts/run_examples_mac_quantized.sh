#!/bin/bash
set -e  # Exit immediately if any command fails

# Script for running Step1X-Edit on M-Series Macs with quantization
# Note: On MPS, quantization uses bfloat16 instead of fp8 for compatibility

# IMPORTANT: Update the model_path below to point to your Step1X-Edit model directory
MODEL_PATH="/path/to/step1x-edit/"  # CHANGE THIS to your actual model path

python inference.py --input_dir ./examples \
    --model_path "$MODEL_PATH" \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_en_mac \
    --device mps \
    --quantized \
    --seed 1234 --size_level 512 --version v1.1

# With offload to reduce memory usage
python inference.py --input_dir ./examples \
    --model_path "$MODEL_PATH" \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn_mac \
    --device mps \
    --quantized \
    --offload \
    --seed 1234 --size_level 512 --version v1.1
