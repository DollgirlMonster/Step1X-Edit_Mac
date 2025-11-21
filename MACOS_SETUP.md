# MacOS Setup Guide for Step1X-Edit

This guide provides detailed instructions for setting up and running Step1X-Edit on M-Series Macs (M1, M2, M3).

## System Requirements

- **Hardware**: M1, M2, or M3 Mac with at least 16GB unified memory
- **OS**: macOS 12.0 (Monterey) or later
- **Python**: 3.10.0 or later

## Memory Recommendations

Based on your Mac's unified memory:

| Unified Memory | Recommended Settings | Resolution |
|----------------|---------------------|------------|
| 16GB | `--offload --quantized` | `--size_level 512` |
| 32GB | `--offload` or `--quantized` | `--size_level 768` |
| 64GB+ | No flags needed | `--size_level 1024` |

## Installation

### 1. Install Python and PyTorch

First, ensure you have Python 3.10+ installed. We recommend using Homebrew:

```bash
brew install python@3.10
```

### 2. Install PyTorch with MPS Support

Install PyTorch 2.3.1 or later with MPS (Metal Performance Shaders) support:

```bash
pip3 install torch>=2.3.1 torchvision>=0.18.1
```

### 3. Install Step1X-Edit Requirements

```bash
cd Step1X-Edit_Mac
pip install -r requirements.txt
```

**Note**: Flash Attention is not available on macOS. The model will automatically use PyTorch's optimized attention instead.

### 4. Download Model Weights

Download the model weights from [HuggingFace](https://huggingface.co/stepfun-ai/Step1X-Edit):

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/stepfun-ai/Step1X-Edit

# Or download manually from the HuggingFace page
```

## Running Step1X-Edit

### Basic Usage

For general image editing with standard quality:

```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/Step1X-Edit \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_mac \
    --device mps \
    --seed 1234 \
    --size_level 512 \
    --version v1.1
```

### Optimized for Lower Memory (16GB Macs)

```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/Step1X-Edit \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_mac \
    --device mps \
    --quantized \
    --offload \
    --seed 1234 \
    --size_level 512 \
    --version v1.1
```

### High Quality (32GB+ Macs)

```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/Step1X-Edit \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_mac \
    --device mps \
    --seed 1234 \
    --size_level 1024 \
    --version v1.1
```

### Using Convenience Scripts

We provide pre-configured scripts for common use cases:

**Standard quality:**
```bash
bash scripts/run_examples_mac.sh
```
(Remember to update the `model_path` in the script first)

**Memory optimized:**
```bash
bash scripts/run_examples_mac_quantized.sh
```

### Gradio Web Interface

Run the interactive web interface:

```bash
# For 16GB Macs
python gradio_app.py \
    --model_path /path/to/Step1X-Edit \
    --device mps \
    --quantized \
    --offload

# For 32GB+ Macs
python gradio_app.py \
    --model_path /path/to/Step1X-Edit \
    --device mps
```

Then open your browser to `http://localhost:32800`

## Command Line Options

### Device Selection
- `--device auto`: Auto-detect best device (recommended)
- `--device mps`: Force MPS (M-Series GPU)
- `--device cpu`: Use CPU only (slow)

### Memory Optimization
- `--offload`: Offload models to CPU when not in use (reduces GPU memory)
- `--quantized`: Use quantized model (bfloat16 on MPS, saves memory)
- `--size_level`: Output resolution (512, 768, or 1024)

### Task Types
- `--task_type edit`: Image editing (default)
- `--task_type t2i`: Text-to-image generation

## Performance Tips

1. **Close other applications** before running to free up unified memory
2. **Start with smaller resolutions** (`--size_level 512`) and increase if you have headroom
3. **Use Activity Monitor** to watch memory pressure during generation
4. **Enable offload** if you experience memory issues
5. **For batch processing**, process images one at a time to avoid memory issues

## Troubleshooting

### Out of Memory Errors

If you encounter memory errors:

1. Add `--offload` flag
2. Add `--quantized` flag  
3. Reduce `--size_level` (try 512)
4. Close other applications
5. Restart your Mac to clear memory

### Slow Performance

If generation is slow:

1. Ensure you're using `--device mps` (not cpu)
2. Check Activity Monitor for memory pressure
3. Try without `--offload` if you have enough memory
4. Update to the latest macOS version

### Import Errors

If you get import errors:

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall --no-cache-dir

# Make sure PyTorch has MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Differences from CUDA Version

1. **Attention Mechanism**: Uses PyTorch's native attention instead of Flash Attention
2. **Quantization**: Uses bfloat16 instead of fp8 (fp8 has limited MPS support)
3. **Memory Management**: Uses `torch.mps.empty_cache()` instead of CUDA equivalents
4. **Multi-GPU**: Not supported on MPS (single GPU only)

## Getting Help

If you encounter issues:

1. Check the [main README](README.md) for general troubleshooting
2. Open an issue on GitHub with:
   - Your Mac model and memory
   - macOS version
   - Full error message
   - Command you ran

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Navigate to the project directory
cd Step1X-Edit_Mac

# 2. Edit an image with custom prompt
python inference.py \
    --input_dir ./my_images \
    --model_path ~/Models/Step1X-Edit \
    --json_path ./my_prompts.json \
    --output_dir ./my_output \
    --device mps \
    --quantized \
    --offload \
    --seed 42 \
    --size_level 512 \
    --version v1.1

# 3. Check the output
open ./my_output
```

Where `my_prompts.json` contains:
```json
{
  "input_image.jpg": "your editing instruction here"
}
```

## Performance Benchmarks (Approximate)

Based on M2 Max with 32GB unified memory:

| Configuration | Resolution | Time per Image |
|---------------|-----------|----------------|
| No optimization | 512x512 | ~25-35s |
| No optimization | 1024x1024 | ~60-80s |
| --quantized --offload | 512x512 | ~35-45s |
| --quantized --offload | 1024x1024 | ~80-110s |

*Note: Actual performance varies based on image complexity and Mac model*
