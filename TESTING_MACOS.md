# Testing Checklist for M-Series MacOS Support

This document outlines the testing that should be performed on actual M-Series Mac hardware to validate the implementation.

## Prerequisites
- M-Series Mac (M1, M2, or M3)
- macOS 12.0 or later
- Python 3.10+
- PyTorch 2.3.1+ with MPS support

## Manual Testing Checklist

### 1. Environment Setup Tests

- [ ] Python installation (3.10+)
- [ ] PyTorch installation with MPS support
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
  ```
- [ ] Requirements installation
  ```bash
  pip install -r requirements.txt
  # Should succeed without flash-attn
  ```
- [ ] Device detection
  ```bash
  python -c "from library.device_utils import HAS_MPS, HAS_CUDA; print(f'CUDA: {HAS_CUDA}, MPS: {HAS_MPS}')"
  # Should show: CUDA: False, MPS: True
  ```

### 2. Basic Functionality Tests

#### Test 2.1: Auto Device Detection
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_auto \
    --device auto \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should automatically detect and use MPS device

#### Test 2.2: Explicit MPS Selection
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_mps \
    --device mps \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should use MPS device successfully

#### Test 2.3: CPU Fallback
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_cpu \
    --device cpu \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should run on CPU (slowly)

### 3. Memory Optimization Tests

#### Test 3.1: Quantization (16GB Mac)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_quant \
    --device mps \
    --quantized \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: 
- Should print warning about using bfloat16 instead of fp8
- Should complete without OOM errors on 16GB+ Macs

#### Test 3.2: Offload (16GB Mac)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_offload \
    --device mps \
    --offload \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should complete without OOM errors, may be slower

#### Test 3.3: Combined Optimization (16GB Mac)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_both \
    --device mps \
    --quantized \
    --offload \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should work on 16GB Macs

### 4. Resolution Tests

#### Test 4.1: Low Resolution (512)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_512 \
    --device mps \
    --seed 1234 --size_level 512 --version v1.1
```
Expected: Should work on all Macs

#### Test 4.2: Medium Resolution (768)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_768 \
    --device mps \
    --quantized \
    --offload \
    --seed 1234 --size_level 768 --version v1.1
```
Expected: Should work on 32GB+ Macs with optimizations

#### Test 4.3: High Resolution (1024)
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_1024 \
    --device mps \
    --seed 1234 --size_level 1024 --version v1.1
```
Expected: Should work on 64GB+ Macs without optimizations

### 5. Gradio Tests

#### Test 5.1: Basic Gradio (32GB+ Mac)
```bash
python gradio_app.py \
    --model_path /path/to/step1x-edit/ \
    --device mps
```
Expected: Should launch on localhost:32800

#### Test 5.2: Optimized Gradio (16GB Mac)
```bash
python gradio_app.py \
    --model_path /path/to/step1x-edit/ \
    --device mps \
    --quantized \
    --offload
```
Expected: Should launch and be usable on 16GB Macs

### 6. Script Tests

#### Test 6.1: Mac Script
```bash
# First update model_path in the script
bash scripts/run_examples_mac.sh
```
Expected: Should execute both commands successfully

#### Test 6.2: Quantized Mac Script
```bash
# First update model_path in the script
bash scripts/run_examples_mac_quantized.sh
```
Expected: Should execute both commands with optimizations

### 7. Edge Case Tests

#### Test 7.1: Missing Device
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_default
```
Expected: Should default to auto detection

#### Test 7.2: Invalid Device
```bash
python inference.py \
    --input_dir ./examples \
    --model_path /path/to/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./test_invalid \
    --device cuda
```
Expected: Should raise ValueError about CUDA not available

### 8. Performance Benchmarks

Record the following for each Mac model:

- [ ] Time to generate 512x512 image (no optimization)
- [ ] Time to generate 512x512 image (with --quantized --offload)
- [ ] Time to generate 1024x1024 image (if applicable)
- [ ] Peak memory usage (check Activity Monitor)
- [ ] Memory pressure level

### 9. Quality Validation

- [ ] Compare output images with CUDA version using same seed
- [ ] Verify images are not corrupted
- [ ] Check that edits match the prompt instructions
- [ ] Verify deterministic output (same seed = same image)

### 10. Error Handling Tests

#### Test 10.1: Out of Memory
- [ ] Try running with high resolution on low memory Mac
- [ ] Verify error message is clear
- [ ] Verify cleanup happens properly

#### Test 10.2: Attention Mode Fallback
- [ ] Verify "torch" mode message appears
- [ ] Verify no flash-attn errors

## Test Results Template

```markdown
## Test Results

**Hardware:**
- Mac Model: [M1/M2/M3 model]
- Unified Memory: [16GB/32GB/64GB/etc]
- macOS Version: [version]

**Software:**
- Python: [version]
- PyTorch: [version]
- MPS Available: [True/False]

**Test Results:**

| Test | Status | Notes |
|------|--------|-------|
| Auto device detection | ✓/✗ | |
| MPS inference 512 | ✓/✗ | Time: Xs |
| MPS inference 1024 | ✓/✗ | Time: Xs |
| Quantized mode | ✓/✗ | |
| Offload mode | ✓/✗ | |
| Gradio app | ✓/✗ | |
| Mac scripts | ✓/✗ | |

**Performance:**
- 512x512 generation time: [X]s
- 1024x1024 generation time: [X]s (if applicable)
- Peak memory usage: [X]GB

**Issues Found:**
1. [List any issues]

**Recommendations:**
[Any recommendations for improvement]
```

## Automated Tests (Future)

The following should be implemented when CI/CD with Mac runners is available:

- [ ] Unit tests for device detection
- [ ] Integration tests for MPS inference
- [ ] Memory usage tests
- [ ] Output quality regression tests
- [ ] Performance regression tests
