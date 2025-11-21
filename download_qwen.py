#!/usr/bin/env python3
"""Download Qwen2.5-VL-7B-Instruct model for Step1X-Edit"""

from huggingface_hub import snapshot_download
import os

if __name__ == "__main__":
    local_dir = "./Step1X-Edit/Qwen2.5-VL-7B-Instruct"
    
    print(f"Downloading Qwen2.5-VL-7B-Instruct to {local_dir}...")
    print("This will take a while (model is ~15GB)...")
    print("Progress:")
    
    try:
        snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("\n✓ Download complete!")
        print(f"Model saved to: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("You can try running this script again to resume the download.")
