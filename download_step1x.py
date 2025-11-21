#!/usr/bin/env python3
"""Download Step1X-Edit model weights"""

from huggingface_hub import snapshot_download
import os

if __name__ == "__main__":
    local_dir = "./Step1X-Edit"
    
    print(f"Downloading Step1X-Edit model to {local_dir}...")
    print("This will take a while (several GB)...")
    print("Progress:")
    
    try:
        snapshot_download(
            repo_id="stepfun-ai/Step1X-Edit",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("\n✓ Download complete!")
        print(f"Model saved to: {os.path.abspath(local_dir)}")
        print("\nTo use this model, update your scripts to use:")
        print(f"  --model_path {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("You can try running this script again to resume the download.")
