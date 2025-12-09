#!/usr/bin/env python3
"""
Simple script to load the HuggingFaceVLA/libero dataset.

Usage:
    python openpi/examples/libero/load_libero_dataset.py

Note:
    Login using e.g. `huggingface-cli login` to access this dataset
"""

import os
from pathlib import Path
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("HuggingFaceVLA/libero")

print("Dataset loaded successfully!")
print(f"Dataset splits: {list(ds.keys())}")

# Show where the dataset is cached
# Check environment variable first, then use default location
cache_dir = os.getenv("HF_DATASETS_CACHE")
if cache_dir is None:
    # Default cache location
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
else:
    cache_dir = Path(cache_dir)

# Expand user path if needed
cache_dir = cache_dir.expanduser()

print(f"\nDataset cache directory: {cache_dir}")

# Try to find the specific dataset directory
# HuggingFace datasets can be stored in different locations depending on format
dataset_name = "HuggingFaceVLA___libero"
dataset_cache_path = cache_dir / dataset_name

# Also check parquet subdirectory (where Arrow files are often stored)
parquet_cache_path = cache_dir / "parquet"

found_path = None
if dataset_cache_path.exists():
    found_path = dataset_cache_path
elif parquet_cache_path.exists():
    # Search for the dataset in parquet subdirectories
    # Datasets are stored in hash-based subdirectories
    for subdir in parquet_cache_path.rglob("*"):
        if subdir.is_dir() and any(f.name.startswith("parquet-") and f.suffix == ".arrow" for f in subdir.iterdir() if f.is_file()):
            found_path = subdir
            break

if found_path:
    print(f"Dataset stored at: {found_path}")
    
    # Check file types
    arrow_files = list(found_path.glob("*.arrow"))
    parquet_files = list(found_path.glob("*.parquet"))
    json_files = list(found_path.glob("*.json"))
    
    if arrow_files:
        print(f"\nFound {len(arrow_files)} Arrow (.arrow) files")
        print("  Note: These are Apache Arrow format files, not Parquet files")
        print("  - HuggingFace datasets library uses Arrow format internally for performance")
        print("  - Arrow is a columnar in-memory format optimized for fast data access")
        print("  - Files may be named 'parquet-*' but are stored as .arrow for efficiency")
        print("  - You can still access the data normally through the datasets library")
        if len(arrow_files) > 0:
            sample_file = arrow_files[0]
            size = sample_file.stat().st_size / (1024**2)  # MB
            print(f"  - Sample file: {sample_file.name} ({size:.1f} MB)")
    
    if parquet_files:
        print(f"\nFound {len(parquet_files)} Parquet (.parquet) files")
    
    if json_files:
        print(f"\nFound {len(json_files)} JSON files (metadata)")
    
    # Check for lock files
    lock_files = list(found_path.rglob("*.lock"))
    if lock_files:
        print(f"\n⚠️  Found {len(lock_files)} lock file(s):")
        for lock_file in lock_files:
            print(f"  {lock_file.relative_to(found_path)}")
        print("\nWhat does this mean?")
        print("  - Lock files indicate the dataset is currently being downloaded or processed")
        print("  - This is normal during the initial download")
        print("  - The lock files prevent multiple processes from corrupting the download")
        print("  - Once the download completes, lock files should be removed automatically")
        print("  - If lock files persist after download, you can safely delete them")
    
    # Show size (excluding lock files)
    data_files = [f for f in found_path.rglob('*') if f.is_file() and not f.suffix == '.lock']
    if data_files:
        total_size = sum(f.stat().st_size for f in data_files)
        print(f"\nDataset size (excluding lock files): {total_size / (1024**3):.2f} GB")
    else:
        print("\n⚠️  No data files found yet - dataset may still be downloading")
else:
    print(f"Note: Dataset may be stored in a subdirectory of: {cache_dir}")
    print("  Try searching in: parquet/ subdirectory")

# Print some basic info about the dataset
for split_name, split_data in ds.items():
    print(f"\n{split_name}:")
    print(f"  Number of examples: {len(split_data)}")
    if len(split_data) > 0:
        print(f"  Features: {list(split_data[0].keys())}")

