"""
Setup LIBERO configuration to point to the correct dataset and asset paths.
Run this once before using the evaluation script.
"""

import os
import pathlib
import yaml


def setup_libero_config():
    """Create ~/.libero/config.yaml with correct paths."""
    
    # Paths relative to this script
    script_dir = pathlib.Path(__file__).parent.resolve()
    openpi_dir = script_dir.parent.parent  # /n/fs/vla-distill/openpi
    workspace_dir = openpi_dir.parent      # /n/fs/vla-distill
    
    # LIBERO library location
    libero_dir = openpi_dir / "third_party" / "libero" / "libero" / "libero"
    
    # Dataset location (where your downloaded datasets are)
    datasets_dir = workspace_dir / "datasets" / "libero_processed"
    
    config = {
        "benchmark_root": str(libero_dir),
        "bddl_files": str(libero_dir / "bddl_files"),
        "init_states": str(libero_dir / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(libero_dir / "assets"),
    }
    
    # Create config directory
    config_dir = pathlib.Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config file
    config_file = config_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    print("✓ LIBERO configuration created successfully!")
    print(f"Config file: {config_file}")
    print("\nPaths configured:")
    for key, value in config.items():
        exists = "✓" if pathlib.Path(value).exists() else "✗"
        print(f"  {exists} {key}: {value}")
    
    # Check if datasets exist
    if not datasets_dir.exists():
        print(f"\n⚠️  WARNING: Datasets directory does not exist: {datasets_dir}")
        print("   You may need to download LIBERO datasets first.")
    
    return config


if __name__ == "__main__":
    setup_libero_config()
