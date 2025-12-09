"""
Diagnostic script to verify HDF5 files contain different images across episodes.
"""
import h5py
import numpy as np
from pathlib import Path
import sys

def check_hdf5_diversity(hdf5_path: str):
    """Check if images in HDF5 file are actually different across episodes."""
    hdf5_path = Path(hdf5_path)

    if hdf5_path.is_file():
        files = [hdf5_path]
    else:
        files = list(hdf5_path.glob("*.hdf5")) + list(hdf5_path.glob("*.h5"))

    print(f"Checking {len(files)} file(s)...\n")

    for file_path in files[:3]:  # Check first 3 files
        print(f"File: {file_path.name}")
        with h5py.File(file_path, 'r') as f:
            demos = sorted(list(f['data'].keys()))[:5]  # Check first 5 demos

            for demo_name in demos:
                demo_group = f['data'][demo_name]
                obs_group = demo_group['obs']

                # Get main camera images
                if 'agentview_rgb' in obs_group:
                    images = obs_group['agentview_rgb'][:]
                elif 'agentview_image' in obs_group:
                    images = obs_group['agentview_image'][:]
                else:
                    print(f"  No agentview found in {demo_name}")
                    continue

                # Check first and last frame
                first_frame = images[0]
                last_frame = images[-1]

                # Compute statistics
                first_mean = first_frame.mean()
                first_std = first_frame.std()
                last_mean = last_frame.mean()
                last_std = last_frame.std()

                # Compute difference between first and last
                frame_diff = np.abs(first_frame.astype(float) - last_frame.astype(float)).mean()

                print(f"  {demo_name}: {len(images)} frames")
                print(f"    First frame: mean={first_mean:.2f}, std={first_std:.2f}")
                print(f"    Last frame:  mean={last_mean:.2f}, std={last_std:.2f}")
                print(f"    Avg pixel diff (first-last): {frame_diff:.2f}")
                print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_hdf5_images.py <path_to_hdf5_file_or_dir>")
        sys.exit(1)

    check_hdf5_diversity(sys.argv[1])