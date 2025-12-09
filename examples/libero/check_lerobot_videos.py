#!/usr/bin/env python3
"""
Check if videos in LeRobot dataset are actually different.
This helps diagnose if the conversion or video extraction is causing identical videos.
"""

import argparse
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq


def check_video_diversity(dataset_path: Path, num_episodes: int = 5):
    """
    Check if different episodes in the LeRobot dataset have different image data.

    Args:
        dataset_path: Path to LeRobot dataset directory
        num_episodes: Number of episodes to check
    """
    print(f"Checking LeRobot dataset: {dataset_path}\n")

    # Find parquet files
    chunk_dirs = sorted(dataset_path.glob("data/chunk-*"))
    if not chunk_dirs:
        print(f"Error: No chunk directories found in {dataset_path}/data/")
        return

    print(f"Found {len(chunk_dirs)} chunk directories\n")

    # Collect episode parquet files
    episode_files = []
    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        episode_files.extend(parquet_files)

    if not episode_files:
        print("Error: No episode parquet files found!")
        return

    print(f"Found {len(episode_files)} episode files")
    print(f"Checking first {min(num_episodes, len(episode_files))} episodes...\n")

    episode_stats = []

    for i, parquet_file in enumerate(episode_files[:num_episodes]):
        print(f"Episode {i} ({parquet_file.name}):")

        try:
            # Read parquet file
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Find image columns
            image_cols = [col for col in df.columns if 'image' in col.lower()]

            if not image_cols:
                print(f"  Warning: No image columns found!")
                continue

            print(f"  Image columns: {image_cols}")
            print(f"  Frames: {len(df)}")

            # Check first image column
            img_col = image_cols[0]

            # Try to get first and last frame stats
            first_val = df[img_col].iloc[0]
            last_val = df[img_col].iloc[-1] if len(df) > 1 else first_val

            # Handle different storage formats
            def extract_stats(val):
                """Extract mean/std from image value."""
                if isinstance(val, dict):
                    # Check for embedded data
                    if 'array' in val:
                        arr = np.array(val['array'])
                        return arr.mean(), arr.std(), arr.shape
                    elif 'data' in val:
                        arr = np.array(val['data'])
                        return arr.mean(), arr.std(), arr.shape
                    elif 'path' in val:
                        return None, None, f"path: {val['path'][:50]}"
                    else:
                        return None, None, f"dict_keys: {list(val.keys())}"
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.array(val)
                    return arr.mean(), arr.std(), arr.shape
                elif isinstance(val, str):
                    return None, None, f"path: {val[:50]}"
                else:
                    return None, None, f"type: {type(val).__name__}"

            first_mean, first_std, first_info = extract_stats(first_val)
            last_mean, last_std, last_info = extract_stats(last_val)

            if first_mean is not None:
                print(f"  First frame: mean={first_mean:.2f}, std={first_std:.2f}, shape={first_info}")
                if last_mean is not None and len(df) > 1:
                    print(f"  Last frame:  mean={last_mean:.2f}, std={last_std:.2f}, shape={last_info}")
                    diff = abs(first_mean - last_mean)
                    print(f"  Pixel diff (first-last): {diff:.2f}")
                    episode_stats.append({
                        'episode': i,
                        'first_mean': first_mean,
                        'last_mean': last_mean,
                        'diff': diff,
                        'frames': len(df)
                    })
                else:
                    episode_stats.append({
                        'episode': i,
                        'first_mean': first_mean,
                        'last_mean': first_mean,
                        'diff': 0,
                        'frames': len(df)
                    })
            else:
                print(f"  First frame info: {first_info}")
                print(f"  Last frame info: {last_info}")
                print(f"  Warning: Could not extract pixel data (images might be stored as paths)")
                episode_stats.append({
                    'episode': i,
                    'first_mean': None,
                    'storage': 'paths',
                    'frames': len(df)
                })

            print()

        except Exception as e:
            print(f"  Error reading episode: {e}\n")
            continue

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not episode_stats:
        print("No episode statistics collected!")
        return

    # Check if all episodes have the same first frame
    valid_stats = [s for s in episode_stats if s.get('first_mean') is not None]

    if valid_stats:
        first_means = [s['first_mean'] for s in valid_stats]
        unique_means = len(set(np.round(first_means, 2)))

        print(f"\nEpisodes checked: {len(valid_stats)}")
        print(f"Unique first-frame means: {unique_means}")

        if unique_means == 1:
            print("\n⚠️  WARNING: All episodes have the SAME first frame mean!")
            print("   This suggests all videos are identical.")
            print("   The bug is likely in the conversion script.")
        elif unique_means == len(valid_stats):
            print("\n✓ All episodes have DIFFERENT first frames - looks good!")
        else:
            print(f"\n⚠️  {unique_means} unique means for {len(valid_stats)} episodes")
            print("   Some episodes might be duplicates.")

        # Show per-episode comparison
        print("\nPer-episode first frame means:")
        for s in valid_stats:
            print(f"  Episode {s['episode']}: {s['first_mean']:.2f} ({s['frames']} frames)")
    else:
        # All stored as paths
        print("\nImages are stored as file paths, not embedded data.")
        print("You'll need to check the actual video files to verify diversity.")
        print("\nTry using the lerobot_to_video.py script to generate videos from")
        print("different episodes and compare them manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if LeRobot dataset videos are diverse")
    parser.add_argument("dataset_path", type=str, help="Path to LeRobot dataset directory")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to check (default: 5)")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        exit(1)

    check_video_diversity(dataset_path, args.num_episodes)