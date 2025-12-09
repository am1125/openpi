#!/usr/bin/env python3
"""
Check the actual video files in a LeRobot dataset to see if they're identical.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess


def get_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def get_video_info(file_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets,width,height,duration',
             '-of', 'csv=p=0', str(file_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return {
                    'width': int(parts[0]),
                    'height': int(parts[1]),
                    'frames': int(parts[2]) if len(parts) > 2 else None,
                    'duration': float(parts[3]) if len(parts) > 3 else None,
                }
    except FileNotFoundError:
        return {'error': 'ffprobe not found'}
    except Exception as e:
        return {'error': str(e)}
    return {}


def check_video_files(dataset_path: Path, num_videos: int = 10):
    """
    Check if video files in LeRobot dataset are identical.

    Args:
        dataset_path: Path to LeRobot dataset directory
        num_videos: Number of videos to check
    """
    print(f"Checking video files in: {dataset_path}\n")

    # Find video directories
    video_dirs = sorted(dataset_path.glob("videos/*"))

    if not video_dirs:
        print(f"No video directories found in {dataset_path}/videos/")
        print("The dataset might not have videos encoded yet, or they might be embedded in parquet files.")
        return

    print(f"Found {len(video_dirs)} video directories\n")

    # Collect all video files
    all_videos = []
    for video_dir in video_dirs:
        video_files = list(video_dir.glob("**/*.mp4"))
        all_videos.extend(video_files)

    if not all_videos:
        print("No .mp4 video files found!")
        return

    print(f"Found {len(all_videos)} video files total")
    print(f"Checking first {min(num_videos, len(all_videos))} videos...\n")

    video_hashes = {}
    video_info = {}

    for i, video_file in enumerate(all_videos[:num_videos]):
        rel_path = video_file.relative_to(dataset_path)
        print(f"Video {i}: {rel_path}")

        # Get file size
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")

        # Get hash
        file_hash = get_file_hash(video_file)
        print(f"  Hash: {file_hash}")

        # Get video metadata
        info = get_video_info(video_file)
        if info:
            if 'error' in info:
                print(f"  Info: {info['error']}")
            else:
                frames = info.get('frames', 'unknown')
                width = info.get('width', 'unknown')
                height = info.get('height', 'unknown')
                print(f"  Dimensions: {width}x{height}, Frames: {frames}")

        video_hashes[str(rel_path)] = file_hash
        video_info[str(rel_path)] = info
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    unique_hashes = len(set(video_hashes.values()))
    total_checked = len(video_hashes)

    print(f"\nVideos checked: {total_checked}")
    print(f"Unique hashes: {unique_hashes}")

    if unique_hashes == 1:
        print("\n⚠️  WARNING: All videos are IDENTICAL (same hash)!")
        print("   All video files have the exact same content.")
        print("   The bug is in the video encoding during conversion.")
    elif unique_hashes == total_checked:
        print("\n✓ All videos are DIFFERENT - looks good!")
    else:
        print(f"\n⚠️  Only {unique_hashes} unique videos for {total_checked} files")
        print("   Some videos are duplicates:")

        # Find duplicates
        hash_to_videos = {}
        for video_path, file_hash in video_hashes.items():
            if file_hash not in hash_to_videos:
                hash_to_videos[file_hash] = []
            hash_to_videos[file_hash].append(video_path)

        for file_hash, videos in hash_to_videos.items():
            if len(videos) > 1:
                print(f"\n  Hash {file_hash[:8]}... appears in {len(videos)} videos:")
                for v in videos[:5]:  # Show first 5
                    print(f"    - {v}")
                if len(videos) > 5:
                    print(f"    ... and {len(videos) - 5} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if LeRobot video files are identical")
    parser.add_argument("dataset_path", type=str, help="Path to LeRobot dataset directory")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to check (default: 10)")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        exit(1)

    check_video_files(dataset_path, args.num_videos)