#!/usr/bin/env python3
"""
Convert LeRobot dataset trajectories to video files.

This script reads image observations from LeRobot dataset parquet files and converts them to video files.

Usage:
    python openpi/examples/libero/lerobot_to_video.py --repo-id your_hf_username/libero --episode-index 0 --output_dir videos/
    python openpi/examples/libero/lerobot_to_video.py --repo-id your_hf_username/libero --episode-index 0 --output_dir videos/ --camera main
"""

import argparse
import io
import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    raise ImportError("pyarrow and pandas are required. Install with: pip install pyarrow pandas")

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


def get_available_cameras(dataset_path: Path, episode_index: int):
    """Get list of available camera/image observation keys for an episode."""
    # Find the parquet file for this episode
    chunk_dirs = sorted(dataset_path.glob("data/chunk-*"))
    for chunk_dir in chunk_dirs:
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            # Find image columns
            cameras = [col for col in df.columns if 'image' in col.lower()]
            return cameras, parquet_path
    
    raise FileNotFoundError(f"Episode {episode_index} not found in dataset")


def extract_images_from_lerobot_api(repo_id: str, episode_index: int, camera=None, dataset_path: Path | None = None):
    """Extract images using LeRobotDataset API (preferred method)."""
    if dataset_path is not None:
        import os
        os.environ["HF_LEROBOT_HOME"] = str(dataset_path.parent)
    
    print(f"Loading dataset (this may take a moment for large datasets)...")
    dataset = LeRobotDataset(repo_id)
    
    # Get episode info to find frame range
    # LeRobotDataset stores episodes sequentially, so we can find the start/end indices
    print(f"Finding episode {episode_index}...")
    
    # Try to get episode info from metadata
    try:
        import json
        episodes_file = dataset_path / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                episodes = [json.loads(line) for line in f]
            # Find the episode
            episode_info = None
            for ep in episodes:
                if ep.get("episode_index") == episode_index:
                    episode_info = ep
                    break
            
            if episode_info:
                start_idx = episode_info.get("index_from")
                end_idx = episode_info.get("index_to")
                print(f"Loading frames {start_idx} to {end_idx}...")
                # Load only frames for this episode
                episode_frames = [dataset[i] for i in range(start_idx, end_idx + 1)]
            else:
                raise ValueError(f"Episode {episode_index} not found in metadata")
        else:
            # Fallback: iterate through dataset (slow but works)
            print("Warning: Using slow iteration method. Consider using --use-parquet flag.")
            episode_frames = []
            for i, frame in enumerate(dataset):
                if frame.get("episode_index", -1) == episode_index:
                    episode_frames.append(frame)
                # Early exit if we've passed this episode
                elif len(episode_frames) > 0:
                    break
    except Exception as e:
        print(f"Error loading episode metadata: {e}")
        raise
    
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode_index} not found in dataset")
    
    # Sort by frame_index
    episode_frames.sort(key=lambda x: x.get("frame_index", 0))
    
    # Get available cameras
    cameras = [key for key in episode_frames[0].keys() if 'image' in key.lower()]
    if len(cameras) == 0:
        raise ValueError(f"No image observations found in episode {episode_index}")
    
    # Select camera
    if camera is None:
        preferred_names = ['observation.images.image', 'image', 'agentview_rgb', 'agentview_image']
        camera = None
        for name in preferred_names:
            if name in cameras:
                camera = name
                break
        if camera is None:
            camera = cameras[0]
    elif camera not in cameras:
        raise ValueError(f"Camera '{camera}' not found. Available cameras: {cameras}")
    
    # Extract images with progress
    print(f"Extracting {len(episode_frames)} frames from camera '{camera}'...")
    images = []
    for i, frame in enumerate(episode_frames):
        if (i + 1) % 10 == 0:
            print(f"  Loading frame {i + 1}/{len(episode_frames)}...")
        img = frame[camera]
        # Convert PIL Image to numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, str):
            img = np.array(Image.open(img))
        elif not isinstance(img, np.ndarray):
            # Try to convert to numpy array
            img = np.array(img)
        
        # Ensure images are uint8 and have correct shape (H, W, C)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                # Likely normalized to [0, 1]
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Handle different image formats
        if len(img.shape) == 2:
            # Grayscale, convert to RGB
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3:
            if img.shape[0] == 3 or img.shape[0] == 4:
                # (C, H, W), convert to (H, W, C)
                img = img.transpose(1, 2, 0)
            if img.shape[2] == 4:
                # Remove alpha channel if present
                img = img[:, :, :3]
            elif img.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        elif len(img.shape) == 4:
            # (1, H, W, C) or (1, C, H, W), squeeze first dimension
            if img.shape[0] == 1:
                img = img[0]
                if len(img.shape) == 3 and (img.shape[0] == 3 or img.shape[0] == 4):
                    img = img.transpose(1, 2, 0)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
        
        images.append(img)
    
    return np.array(images), episode_index, camera


def extract_images_from_lerobot(dataset_path: Path, episode_index: int, camera=None):
    """
    Extract images from a LeRobot dataset episode.
    
    Args:
        dataset_path: Path to the LeRobot dataset directory
        episode_index: Index of the episode to extract
        camera: Camera/view to extract (if None, tries to find observation.images.image or main camera)
    
    Returns:
        tuple: (images array, episode_index, camera_name)
    """
    # Find the parquet file for this episode
    chunk_dirs = sorted(dataset_path.glob("data/chunk-*"))
    parquet_path = None
    for chunk_dir in chunk_dirs:
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        if parquet_path.exists():
            break
    else:
        raise FileNotFoundError(f"Episode {episode_index} not found in dataset")
    
    # Read parquet file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Get available cameras
    cameras = [col for col in df.columns if 'image' in col.lower()]
    if len(cameras) == 0:
        raise ValueError(f"No image observations found in episode {episode_index}")
    
    # Select camera
    if camera is None:
        # Try common camera names
        preferred_names = ['observation.images.image', 'image', 'agentview_rgb', 'agentview_image']
        camera = None
        for name in preferred_names:
            if name in cameras:
                camera = name
                break
        if camera is None:
            camera = cameras[0]
    elif camera not in cameras:
        raise ValueError(f"Camera '{camera}' not found. Available cameras: {cameras}")
    
    # Extract images
    images = []
    for idx in range(len(df)):
        img_val = df[camera].iloc[idx]
        
        # Handle different image storage formats
        if isinstance(img_val, dict):
            # LeRobot stores images as dictionaries - check for embedded data first
            img = None
            
            # 1. Check for embedded numpy array or list
            if 'array' in img_val:
                img = np.array(img_val['array'])
            elif 'data' in img_val:
                img = np.array(img_val['data'])
            elif 'bytes' in img_val:
                # Raw image bytes
                img = np.array(Image.open(io.BytesIO(img_val['bytes'])))
            else:
                # Try to extract numpy array from any dict value
                for key, val in img_val.items():
                    if isinstance(val, (np.ndarray, list)):
                        img = np.array(val)
                        break
                    elif isinstance(val, (bytes, bytearray)):
                        try:
                            img = np.array(Image.open(io.BytesIO(val)))
                            break
                        except:
                            pass
            
            # 2. If no embedded data found, try loading from path
            if img is None and 'path' in img_val:
                img_path_str = img_val['path']
                chunk_dirs = sorted(dataset_path.glob("data/chunk-*"))
                # Try different possible path locations
                # 1. Relative to dataset root
                img_path = dataset_path / img_path_str
                # 2. Relative to chunk directory (where parquet file is)
                if not img_path.exists():
                    for chunk_dir in chunk_dirs:
                        test_path = chunk_dir / img_path_str
                        if test_path.exists():
                            img_path = test_path
                            break
                # 3. In an images subdirectory
                if not img_path.exists():
                    img_path = dataset_path / "images" / img_path_str
                # 4. In chunk/images subdirectory
                if not img_path.exists():
                    for chunk_dir in chunk_dirs:
                        test_path = chunk_dir / "images" / img_path_str
                        if test_path.exists():
                            img_path = test_path
                            break
                # 5. Absolute path
                if not img_path.exists():
                    img_path = Path(img_path_str)
                # 6. Relative to parent of dataset
                if not img_path.exists():
                    img_path = dataset_path.parent.parent / img_path_str
                # 7. Try just the filename in various locations
                if not img_path.exists():
                    filename = Path(img_path_str).name
                    for chunk_dir in chunk_dirs:
                        test_path = chunk_dir / filename
                        if test_path.exists():
                            img_path = test_path
                            break
                    if not img_path.exists():
                        test_path = dataset_path / "images" / filename
                        if test_path.exists():
                            img_path = test_path
                
                if img_path.exists():
                    img = np.array(Image.open(img_path))
                else:
                    # Path doesn't exist - images might be embedded, suggest using API
                    # Print debug info about the dict structure
                    print(f"\nDebug: Image dict structure for first frame:")
                    print(f"  Keys: {list(img_val.keys())}")
                    for k, v in img_val.items():
                        print(f"    {k}: {type(v).__name__}")
                        if hasattr(v, 'shape'):
                            print(f"      shape: {v.shape}, dtype: {v.dtype}")
                        elif isinstance(v, (bytes, bytearray)):
                            print(f"      bytes length: {len(v)}")
                        elif isinstance(v, str):
                            print(f"      str value: {v[:100] if len(v) > 100 else v}")
                    raise FileNotFoundError(
                        f"Image file not found: {img_path_str}\n"
                        f"Images appear to be referenced by path but files don't exist.\n"
                        f"This suggests images are embedded in parquet or stored elsewhere.\n"
                        f"Try using --use-api flag to use LeRobotDataset API which handles this automatically:\n"
                        f"  --use-api"
                    )
            
            if img is None:
                raise ValueError(
                    f"Could not extract image from dict with keys: {list(img_val.keys())}\n"
                    f"Expected keys: 'array', 'data', 'bytes', or 'path'"
                )
        elif isinstance(img_val, str):
            # Image path - load it
            img_path = dataset_path.parent.parent / img_val
            if img_path.exists():
                img = np.array(Image.open(img_path))
            else:
                # Try relative to dataset path
                img_path = dataset_path / img_val
                if img_path.exists():
                    img = np.array(Image.open(img_path))
                else:
                    raise FileNotFoundError(f"Image not found: {img_val}")
        elif isinstance(img_val, (list, np.ndarray)):
            # Image data directly
            img = np.array(img_val)
        else:
            raise ValueError(f"Unexpected image type: {type(img_val)}")
        
        # Ensure images are uint8 and have correct shape
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                # Likely normalized to [0, 1]
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Handle different image formats
        if len(img.shape) == 2:
            # Grayscale, convert to RGB
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3:
            if img.shape[0] == 3:
                # (C, H, W), convert to (H, W, C)
                img = img.transpose(1, 2, 0)
            elif img.shape[2] == 4:
                # Remove alpha channel if present
                img = img[:, :, :3]
            elif img.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        elif len(img.shape) == 4:
            # (1, H, W, C) or (1, C, H, W), squeeze first dimension
            if img.shape[0] == 1:
                img = img[0]
                if len(img.shape) == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        
        images.append(img)
    
    images = np.array(images)
    
    return images, episode_index, camera


def lerobot_to_video(dataset_path: Path, episode_index: int, output_dir: Path, camera=None, fps=30, filename=None, repo_id: str | None = None, use_api_flag: bool = False):
    """
    Convert LeRobot dataset episode to video.
    
    Args:
        dataset_path: Path to LeRobot dataset directory
        episode_index: Index of the episode to convert
        output_dir: Directory to save output video
        camera: Camera/view to use (if None, auto-detects)
        fps: Frames per second for output video
        filename: Output filename (if None, auto-generates from episode and camera names)
        repo_id: Repository ID for LeRobotDataset API (if None, tries to infer from dataset_path)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Use parquet mode by default (faster), API only if explicitly requested
    use_api = LEROBOT_AVAILABLE and repo_id and use_api_flag
    if use_api:
        try:
            print(f"Reading LeRobot dataset using API: {dataset_path}")
            images, actual_episode_index, actual_camera = extract_images_from_lerobot_api(
                repo_id, episode_index, camera, dataset_path
            )
        except Exception as e:
            print(f"Warning: LeRobotDataset API failed ({e}), falling back to direct parquet reading")
            images, actual_episode_index, actual_camera = extract_images_from_lerobot(
                dataset_path, episode_index, camera
            )
    else:
        # Extract images directly from parquet (faster for large datasets)
        print(f"Reading LeRobot dataset directly from parquet: {dataset_path}")
        images, actual_episode_index, actual_camera = extract_images_from_lerobot(
            dataset_path, episode_index, camera
        )
    print(f"Extracted {len(images)} frames from episode {actual_episode_index}, camera '{actual_camera}'")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if filename is None:
        filename = f"episode_{actual_episode_index:06d}_{actual_camera.replace('.', '_')}.mp4"
    elif not filename.endswith('.mp4'):
        filename = f"{filename}.mp4"
    
    output_path = output_dir / filename
    
    # Write video
    print(f"Writing video to: {output_path}")
    print(f"Video: {len(images)} frames, {fps} fps, shape: {images[0].shape}")
    
    with imageio.get_writer(output_path, fps=fps) as writer:
        for img in images:
            writer.append_data(img)
    
    print(f"Successfully created video: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset trajectories to video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert episode 0 with auto-detected camera
  python openpi/examples/libero/lerobot_to_video.py --repo-id your_hf_username/libero --episode-index 0 --output_dir videos/
  
  # Convert specific episode and camera
  python openpi/examples/libero/lerobot_to_video.py --repo-id your_hf_username/libero --episode-index 5 --output_dir videos/ --camera observation.images.image
  
  # Convert with custom fps and filename
  python openpi/examples/libero/lerobot_to_video.py --repo-id your_hf_username/libero --episode-index 0 --output_dir videos/ --fps 60 --filename my_video.mp4
        """
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repository ID (e.g., 'your_hf_username/libero')"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Index of the episode to convert (e.g., 0, 1, 2, ...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output video"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset directory (if None, uses HF_LEROBOT_HOME/repo_id)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera/view to use (e.g., 'observation.images.image', 'observation.images.image2'). If not specified, auto-detects."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video (default: 30)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output video filename (default: auto-generated from episode and camera names)"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras for the episode, then exit"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use LeRobotDataset API instead of direct parquet reading (slower but handles image paths automatically)"
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset_path is None:
        hf_home = os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
        dataset_path = Path(hf_home).expanduser() / args.repo_id
    else:
        dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at: {dataset_path}")
        print(f"\nHint: Use --dataset-path to specify the dataset location, e.g.:")
        print(f"  --dataset-path /scratch/am1125/.cache/huggingface/lerobot/your_hf_username/libero")
        print(f"\nOr set the HF_LEROBOT_HOME environment variable:")
        print(f"  export HF_LEROBOT_HOME=/scratch/am1125/.cache/huggingface/lerobot")
        return 1
    
    # List cameras mode
    if args.list_cameras:
        try:
            cameras, parquet_path = get_available_cameras(dataset_path, args.episode_index)
            print(f"\nAvailable cameras for episode {args.episode_index}:")
            for cam in cameras:
                print(f"  {cam}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
        return 0
    
    # Convert to video
    try:
        lerobot_to_video(
            dataset_path,
            args.episode_index,
            Path(args.output_dir),
            camera=args.camera,
            fps=args.fps,
            filename=args.filename,
            repo_id=args.repo_id,
            use_api_flag=args.use_api
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

