"""
Script for converting Libero dataset from local HDF5 format to LeRobot format.

The Libero dataset is stored in HDF5 format and contains:
- data/demo_X/obs/agentview_rgb (256x256x3) - main camera image
- data/demo_X/obs/eye_in_hand_rgb (256x256x3) - wrist camera image  
- data/demo_X/states (N, D) - robot states (shape detected dynamically)
- data/demo_X/actions (N, 7) - robot actions (shape detected dynamically)

The output follows LeRobot format with nested feature names:
- observation.images.image - main camera
- observation.images.image2 - wrist camera
- observation.state - robot state
- action - robot actions
- timestamp, frame_index, episode_index, index, task_index - metadata

Usage:
python openpi/examples/libero/convert_hf_libero_to_lerobot.py --hdf5_path /path/to/dataset.hdf5 --repo_id your_hf_username/libero

To save to a specific local directory:
python openpi/examples/libero/convert_hf_libero_to_lerobot.py --hdf5_path /path/to/dataset.hdf5 --repo_id your_hf_username/libero --output-dir /path/to/output

If you want to push your dataset to the Hugging Face Hub:
python openpi/examples/libero/convert_hf_libero_to_lerobot.py --hdf5_path /path/to/dataset.hdf5 --repo_id your_hf_username/libero --push_to_hub

By default, the dataset is saved to $HF_LEROBOT_HOME, or use --output-dir to specify a custom location.
"""

import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro

# Import HF_LEROBOT_HOME function to get it dynamically
def get_hf_lerobot_home():
    """Get HF_LEROBOT_HOME, reading from environment variable if set."""
    # Force re-import to get updated value after env var change
    import importlib
    import lerobot.common.datasets.lerobot_dataset as lerobot_module
    importlib.reload(lerobot_module)
    return lerobot_module.HF_LEROBOT_HOME


def main(
    hdf5_path: str,
    repo_id: str = "your_hf_username/libero",
    *,
    output_dir: str | None = None,
    push_to_hub: bool = False,
):
    """
    Convert Libero dataset from local HDF5 format to LeRobot format.
    
    Args:
        hdf5_path: Path to the HDF5 file or directory containing HDF5 files
        repo_id: The LeRobot repo ID for the output dataset
        output_dir: Local directory path to save the dataset (if None, uses HF_LEROBOT_HOME)
        push_to_hub: Whether to push the converted dataset to HuggingFace Hub
    """
    # Determine output directory
    # Note: LeRobotDataset.create() uses the HF_LEROBOT_HOME environment variable
    # internally to determine where to save. It doesn't accept a custom path parameter.
    # So we temporarily set the environment variable if a custom output_dir is specified.
    original_hf_home = None
    if output_dir is not None:
        output_dir_path = Path(output_dir).resolve()
        # Create the output directory if it doesn't exist
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / repo_id
        # Save original value to restore later
        original_hf_home = os.environ.get("HF_LEROBOT_HOME")
        # Set it to our custom directory so LeRobotDataset saves there
        os.environ["HF_LEROBOT_HOME"] = str(output_dir_path)
        print(f"Using custom output directory: {output_dir_path}")
        print(f"Dataset will be saved to: {output_path}")
    else:
        # Get HF_LEROBOT_HOME dynamically (after potential env var changes)
        hf_home = get_hf_lerobot_home()
        output_path = hf_home / repo_id
        print(f"Using default HF_LEROBOT_HOME: {hf_home}")
        print(f"Dataset will be saved to: {output_path}")
    
    # Remove existing dataset if it exists (LeRobotDataset.create() doesn't like existing dirs)
    # Also check the default location in case a previous run left it there
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Also check and clean up any existing dataset at the location LeRobotDataset will use
    # (in case environment variable wasn't set properly in a previous run)
    # We need to check this right before create() to ensure we have the correct path

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 path not found: {hdf5_path}")
    
    # Collect all HDF5 files
    if hdf5_path.is_file() and hdf5_path.suffix in ['.hdf5', '.h5']:
        hdf5_files = [hdf5_path]
    elif hdf5_path.is_dir():
        hdf5_files = list(hdf5_path.glob("*.hdf5")) + list(hdf5_path.glob("*.h5"))
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in directory: {hdf5_path}")
    else:
        raise ValueError(f"Invalid HDF5 path: {hdf5_path}")
    
    print(f"Found {len(hdf5_files)} HDF5 file(s)")
    
    # Scan all files to find maximum state and action dimensions
    # (different files may have different state dimensions)
    print("Scanning all HDF5 files to detect maximum state and action dimensions...")
    max_state_dim = 0
    max_action_dim = 0
    state_shape = None
    action_shape = None
    
    for hdf5_file in tqdm(hdf5_files, desc="Scanning files"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if 'data' not in f:
                    continue
                demos = sorted(list(f['data'].keys()))
                if len(demos) == 0:
                    continue
                
                # Check first demo in this file
                demo_group = f[f'data/{demos[0]}']
                
                # Check actions
                if 'actions' in demo_group:
                    action_dim = demo_group['actions'].shape[1] if len(demo_group['actions'].shape) > 1 else demo_group['actions'].shape[0]
                    if action_dim > max_action_dim:
                        max_action_dim = action_dim
                        action_shape = demo_group['actions'].shape[1:] if len(demo_group['actions'].shape) > 1 else demo_group['actions'].shape
                
                # Check states
                if 'states' in demo_group:
                    state_dim = demo_group['states'].shape[1] if len(demo_group['states'].shape) > 1 else demo_group['states'].shape[0]
                    if state_dim > max_state_dim:
                        max_state_dim = state_dim
                        state_shape = demo_group['states'].shape[1:] if len(demo_group['states'].shape) > 1 else demo_group['states'].shape
                elif 'robot_states' in demo_group:
                    state_dim = demo_group['robot_states'].shape[1] if len(demo_group['robot_states'].shape) > 1 else demo_group['robot_states'].shape[0]
                    if state_dim > max_state_dim:
                        max_state_dim = state_dim
                        state_shape = demo_group['robot_states'].shape[1:] if len(demo_group['robot_states'].shape) > 1 else demo_group['robot_states'].shape
        except Exception as e:
            print(f"Warning: Error scanning {hdf5_file}: {e}")
            continue
    
    if state_shape is None:
        raise ValueError("Could not find 'states' or 'robot_states' in any HDF5 file")
    if action_shape is None:
        raise ValueError("Could not find 'actions' in any HDF5 file")
    
    print(f"\nDetected maximum dimensions:")
    print(f"  State shape: {state_shape} (max dimension: {max_state_dim})")
    print(f"  Action shape: {action_shape} (max dimension: {max_action_dim})")
    print(f"\nNote: States/actions with smaller dimensions will be padded with zeros.\n")

    # Double-check and clean up the directory right before create() to ensure it's empty
    # LeRobotDataset.create() may read HF_LEROBOT_HOME at import time, so we need to clean up
    # both the custom location (if set) and the default location to be safe
    if output_dir is not None:
        # Clean up custom location
        custom_output_path = Path(output_dir).resolve() / repo_id
        if custom_output_path.exists():
            print(f"Removing existing dataset at custom location: {custom_output_path}")
            shutil.rmtree(custom_output_path)
    
    # Also clean up default location (in case LeRobotDataset uses cached import-time value)
    default_hf_home = get_hf_lerobot_home()
    default_output_path = default_hf_home / repo_id
    if default_output_path.exists():
        print(f"Removing existing dataset at default location: {default_output_path}")
        shutil.rmtree(default_output_path)
    
    # Create LeRobot dataset, define features to store
    # Based on the Libero HDF5 dataset structure and LeRobot format:
    # - obs/agentview_rgb -> observation.images.image (main camera)
    # - obs/eye_in_hand_rgb -> observation.images.image2 (wrist camera)
    # - states -> observation.state (dynamic shape detected from file)
    # - actions -> action (dynamic shape detected from file)
    # - Metadata: timestamp, frame_index, episode_index, index, task_index
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,  # From the dataset metadata
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["actions"],
            },
        },
    )

    # Convert HDF5 files to LeRobot format
    print("Converting HDF5 files to LeRobot format...")
    total_episodes = 0
    total_frames = 0
    
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        with h5py.File(hdf5_file, 'r') as f:
            data_group = f['data']
            demos = sorted(list(data_group.keys()))
            
            # Try to get task description from dataset attributes
            task_description = ""
            if 'problem_info' in data_group.attrs:
                try:
                    problem_info = json.loads(data_group.attrs['problem_info'])
                    # Extract language instruction if available
                    if isinstance(problem_info, dict) and 'language_instruction' in problem_info:
                        lang_inst = problem_info['language_instruction']
                        if isinstance(lang_inst, list):
                            task_description = " ".join(lang_inst)
                        elif isinstance(lang_inst, str):
                            task_description = lang_inst
                except:
                    pass
            
            # If no task description found, use the HDF5 filename or a default
            if not task_description:
                task_description = Path(hdf5_file).stem or "libero_task"
            
            # Debug: print file info for first few files
            if len([f for f in hdf5_files if f == hdf5_file][:1]) or total_episodes < 3:
                print(f"\nProcessing HDF5 file: {hdf5_file.name}")
                print(f"  Task: {task_description}")
                print(f"  Number of demos: {len(demos)}")
                if len(demos) <= 5:
                    print(f"  Demos: {demos}")
                else:
                    print(f"  Demos: {demos[:3]} ... {demos[-2:]}")
            
            for episode_idx, demo_name in enumerate(demos):
                demo_group = data_group[demo_name]
                
                # Get observations
                obs_group = demo_group['obs']
                
                # Get images - try common key names
                if 'agentview_rgb' in obs_group:
                    images = obs_group['agentview_rgb'][:]
                elif 'agentview_image' in obs_group:
                    images = obs_group['agentview_image'][:]
                else:
                    raise ValueError(f"Could not find main camera image in demo {demo_name}. Available keys: {list(obs_group.keys())}")
                
                if 'eye_in_hand_rgb' in obs_group:
                    wrist_images = obs_group['eye_in_hand_rgb'][:]
                elif 'robot0_eye_in_hand_image' in obs_group:
                    wrist_images = obs_group['robot0_eye_in_hand_image'][:]
                else:
                    raise ValueError(f"Could not find wrist camera image in demo {demo_name}. Available keys: {list(obs_group.keys())}")
                
                # Convert images to uint8 in bulk (much faster than per-frame)
                # Check dtype first to avoid expensive max() call if already uint8
                if images.dtype != np.uint8:
                    # Sample first frame to determine if scaling is needed (faster than checking all frames)
                    sample_max = float(images[0].max())
                    if sample_max <= 1.0:
                        images = (images * 255).astype(np.uint8)
                    else:
                        images = images.astype(np.uint8)
                
                if wrist_images.dtype != np.uint8:
                    sample_max = float(wrist_images[0].max())
                    if sample_max <= 1.0:
                        wrist_images = (wrist_images * 255).astype(np.uint8)
                    else:
                        wrist_images = wrist_images.astype(np.uint8)
                
                # Debug: verify we're reading different data (print first demo of each file)
                file_idx = hdf5_files.index(hdf5_file)
                if file_idx < 3 or (file_idx < 10 and episode_idx == 0):
                    print(f"  Demo {demo_name}: {len(images)} frames, first_image_mean={images[0].mean():.3f}, last_image_mean={images[-1].mean():.3f}, first_image_hash={hash(images[0].tobytes())}")
                
                # Get states and actions
                # Prefer 'states' over 'robot_states' if both exist
                if 'states' in demo_group:
                    states = demo_group['states'][:]
                elif 'robot_states' in demo_group:
                    states = demo_group['robot_states'][:]
                else:
                    raise ValueError(f"Could not find 'states' or 'robot_states' in demo {demo_name}")
                
                actions = demo_group['actions'][:]
                
                # Convert to float32 in bulk if needed
                if states.dtype != np.float32:
                    states = states.astype(np.float32)
                if actions.dtype != np.float32:
                    actions = actions.astype(np.float32)
                
                # Pre-compute padding needs (only once per episode)
                expected_state_dim = state_shape[0] if isinstance(state_shape, tuple) else state_shape
                expected_action_dim = action_shape[0] if isinstance(action_shape, tuple) else action_shape
                needs_state_padding = len(states[0]) < expected_state_dim
                needs_action_padding = len(actions[0]) < expected_action_dim
                needs_state_truncation = len(states[0]) > expected_state_dim
                needs_action_truncation = len(actions[0]) > expected_action_dim
                
                # Verify shapes match
                num_frames = len(actions)
                if len(images) != num_frames:
                    raise ValueError(f"Image count ({len(images)}) doesn't match action count ({num_frames}) in {demo_name}")
                if len(wrist_images) != num_frames:
                    raise ValueError(f"Wrist image count ({len(wrist_images)}) doesn't match action count ({num_frames}) in {demo_name}")
                if len(states) != num_frames:
                    raise ValueError(f"State count ({len(states)}) doesn't match action count ({num_frames}) in {demo_name}")
                
                # Add each frame to the dataset
                for frame_idx in range(num_frames):
                    image = images[frame_idx]
                    wrist_image = wrist_images[frame_idx]
                    state = states[frame_idx]
                    action = actions[frame_idx]
                    
                    # Handle padding/truncation only if needed (pre-computed above)
                    if needs_state_padding:
                        padding = np.zeros(expected_state_dim - len(state), dtype=np.float32)
                        state = np.concatenate([state, padding])
                    elif needs_state_truncation:
                        state = state[:expected_state_dim]
                    
                    if needs_action_padding:
                        padding = np.zeros(expected_action_dim - len(action), dtype=np.float32)
                        action = np.concatenate([action, padding])
                    elif needs_action_truncation:
                        action = action[:expected_action_dim]
                    
                    # Add frame to dataset with LeRobot format naming
                    # Note: LeRobotDataset automatically handles metadata like episode_index, frame_index, etc.
                    # Task is added to each frame (LeRobotDataset handles it automatically, no need to define in features)
                    lerobot_dataset.add_frame(
                        {
                            "observation.images.image": image,
                            "observation.images.image2": wrist_image,
                            "observation.state": state,
                            "action": action,
                            "task": task_description,
                        }
                    )
                
                # Save the episode
                lerobot_dataset.save_episode()
                total_episodes += 1
                total_frames += num_frames

    print(f"Conversion complete! Dataset saved to {output_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    
    # Restore original HF_LEROBOT_HOME if we changed it
    if output_dir is not None:
        if original_hf_home is not None:
            os.environ["HF_LEROBOT_HOME"] = original_hf_home
        elif "HF_LEROBOT_HOME" in os.environ:
            del os.environ["HF_LEROBOT_HOME"]

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("Pushing dataset to HuggingFace Hub...")
        lerobot_dataset.push_to_hub(
            tags=["libero", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed to HuggingFace Hub!")


if __name__ == "__main__":
    tyro.cli(main)

