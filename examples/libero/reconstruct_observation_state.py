#!/usr/bin/env python3
"""
Reconstruct the 8-dimensional observation_state from old rollouts that don't have it.

This script reads HDF5 rollout files and adds the observation_state field by
reconstructing it from the saved raw observations:
- obs/ee_pos (3 dims)
- obs/ee_ori (3 dims - already axis-angle)
- obs/gripper_states (2 dims)

Total: 3 + 3 + 2 = 8 dimensions
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import tqdm


def reconstruct_observation_state(hdf5_path, in_place=False):
    """
    Reconstruct observation_state from ee_pos, ee_ori, and gripper_states.
    
    Args:
        hdf5_path: Path to HDF5 file
        in_place: If True, modify the file in place. If False, create a new file with '_reconstructed' suffix.
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        return
    
    if in_place:
        output_path = hdf5_path
        mode = 'r+'
    else:
        output_path = hdf5_path.parent / f"{hdf5_path.stem}_reconstructed{hdf5_path.suffix}"
        mode = 'r'
    
    print(f"\nProcessing: {hdf5_path.name}")
    print(f"Output: {output_path.name}")
    
    with h5py.File(hdf5_path, mode) as f:
        if 'data' not in f:
            print("Error: No 'data' group found in HDF5 file")
            return
        
        data_group = f['data']
        demos = sorted(list(data_group.keys()))
        print(f"Found {len(demos)} demos")
        
        reconstructed_count = 0
        skipped_count = 0
        
        for demo_name in tqdm.tqdm(demos, desc="Processing demos"):
            demo_group = data_group[demo_name]
            
            # Check if observation_state already exists
            if 'obs' in demo_group and 'observation_state' in demo_group['obs']:
                print(f"  {demo_name}: observation_state already exists, skipping")
                skipped_count += 1
                continue
            
            # Check if required fields exist
            obs_group = demo_group.get('obs')
            if obs_group is None:
                print(f"  {demo_name}: No 'obs' group found, skipping")
                skipped_count += 1
                continue
            
            required_fields = ['ee_pos', 'ee_ori', 'gripper_states']
            missing_fields = [f for f in required_fields if f not in obs_group]
            if missing_fields:
                print(f"  {demo_name}: Missing required fields: {missing_fields}, skipping")
                skipped_count += 1
                continue
            
            # Reconstruct observation_state
            ee_pos = obs_group['ee_pos'][:]  # (T, 3)
            ee_ori = obs_group['ee_ori'][:]  # (T, 3) - already axis-angle
            gripper_states = obs_group['gripper_states'][:]  # (T, 2)
            
            # Verify shapes
            num_steps = len(ee_pos)
            if len(ee_ori) != num_steps or len(gripper_states) != num_steps:
                print(f"  {demo_name}: Shape mismatch, skipping")
                skipped_count += 1
                continue
            
            # Concatenate to form 8-dim state: [ee_pos(3), ee_ori(3), gripper(2)]
            observation_state = np.concatenate([
                ee_pos,      # (T, 3)
                ee_ori,      # (T, 3)
                gripper_states  # (T, 2)
            ], axis=1)  # Result: (T, 8)
            
            # Verify it's 8 dimensions
            assert observation_state.shape[1] == 8, f"Expected 8 dims, got {observation_state.shape[1]}"
            
            # Save to HDF5
            if in_place:
                # Modify in place
                obs_group.create_dataset('observation_state', data=observation_state)
            else:
                # Need to copy to new file
                # This is more complex, so for now we'll just modify in place
                # or create a new file using copy
                pass
            
            reconstructed_count += 1
        
        if not in_place:
            # Create new file with reconstructed data
            print(f"\nCreating new file with reconstructed observation_state...")
            with h5py.File(output_path, 'w') as out_f:
                # Copy all data
                def copy_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        out_f[name] = obj[:]
                    elif isinstance(obj, h5py.Group):
                        if name not in out_f:
                            out_f.create_group(name)
                        for key, value in obj.attrs.items():
                            out_f[name].attrs[key] = value
                
                f.visititems(copy_item)
                
                # Now add observation_state to each demo
                for demo_name in demos:
                    demo_group = f[f'data/{demo_name}']
                    obs_group = demo_group.get('obs')
                    
                    if obs_group is None:
                        continue
                    
                    if 'observation_state' in obs_group:
                        continue  # Already exists
                    
                    required_fields = ['ee_pos', 'ee_ori', 'gripper_states']
                    if not all(f in obs_group for f in required_fields):
                        continue
                    
                    ee_pos = obs_group['ee_pos'][:]
                    ee_ori = obs_group['ee_ori'][:]
                    gripper_states = obs_group['gripper_states'][:]
                    
                    observation_state = np.concatenate([
                        ee_pos,
                        ee_ori,
                        gripper_states
                    ], axis=1)
                    
                    out_f[f'data/{demo_name}/obs/observation_state'] = observation_state
        
        print(f"\nSummary:")
        print(f"  Reconstructed: {reconstructed_count} demos")
        print(f"  Skipped: {skipped_count} demos")
        if not in_place:
            print(f"  New file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 8-dimensional observation_state from old rollouts"
    )
    parser.add_argument(
        "hdf5_path",
        type=str,
        help="Path to HDF5 rollout file"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the file in place (default: create new file with '_reconstructed' suffix)"
    )
    args = parser.parse_args()
    
    reconstruct_observation_state(args.hdf5_path, in_place=args.in_place)


if __name__ == "__main__":
    main()

