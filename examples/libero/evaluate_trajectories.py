"""
Evaluate pi-0.5 on LIBERO dataset by rolling out from trajectory initial states.

For each trajectory:
1. Load initial state from the trajectory
2. Roll out pi-0.5 from that state 10 times
3. Record success/failure for each rollout
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib

import h5py
import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import utils as libero_utils
import numpy as np
import pandas as pd
import tqdm
import tyro

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Task parameters
    task_suite_name: str = "libero_spatial"
    task_id: int = 0  # Which task to evaluate (set to None to evaluate all tasks)
    
    # Rollout parameters
    num_rollouts_per_trajectory: int = 10
    num_steps_wait: int = 10  # Steps to wait for physics to stabilize
    max_trajectories: int = None  # Limit number of trajectories (None = all, useful for testing)
    
    # Output
    output_dir: str = "data/libero/evaluation"
    save_videos: bool = False  # Set to True to save videos (slow)
    
    seed: int = 7


def load_policy():
    """Load the pi-0.5 LIBERO policy using direct inference (not server)."""
    logging.info("Loading pi-0.5 LIBERO policy...")
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logging.info("✓ Policy loaded")
    return policy


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_observation_dict(obs, task_description, resize_size=224):
    """Convert raw environment observation to policy input format."""
    # Import here to avoid making it a required dependency at module level
    from openpi_client import image_tools
    
    # Get and preprocess images (rotate 180 degrees to match training)
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    
    # Create observation dict in the format expected by the policy
    obs_dict = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(task_description),
    }
    
    return obs_dict


def rollout_from_initial_state(env, policy, initial_state, model_xml, task_description, args, max_steps):
    """
    Perform a single rollout from an initial state.
    Returns (success, num_steps, replay_images, trajectory_data).
    """
    # Reset environment to initial state
    env.reset()
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    
    # Re-compute observations
    env._post_process()
    env._update_observables(force=True)
    obs = env.env._get_observations()
    
    replay_images = []
    replan_steps = 5  # Replan every 5 steps
    
    # Store full trajectory data
    trajectory_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
        "sim_states": [],
    }
    
    for t in range(max_steps + args.num_steps_wait):
        # Wait for physics stabilization
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            continue
        
        # Get observation in policy format
        obs_dict = get_observation_dict(obs, task_description)
        
        # Save image for video
        if args.save_videos:
            replay_images.append(obs_dict["observation/image"])
        
        # Get action from policy
        if t % replan_steps == 0 or t == args.num_steps_wait:
            # Query policy for action chunk
            result = policy.infer(obs_dict)
            actions = result["actions"]
            action_idx = 0
        
        # Execute action
        action = actions[action_idx]
        
        # Save trajectory data BEFORE stepping
        # Environment already returns RGB format (matches LIBERO datasets), so save directly
        trajectory_data["observations"].append({
            "agentview_image": obs["agentview_image"].copy(),
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"].copy(),
            "robot0_eef_pos": obs["robot0_eef_pos"],
            "robot0_eef_quat": obs["robot0_eef_quat"],
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
        })
        trajectory_data["actions"].append(action.copy())
        trajectory_data["sim_states"].append(env.sim.get_state())
        
        obs, reward, done, info = env.step(action.tolist())
        
        trajectory_data["rewards"].append(reward)
        trajectory_data["dones"].append(done)
        trajectory_data["infos"].append(info)
        
        action_idx += 1
        if action_idx >= len(actions):
            action_idx = 0  # Will replan on next iteration
        
        if done:
            return True, t, replay_images, trajectory_data
    
    return False, max_steps, replay_images, trajectory_data


def load_trajectory_data(dataset_path):
    """
    Load all trajectories from a dataset file.
    Returns list of (demo_name, initial_state, model_xml) tuples.
    """
    with h5py.File(dataset_path, "r") as f:
        demos = sorted(list(f["data"].keys()))
        
        trajectories = []
        for demo_name in demos:
            model_xml = f[f"data/{demo_name}"].attrs["model_file"]
            # Postprocess XML to fix hardcoded paths
            model_xml = fix_model_xml_paths(model_xml)
            states = f[f"data/{demo_name}/states"][()]
            initial_state = states[0]
            trajectories.append((demo_name, initial_state, model_xml))
    
    return trajectories


def fix_model_xml_paths(xml_str):
    """
    Fix hardcoded paths in model XML to point to local installation.
    Handles both robosuite and LIBERO asset paths.
    """
    import xml.etree.ElementTree as ET
    import robosuite
    import os
    
    # Get correct paths
    robosuite_path = os.path.split(robosuite.__file__)[0]
    libero_path = pathlib.Path(__file__).parent.parent.parent / "third_party" / "libero" / "libero" / "libero"
    
    # Parse XML
    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is None:
        return xml_str
    
    # Fix all mesh and texture paths
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    
    for elem in meshes + textures:
        old_path = elem.get("file")
        if old_path is None:
            continue
        
        old_path_split = old_path.split("/")
        
        # Fix robosuite paths
        if "robosuite" in old_path_split:
            ind = max(loc for loc, val in enumerate(old_path_split) if val == "robosuite")
            new_path_split = robosuite_path.split("/") + old_path_split[ind + 1:]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        
        # Fix LIBERO paths (look for assets folder)
        elif "assets" in old_path_split:
            ind = max(loc for loc, val in enumerate(old_path_split) if val == "assets")
            # Everything after "assets" in the old path
            relative_path = "/".join(old_path_split[ind:])
            new_path = str(libero_path / relative_path)
            elem.set("file", new_path)
    
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def evaluate_task(task, policy, args, max_steps):
    """Evaluate policy on all trajectories for a single task."""
    # Get dataset path
    # dataset_path = pathlib.Path(get_libero_path("datasets")) / "libero" / task.problem_folder / f"{task.name}_demo.hdf5"
    dataset_path = pathlib.Path(get_libero_path("datasets")) / "libero_processed" / f"{args.task_suite_name}_openvla_processed" / f"{task.name}_demo.hdf5"
    
    if not dataset_path.exists():
        logging.warning(f"Dataset not found: {dataset_path}")
        return []
    
    logging.info(f"Dataset: {dataset_path}")
    
    # Load trajectories
    trajectories = load_trajectory_data(dataset_path)
    logging.info(f"Found {len(trajectories)} trajectories")
    
    # Limit trajectories if requested (for testing)
    if args.max_trajectories is not None:
        trajectories = trajectories[:args.max_trajectories]
        logging.info(f"Limiting to {len(trajectories)} trajectories for testing")
    
    # Initialize environment
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": LIBERO_ENV_RESOLUTION,
        "camera_widths": LIBERO_ENV_RESOLUTION
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)
    
    # Evaluate each trajectory
    results = []
    trajectories_dir = pathlib.Path(args.output_dir) / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    # Create single HDF5 file for all rollouts of this task (matching LIBERO format)
    task_hdf5_file = trajectories_dir / f"{task.name}_rollouts.hdf5"
    
    with h5py.File(task_hdf5_file, 'w') as task_f:
        # Create data group (matching LIBERO structure exactly)
        data_group = task_f.create_group('data')
        
        # Add dataset-level metadata (matching LIBERO format exactly)
        data_group.attrs['bddl_file_name'] = str(task_bddl_file)
        data_group.attrs['env_args'] = json.dumps({
            "type": 1,
            "env_name": "Libero_Tabletop_Manipulation",
            "problem_name": "libero_tabletop_manipulation",
            "bddl_file": str(task_bddl_file),
            "env_kwargs": {
                "robots": ["Panda"],
                "controller_configs": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping_ratio": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_ratio_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2
                },
                "bddl_file_name": str(task_bddl_file),
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "ignore_done": True,
                "use_camera_obs": True,
                "camera_depths": False,
                "camera_names": ["robot0_eye_in_hand", "agentview"],
                "reward_shaping": True,
                "control_freq": 20,
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
                "camera_segmentations": None
            }
        })
        data_group.attrs['env_name'] = 'Libero_Tabletop_Manipulation'
        data_group.attrs['macros_image_convention'] = 'opengl'
        data_group.attrs['problem_info'] = json.dumps({
            "problem_name": "libero_tabletop_manipulation",
            "domain_name": "robosuite",
            "language_instruction": task_description
        })
        data_group.attrs['tag'] = 'libero-v1'
        
        demo_counter = 0
        
        for traj_idx, (demo_name, initial_state, model_xml) in enumerate(tqdm.tqdm(
            trajectories, desc="Trajectories", leave=False
        )):
            for rollout_idx in range(args.num_rollouts_per_trajectory):
                try:
                    success, num_steps, replay_images, trajectory_data = rollout_from_initial_state(
                        env, policy, initial_state, model_xml,
                        task_description, args, max_steps
                    )
                    
                    results.append({
                        "task_name": task.name,
                        "task_description": task_description,
                        "trajectory_name": demo_name,
                        "trajectory_idx": traj_idx,
                        "rollout_idx": rollout_idx,
                        "success": success,
                        "num_steps": num_steps,
                    })
                    
                    # Add this rollout as a demo in the single HDF5 file (matching LIBERO format exactly)
                    demo_name_libero = f"demo_{demo_counter}"
                    demo_group = data_group.create_group(demo_name_libero)
                    
                    # Create obs group (matching LIBERO structure exactly)
                    obs_group = demo_group.create_group('obs')
                    
                    # Convert observations to LIBERO format
                    obs_arrays = {}
                    for key in trajectory_data["observations"][0].keys():
                        obs_arrays[key] = np.array([o[key] for o in trajectory_data["observations"]])
                    
                    # Save observations with EXACT LIBERO naming and format
                    obs_group.create_dataset('agentview_rgb', data=obs_arrays["agentview_image"])  # LIBERO uses 'agentview_rgb'
                    obs_group.create_dataset('eye_in_hand_rgb', data=obs_arrays["robot0_eye_in_hand_image"])  # LIBERO uses 'eye_in_hand_rgb'
                    
                    # Robot state information (matching LIBERO format exactly)
                    obs_group.create_dataset('ee_pos', data=obs_arrays["robot0_eef_pos"])  # End-effector position
                    
                    # Convert quaternion to axis-angle (LIBERO uses axis-angle for ee_ori)
                    ee_ori_axis_angle = np.array([_quat2axisangle(quat) for quat in obs_arrays["robot0_eef_quat"]])
                    obs_group.create_dataset('ee_ori', data=ee_ori_axis_angle)  # End-effector orientation (axis-angle)
                    
                    # ee_states combines position + orientation (axis-angle)
                    obs_group.create_dataset('ee_states', data=np.concatenate([
                        obs_arrays["robot0_eef_pos"], 
                        ee_ori_axis_angle
                    ], axis=1))  # Combined position + orientation
                    obs_group.create_dataset('gripper_states', data=obs_arrays["robot0_gripper_qpos"])  # Gripper finger positions
                    
                    # Joint states (extract from stored sim states)
                    joint_states = np.array([state.qpos[:7] for state in trajectory_data["sim_states"]])
                    obs_group.create_dataset('joint_states', data=joint_states)
                    
                    # Save main trajectory data (matching LIBERO format exactly)
                    demo_group.create_dataset('actions', data=np.array(trajectory_data["actions"]))
                    demo_group.create_dataset('rewards', data=np.array(trajectory_data["rewards"]).astype(np.uint8))  # LIBERO uses uint8
                    demo_group.create_dataset('dones', data=np.array(trajectory_data["dones"]).astype(np.uint8))  # LIBERO uses uint8
                    
                    # Save full MuJoCo states (matching LIBERO format exactly)
                    states_array = np.array([state.flatten() for state in trajectory_data["sim_states"]])
                    demo_group.create_dataset('states', data=states_array)
                    
                    # Robot states (matching LIBERO format exactly)
                    # LIBERO robot_states: [gripper_left, gripper_right, ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
                    robot_states = np.array([np.concatenate([
                        obs_arrays["robot0_gripper_qpos"][i],  # 2 dims: [left, right]
                        obs_arrays["robot0_eef_pos"][i],      # 3 dims: [x, y, z]
                        obs_arrays["robot0_eef_quat"][i]      # 4 dims: [w, x, y, z] (quaternion)
                    ]) for i in range(len(trajectory_data["observations"]))])
                    demo_group.create_dataset('robot_states', data=robot_states)
                    
                    # Demo attributes (matching LIBERO format exactly)
                    demo_group.attrs['init_state'] = initial_state
                    demo_group.attrs['model_file'] = model_xml
                    demo_group.attrs['num_samples'] = len(trajectory_data["observations"])
                    
                    demo_counter += 1
                    
                    # Save video if requested
                    if args.save_videos and replay_images:
                        video_dir = pathlib.Path(args.output_dir) / "videos" / task.name
                        video_dir.mkdir(parents=True, exist_ok=True)
                        suffix = "success" if success else "failure"
                        video_path = video_dir / f"{demo_name}_rollout{rollout_idx}_{suffix}.mp4"
                        imageio.mimwrite(video_path, replay_images, fps=10)
                    
                except Exception as e:
                    logging.error(f"Error in rollout: {e}")
                    results.append({
                        "task_name": task.name,
                        "task_description": task_description,
                        "trajectory_name": demo_name,
                        "trajectory_idx": traj_idx,
                        "rollout_idx": rollout_idx,
                        "success": False,
                        "num_steps": 0,
                        "error": str(e),
                    })
        
        # Finalize dataset metadata
        data_group.attrs['num_demos'] = demo_counter
        data_group.attrs['total'] = sum(demo_group.attrs['num_samples'] for demo_group in data_group.values())
    
    logging.info(f"Saved {demo_counter} rollouts to {task_hdf5_file}")
    
    env.close()
    return results


def main(args: Args):
    # Setup
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    policy = load_policy()
    
    # Initialize task suite
    logging.info(f"Initializing task suite: {args.task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Determine which tasks to evaluate
    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = range(task_suite.n_tasks)
    
    # Determine max steps based on suite
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(args.task_suite_name, 400)
    
    # Evaluate each task
    all_results = []
    for task_id in tqdm.tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(task_id)
        logging.info(f"\nTask {task_id}: {task.language}")
        
        results = evaluate_task(task, policy, args, max_steps)
        all_results.extend(results)
        
        # Save intermediate results
        if all_results:
            df = pd.DataFrame(all_results)
            csv_path = output_dir / f"{args.task_suite_name}_{task.name}_results.csv"
            df.to_csv(csv_path, index=False)
            
            # Print progress
            successes = df["success"].sum()
            total = len(df)
            logging.info(f"Progress: {successes}/{total} = {100.0 * successes / total:.1f}% success")
        else:
            logging.warning(f"No results collected for task {task_id}")
    
    # Final results
    if not all_results:
        print("\n" + "="*80)
        print("ERROR: No results collected!")
        print("="*80)
        print("Please check:")
        print("  1. Run setup_libero_paths.py to configure LIBERO paths")
        print("  2. Ensure datasets exist at the configured location")
        return
    
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"{args.task_suite_name}_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    total = len(df)
    successes = df["success"].sum()
    print(f"\nOverall: {successes}/{total} = {100.0 * successes / total:.2f}%")
    
    # Per-task summary
    print("\nPer-task results:")
    task_stats = df.groupby(["task_name", "task_description"])["success"].agg(["sum", "count", "mean"])
    task_stats.columns = ["successes", "total", "success_rate"]
    print(task_stats)
    
    print(f"\nDetailed results saved to: {csv_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    tyro.cli(main)
