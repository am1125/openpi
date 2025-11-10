# Evaluating π-0.5 on LIBERO Trajectories

This script evaluates the π-0.5 policy on LIBERO dataset trajectories by rolling out from initial states.

## Setup (One-time)

Before running the evaluation, configure LIBERO paths:

```bash
cd /n/fs/vla-distill/openpi
python examples/libero/setup_libero_paths.py
```

This creates `~/.libero/config.yaml` pointing to the correct dataset locations.

## What it does

For each trajectory in the dataset:
1. Loads the initial state
2. Rolls out π-0.5 from that state **10 times**
3. Records success/failure for each rollout
4. Saves results to CSV

## Usage

### Quick Test (1 trajectory, 1 rollout)
**Recommended first!** Test everything works before running the full evaluation:
```bash
cd /n/fs/vla-distill/openpi
python examples/libero/evaluate_trajectories.py \
    --max-trajectories 1 \
    --num-rollouts-per-trajectory 1
```
This takes ~2-3 minutes and verifies the setup works.

### Basic (evaluate task 0 of libero_spatial)
```bash
cd /n/fs/vla-distill/openpi
python examples/libero/evaluate_trajectories.py
```
Full evaluation: ~50 trajectories × 10 rollouts = 500 rollouts (~1-1.5 hours)

### Evaluate a specific task
```bash
uv run python examples/libero/evaluate_trajectories.py --task-id 3
```

### Evaluate all tasks in a suite
```bash
uv run python examples/libero/evaluate_trajectories.py --task-id None
```

### Different task suite
```bash
uv run python examples/libero/evaluate_trajectories.py \
    --task-suite-name libero_10 \
    --task-id 0
```

### Save videos (slow!)
```bash
uv run python examples/libero/evaluate_trajectories.py \
    --save-videos True
```

### Test with a few trajectories (faster testing)
```bash
python examples/libero/evaluate_trajectories.py \
    --max-trajectories 5 \
    --num-rollouts-per-trajectory 2
```
5 trajectories × 2 rollouts = 10 rollouts (~10 minutes)

### Custom number of rollouts per trajectory
```bash
python examples/libero/evaluate_trajectories.py \
    --num-rollouts-per-trajectory 20
```

## Output

### 1. Results CSV
`data/libero/evaluation/{suite}_results.csv` with columns:
- `task_name`: Name of the task
- `task_description`: Natural language description
- `trajectory_name`: Name of the demo (e.g., "demo_0")
- `trajectory_idx`: Index of the trajectory
- `rollout_idx`: Which rollout (0-9 for 10 rollouts)
- `success`: Boolean - did it succeed?
- `num_steps`: How many steps it took

### 2. Full Trajectory Data (for training)
`data/libero/evaluation/trajectories/{task_name}/{demo_name}_rollout{i}.npz`

Each `.npz` file contains:
- `actions`: (T, 7) - actions taken
- `agentview_images`: (T, 256, 256, 3) - third-person camera
- `wrist_images`: (T, 256, 256, 3) - wrist camera
- `eef_pos`: (T, 3) - end-effector position
- `eef_quat`: (T, 4) - end-effector orientation
- `gripper_qpos`: (T, 2) - gripper state
- `rewards`: (T,) - rewards at each step
- `dones`: (T,) - done flags
- `task_description`: str - task description
- `success`: bool - final success
- `initial_state`: array - MuJoCo initial state

**Example: Load trajectory for training**
```python
import numpy as np

# Load a trajectory
data = np.load("data/libero/evaluation/trajectories/pick_up.../demo_0_rollout0.npz")

# Access data
actions = data["actions"]  # (T, 7)
images = data["agentview_images"]  # (T, 256, 256, 3)
success = data["success"]  # True/False

print(f"Trajectory length: {len(actions)}")
print(f"Success: {success}")
```

## Performance

- **Without videos**: ~5-10 seconds per rollout
- **With videos**: ~15-20 seconds per rollout

For libero_spatial (10 tasks × ~50 trajectories × 10 rollouts = 5000 rollouts):
- Estimated time: 7-14 hours without videos

## Tips

1. **Start small**: Test on one task first
   ```bash
   uv run python examples/libero/evaluate_trajectories.py --task-id 0
   ```

2. **Use SLURM**: For full evaluation, use job arrays to parallelize across tasks

3. **Monitor progress**: Results are saved incrementally, so you can check progress while it runs
   ```bash
   tail -f data/libero/evaluation/libero_spatial_results.csv
   ```
