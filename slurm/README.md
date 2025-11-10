# SLURM Scripts for LIBERO Evaluation

This directory contains SLURM scripts for running π-0.5 policy evaluation on the LIBERO dataset.

## Directory Structure

```
slurm/
├── scripts/
│   ├── evaluate_test.slurm                    # Quick test (1 traj, 1 rollout)
│   ├── evaluate_single_task.slurm             # Single task evaluation
│   ├── evaluate_all_tasks_parallel.slurm      # All tasks in parallel
│   └── combine_parallel_results.py            # Combine results from parallel jobs
├── logs/                                       # SLURM output logs (auto-created)
└── README.md                                   # This file
```

## Quick Start

### 1. Test First! (Recommended)

Always run a test job first to verify everything works:

```bash
cd /n/fs/vla-distill/openpi
sbatch slurm/scripts/evaluate_test.slurm
```

**What it does:**
- 1 trajectory × 1 rollout (~2-3 minutes)
- Saves video to verify behavior
- Tests GPU, environment, data loading

**Monitor:**
```bash
squeue -u $USER                          # Check job status
tail -f slurm/logs/libero_test_*.log    # Watch progress
```

**Verify output:**
```bash
ls data/libero/test_evaluation/
# Should see: libero_spatial_results.csv, trajectories/, videos/
```

### 2. Single Task Evaluation

Evaluate one task (50 trajectories × 10 rollouts = 500 rollouts, ~1-2 hours):

```bash
sbatch slurm/scripts/evaluate_single_task.slurm
```

### 3. All Tasks in Parallel (Recommended for Full Evaluation)

Run all 10 tasks simultaneously (10× faster):

```bash
sbatch slurm/scripts/evaluate_all_tasks_parallel.slurm
```

**What happens:**
- Submits 10 jobs (one per task)
- Each job runs independently on its own GPU
- Results saved per task

**Monitor all jobs:**
```bash
squeue -u $USER
watch -n 5 'squeue -u $USER'  # Auto-refresh every 5 seconds
```

**After completion, combine results:**
```bash
python slurm/scripts/combine_results.py
```

This creates: `data/libero/combined_results.csv`

---

## Customizing Jobs

### Modify Task or Parameters

Edit the `.slurm` file or pass parameters:

```bash
# Evaluate task 3 only
sbatch slurm/scripts/evaluate_single_task.slurm --export=TASK_ID=3

# Or edit the file directly:
# Change: --task-id 0
# To:     --task-id 3
```

### Run Subset of Tasks

Modify array range in `evaluate_all_tasks_parallel.slurm`:

```bash
#SBATCH --array=0-9    # All 10 tasks
#SBATCH --array=0-4    # First 5 tasks only
#SBATCH --array=0,3,7  # Tasks 0, 3, and 7 only
```

### Change Resources

Edit the `#SBATCH` directives:

```bash
#SBATCH --time=12:00:00      # Max time
#SBATCH --mem=64G            # More memory
#SBATCH --gres=gpu:2         # 2 GPUs
#SBATCH --cpus-per-task=16   # More CPUs
```

### Save Videos

Add `--save-videos` flag in the script:

```bash
python examples/libero/evaluate_trajectories.py \
    # ... other args ...
    --save-videos
```

⚠️ **Warning:** Videos are large (~10-20 MB each). 500 rollouts = ~5-10 GB.

---

## Monitoring Jobs

### Check Job Status

```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <JOB_ID>

# Job array status
squeue --array -j <ARRAY_JOB_ID>
```

### Check Logs (Live)

```bash
# Single task
tail -f slurm/logs/libero_eval_*.log

# Job array (task 3)
tail -f slurm/logs/libero_eval_task3_*.log
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job array
scancel <ARRAY_JOB_ID>

# Cancel specific task in array
scancel <ARRAY_JOB_ID>_<TASK_ID>
```

---

## Output Files

### Per-Task Outputs

Each task creates:
```
data/libero/evaluation/
├── libero_spatial_results.csv       # Statistics
└── trajectories/
    └── <task_name>/
        ├── demo_0_rollout0.npz
        ├── demo_0_rollout1.npz
        └── ...
```

### Combined Output (After Parallel Run)

```bash
python slurm/scripts/combine_results.py
```

Creates:
```
data/libero/
├── evaluation_task0/
├── evaluation_task1/
├── ...
└── combined_results.csv  # All tasks merged
```

---

## Expected Runtime

| Job Type | Trajectories | Rollouts | Time (1 GPU) | GPUs Needed |
|----------|--------------|----------|--------------|-------------|
| Test | 1 | 1 | ~3 min | 1 |
| Single Task | 50 | 500 | ~1-2 hours | 1 |
| All Tasks (Parallel) | 500 | 5,000 | ~1-2 hours | 10 |
| All Tasks (Sequential) | 500 | 5,000 | ~10-20 hours | 1 |

**Recommendation:** Use parallel for full evaluation (10× speedup).

---

## Troubleshooting

### Job Fails Immediately

Check logs:
```bash
cat slurm/logs/libero_eval_*.err
```

Common issues:
- **Module not found:** Virtual environment not activated
- **CUDA error:** Wrong GPU type or out of memory
- **Dataset not found:** Run `python examples/libero/setup_libero_paths.py`

### Job Hangs

1. Check if actually running:
   ```bash
   ssh <node_name>
   nvidia-smi  # Should show GPU usage
   top -u $USER  # Should show Python process
   ```

2. Kill if stuck:
   ```bash
   scancel <JOB_ID>
   ```

### Out of Memory

Reduce batch/image size or request more memory:
```bash
#SBATCH --mem=64G  # Instead of 32G
```

### No Results Saved

Check:
1. Job completed successfully (EXIT_CODE=0 in logs)
2. Output directory exists and is writable
3. No errors in `.err` log files

---

## Example Workflow

```bash
# 1. Test (verify setup)
sbatch slurm/scripts/evaluate_test.slurm
# Wait ~3 minutes, check output

# 2. Run single task (prototype)
sbatch slurm/scripts/evaluate_single_task.slurm
# Wait ~1-2 hours, analyze results

# 3. Run all tasks in parallel (production)
sbatch slurm/scripts/evaluate_all_tasks_parallel.slurm
# Wait ~1-2 hours for all 10 jobs

# 4. Combine results
python slurm/scripts/combine_results.py

# 5. Analyze
python
>>> import pandas as pd
>>> df = pd.read_csv("data/libero/combined_results.csv")
>>> print(f"Overall success: {df['success'].mean():.2%}")
```

---

## Tips

✅ **Always test first** with `evaluate_test.slurm`  
✅ **Use parallel jobs** for full evaluation (much faster)  
✅ **Monitor logs** with `tail -f` to catch errors early  
✅ **Start small** - don't run all 10 tasks until single task works  
✅ **Check disk space** before running (trajectories are large)  

❌ Don't save videos for full evaluation (too much storage)  
❌ Don't run on login node (use `sbatch`, not `python` directly)  
❌ Don't forget to combine results after parallel runs  

---

## Questions?

- Check logs in `slurm/logs/`
- See main README: `examples/libero/EVALUATION_README.md`
- Script source: `examples/libero/evaluate_trajectories.py`
