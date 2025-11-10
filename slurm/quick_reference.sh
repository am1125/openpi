#!/bin/bash
# Quick reference for SLURM commands

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║           LIBERO Evaluation - SLURM Quick Reference          ║
╚══════════════════════════════════════════════════════════════╝

SUBMIT JOBS
───────────
  sbatch slurm/scripts/evaluate_test.slurm              # Test (1 rollout, ~3 min)
  sbatch slurm/scripts/evaluate_single_task.slurm       # One task (~1-2 hours)
  sbatch slurm/scripts/evaluate_all_tasks_parallel.slurm # All tasks parallel (~1-2 hours)

MONITOR JOBS
────────────
  squeue -u $USER                                        # Your jobs
  squeue --array -j <ARRAY_JOB_ID>                      # Job array status
  tail -f slurm/logs/libero_eval_*.log                  # Watch logs
  watch -n 5 'squeue -u $USER'                          # Auto-refresh status

CANCEL JOBS
───────────
  scancel <JOB_ID>                                       # Cancel specific job
  scancel -u $USER                                       # Cancel all your jobs
  scancel <ARRAY_JOB_ID>                                # Cancel entire array

CHECK OUTPUTS
─────────────
  ls -lh slurm/logs/                                    # Check log files
  cat slurm/logs/libero_eval_*.err                      # Check errors
  ls data/libero/evaluation/                            # Check results
  head data/libero/evaluation/libero_spatial_results.csv # View results

AFTER PARALLEL RUN
──────────────────
  python slurm/scripts/combine_parallel_results.py      # Combine results

ANALYZE RESULTS
───────────────
  python -c "
import pandas as pd
df = pd.read_csv('data/libero/evaluation/libero_spatial_all_tasks_combined.csv')
print(f'Success rate: {df[\"success\"].mean():.1%}')
print(df.groupby('task_name')['success'].mean())
"

COMMON WORKFLOWS
────────────────
  # Test → Single Task → All Tasks → Combine
  sbatch slurm/scripts/evaluate_test.slurm
  # (wait, verify)
  sbatch slurm/scripts/evaluate_all_tasks_parallel.slurm
  # (wait for completion)
  python slurm/scripts/combine_parallel_results.py

TROUBLESHOOTING
───────────────
  scontrol show job <JOB_ID>                            # Job details
  sacct -j <JOB_ID> --format=JobID,State,ExitCode      # Job history
  ssh <node> nvidia-smi                                 # Check GPU on node

───────────────────────────────────────────────────────────────
For full documentation: cat slurm/README.md
EOF
