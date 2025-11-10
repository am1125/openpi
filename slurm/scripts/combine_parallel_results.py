#!/usr/bin/env python3
"""
Combine results from parallel SLURM job array runs.

After running evaluate_all_tasks_parallel.slurm, each task saves its own CSV.
This script combines them into one master CSV.

Usage:
    python slurm/scripts/combine_parallel_results.py
"""

import pandas as pd
import glob
from pathlib import Path


def combine_results(data_dir="data/libero/evaluation", output_file=None):
    """Combine all libero_spatial_results.csv files into one."""
    
    # Find all result CSVs
    csv_files = sorted(glob.glob(f"{data_dir}/**/libero_spatial_results.csv", recursive=True))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load and combine
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {csv_file}")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by task and trajectory
    combined = combined.sort_values(['task_name', 'trajectory_idx', 'rollout_idx'])
    
    # Save
    if output_file is None:
        output_file = f"{data_dir}/libero_spatial_all_tasks_combined.csv"
    
    combined.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print('='*60)
    print(f"Total rollouts: {len(combined)}")
    print(f"Unique tasks: {combined['task_name'].nunique()}")
    print(f"Unique trajectories: {combined['trajectory_name'].nunique()}")
    print(f"\nSuccess rate: {combined['success'].sum()}/{len(combined)} = {100*combined['success'].mean():.1f}%")
    
    # Per-task breakdown
    print(f"\nPer-task success rates:")
    task_stats = combined.groupby('task_name')['success'].agg(['sum', 'count', 'mean'])
    task_stats.columns = ['successes', 'total', 'rate']
    task_stats['rate'] = task_stats['rate'] * 100
    print(task_stats)
    
    print(f"\nSaved to: {output_file}")
    print('='*60)


if __name__ == "__main__":
    combine_results()
