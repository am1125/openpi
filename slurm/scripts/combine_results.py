#!/usr/bin/env python3
"""
Combine results from parallel task evaluations into a single CSV file.
"""

import pandas as pd
import pathlib
import sys
from typing import List

def combine_task_results(base_dir: str = "data/libero", task_suite_name: str = "libero_10", output_file: str = None):
    """
    Combine CSV results from multiple task directories.
    
    Args:
        base_dir: Base directory containing task suite folders
        task_suite_name: Name of the task suite (e.g., "libero_10")
        output_file: Output CSV file path (default: base_dir/task_suite_name/combined_results.csv)
    """
    base_path = pathlib.Path(base_dir)
    task_suite_path = base_path / task_suite_name
    
    if output_file is None:
        output_file = task_suite_path / "combined_results.csv"
    else:
        output_file = pathlib.Path(output_file)
    
    # Find all task directories within the task suite
    task_dirs = sorted([d for d in task_suite_path.glob("evaluation_task*") if d.is_dir()])
    
    if not task_dirs:
        print(f"No evaluation_task* directories found in {task_suite_path}")
        return
    
    print(f"Found {len(task_dirs)} task directories:")
    for task_dir in task_dirs:
        print(f"  - {task_dir.name}")
    
    # Collect all CSV files
    all_dfs = []
    
    for task_dir in task_dirs:
        csv_files = list(task_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {task_dir}")
            continue
            
        for csv_file in csv_files:
            print(f"Loading {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  - Loaded {len(df)} rows")
                else:
                    print(f"  - Empty file, skipping")
            except Exception as e:
                print(f"  - Error loading {csv_file}: {e}")
    
    if not all_dfs:
        print("No valid CSV files found!")
        return
    
    # Combine all dataframes
    print(f"\nCombining {len(all_dfs)} dataframes...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined results saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    
    # Print summary by task
    if 'task_name' in combined_df.columns:
        print("\nResults by task:")
        task_summary = combined_df.groupby('task_name').agg({
            'success': ['count', 'sum', 'mean'],
            'num_steps': 'mean'
        }).round(2)
        print(task_summary)
    
    return combined_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine parallel task evaluation results")
    parser.add_argument("--base-dir", default="data/libero", 
                       help="Base directory containing task suite folders")
    parser.add_argument("--task-suite", default="libero_10",
                       help="Task suite name (e.g., libero_10)")
    parser.add_argument("--output", default=None,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    combine_task_results(args.base_dir, args.task_suite, args.output)
