#!/usr/bin/env python3
"""
Compare two dataset files to see their differences.

Usage:
    python openpi/examples/libero/compare_datasets.py <file1> <file2>
    python openpi/examples/libero/compare_datasets.py /path/to/file1.arrow /path/to/file2.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Error: datasets library required. Install with: pip install datasets")

try:
    import pyarrow.parquet as pq
    import pandas as pd
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def load_dataset_file(file_path: Path):
    """Load a dataset file using the best available method."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading: {file_path}")
    
    # Try datasets library first (works for both .arrow and .parquet)
    if DATASETS_AVAILABLE:
        try:
            if file_path.is_file():
                dataset = Dataset.from_file(str(file_path))
                print(f"  ✓ Loaded with Dataset.from_file")
                return dataset
            else:
                raise ValueError("Path must be a file")
        except Exception as e1:
            # Try loading as parquet with datasets
            try:
                if file_path.suffix == '.parquet':
                    dataset = Dataset.from_file(str(file_path), format='parquet')
                    print(f"  ✓ Loaded with Dataset.from_file (parquet format)")
                    return dataset
            except Exception as e2:
                print(f"  ⚠ datasets library failed: {e1}")
    
    # Fallback to pyarrow + pandas
    if PYARROW_AVAILABLE:
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            print(f"  ✓ Loaded with pyarrow (parquet)")
            # Convert to Dataset for consistent interface
            if DATASETS_AVAILABLE:
                return Dataset.from_pandas(df)
            else:
                return df
        except Exception as e:
            raise Exception(f"Failed to load with pyarrow: {e}")
    
    raise Exception("No available method to load the file")


def get_schema_info(dataset):
    """Extract schema information from a dataset."""
    if hasattr(dataset, 'features'):
        # HuggingFace Dataset
        features = dataset.features
        schema = {}
        for key, feature in features.items():
            schema[key] = {
                'type': str(feature),
                'dtype': str(type(feature).__name__)
            }
        return schema, len(dataset)
    elif isinstance(dataset, pd.DataFrame):
        # Pandas DataFrame
        schema = {}
        for col in dataset.columns:
            schema[col] = {
                'type': str(dataset[col].dtype),
                'dtype': str(dataset[col].dtype)
            }
        return schema, len(dataset)
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")


def get_sample_data(dataset, num_samples=3):
    """Get sample data from dataset."""
    samples = []
    if hasattr(dataset, '__getitem__'):
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                samples.append(sample)
            except Exception as e:
                print(f"  Warning: Could not get sample {i}: {e}")
                break
    return samples


def compare_schemas(schema1: Dict, schema2: Dict, name1: str, name2: str):
    """Compare two schemas and show differences."""
    print(f"\n{'='*80}")
    print("SCHEMA COMPARISON")
    print(f"{'='*80}\n")
    
    keys1 = set(schema1.keys())
    keys2 = set(schema2.keys())
    
    # Common keys
    common_keys = keys1 & keys2
    # Keys only in schema1
    only_in_1 = keys1 - keys2
    # Keys only in schema2
    only_in_2 = keys2 - keys1
    
    print(f"Total columns in {name1}: {len(keys1)}")
    print(f"Total columns in {name2}: {len(keys2)}")
    print(f"Common columns: {len(common_keys)}")
    print(f"Columns only in {name1}: {len(only_in_1)}")
    print(f"Columns only in {name2}: {len(only_in_2)}")
    
    # Show columns only in first dataset
    if only_in_1:
        print(f"\n{'─'*80}")
        print(f"Columns only in {name1}:")
        print(f"{'─'*80}")
        for key in sorted(only_in_1):
            print(f"  {key}: {schema1[key]['type']}")
    
    # Show columns only in second dataset
    if only_in_2:
        print(f"\n{'─'*80}")
        print(f"Columns only in {name2}:")
        print(f"{'─'*80}")
        for key in sorted(only_in_2):
            print(f"  {key}: {schema2[key]['type']}")
    
    # Compare common columns
    if common_keys:
        print(f"\n{'─'*80}")
        print(f"Common columns (comparing types):")
        print(f"{'─'*80}")
        type_differences = []
        for key in sorted(common_keys):
            type1 = schema1[key]['type']
            type2 = schema2[key]['type']
            if type1 != type2:
                type_differences.append(key)
                print(f"  ⚠ {key}:")
                print(f"    {name1}: {type1}")
                print(f"    {name2}: {type2}")
        
        if not type_differences:
            print("  ✓ All common columns have matching types")
        else:
            print(f"\n  Found {len(type_differences)} columns with type differences")


def compare_sample_data(samples1: List, samples2: List, name1: str, name2: str):
    """Compare sample data from both datasets."""
    print(f"\n{'='*80}")
    print("SAMPLE DATA COMPARISON")
    print(f"{'='*80}\n")
    
    num_samples = min(len(samples1), len(samples2))
    if num_samples == 0:
        print("No samples available for comparison")
        return
    
    print(f"Comparing first {num_samples} examples:\n")
    
    for i in range(num_samples):
        print(f"{'─'*80}")
        print(f"Example {i}:")
        print(f"{'─'*80}")
        
        sample1 = samples1[i]
        sample2 = samples2[i]
        
        # Get keys from both samples
        keys1 = set(sample1.keys() if isinstance(sample1, dict) else sample1.index if hasattr(sample1, 'index') else [])
        keys2 = set(sample2.keys() if isinstance(sample2, dict) else sample2.index if hasattr(sample2, 'index') else [])
        
        common_keys = keys1 & keys2
        
        # Show values for common keys
        if common_keys:
            print("Common fields:")
            for key in sorted(common_keys):
                val1 = sample1[key] if isinstance(sample1, dict) else sample1.get(key, 'N/A')
                val2 = sample2[key] if isinstance(sample2, dict) else sample2.get(key, 'N/A')
                
                # Format values for display
                def format_val(v):
                    if isinstance(v, (list, tuple)):
                        if len(v) > 0:
                            if isinstance(v[0], (int, float)):
                                return f"{type(v).__name__}[{len(v)}] = {v[:5]}{'...' if len(v) > 5 else ''}"
                            else:
                                return f"{type(v).__name__}[{len(v)}] (first: {type(v[0]).__name__})"
                        else:
                            return f"{type(v).__name__}[]"
                    elif isinstance(v, str):
                        return v[:100] + ('...' if len(v) > 100 else '')
                    else:
                        return str(v)[:100] + ('...' if len(str(v)) > 100 else '')
                
                val1_str = format_val(val1)
                val2_str = format_val(val2)
                
                if val1_str == val2_str:
                    print(f"  ✓ {key}: {val1_str}")
                else:
                    print(f"  ⚠ {key}:")
                    print(f"    {name1}: {val1_str}")
                    print(f"    {name2}: {val2_str}")
        
        # Show keys only in first sample
        only_in_1 = keys1 - keys2
        if only_in_1:
            print(f"\nFields only in {name1}:")
            for key in sorted(only_in_1):
                val = sample1[key] if isinstance(sample1, dict) else sample1.get(key, 'N/A')
                print(f"  {key}: {format_val(val)}")
        
        # Show keys only in second sample
        only_in_2 = keys2 - keys1
        if only_in_2:
            print(f"\nFields only in {name2}:")
            for key in sorted(only_in_2):
                val = sample2[key] if isinstance(sample2, dict) else sample2.get(key, 'N/A')
                print(f"  {key}: {format_val(val)}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two dataset files to see their differences",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to first dataset file"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to second dataset file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample examples to compare (default: 3)"
    )
    
    args = parser.parse_args()
    
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    # Load both datasets
    print(f"{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}\n")
    
    try:
        dataset1 = load_dataset_file(file1_path)
        print()
        dataset2 = load_dataset_file(file2_path)
    except Exception as e:
        print(f"\nError loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get schema information
    print(f"\n{'='*80}")
    print("EXTRACTING SCHEMA INFORMATION")
    print(f"{'='*80}\n")
    
    schema1, len1 = get_schema_info(dataset1)
    schema2, len2 = get_schema_info(dataset2)
    
    print(f"Dataset 1: {len1} examples, {len(schema1)} columns")
    print(f"Dataset 2: {len2} examples, {len(schema2)} columns")
    
    # Compare schemas
    name1 = file1_path.name
    name2 = file2_path.name
    compare_schemas(schema1, schema2, name1, name2)
    
    # Get sample data
    print(f"\n{'='*80}")
    print("EXTRACTING SAMPLE DATA")
    print(f"{'='*80}\n")
    
    samples1 = get_sample_data(dataset1, args.samples)
    samples2 = get_sample_data(dataset2, args.samples)
    
    print(f"Extracted {len(samples1)} samples from dataset 1")
    print(f"Extracted {len(samples2)} samples from dataset 2")
    
    # Compare sample data
    compare_sample_data(samples1, samples2, name1, name2)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Dataset 1 ({name1}):")
    print(f"  - {len1} examples")
    print(f"  - {len(schema1)} columns")
    print(f"  - Columns: {', '.join(sorted(schema1.keys()))}")
    print(f"\nDataset 2 ({name2}):")
    print(f"  - {len2} examples")
    print(f"  - {len(schema2)} columns")
    print(f"  - Columns: {', '.join(sorted(schema2.keys()))}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

