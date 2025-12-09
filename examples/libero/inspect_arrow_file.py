#!/usr/bin/env python3
"""
Inspect the contents of an Apache Arrow file.

Usage:
    python openpi/examples/libero/inspect_arrow_file.py <path_to_arrow_file>
    python openpi/examples/libero/inspect_arrow_file.py /path/to/file.arrow
    python openpi/examples/libero/inspect_arrow_file.py /path/to/file.arrow --sample 5
"""

import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("Warning: pyarrow not available. Install with: pip install pyarrow")

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def detect_file_format(file_path: Path):
    """Try to detect the file format by reading the header."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            # Check for Parquet magic bytes
            if header[:4] == b'PAR1':
                return "parquet"
            # Check for Arrow IPC magic bytes
            elif header[:6] == b'ARROW1':
                return "arrow_ipc"
            # Check for Arrow stream format
            elif b'ARROW' in header:
                return "arrow_stream"
            else:
                return "unknown"
    except Exception:
        return "unknown"


def inspect_with_pyarrow(arrow_path: Path, num_samples: int = 3):
    """Inspect Arrow file using pyarrow."""
    print(f"\n{'='*80}")
    print(f"Inspecting Arrow file: {arrow_path}")
    print(f"{'='*80}\n")
    
    # Read the Arrow file
    try:
        # Try reading as Arrow file
        with pa.ipc.open_file(arrow_path) as reader:
            table = reader.read_all()
            print("✓ Successfully read as Arrow IPC file")
    except Exception as e1:
        try:
            # Try reading as Parquet file (sometimes .arrow files are actually parquet)
            table = pq.read_table(arrow_path)
            print("✓ Successfully read as Parquet file")
        except Exception as e2:
            print(f"✗ Failed to read as Arrow IPC: {e1}")
            print(f"✗ Failed to read as Parquet: {e2}")
            return
    
    # Get basic info
    print(f"\nTable shape: {table.num_rows} rows × {table.num_columns} columns")
    print(f"Total size: {table.nbytes / (1024**2):.2f} MB")
    
    # Show schema
    print(f"\n{'='*80}")
    print("Schema:")
    print(f"{'='*80}")
    print(table.schema)
    
    # Show column names and types
    print(f"\n{'='*80}")
    print("Columns:")
    print(f"{'='*80}")
    for i, field in enumerate(table.schema):
        col = table.column(i)
        print(f"  {i}. {field.name} ({field.type})")
        if col.num_chunks > 0:
            chunk = col.chunk(0)
            if hasattr(chunk, 'null_count'):
                print(f"     - Nulls: {chunk.null_count}/{len(chunk)}")
            if hasattr(chunk, 'nbytes'):
                print(f"     - Size: {chunk.nbytes / 1024:.1f} KB")
    
    # Show sample data
    print(f"\n{'='*80}")
    print(f"Sample data (first {num_samples} rows):")
    print(f"{'='*80}")
    
    # Convert to pandas for easier viewing
    try:
        import pandas as pd
        df = table.to_pandas()
        print(df.head(num_samples).to_string())
        
        # Show more detailed info about each column
        print(f"\n{'='*80}")
        print("Column details:")
        print(f"{'='*80}")
        for col_name in df.columns:
            col_data = df[col_name]
            print(f"\n{col_name}:")
            print(f"  Type: {col_data.dtype}")
            print(f"  Non-null count: {col_data.notna().sum()}/{len(col_data)}")
            if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                print(f"  Min: {col_data.min()}, Max: {col_data.max()}")
                print(f"  Mean: {col_data.mean():.4f}")
            elif col_data.dtype == 'object':
                # Check if it's string or something else
                sample_val = col_data.dropna().iloc[0] if len(col_data.dropna()) > 0 else None
                if sample_val is not None:
                    print(f"  Sample value type: {type(sample_val).__name__}")
                    if isinstance(sample_val, str):
                        print(f"  Sample value: {sample_val[:100]}{'...' if len(sample_val) > 100 else ''}")
                    elif isinstance(sample_val, (list, tuple)):
                        print(f"  Sample value: {str(sample_val)[:100]}{'...' if len(str(sample_val)) > 100 else ''}")
                    else:
                        print(f"  Sample value: {str(sample_val)[:100]}{'...' if len(str(sample_val)) > 100 else ''}")
    except ImportError:
        print("pandas not available. Showing raw table data...")
        # Show raw data without pandas
        for i in range(min(num_samples, table.num_rows)):
            print(f"\nRow {i}:")
            for j, field in enumerate(table.schema):
                col = table.column(j)
                val = col[i].as_py() if hasattr(col[i], 'as_py') else str(col[i])
                print(f"  {field.name}: {val}")


def inspect_with_datasets(arrow_path: Path, num_samples: int = 3):
    """Inspect Arrow file using HuggingFace datasets library."""
    print(f"\n{'='*80}")
    print(f"Inspecting with datasets library: {arrow_path}")
    print(f"{'='*80}\n")
    
    try:
        # Detect file format
        if arrow_path.is_file():
            file_format = detect_file_format(arrow_path)
            print(f"Detected file format: {file_format}")
            print(f"Loading single Arrow file...")
            
            # Try loading as a single file
            try:
                dataset = Dataset.from_file(str(arrow_path))
            except Exception as e1:
                # Try loading with explicit arrow format
                try:
                    from datasets import load_dataset
                    print("Trying load_dataset with arrow format...")
                    dataset = load_dataset("arrow", data_files=str(arrow_path), split="train")
                except Exception as e2:
                    # Try loading the parent directory (HuggingFace datasets often need the full directory)
                    parent_dir = arrow_path.parent
                    print(f"Trying to load from parent directory: {parent_dir}")
                    try:
                        from datasets import load_dataset
                        # Load all arrow files from the directory
                        arrow_files = list(parent_dir.glob("*.arrow"))
                        if arrow_files:
                            print(f"Found {len(arrow_files)} arrow files in parent directory")
                            dataset = load_dataset("arrow", data_files=[str(f) for f in arrow_files], split="train")
                        else:
                            raise ValueError("No arrow files in parent directory")
                    except Exception as e3:
                        raise Exception(
                            f"All loading methods failed:\n"
                            f"  1. Dataset.from_file: {e1}\n"
                            f"  2. load_dataset('arrow'): {e2}\n"
                            f"  3. load_dataset from parent dir: {e3}\n"
                            f"\nNote: HuggingFace datasets may require loading the entire dataset directory, not individual files."
                        )
        elif arrow_path.is_dir():
            # If it's a directory, try to load all arrow files in it
            arrow_files = list(arrow_path.glob("*.arrow"))
            if arrow_files:
                print(f"Found {len(arrow_files)} arrow files in directory")
                print("Loading first file for inspection...")
                dataset = Dataset.from_file(str(arrow_files[0]))
            else:
                raise ValueError(f"No .arrow files found in directory: {arrow_path}")
        else:
            raise ValueError(f"Path is neither a file nor directory: {arrow_path}")
        
        print(f"✓ Successfully loaded dataset")
        print(f"  Number of examples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")
        
        print(f"\n{'='*80}")
        print(f"Sample data (first {num_samples} examples):")
        print(f"{'='*80}")
        
        for i in range(min(num_samples, len(dataset))):
            print(f"\nExample {i}:")
            example = dataset[i]
            for key, value in example.items():
                if isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        print(f"  {key}: {type(value).__name__} of length {len(value)}")
                        if isinstance(value[0], (int, float)):
                            print(f"    Sample: {value[:5]}{'...' if len(value) > 5 else ''}")
                        else:
                            print(f"    First element type: {type(value[0]).__name__}")
                    else:
                        print(f"  {key}: empty {type(value).__name__}")
                elif isinstance(value, str):
                    print(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
                else:
                    print(f"  {key}: {value}")
        
        # Show feature details
        print(f"\n{'='*80}")
        print("Feature details:")
        print(f"{'='*80}")
        print(dataset.features)
        
    except Exception as e:
        print(f"✗ Failed to load with datasets library: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the contents of an Apache Arrow file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "arrow_file",
        type=str,
        help="Path to the .arrow file to inspect"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of sample rows/examples to show (default: 3)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "pyarrow", "datasets"],
        default="auto",
        help="Method to use for inspection (default: auto, tries both)"
    )
    
    args = parser.parse_args()
    
    arrow_path = Path(args.arrow_file)
    
    if not arrow_path.exists():
        print(f"Error: File not found: {arrow_path}")
        return 1
    
    if not arrow_path.is_file():
        print(f"Error: Not a file: {arrow_path}")
        return 1
    
    # Try inspection methods
    # For HuggingFace datasets, use datasets library first (it handles their custom format)
    if args.method == "auto":
        if DATASETS_AVAILABLE:
            try:
                inspect_with_datasets(arrow_path, args.sample)
            except Exception as e:
                print(f"\nWarning: datasets library method failed: {e}")
                print("Trying pyarrow method...\n")
                if PYARROW_AVAILABLE:
                    inspect_with_pyarrow(arrow_path, args.sample)
                else:
                    print("Error: pyarrow not available. Install with: pip install pyarrow")
                    return 1
        elif PYARROW_AVAILABLE:
            inspect_with_pyarrow(arrow_path, args.sample)
        else:
            print("Error: Neither pyarrow nor datasets library is available.")
            print("Install one of them:")
            print("  pip install pyarrow")
            print("  pip install datasets")
            return 1
    elif args.method == "pyarrow":
        if not PYARROW_AVAILABLE:
            print("Error: pyarrow not available. Install with: pip install pyarrow")
            return 1
        inspect_with_pyarrow(arrow_path, args.sample)
    elif args.method == "datasets":
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not available. Install with: pip install datasets")
            return 1
        inspect_with_datasets(arrow_path, args.sample)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

