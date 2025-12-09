#!/bin/bash
#SBATCH --job-name=libero_eval
#SBATCH --output=slurm/logs/libero_eval_%j.log
#SBATCH --error=slurm/logs/libero_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Job info
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"
echo ""

# Navigate to project directory
cd /n/fs/vla-distill/openpi

# Activate virtual environment
source .venv_clone_uv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# JAX configuration to help with CUDA initialization
# Disable preallocation to avoid CUDA context issues in SLURM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Verify GPU
nvidia-smi

echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
# Default: Task 0, all trajectories, 10 rollouts per trajectory
# Can override with: sbatch --export=TASK_ID=3 evaluate_single_task.slurm
TASK_ID=${TASK_ID:-0}  # Default to task 0 if not set
TASK_SUITE_NAME="libero_spatial"  # Match the parallel script
OUTPUT_DIR="data/libero/test_evaluation_color_processed/${TASK_SUITE_NAME}"

echo "Task ID: $TASK_ID"
echo "Task Suite: $TASK_SUITE_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

python examples/libero/evaluate_trajectories.py \
    --args.task-suite-name $TASK_SUITE_NAME \
    --args.task-id $TASK_ID \
    --args.num-rollouts-per-trajectory 1 \
    --args.output-dir $OUTPUT_DIR \
    --args.save-videos

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
