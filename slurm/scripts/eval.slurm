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
source .venv/bin/activate

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
# Default parameters can be overridden with environment variables:
#   HOST, PORT, TASK_SUITE_NAME, NUM_TRIALS_PER_TASK, VIDEO_OUT_PATH, etc.
# Example: sbatch --export=HOST=localhost,PORT=8001,TASK_SUITE_NAME=libero_10 eval.slurm

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_spatial}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-data/libero/videos}
RESIZE_SIZE=${RESIZE_SIZE:-224}
REPLAN_STEPS=${REPLAN_STEPS:-5}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SEED=${SEED:-7}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Task Suite: $TASK_SUITE_NAME"
echo "  Number of trials per task: $NUM_TRIALS_PER_TASK"
echo "  Video output path: $VIDEO_OUT_PATH"
echo "  Resize size: $RESIZE_SIZE"
echo "  Replan steps: $REPLAN_STEPS"
echo "  Number of steps wait: $NUM_STEPS_WAIT"
echo "  Seed: $SEED"
echo ""

python examples/libero/main.py \
    --host $HOST \
    --port $PORT \
    --task-suite-name $TASK_SUITE_NAME \
    --num-trials-per-task $NUM_TRIALS_PER_TASK \
    --video-out-path $VIDEO_OUT_PATH \
    --resize-size $RESIZE_SIZE \
    --replan-steps $REPLAN_STEPS \
    --num-steps-wait $NUM_STEPS_WAIT \
    --seed $SEED

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
