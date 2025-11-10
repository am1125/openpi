#!/bin/bash
#SBATCH --job-name=libero_eval_array
#SBATCH --output=slurm/logs/libero_eval_task%a_%A.log
#SBATCH --error=slurm/logs/libero_eval_task%a_%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9  # 10 tasks for libero_spatial (0-9)

# Job info
echo "======================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"
echo ""

# Navigate to project directory
cd /n/fs/vla-distill/openpi

# Activate virtual environment
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Verify GPU
nvidia-smi

echo ""
echo "Starting evaluation for task $SLURM_ARRAY_TASK_ID..."
echo ""

# Run evaluation for this specific task
# Create task-specific output directory to avoid conflicts
# Structure: data/libero/task-suite-name/evaluation_task_${SLURM_ARRAY_TASK_ID}
TASK_SUITE_NAME="libero_10"
OUTPUT_DIR="/n/fs/vla-distill/datasets/rollouts/${TASK_SUITE_NAME}"
# "data/libero/${TASK_SUITE_NAME}/evaluation_task_${SLURM_ARRAY_TASK_ID}"

python examples/libero/evaluate_trajectories.py \
    --args.task-suite-name $TASK_SUITE_NAME \
    --args.task-id $SLURM_ARRAY_TASK_ID \
    --args.num-rollouts-per-trajectory 10 \
    --args.output-dir $OUTPUT_DIR \
    --args.seed 7

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
