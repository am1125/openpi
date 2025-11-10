#!/bin/bash
#SBATCH --job-name=libero_test
#SBATCH --output=slurm/logs/libero_test_%j.log
#SBATCH --error=slurm/logs/libero_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Quick test job: 1 trajectory, 1 rollout, with video
# Use this to verify everything works before running full evaluation

echo "======================================"
echo "TEST JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"
echo ""

cd /n/fs/vla-distill/openpi
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

nvidia-smi

echo ""
echo "Running test evaluation (1 trajectory, 1 rollout)..."
echo ""

python examples/libero/evaluate_trajectories.py \
    --args.output-dir data/libero/libero_spatial_rollouts \

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Test finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED - Ready for full evaluation!"
    echo ""
    echo "Check outputs:"
    echo "  - CSV: data/libero/test_evaluation/libero_spatial_results.csv"
    echo "  - Trajectory: data/libero/test_evaluation/trajectories/..."
    echo "  - Video: data/libero/test_evaluation/videos/..."
else
    echo "✗ Test FAILED - Check logs above"
fi
echo "======================================"

exit $EXIT_CODE
