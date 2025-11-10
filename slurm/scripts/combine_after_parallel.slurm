#!/bin/bash
#SBATCH --job-name=combine_results
#SBATCH --output=slurm/logs/combine_results_%j.log
#SBATCH --error=slurm/logs/combine_results_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --dependency=afterok:JOB_ID_PLACEHOLDER  # Replace with actual job ID

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

echo "Combining results from parallel task evaluations..."
echo ""

# Run combination script
python slurm/scripts/combine_results.py \
    --base-dir data/libero \
    --task-suite libero_10 \
    --output data/libero/libero_10/combined_results.csv

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Combination finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
