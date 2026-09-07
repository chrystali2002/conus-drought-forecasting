#!/bin/bash --login
#SBATCH --job-name=eval_multiseed
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --account=bassolab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=300G
#SBATCH --partition=general-short-gpu
#SBATCH --time=48:00:00    #48:00:00
#SBATCH --mail-user=olusegu3@msu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1

set -eo pipefail

module purge
module load Conda
conda activate /mnt/ffs24/home/olusegu3/miniconda3/envs/drought_forecast
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
unset PYTHONPATH

cd /mnt/gs21/scratch/olusegu3/drought/conus_drought
mkdir -p logs

ARMS=(corrected uncorrected)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "=== evaluating arm: $ARM ==="

# fail now, with a clear message, rather than part-way through inference
for s in 42 123 777 2024 31337  7 99 512 1234 4096 8191 20250 60613 77777 90210; do
  f="models/${ARM}/seed_${s}_convlstm.pt"
  [[ -f "$f" ]] || { echo "MISSING $f — run Step 15 for arm $ARM first"; exit 1; }
done

# Checkpoints exist from the first improving epoch onward, so their presence does
# not mean training finished. multiseed_results.json is written only after the
# final seed completes, so it is the real completion test.
SUMMARY="outputs/metrics/${ARM}/multiseed_results.json"
[[ -f "$SUMMARY" ]] || { echo "Step 15 for $ARM did not complete — $SUMMARY missing"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DROUGHT_ARM=$ARM python 16_evaluate_multiseed.py 2>&1 | tee "logs/eval_${ARM}.log"





