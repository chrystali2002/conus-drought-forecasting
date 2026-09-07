#!/bin/bash --login
#SBATCH --job-name=multiseed
#SBATCH --output=logs/multiseed_%A_%a.out
#SBATCH --error=logs/multiseed_%A_%a.err
#SBATCH --account=bassolab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=300G
#SBATCH --partition=general-long-gpu
#SBATCH --time=96:00:00
#SBATCH --mail-user=olusegu3@msu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

set -eo pipefail          # abort on error; make failures inside a pipe fail the job

module purge
module load Conda
conda activate /mnt/ffs24/home/olusegu3/miniconda3/envs/drought_forecast
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
unset PYTHONPATH           # module tree leaks Python 3.11 packages into the 3.10 env

cd /mnt/gs21/scratch/olusegu3/drought/conus_drought
mkdir -p logs              # tee fails if this is absent

ARMS=(corrected uncorrected)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "=== arm: $ARM  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="

python3 -c "
import torch
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)')
if mem < 60: raise SystemExit(f'ERROR: Got {mem:.1f}GB. Need 80GB.')
print('GPU OK.')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DROUGHT_ARM=$ARM python 15_train_multiseed.py 2>&1 | tee "logs/train_${ARM}.log"
