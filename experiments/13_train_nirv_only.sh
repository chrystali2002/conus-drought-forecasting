#!/bin/bash --login
#SBATCH --job-name=nirv_only
#SBATCH --output=nirv_only_%j.out
#SBATCH --error=nirv_only_%j.err
#SBATCH --account=bassolab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=300G

#SBATCH --time=24:00:00
#SBATCH --mail-user=olusegu3@msu.edu
#SBATCH --mail-type=ALL

module purge
module load Conda
conda activate /mnt/ffs24/home/olusegu3/miniconda3/envs/drought_forecast
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# GPU guard — ensure 80GB A100
python3 -c "
import torch
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)')
if mem < 60:
    raise SystemExit(f'ERROR: Got {mem:.1f}GB GPU. Need 80GB.')
print('GPU OK.')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/gs21/scratch/olusegu3/drought/conus_drought
python 13_train_nirv_only.py
