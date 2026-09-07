#!/bin/bash --login
#SBATCH --job-name=convlstm
#SBATCH --output=convlstm_%j.out
#SBATCH --error=convlstm_%j.err
#SBATCH --account=bassolab

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1

#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --mail-user=olusegu3@msu.edu
#SBATCH --mail-type=ALL

module purge
module load Conda

conda activate /mnt/ffs24/home/olusegu3/miniconda3/envs/drought_forecast
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load powertools

python - << EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# Adding this for GPU size check so the job fails fast instead of wasting queue time if it lands on the wrong cardright of less than required memory size of 80GB
python3 -c "
import torch; mem=torch.cuda.get_device_properties(0).total_memory/1e9
print(f'GPU: {torch.cuda.get_device_name(0)} ({mem:.1f}GB)')
if mem < 60: raise SystemExit(f'Wrong GPU ({mem:.1f}GB). Need 80GB.')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python  11_train.py

# Job info
scontrol show job $SLURM_JOB_ID
js -j $SLURM_JOB_ID







