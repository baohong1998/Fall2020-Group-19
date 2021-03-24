#!/bin/bash

#SBATCH --job-name="noisy-high-gamma-2"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1
#SBATCH --output=everglades-DQN-train-slurm-%J.out

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
echo "Loading anaconda3 (with TensorFlow)..."
module load anaconda/anaconda3
echo "TensorFlow module loaded."

# List the modules that are loaded
echo "Listing loaded modules..."
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
echo "Use nvidia-smi to tell us GPU/CPU mapping..."
nvidia-smi topo -m
echo

# Activate the GPU version of TensorFlow
echo "Activating gpu version of TensorFlow..."
source activate group19-env


# Run TensorFlow
echo
echo "Running..."
time python train.py config/no-noisy.json
# --train_dir=./SLURM --tmp_dir=./SLURM_tmp
echo

# You're done!
echo "Ending script..."
date


