#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-sensor-dropout-train-perceiver"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=not_tracked_dir/slurm/%j_slurm_%x.out

# Resource allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G

# Node configurations (commented out)
## falcone configurations
#SBATCH --gres=gpu:tesla:4
#SBATCH --cpus-per-task=4

## Pegasus configuration
##SBATCH --gres=gpu:a100-40g:4
##SBATCH --cpus-per-task=24

## Pegasus2 configuration
##SBATCH --gres=gpu:a100-80g:2
##SBATCH --cpus-per-task=16

module load miniconda3
conda activate sensor_dropout

# Default experiment. Frame drop out is disabled.

model='perceiver'
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model}_${timestamp}"

python train.py \
    --output_dir $output_dir \
    --train_dataset_fraction 1 \
    --sampler_steps \
    --frame_dropout_probs \
    --num_objects 10 \
    --img_size 128