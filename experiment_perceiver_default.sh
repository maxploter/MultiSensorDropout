#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-multi-sensor-dropout-train-perceiver"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=not_tracked_dir/slurm/%j_slurm_%x.out

# Resource allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G

# Node configurations (commented out)
## falcone configurations
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4

## Pegasus configuration
##SBATCH --gres=gpu:a100-40g:1
##SBATCH --cpus-per-task=24

## Pegasus2 configuration
##SBATCH --gres=gpu:a100-80g:1
##SBATCH --cpus-per-task=16

module load miniconda3
conda activate multi_sensor_dropout

# Default experiment. Frame drop out is disabled.

model='perceiver'
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model}_${timestamp}"

num_objects=2
self_per_cross_attn=1
hidden_dim=128 # Perceiver hidden size
learning_rate=1e-3
grid_size="1 1"
tile_overlap=0.0
train_dataset_fraction=0.5
num_queries=256

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --num_objects) num_objects="$2"; shift ;;
    --self_per_cross_attn) self_per_cross_attn="$2"; shift ;;
    --hidden_dim) hidden_dim="$2"; shift ;;
    --learning_rate) learning_rate="$2"; shift ;;
    --grid_size) grid_size="$2"; shift ;;
    --tile_overlap) tile_overlap="$2"; shift ;;
    --train_dataset_fraction) train_dataset_fraction="$2"; shift ;;
    --num_queries) num_queries="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python train.py \
    --model $model \
    --backbone 'cnn' \
    --learning_rate $learning_rate \
    --output_dir $output_dir \
    --train_dataset_fraction $train_dataset_fraction \
    --num_workers 4 \
    --hidden_dim $hidden_dim \
    --sampler_steps \
    --frame_dropout_probs \
    --num_objects $num_objects \
    --self_per_cross_attn $self_per_cross_attn \
    --img_size 128 \
    --grid_size $grid_size \
    --tile_overlap $tile_overlap \
    --num_queries $num_queries