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

# Blind experiment. Drop frames during training.

model='perceiver'
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model}_${timestamp}"

num_objects=2
grid_size="1 1"
tile_overlap=0.0
train_dataset_fraction=0.5
test_dataset_fraction=1.0
num_queries=256
weight_loss_center_point=5
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --num_objects) num_objects="$2"; shift ;;  # Parse --num_objects and its value
    --grid_size) grid_size="$2"; shift ;;
    --tile_overlap) tile_overlap="$2"; shift ;;
    --train_dataset_fraction) train_dataset_fraction="$2"; shift ;;
    --test_dataset_fraction) test_dataset_fraction="$2"; shift ;;
    --num_queries) num_queries="$2"; shift ;;
    --weight_loss_center_point) weight_loss_center_point="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python train.py \
    --model $model \
    --backbone 'cnn' \
    --output_dir $output_dir \
    --train_dataset_fraction $train_dataset_fraction \
    --test_dataset_fraction $test_dataset_fraction \
    --num_workers 4 \
    --frame_dropout_pattern '00001111' \
    --num_objects $num_objects \
    --img_size 128 \
    --grid_size $grid_size \
    --tile_overlap $tile_overlap \
    --num_queries $num_queries \
    --multi_classification_heads \
    --weight_loss_center_point $weight_loss_center_point