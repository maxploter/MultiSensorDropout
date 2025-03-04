#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-multi-sensor-dropout-train-perceiver"
#SBATCH --time=4:00:00
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
output_dir=""
resume=""
dataset_path='Max-Ploter/detection-moving-mnist-easy'

grid_size="1 1"
tile_overlap=0.0
num_queries=256
weight_loss_center_point=5
num_frames=12
frame_dropout_pattern='000000111111'

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_path) dataset_path="$2"; shift ;;
    --num_frames) num_frames="$2"; shift ;;
    --frame_dropout_pattern) frame_dropout_pattern="$2"; shift ;;
    --output_dir) output_dir="$2"; shift ;;
    --resume) resume="$2"; shift ;;
    --grid_size) grid_size="$2"; shift ;;
    --tile_overlap) tile_overlap="$2"; shift ;;
    --num_queries) num_queries="$2"; shift ;;
    --weight_loss_center_point) weight_loss_center_point="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python train.py \
  --eval \
  --model $model \
  --dataset_path $dataset_path \
  --backbone 'cnn' \
  --num_frames $num_frames \
  --frame_dropout_pattern $frame_dropout_pattern \
  --output_dir $output_dir \
  --resume $resume \
  --num_workers 4 \
  --grid_size $grid_size \
  --tile_overlap $tile_overlap \
  --num_queries $num_queries \
  --weight_loss_center_point $weight_loss_center_point