#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-sensor-dropout-train-lstm"
#SBATCH --time=5:00:00
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

model='lstm'
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model}_${timestamp}"

num_objects=2
lstm_hidden_size=64
img_size=128
learning_rate=1e-3

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --num_objects) num_objects="$2"; shift ;;
    --lstm_hidden_size) lstm_hidden_size="$2"; shift ;;
    --img_size) img_size="$2"; shift ;;
    --learning_rate) learning_rate="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python train.py \
    --model $model \
    --output_dir $output_dir \
    --learning_rate $learning_rate \
    --eval_interval 4 \
    --train_dataset_fraction 0.5 \
    --num_workers 4 \
    --lstm_hidden_size $lstm_hidden_size \
    --sampler_steps \
    --frame_dropout_probs \
    --num_objects $num_objects \
    --img_size $img_size