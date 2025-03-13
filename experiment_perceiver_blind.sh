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
resume=null
wandb_id=''
dataset_path='Max-Ploter/detection-moving-mnist-easy'

hidden_dim=128 # Perceiver hidden size
learning_rate=1e-3
learning_rate_backbone=1e-4
grid_size="1 1"
tile_overlap=0.0
train_dataset_fraction=0.5
test_dataset_fraction=1.0
num_queries=256
weight_loss_center_point=5
eval_interval=1
num_frames=12
scheduler_step_size=12
epochs=18
view_dropout_probs='0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85'
sampler_steps='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15'
shuffle_views=''

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --epochs) epochs="$2"; shift ;;
    --dataset_path) dataset_path="$2"; shift ;;
    --num_frames) num_frames="$2"; shift ;;
    --hidden_dim) hidden_dim="$2"; shift ;;
    --learning_rate) learning_rate="$2"; shift ;;
    --learning_rate_backbone) learning_rate_backbone="$2"; shift ;;
    --grid_size) grid_size="$2"; shift ;;
    --tile_overlap) tile_overlap="$2"; shift ;;
    --train_dataset_fraction) train_dataset_fraction="$2"; shift ;;
    --test_dataset_fraction) test_dataset_fraction="$2"; shift ;;
    --num_queries) num_queries="$2"; shift ;;
    --weight_loss_center_point) weight_loss_center_point="$2"; shift ;;
    --eval_interval) eval_interval="$2"; shift ;;
    --scheduler_step_size) scheduler_step_size="$2"; shift ;;
    --view_dropout_probs) view_dropout_probs="$2"; shift ;;
    --sampler_steps) sampler_steps="$2"; shift ;;
    --wandb_id) wandb_id="$2"; shift ;;
    --output_dir) output_dir="$2"; shift ;;
    --resume) resume="$2"; shift ;;
    --shuffle_views) shuffle_views="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python_command="python train.py \
    --model $model \
    --backbone 'cnn' \
    --dataset_path $dataset_path \
    --epochs $epochs \
    --eval_interval $eval_interval \
    --num_frames $num_frames \
    --learning_rate $learning_rate \
    --learning_rate_backbone $learning_rate_backbone \
    --output_dir $output_dir \
    --train_dataset_fraction $train_dataset_fraction \
    --test_dataset_fraction $test_dataset_fraction \
    --generate_dataset_runtime \
    --num_workers 4 \
    --hidden_dim $hidden_dim \
    --grid_size $grid_size \
    --tile_overlap $tile_overlap \
    --num_queries $num_queries \
    --scheduler_step_size $scheduler_step_size \
    --view_dropout_probs $view_dropout_probs \
    --sampler_steps $sampler_steps \
    --weight_loss_center_point $weight_loss_center_point"

if [[ -n "$resume" ]]; then
    python_command="$python_command --resume $resume --wandb_id $wandb_id"
fi

if [[ -n "$shuffle_views" ]]; then
    python_command="$python_command --shuffle_views"
fi

# Execute the python command
eval "$python_command"