#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-multi-sensor-detection"
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
output_dir="not_tracked_dir/output_${model}_detection_${timestamp}"
resume=''
wandb_id=''
dataset_path='Max-Ploter/detection-moving-mnist-easy'
backbone='cnn'

self_per_cross_attn=1
hidden_dim=128 # Perceiver hidden size
learning_rate=1e-4
learning_rate_backbone=1e-4
train_dataset_fraction=0.5
test_dataset_fraction=1.0
num_queries=16
eval_interval=1
num_frames=20
scheduler_step_size=12
epochs=18
dropout=0.0
enc_layers=1
resize_frame=''
random_digits_placement=''
max_freq=10
num_freq_bands=6

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --backbone) backbone="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --dataset_path) dataset_path="$2"; shift ;;
    --num_frames) num_frames="$2"; shift ;;
    --self_per_cross_attn) self_per_cross_attn="$2"; shift ;;
    --hidden_dim) hidden_dim="$2"; shift ;;
    --learning_rate) learning_rate="$2"; shift ;;
    --learning_rate_backbone) learning_rate_backbone="$2"; shift ;;
    --train_dataset_fraction) train_dataset_fraction="$2"; shift ;;
    --test_dataset_fraction) test_dataset_fraction="$2"; shift ;;
    --num_queries) num_queries="$2"; shift ;;
    --eval_interval) eval_interval="$2"; shift ;;
    --wandb_id) wandb_id="$2"; shift ;;
    --output_dir) output_dir="$2"; shift ;;
    --resume) resume="$2"; shift ;;
    --scheduler_step_size) scheduler_step_size="$2"; shift ;;
    --dropout) dropout="$2"; shift ;;
    --enc_layers) enc_layers="$2"; shift ;;
    --resize_frame) resize_frame="$2"; shift ;;
    --max_freq) max_freq="$2"; shift ;;
    --num_freq_bands) num_freq_bands="$2"; shift ;;
    --random_digits_placement) random_digits_placement="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

python_command="python train.py \
    --model $model \
    --backbone $backbone \
    --object_detection \
    --generate_dataset_runtime \
    --dataset_path $dataset_path \
    --epochs $epochs \
    --dropout $dropout \
    --eval_interval $eval_interval \
    --num_frames $num_frames \
    --learning_rate $learning_rate \
    --learning_rate_backbone $learning_rate_backbone \
    --output_dir $output_dir \
    --train_dataset_fraction $train_dataset_fraction \
    --test_dataset_fraction $test_dataset_fraction \
    --num_workers 4 \
    --hidden_dim $hidden_dim \
    --enc_layers $enc_layers \
    --self_per_cross_attn $self_per_cross_attn \
    --num_queries $num_queries \
    --max_freq $max_freq \
    --num_freq_bands $num_freq_bands \
    --scheduler_step_size $scheduler_step_size"

if [[ -n "$resize_frame" ]]; then
    python_command="$python_command --resize_frame $resize_frame"
fi

# Conditionally add --resume
if [[ -n "$resume" ]]; then
    python_command="$python_command --resume $resume --wandb_id $wandb_id"
fi

# Conditionally add --random_digits_placement
if [[ -n "$random_digits_placement" ]]; then
    python_command="$python_command --random_digits_placement"
fi

# Execute the python command
eval "$python_command"