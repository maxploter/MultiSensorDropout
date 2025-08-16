#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-multi-sensor-detection-eval"
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
backbone='yolo'
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir=""
resume=""
dataset_path='Max-Ploter/detection-moving-mnist-medium'
dataset='moving-mnist-medium'
wandb_id=''
hidden_dim=128
grid_size="1 1"
tile_overlap=0.0
num_queries=256
weight_loss_center_point=5
num_frames=20
frame_dropout_pattern=''
shuffle_views=''
enc_layers=1
self_per_cross_attn=1
max_freq=10
num_freq_bands=6
test_dataset_fraction=1.0
resize_frame=''
evaluators=''
batch_size=1
backbone_checkpoint=''
enc_nheads_cross=1
input_axis=2
disable_fourier_encoding=''
disable_recurrence=''
detr_nheads=4
detr_enc_layers=3
detr_dec_layers=3
dropout=0.0


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=$((12000 + RANDOM % 1000))
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

echo SLURM_PROCID=$SLURM_PROCID
echo RANK=$RANK
echo output_dir=$output_dir
echo OMP_NUM_THREADS=$OMP_NUM_THREADS

export NCCL_DEBUG=INFO

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
    --wandb_id) wandb_id="$2"; shift ;;
    --shuffle_views) shuffle_views="$2"; shift ;;
    --enc_layers) enc_layers="$2"; shift ;;
    --hidden_dim) hidden_dim="$2"; shift ;;
    --backbone) backbone="$2"; shift ;;
    --dataset) dataset="$2"; shift ;;
    --self_per_cross_attn) self_per_cross_attn="$2"; shift ;;
    --max_freq) max_freq="$2"; shift ;;
    --num_freq_bands) num_freq_bands="$2"; shift ;;
    --test_dataset_fraction) test_dataset_fraction="$2"; shift ;;
    --resize_frame) resize_frame="$2"; shift ;;
    --evaluators) evaluators="$2"; shift ;;
    --batch_size) batch_size="$2"; shift ;;
    --backbone_checkpoint) backbone_checkpoint="$2"; shift ;;
    --enc_nheads_cross) enc_nheads_cross="$2"; shift ;;
    --input_axis) input_axis="$2"; shift ;;
    --disable_fourier_encoding) disable_fourier_encoding="--disable_fourier_encoding"; ;;
    --disable_recurrence) disable_recurrence="--disable_recurrence"; ;;
    --detr_nheads) detr_nheads="$2"; shift ;;
    --detr_enc_layers) detr_enc_layers="$2"; shift ;;
    --detr_dec_layers) detr_dec_layers="$2"; shift ;;
    --dropout) dropout="$2"; shift ;;
    --model) model="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

evaluate_checkpoint() {
    local checkpoint=$1
    local checkpoint_file=$(basename "$checkpoint")
    echo "Evaluating checkpoint: $checkpoint_file"
    python_command="python train.py \
      --eval \
      --model $model \
      --backbone $backbone \
      --object_detection \
      --dataset_path $dataset_path \
      --dataset $dataset \
      --test_dataset_fraction $test_dataset_fraction \
      --num_frames $num_frames \
      --view_dropout_probs \
      --sampler_steps \
      --output_dir $output_dir \
      --resume $checkpoint_file \
      --hidden_dim $hidden_dim \
      --enc_layers $enc_layers \
      --self_per_cross_attn $self_per_cross_attn \
      --num_queries $num_queries \
      --max_freq $max_freq \
      --num_freq_bands $num_freq_bands \
      --num_workers 4 \
      --grid_size $grid_size \
      --tile_overlap $tile_overlap \
      --weight_loss_center_point $weight_loss_center_point \
      --world_size $WORLD_SIZE \
      --batch_size $batch_size \
      --enc_nheads_cross $enc_nheads_cross \
      --input_axis $input_axis \
      --detr_nheads $detr_nheads \
      --detr_enc_layers $detr_enc_layers \
      --detr_dec_layers $detr_dec_layers \
      --dropout $dropout"

    if [[ -n "$wandb_id" ]]; then
        python_command="$python_command --wandb_id $wandb_id"
    fi

    if [[ -n "$shuffle_views" ]]; then
        python_command="$python_command --shuffle_views"
    fi

    if [[ -n "$resize_frame" ]]; then
        python_command="$python_command --resize_frame $resize_frame"
    fi

    if [[ -n "$frame_dropout_pattern" ]]; then
        python_command="$python_command --frame_dropout_pattern $frame_dropout_pattern"
    fi
    
    if [[ -n "$evaluators" ]]; then
        python_command="$python_command --evaluators $evaluators"
    fi

    if [[ -n "$disable_fourier_encoding" ]]; then
        python_command="$python_command $disable_fourier_encoding"
    fi

    if [[ -n "$disable_recurrence" ]]; then
        python_command="$python_command $disable_recurrence"
    fi

    if [[ -n "$backbone_checkpoint" ]]; then
        python_command="$python_command --backbone_checkpoint $backbone_checkpoint"
    fi

    eval "$python_command"
}

if [[ -n "$resume" ]]; then
    evaluate_checkpoint "$resume"
else
    # Find all checkpoint files in output_dir
    if [[ -d "$output_dir" ]]; then
        checkpoints=($(ls $output_dir/checkpoint_epoch_*.pth 2>/dev/null))
        for checkpoint in "${checkpoints[@]}"; do
            evaluate_checkpoint "$checkpoint"
        done
    else
        echo "Output directory does not exist or is not set correctly."
        exit 1
    fi
fi