import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from dataset import build_dataset
from engine import train_one_epoch, evaluate
from models import build_model
from models.ade_post_processor import PostProcessTrajectory, MultiHeadPostProcessTrajectory
from models.perceiver import PostProcess
from models.set_criterion import build_criterion
from util.misc import collate_fn, is_main_process, get_sha, get_rank
import util.misc as utils

def _get_parser():
    parser = argparse.ArgumentParser(description="Train MultiSensor dropout")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=18, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--learning_rate_backbone', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--learning_rate_backbone_names', default=["backbone"], type=str, nargs='+')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--scheduler_step_size', type=int, default=12, help='Scheduler step size')
    parser.add_argument('--eval_interval', type=int, default=1, help='Eval every interval')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement')
    parser.add_argument('--model', type=str, default='perceiver', help='Model type')
    parser.add_argument('--backbone', type=str, help='Backbone type')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--weight_loss_center_point', type=int, default=5, help='Weight loss center point')
    parser.add_argument('--weight_loss_bce', type=int, default=1, help='Weight loss binary cross entropy')
    parser.add_argument('--shuffle_views', action='store_true', help='Shuffle views during inference')
    parser.add_argument('--object_detection', action='store_true', help='Use object detection prediction head')
    parser.add_argument('--disable_recurrence', action='store_true', help='Disable recurrent module for single frame processing')
    parser.add_argument('--resize_frame', type=int, help='Resize frame to this size')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)") # resnet specific
    parser.add_argument('--disable_fourier_encoding', action='store_true',
                        help='Disable Fourier encoding in the Perceiver model')

    # DETR specific parameters
    parser.add_argument('--detr_nheads', type=int, default=4, help='Number of attention heads in DETR transformer')
    parser.add_argument('--detr_enc_layers', type=int, default=3, help='Number of encoder layers in DETR transformer')
    parser.add_argument('--detr_dec_layers', type=int, default=3, help='Number of decoder layers in DETR transformer')


    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', default=2, type=float, help='Focal loss gamma')

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float,
                        help="L1 box coefficient")
    parser.add_argument('--giou_loss_coef', default=2, type=float,
                        help="GIoU box coefficient")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--output_dir', type=str, default=None, required=True, help='Output directory')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cpu or cuda)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='moving-mnist', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default="Max-Ploter/detection-moving-mnist-easy", help='Dataset path')
    parser.add_argument('--generate_dataset_runtime', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames.')
    parser.add_argument('--train_dataset_fraction', type=float, default=1.0, help='Train dataset fraction')
    parser.add_argument('--test_dataset_fraction', type=float, default=1.0, help='Test dataset fraction')
    parser.add_argument('--frame_dropout_pattern', type=str, required=False, help='Frame dropout pattern')
    parser.add_argument('--view_dropout_probs', nargs='*', type=float,
                        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], help='List of frame dropout probabilities')
    parser.add_argument('--sampler_steps', nargs='*', type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], help='Sampler steps')
    parser.add_argument('--grid_size', type=int, nargs=2, default=(1,1),
                        help='Grid size for splitting frames into tiles (rows, cols)')
    parser.add_argument('--tile_overlap', type=float, default=0.0,
                        help='Overlap ratio between tiles (0.0 to 1.0)')

    # wandb
    parser.add_argument('--wandb_project', type=str, default='multi-sensor-dropout', help='Wandb project')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb ID resume training')

    # Perceiver model specific arguments
    parser.add_argument('--num_freq_bands', type=int, default=6, help='Number of frequency bands for Fourier encoding')
    parser.add_argument('--max_freq', type=int, default=10, help='Maximum frequency for Fourier encoding')
    parser.add_argument('--enc_layers', type=int, default=1, help='Number of layers in Perceiver encoder')

    parser.add_argument('--num_queries', type=int, default=10, help='Number of latents, or induced set points, or centroids')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Latent dimension size')
    parser.add_argument('--enc_nheads_cross', type=int, default=1, help='Number of cross-attention heads')
    parser.add_argument('--nheads', type=int, default=1, help='Number of latent self-attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--self_per_cross_attn', type=int, default=1, help='Number of self-attention blocks per cross-attention block')
    parser.add_argument('--multi_classification_heads', action='store_true')
    parser.add_argument('--input_axis', type=int, default=2, help='Number of axes for input data (1 text, 2 for images, 3 for video)')

    # LSTM model specific arguments
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='Hidden size of LSTM')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--backbone_checkpoint', type=str, default=None, help='Path to pretrained backbone checkpoint')
    return parser

def parse_args():
    parser = _get_parser()
    args = parser.parse_args()
    return args

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    print("git:\n  {}\n".format(get_sha()))

    if is_main_process():
        print(f'Init wandb in process rank: {get_rank()}')
        wandb.init(**get_wandb_init_config(args))

    # Set seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    # Paths and directories
    output_dir = Path(args.output_dir)
    if args.output_dir and not args.eval and utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    # Dataset and dataloaders
    dataset_train = build_dataset('train', args)
    dataset_test = build_dataset('test', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size,
                                collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    dataloader_test_blind = None
    dataset_test_blind = None
    if args.frame_dropout_pattern is not None:
        dataset_test_blind = build_dataset('test', args, frame_dropout_pattern=args.frame_dropout_pattern)
        sampler_test_blind = torch.utils.data.SequentialSampler(dataset_test_blind)
        dataloader_test_blind = DataLoader(dataset_test_blind, sampler=sampler_test_blind, batch_size=args.batch_size,
                                          collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # Model, criterion, optimizer, and scheduler
    model = build_model(args, dataset_train.input_image_view_size)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.object_detection:
        postprocessors = {'bbox': PostProcess()}
    else:
        postprocessors = {'trajectory': MultiHeadPostProcessTrajectory() if hasattr(args,
                                                                                'multi_classification_heads') and args.multi_classification_heads else PostProcessTrajectory()}

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if not match_name_keywords(n, args.learning_rate_backbone_names) and p.requires_grad],
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.learning_rate_backbone_names) and p.requires_grad],
            "lr": args.learning_rate_backbone,
        },
    ]

    print(f'Params sizes: {[len(p["params"]) for p in param_dicts]}')

    optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = build_criterion(args)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.1)

    patience = args.patience
    current_patience = 0
    start_epoch = 0
    best_val_loss = float('inf')
    # Resume from checkpoint
    if args.resume:
        checkpoint_path = output_dir / args.resume
        print(f'Resuming from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if lr_scheduler.step_size != args.scheduler_step_size:
                print(f'Overwriting scheduler_step_size {args.scheduler_step_size}')
                lr_scheduler.step_size = args.scheduler_step_size

            start_epoch = checkpoint['epoch'] + 1
            current_patience = checkpoint['current_patience']
            best_val_loss = checkpoint['best_val_loss']
        else:
            start_epoch = checkpoint['epoch'] # eval epoch

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    criterion = criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Training loop
    print("Start training")
    start_time = time.time()

    dataset_train.set_epoch(start_epoch)
    dataset_test.set_epoch(start_epoch)

    if dataset_test_blind:
        dataset_test_blind.set_epoch(start_epoch)

    if args.eval:
        test_stats = evaluate(model, dataloader_test, criterion, postprocessors, start_epoch, device)
        blind_stats = {}
        if args.frame_dropout_pattern is not None:
            dataset_test_blind = build_dataset('test', args, frame_dropout_pattern=args.frame_dropout_pattern)
            sampler_test_blind = torch.utils.data.SequentialSampler(dataset_test_blind)
            dataloader_test_blind = DataLoader(dataset_test_blind, sampler=sampler_test_blind, batch_size=args.batch_size,
                                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
            blind_stats = evaluate(model, dataloader_test_blind, criterion, postprocessors, start_epoch, device)

        log_stats = {
            **{f'test_default_{k}': v for k, v in test_stats.items()},
            **{f'test_blind_{k}': v for k, v in blind_stats.items()},
            'epoch': start_epoch,
            'n_parameters': n_parameters,
            'output_dir': args.output_dir,
            'resume': args.resume,
        }

        if is_main_process():
            print(json.dumps(log_stats, indent=2))
            wandb.log(log_stats, step=start_epoch)

        return

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, dataloader_train, optimizer, criterion, epoch, device)

        blind_stats = {}
        test_stats = {}
        if epoch % args.eval_interval == 0:
            test_stats = evaluate(model, dataloader_test, criterion, postprocessors, epoch, device)
            if dataloader_test_blind:
                blind_stats = evaluate(model, dataloader_test_blind, criterion, postprocessors, epoch, device)

        lr_scheduler.step()

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:02}.pth"
        val_loss = test_stats.get("loss", float("inf"))

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_default_{k}': v for k, v in test_stats.items()},
            **{f'test_blind_{k}': v for k, v in blind_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'view_dropout_prob': dataset_train.view_dropout_prob,
        }

        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if is_main_process():
            print(json.dumps(log_stats, indent=2))
            wandb.log(log_stats, step=epoch)

        if val_loss < best_val_loss or epoch % 2 == 0 or epoch + 1 == args.epochs:
            utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'current_patience': current_patience,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            best_val_loss = val_loss
            current_patience = 0
            print(f"Checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}")
        else:
            current_patience += 1
            print(f'Current patience: {current_patience}')


        if current_patience >= patience:
            print('Early stopping triggered')
            break

        dataset_train.step_epoch()
        dataset_test.step_epoch()
        if dataset_test_blind:
            dataset_test_blind.step_epoch()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_wandb_init_config(args):
    result = {
        'project': args.wandb_project
    }

    if args.wandb_id:
        result['id'] = args.wandb_id
        result['resume'] = 'must'
    else:
        notes = f'model:{args.model}'

        if args.backbone is not None:
            notes += f',backbone:{args.backbone}'

        if args.view_dropout_probs is not None and len(args.view_dropout_probs) > 0:
            notes += f',view_dropout_probs:{len(args.view_dropout_probs)}'

        if args.hidden_dim != 128:
            notes += f',hidden_dim:{args.hidden_dim}'

        if args.dropout > 0:
            notes += f',dropout:{args.dropout}'

        if args.eval:
            notes += f',eval'
            notes += f',output_dir:{args.output_dir}'
            notes += f',resume:{args.resume}'

        if args.multi_classification_heads:
            notes += f',multi_classification_heads'

        if args.object_detection:
            notes += f',object_detection'

        if args.focal_loss:
            notes += f',focal_loss'
        if args.shuffle_views:
            notes += f',shuffle_views'

        if args.generate_dataset_runtime:
            notes += f',generate_dataset_runtime'
          
        if args.num_frames:
            notes += f',num_frames:{args.num_frames}'

        if args.self_per_cross_attn > 1:
            notes += f',self_per_cross_attn:{args.self_per_cross_attn}'

        result['notes'] = notes

    return result


if __name__ == '__main__':
    args = parse_args()
    args.focal_loss = True
    main(args)