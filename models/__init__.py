import re
from types import SimpleNamespace

import numpy as np
from torchvision.datasets import MNIST

from datasets.moving_mnist import MovingMNIST
from models.center_point_lstm import SimpleCenterNetWithLSTM
from models.perceiver_ar import build_perceiver_ar_model


def build_model(args):
    assert 'moving-mnist' in args.dataset.lower()
    num_classes = 10

    if args.model == 'lstm':
        return SimpleCenterNetWithLSTM(num_objects=args.num_objects, num_classes=num_classes, lstm_hidden_size=args.lstm_hidden_size)

    elif args.model == 'perceiver':
        return build_perceiver_ar_model(args, num_classes=num_classes)


def build_dataset(split, args, frame_dropout_pattern=None):

    dataset_name = args.dataset.lower()
    assert dataset_name.startswith('moving-mnist')

    match = re.search(r'(\d+)digit', dataset_name)
    if match:
        num_digits = [int(match.group(1))]
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')

    split_indices_attr = f"{split}_split_indices"

    if not hasattr(args, split_indices_attr) or getattr(args, f"{split}_split_indices") == None or len(getattr(args, f"{split}_split_indices")) == 0:
        # Load the MNIST dataset only once
        full_dataset = MNIST(".", download=True)
        num_samples = len(full_dataset)

        # Create train and validation indices
        indices = list(range(num_samples))
        indices_split = int(args.train_val_split_ratio * num_samples)
        np.random.shuffle(indices)

        train_indices = indices[:indices_split]
        val_indices = indices[indices_split:]
        if args.debug:
          downsample_factor = 0.005
          print(f'Downsample factor: {downsample_factor}')
          val_indices = val_indices[:int(len(val_indices) * downsample_factor)]
          train_indices = train_indices[:int(len(train_indices) * downsample_factor)]

        args.train_split_indices = train_indices
        args.val_split_indices = val_indices


    split_indices = getattr(args, f"{split}_split_indices")

    affine_params = SimpleNamespace(
      angle=(0, 0),
      translate=((-5, 5), (-5, 5)),
      scale=(1, 1),
      shear=(0, 0),
    )
    bounce = False
    use_center_bounce = False
    if split == 'train':
        dataset = MovingMNIST(
            normalize=True,
            bounce=bounce,
            use_center_bounce=use_center_bounce,
            num_digits=num_digits,
            num_frames=args.num_frames,
            split_indices=split_indices,
            frame_dropout_probs=args.frame_dropout_probs,
            sampler_steps=args.sampler_steps,
            affine_params=affine_params,
            dataset_fraction=args.train_dataset_fraction,
        )
    elif split == 'val':
        dataset = MovingMNIST(
            normalize=True,
            bounce=bounce,
            use_center_bounce=use_center_bounce,
            num_digits=num_digits,
            num_frames=args.num_frames,
            split_indices=split_indices,
            frame_dropout_pattern=frame_dropout_pattern,
            affine_params=affine_params,
        )
    else:
        raise ValueError(f'unknown {split}')

    return dataset
