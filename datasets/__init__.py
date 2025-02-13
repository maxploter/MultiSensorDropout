from types import SimpleNamespace

import numpy as np
from torchvision.datasets import MNIST

from datasets.moving_mnist import MovingMNIST


def build_dataset(split, args, frame_dropout_pattern=None):

    dataset_name = args.dataset.lower()
    assert dataset_name.startswith('moving-mnist')

    num_digits = args.num_objects

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
    if split == 'train':
        dataset = MovingMNIST(
            normalize=True,
            bounce=args.bounce,
            num_digits=num_digits,
            img_size=args.img_size,
            num_frames=args.num_frames,
            split_indices=split_indices,
            frame_dropout_probs=args.frame_dropout_probs,
            sampler_steps=args.sampler_steps,
            affine_params=affine_params,
            dataset_fraction=args.train_dataset_fraction,
            overlap_free_initial_translation=args.overlap_free_initial_position,
            grid_size=args.grid_size,
            tile_overlap=args.tile_overlap,
        )
    elif split == 'val':
        dataset = MovingMNIST(
            normalize=True,
            bounce=args.bounce,
            num_digits=num_digits,
            img_size=args.img_size,
            num_frames=args.num_frames,
            split_indices=split_indices,
            frame_dropout_pattern=frame_dropout_pattern,
            affine_params=affine_params,
            overlap_free_initial_translation=args.overlap_free_initial_position,
            grid_size=args.grid_size,
            tile_overlap=args.tile_overlap,
        )
    else:
        raise ValueError(f'unknown {split}')

    return dataset
