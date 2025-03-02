from datasets import load_dataset

from dataset.moving_mnist import MovingMNIST


def build_dataset(split, args, frame_dropout_pattern=None):
	dataset_name = args.dataset.lower()
	assert dataset_name.startswith('moving-mnist')

	# Load Hugging Face dataset split
	hf_dataset = load_dataset(args.dataset_path, split=split)

	# Determine dataset fraction
	dataset_fraction = args.train_dataset_fraction if split == 'train' else args.test_dataset_fraction

	return MovingMNIST(
		hf_split=hf_dataset,
		train=split == 'train',
		frame_dropout_pattern=frame_dropout_pattern,
		grid_size=args.grid_size,
		tile_overlap=args.tile_overlap,
		sampler_steps=args.sampler_steps,
		view_dropout_probs=args.view_dropout_probs,
		dataset_fraction=dataset_fraction
	)