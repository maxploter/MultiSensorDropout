from types import SimpleNamespace

from datasets import load_dataset
from detection_moving_mnist.mmnist.trajectory import SimpleLinearTrajectory, BouncingTrajectory

from dataset.moving_mnist import MovingMNISTWrapper
from detection_moving_mnist.mmnist.mmnist import MovingMNIST

from dataset.moving_mnist_dynamic import MovingMNISTDynamicAdapter
from dataset.moving_mnist_huggingface import MovingMNISTHuggingFaceAdapter

CONFIGS = {
	"easy": {
		"angle": (0, 0),  # No rotation
		"translate": ((-5, 5), (-5, 5)),
		"scale": (1, 1),  # No scaling
		"shear": (0, 0),  # No deformation on z-axis
		"num_digits": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
	},
}
TRAJECTORIES = {
	"easy": SimpleLinearTrajectory,
}

def build_dataset(split, args, frame_dropout_pattern=None):
	dataset_name = args.dataset.lower()
	assert dataset_name.startswith('moving-mnist')

	if args.generate_dataset_runtime and split == 'train':
		print(f"Generating {split} dynamic MovingMNIST dataset...")
		affine_params = SimpleNamespace(**CONFIGS['easy'])
		trajectory = TRAJECTORIES['easy']
		dataset = MovingMNISTDynamicAdapter(
			dynamic_dataset=MovingMNIST(
				trajectory=trajectory,
				train=True,
				path='.',
				affine_params=affine_params,
				num_digits=CONFIGS['easy']["num_digits"],
				num_frames=20,
			),
			num_of_videos=60_000
		)
	else:
		print(f"Generating {split} huggingface MovingMNIST dataset...")
		# Load Hugging Face dataset split
		hf_dataset = load_dataset(args.dataset_path, split=split)
		dataset = MovingMNISTHuggingFaceAdapter(hf_dataset)

	# Determine dataset fraction
	dataset_fraction = args.train_dataset_fraction if split == 'train' else args.test_dataset_fraction

	return MovingMNISTWrapper(
		dataset=dataset,
		train=split == 'train',
		frame_dropout_pattern=frame_dropout_pattern,
		grid_size=args.grid_size,
		tile_overlap=args.tile_overlap,
		sampler_steps=args.sampler_steps,
		view_dropout_probs=args.view_dropout_probs,
		dataset_fraction=dataset_fraction
	)
