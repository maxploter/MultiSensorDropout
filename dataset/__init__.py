from types import SimpleNamespace

from datasets import load_dataset
from detection_moving_mnist.mmnist.trajectory import SimpleLinearTrajectory, BouncingTrajectory

from dataset.moving_mnist import MovingMNISTWrapper, make_mmist_transforms
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
				num_frames=args.num_frames,
			),
			num_of_videos=60_000
		)
	else:
		print(f"Generating {split} huggingface MovingMNIST dataset...")
		# Load Hugging Face dataset split
		hf_dataset = load_dataset(args.dataset_path, split=split)
		dataset = MovingMNISTHuggingFaceAdapter(hf_dataset, num_frames=args.num_frames)

	# Determine dataset fraction
	dataset_fraction = args.train_dataset_fraction if split == 'train' else args.test_dataset_fraction

	video, _ = dataset[0]
	img_size = video.shape[2]
	assert video.shape[2] == video.shape[3]

	transforms, norm_transforms = make_mmist_transforms(split, img_size, args)

	return MovingMNISTWrapper(
		dataset=dataset,
		transforms=transforms,
		norm_transforms=norm_transforms,
		train=split == 'train',
		dataset_fraction=dataset_fraction
	)