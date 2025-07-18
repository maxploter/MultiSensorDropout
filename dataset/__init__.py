from types import SimpleNamespace

from datasets import load_dataset
from detection_moving_mnist.mmnist.trajectory import SimpleLinearTrajectory, NonLinearTrajectory

from dataset.detection_moving_mnist_easy import DetectionMovingMNISTEasyWrapper, \
	make_mmist_transforms as make_mmist_transforms_detection
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
	"medium": {
		"angle": (0, 0),  # No rotation
		"translate": ((-5, 5), (-5, 5)),
		"scale": (1, 1),  # No scaling
		"shear": (0, 0),  # No deformation on z-axis
		"num_digits": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
	},
}
TRAJECTORIES = {
	"easy": SimpleLinearTrajectory,
	'medium': NonLinearTrajectory,
}


def build_dataset(split, args, frame_dropout_pattern=None):
	dataset_name = args.dataset.lower()
	assert dataset_name.startswith('moving-mnist')

	if args.generate_dataset_runtime and split == 'train':
		version = dataset_name[len('moving-mnist-'):].lower()
		affine_params = SimpleNamespace(**CONFIGS[version])
		trajectory = TRAJECTORIES[version]
		print(f"Generating {split} dynamic MovingMNIST dataset version {version}...")
		dataset = MovingMNISTDynamicAdapter(
			dynamic_dataset=MovingMNIST(
				trajectory=trajectory,
				train=True,
				path='.',
				affine_params=affine_params,
				num_digits=CONFIGS['easy']["num_digits"],
				num_frames=args.num_frames,
			),
			num_of_videos=60_000,
			detection=args.object_detection
		)
	else:
		print(f"Generating {split} huggingface MovingMNIST dataset {args.dataset_path}...")
		# Load Hugging Face dataset split
		hf_dataset = load_dataset(args.dataset_path, split=split)
		dataset = MovingMNISTHuggingFaceAdapter(hf_dataset, num_frames=args.num_frames, detection=args.object_detection)

	# Determine dataset fraction
	dataset_fraction = args.train_dataset_fraction if split == 'train' else args.test_dataset_fraction

	video, _ = dataset[0]
	img_size = video.shape[2]
	assert video.shape[2] == video.shape[3]

	if args.object_detection:
		print("Using object detection mode")
		transforms, norm_transforms = make_mmist_transforms_detection(args)
		return DetectionMovingMNISTEasyWrapper(
			dataset=dataset,
			transforms=transforms,
			norm_transforms=norm_transforms,
			train=split == 'train',
			dataset_fraction=dataset_fraction
		)

	transforms, norm_transforms = make_mmist_transforms(split, img_size, frame_dropout_pattern, args)
	return MovingMNISTWrapper(
		dataset=dataset,
		transforms=transforms,
		norm_transforms=norm_transforms,
		train=split == 'train',
		dataset_fraction=dataset_fraction
	)
