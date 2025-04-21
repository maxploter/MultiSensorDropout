import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from dataset.transformations import Compose, TilingTransform, ViewDropoutTransform, ToCenterCoordinateSystemTransform, \
	NormBoxesTransform, RandomResize

mmnist_stat = (
	[
		0.023958550628466375
	],
	[
		0.14140212075592035
	]
)  # mean, std


class DetectionMovingMNISTEasyWrapper(Dataset):
	def __init__(
			self,
			dataset,
			transforms,
			norm_transforms,
			train,
			dataset_fraction=1,
	):
		self.dataset = dataset
		self.transforms = transforms
		self.norm_transforms = norm_transforms
		self.dataset_fraction = dataset_fraction
		self.train = train
		self.input_image_view_size = (0,0) # stub for compatibility with other datasets
		self.view_dropout_prob = -1 # stub for compatibility with other datasets

		self.indices = list(range(len(self.dataset)))
		if self.dataset_fraction < 1:
			self.indices = self.indices[:int(len(self.dataset) * self.dataset_fraction)]

	def step_epoch(self):
		pass

	def set_epoch(self, epoch):
		pass

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample_idx = self.indices[idx]
		video, targets = self.dataset[sample_idx]
		video = self.norm_transforms(video) #TODO: move in the end of the pipeline
		video, targets = self.transforms(video, targets)

		return video, targets


def make_mmist_transforms(args):
	norm_transforms = T.Compose([
		T.ConvertImageDtype(torch.float32),
		T.Normalize(*mmnist_stat)
	])

	transforms_list = []

	if args.resize_frame:
		print(f"Resizing frames to {args.resize_frame}x{args.resize_frame}")
		transforms_list.append(
			RandomResize([args.resize_frame], max_size=args.resize_frame),
		)

	transforms_list.append(
		NormBoxesTransform()
	)

	transforms = Compose(transforms_list)

	print(f"Transforms: {transforms}")

	return transforms, norm_transforms
