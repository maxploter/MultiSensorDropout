import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from dataset.transformations import Compose, TilingTransform, ViewDropoutTransform, ToCenterCoordinateSystemTransform

mmnist_stat = (
	[
		0.023958550628466375
	],
	[
		0.14140212075592035
	]
)  # mean, std


class MovingMNISTWrapper(Dataset):
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

		self.indices = list(range(len(self.dataset)))
		if self.dataset_fraction < 1:
			self.indices = self.indices[:int(len(self.dataset) * self.dataset_fraction)]

		# Infer dataset properties from the first sample
		self.coordinate_norm_const = next(
			(t.coordinate_norm_const for t in self.transforms if hasattr(t, 'coordinate_norm_const')), None
		)

		self.input_image_view_size = next(
			(t.input_image_view_size for t in self.transforms if hasattr(t, 'input_image_view_size')), None
		)

		self.set_epoch(0)

	def step_epoch(self):
		print("Dataset: epoch {} finishes".format(self.current_epoch))
		self.set_epoch(self.current_epoch + 1)

	def set_epoch(self, epoch):
		self.current_epoch = epoch
		for t in self.transforms:
			if hasattr(t, 'set_epoch'):
				t.set_epoch(epoch)

		self.view_dropout_prob = next((t.view_dropout_prob for t in self.transforms if hasattr(t, 'view_dropout_prob')),
		                              None)

		if self.dataset_fraction < 1 and self.train:
			full_length = len(self.dataset)
			self.indices = list(range(full_length))
			random_start_index = random.randint(0, full_length - 1)
			number_of_elements = int(full_length * self.dataset_fraction)

			indices_from_start = []
			if random_start_index + number_of_elements > full_length:
				# Take elements from end of array
				indices_from_start = self.indices[random_start_index:full_length]
				# Take remaining elements from beginning
				remaining = number_of_elements - len(indices_from_start)
				indices_from_start.extend(self.indices[:remaining])
			else:
				indices_from_start = self.indices[random_start_index:random_start_index + number_of_elements]

			self.indices = indices_from_start

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample_idx = self.indices[idx]
		video, targets = self.dataset[sample_idx]
		video = self.norm_transforms(video)
		video, targets = self.transforms(video, targets)

		return video, targets


def make_mmist_transforms(split, img_size, frame_dropout_pattern, args):
	norm_transforms = T.Compose([
		T.ConvertImageDtype(torch.float32),
		T.Normalize(*mmnist_stat)
	])

	tiling = TilingTransform(
		img_size=img_size,
		grid_size=args.grid_size,
		tile_overlap=args.tile_overlap,
	)

	view_dropout = ViewDropoutTransform(
		sampler_steps=args.sampler_steps,
		frame_dropout_pattern=frame_dropout_pattern,
		view_dropout_probs=args.view_dropout_probs,
		grid_size=args.grid_size,
	)

	to_ccst = ToCenterCoordinateSystemTransform(img_size)

	transforms = Compose([
		tiling,
		view_dropout,
		to_ccst,
	])

	return transforms, norm_transforms
