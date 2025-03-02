from types import SimpleNamespace
import math
import random
import copy
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from datasets import load_dataset

mmnist_stat = ([0.0321, 0.0321, 0.0321], [0.1631, 0.1631, 0.1631])  # mean, std


def split_frame_into_tiles(frame, grid_size, overlap_ratio=0.0):
	"""Split a frame into tiles based on grid size with optional overlap

	Args:
			frame: Tensor of shape (C, H, W)
			grid_size: Tuple of (rows, cols) for the grid
			overlap_ratio: Float between 0 and 1 indicating how much tiles should overlap

	Returns:
			List of tiles, each tile is a tensor of shape (C, tile_h, tile_w)
	"""
	_, H, W = frame.shape
	rows, cols = grid_size

	# Calculate tile dimensions with overlap factor
	overlap = 1 + overlap_ratio  # e.g., 1.2 for 20% overlap
	tile_w = int((W // cols) * overlap)
	tile_h = int((H // rows) * overlap)

	delta_x = (W - tile_w) // (cols - 1)
	delta_y = (H - tile_h) // (rows - 1)
	tiles = []
	tile_center_point = None

	for i in range(rows):
		if tile_center_point is None:
			center_x = tile_w // 2
			center_y = tile_h // 2
			tile_center_point = (center_x, center_y)
		else:
			center_x = tile_w // 2
			center_y = tile_center_point[1] + delta_y
			tile_center_point = (center_x, center_y)

		for j in range(cols):
			# Calculate tile boundaries around center
			start_h = int(tile_center_point[1] - tile_h // 2)
			end_h = int(start_h + tile_h)
			start_w = int(tile_center_point[0] - tile_w // 2)
			end_w = int(start_w + tile_w)

			tile = frame[:, start_h:end_h, start_w:end_w]
			tiles.append(tile)

			tile_center_point = (tile_center_point[0] + delta_x, tile_center_point[1])

	return tiles


class MovingMNIST(Dataset):
	def __init__(
			self,
			hf_split,
			train,
			normalize=True,
			frame_dropout_pattern=None,
			grid_size=(1, 1),
			tile_overlap=0.0,
			sampler_steps=None,
			view_dropout_probs=None,
			dataset_fraction=1,
	):

		self.hf_split = hf_split
		self.normalize = normalize
		self.grid_size = grid_size
		self.tile_overlap = tile_overlap
		self.dataset_fraction = dataset_fraction
		self.sampler_steps = sampler_steps
		self.view_dropout_probs = view_dropout_probs
		self.train = train
		self.indices = list(range(len(self.hf_split)))
		if self.dataset_fraction < 1:
			self.indices = self.indices[:int(len(self.hf_split) * self.dataset_fraction)]

		# Infer dataset properties from the first sample
		sample = self.hf_split[0]
		video = sample['video']
		video_frame = video[-1].asnumpy()
		self.num_frames = len(video)
		self.img_size = video_frame.shape[0]
		assert video_frame.shape[0] == video_frame.shape[1]
		self.channels = video_frame.shape[2]

		self.num_views = self.grid_size[0] * self.grid_size[1]
		rows, cols = grid_size
		overlap = 1 + tile_overlap  # e.g., 1.2 for 20% overlap
		self.input_image_view_size = (
			int((self.img_size // rows) * overlap),
			int((self.img_size // cols) * overlap)
		)

		if self.sampler_steps and frame_dropout_pattern is None:
			assert len(self.view_dropout_probs) == len(self.sampler_steps) + 1
			for i in range(len(self.sampler_steps) - 1):
				assert self.sampler_steps[i] < self.sampler_steps[i + 1]

		self.keep_view_mask = None
		if frame_dropout_pattern is not None:
			drop_frame_mask = torch.tensor([int(c) for c in frame_dropout_pattern])
			num_views = grid_size[0] * grid_size[1]
			num_drop_frames = drop_frame_mask.sum().item()
			self.keep_view_mask = torch.cat([
				torch.ones((self.num_frames - num_drop_frames) * num_views).reshape(-1, num_views),
				torch.zeros(num_drop_frames * num_views).reshape(-1, num_views)
			])

		# Normalization transforms
		self.batch_tfms = T.Compose([
			T.ConvertImageDtype(torch.float32),
			T.Normalize(*mmnist_stat) if normalize else T.Lambda(lambda x: x)
		])

		self.set_epoch(0)

	def step_epoch(self):
		print("Dataset: epoch {} finishes".format(self.current_epoch))
		self.set_epoch(self.current_epoch + 1)

	def set_epoch(self, epoch):
		self.current_epoch = epoch
		if not self.view_dropout_probs:
			self.view_dropout_prob = 0
			return

		period_idx = 0
		for i in range(len(self.sampler_steps)):
			if epoch >= self.sampler_steps[i]:
				period_idx = i + 1
		print("set epoch: epoch {} period_idx={}".format(epoch, period_idx))
		self.view_dropout_prob = self.view_dropout_probs[period_idx]

		# Shuffle indices if dataset_fraction < 1
		if self.dataset_fraction < 1 and self.train:
			self.indices = list(range(len(self.hf_split)))
			random.shuffle(self.indices)
			self.indices = self.indices[:int(len(self.hf_split) * self.dataset_fraction)]

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample_idx = self.indices[idx]
		sample = self.hf_split[sample_idx]

		video_np = sample['video'].get_batch(range(self.num_frames)).asnumpy()

		video = torch.tensor(video_np).permute(0, 3, 1, 2)  # T, C, H, W

		targets = copy.deepcopy(sample['targets'])

		video = self.batch_tfms(video)

		if self.keep_view_mask is None:
			view_keep_flags = torch.ones((self.num_frames, self.num_views), dtype=torch.int)
			if self.view_dropout_prob > 0:
				num_potential_drop = self.num_frames // 2
				view_probs = torch.rand(num_potential_drop * self.num_views)
				view_flags = (view_probs > self.view_dropout_prob).int().view(num_potential_drop, self.num_views)
				view_keep_flags = torch.cat([
					torch.ones(self.num_frames - num_potential_drop, self.num_views),
					view_flags
				])
		else:
			view_keep_flags = self.keep_view_mask.int()

		# Process targets
		for frame_idx, target in enumerate(targets):
			target['keep_frame'] = torch.tensor(1)
			target['active_views'] = view_keep_flags[frame_idx]
			target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

			if 'center_points' in target and len(target['center_points']) > 0:
				n = self.img_size / 2  # center
				target['center_points'] = torch.tensor(target['center_points'], dtype=torch.float32) / n
			else:
				target['center_points'] = torch.tensor([])

		# Split into tiles
		tiled_frames = []
		for frame in video:
			tiles = split_frame_into_tiles(frame, self.grid_size, self.tile_overlap)
			tiled_frames.append(torch.stack(tiles))
		tiled_video = torch.stack(tiled_frames)  # T, num_tiles, C, H, W

		return tiled_video, targets
