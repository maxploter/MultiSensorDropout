import torch

from util.box_ops import box_xyxy_to_cxcywh


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

	if rows == 1 and cols == 1:
		return [frame]

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


class TilingTransform:
	"""Splits video frames into tiles with overlap"""

	def __init__(self, img_size, grid_size=(1, 1), tile_overlap=0.0):
		self.grid_size = grid_size
		self.tile_overlap = tile_overlap
		rows, cols = self.grid_size
		overlap = 1 + tile_overlap  # e.g., 1.2 for 20% overlap
		self.input_image_view_size = (
			int((img_size // rows) * overlap),
			int((img_size // cols) * overlap)
		)

	def __call__(self, video, targets):
		tiled_frames = []
		for frame in video:
			tiles = split_frame_into_tiles(frame, self.grid_size, self.tile_overlap)
			tiled_frames.append(torch.stack(tiles))
		return torch.stack(tiled_frames), targets


class ViewDropoutTransform:
	def __init__(self, sampler_steps=None, frame_dropout_pattern=None, view_dropout_probs=None, grid_size=None):
		self.view_dropout_probs = view_dropout_probs if not frame_dropout_pattern else None
		self.sampler_steps = sampler_steps if not frame_dropout_pattern else None
		self.grid_size = grid_size

		if self.sampler_steps and frame_dropout_pattern is None:
			assert len(self.view_dropout_probs) == len(self.sampler_steps) + 1
			for i in range(len(self.sampler_steps) - 1):
				assert self.sampler_steps[i] < self.sampler_steps[i + 1]

		self.keep_view_mask = None
		if frame_dropout_pattern:
			drop_frame_mask = torch.tensor([int(c) for c in frame_dropout_pattern])
			num_views = grid_size[0] * grid_size[1]
			num_drop_frames = drop_frame_mask.sum().item()
			self.keep_view_mask = torch.cat([
				torch.ones((len(frame_dropout_pattern) - num_drop_frames) * num_views).reshape(-1, num_views),
				torch.zeros(num_drop_frames * num_views).reshape(-1, num_views)
			])

	def set_epoch(self, epoch):
		self.current_epoch = epoch

		if not self.view_dropout_probs:
			self.view_dropout_prob = 0
		else:
			period_idx = 0
			for i in range(len(self.sampler_steps)):
				if epoch >= self.sampler_steps[i]:
					period_idx = i + 1
			print("set epoch: epoch {} period_idx={}".format(epoch, period_idx))
			self.view_dropout_prob = self.view_dropout_probs[period_idx]

	def __call__(self, video, targets):
		num_frames, num_views = video.shape[:2]

		if self.keep_view_mask is None:
			view_keep_flags = torch.ones((num_frames, num_views), dtype=torch.uint8)
			if self.view_dropout_prob > 0:
				num_potential_frames_with_drops = num_frames // 2
				view_probs = torch.rand(num_potential_frames_with_drops * num_views)
				view_flags = (view_probs > self.view_dropout_prob).int().view(num_potential_frames_with_drops, num_views)
				view_keep_flags = torch.cat([
					torch.ones(num_frames - num_potential_frames_with_drops, num_views),
					view_flags
				])
		else:
			view_keep_flags = self.keep_view_mask.int()

		for i in range(len(targets)):
			targets[i]['active_views'] = view_keep_flags[i]

		return video, targets


class ToCenterCoordinateSystemTransform:

	def __init__(self, img_size):
		self.img_size = img_size
		self.coordinate_norm_const = self.img_size / 2

	def __call__(self, video, targets):
		for i in range(len(targets)):
			targets[i]['center_points'] = targets[i]['center_points'] / self.coordinate_norm_const

		return video, targets


class NormBoxesTransform:

	def __init__(self, img_size):
		self.img_size = img_size

	def __call__(self, video, targets):
		for i in range(len(targets)):
			boxes = targets[i]['boxes']
			if len(boxes) == 0:
				continue
			boxes[:, 2:] += boxes[:, :2]
			boxes = box_xyxy_to_cxcywh(boxes)
			w, h = self.img_size, self.img_size
			boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
			targets[i]['boxes'] = boxes

		return video, targets

class Compose:

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, video, targets=None):
		for t in self.transforms:
			video, targets = t(video, targets)
		return video, targets

	def __iter__(self):
		return iter(self.transforms)

	def __repr__(self):
		format_string = self.__class__.__name__ + "("
		for t in self.transforms:
			format_string += "\n"
			format_string += "    {0}".format(t)
		format_string += "\n)"
		return format_string
