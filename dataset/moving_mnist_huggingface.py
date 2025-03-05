import torch


class MovingMNISTHuggingFaceAdapter:
	def __init__(self, hf_split, num_frames):
		self.hf_split = hf_split
		self.num_frames = num_frames

	def __len__(self):
		return len(self.hf_split)

	def __getitem__(self, idx):
		sample = self.hf_split[idx]

		video = torch.tensor(sample['video'], dtype=torch.float32).unsqueeze(1)[:self.num_frames] / 255.0  # T, C, H, W

		targets = []
		for i in range(self.num_frames):
			targets.append({
				'labels': torch.tensor(sample['labels'][i], dtype=torch.int64),
				'center_points': torch.tensor(sample['center_points'][i], dtype=torch.float32),
			})

		return video, targets
