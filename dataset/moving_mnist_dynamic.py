import torch


class MovingMNISTDynamicAdapter:
	def __init__(self, dynamic_dataset, num_of_videos):
		self.dynamic_dataset = dynamic_dataset
		self.num_of_videos = num_of_videos

	def __len__(self):
		return self.num_of_videos

	def __getitem__(self, idx):
		video, targets, _ = self.dynamic_dataset[idx] # T, C, H, W

		targets_result = []
		for t in targets:
			targets_result.append({
				'labels': torch.tensor(t['labels'], dtype=torch.int64),
				'center_points': torch.tensor(t['center_points'], dtype=torch.float32),
			})

		return video, targets_result
