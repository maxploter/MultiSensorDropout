import torch

class MovingMNISTHuggingFaceAdapter:
		def __init__(self, hf_split):
				self.hf_split = hf_split
		def __len__(self):
				return len(self.hf_split)

		def __getitem__(self, idx):
			sample = self.hf_split[idx]

			video = torch.tensor(sample['video']).unsqueeze(1)  # T, C, H, W

			targets = []
			for i in range(video.shape[0]):
				targets.append({
					'labels': torch.tensor(sample['labels'][i], dtype=torch.int64),
					'center_points': torch.tensor(sample['center_points'][i], dtype=torch.float32),
				})

			return video, targets