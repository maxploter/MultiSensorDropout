import torch


class MovingMNISTHuggingFaceAdapter:
	def __init__(self, hf_split, num_frames, detection=False):
		self.hf_split = hf_split
		self.num_frames = num_frames
		self.detection = detection

	def __len__(self):
		return len(self.hf_split)

	def __getitem__(self, idx):
		sample = self.hf_split[idx]

		video = torch.tensor(sample['video'], dtype=torch.float32).unsqueeze(1)[:self.num_frames] / 255.0  # T, C, H, W

		targets = []
		for i in range(self.num_frames):
			if self.detection:
				targets.append({
					'boxes': torch.tensor(sample['bboxes'][i], dtype=torch.float),
					'labels': torch.tensor(sample['bboxes_labels'][i], dtype=torch.int64),
					'orig_size': torch.tensor([128,128], dtype=torch.int64), # TODO: fix this
                    'track_ids': torch.tensor(sample['track_ids'][i], dtype=torch.int64),
				})
			else:
				targets.append({
					'labels': torch.tensor(sample['labels'][i], dtype=torch.int64),
					'center_points': torch.tensor(sample['center_points'][i], dtype=torch.float32),
				})

		return video, targets
