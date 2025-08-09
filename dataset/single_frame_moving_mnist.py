import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.transformations import Compose, TilingTransform, ViewDropoutTransform, ToCenterCoordinateSystemTransform, \
    NormBoxesTransform, RandomResize

mmnist_stat = {
    'easy': (
        [
            0.023958550628466375
        ],
        [
            0.14140212075592035
        ]
    ),
    'medium': (
        [
            0.030643088476670285
        ],
        [
            0.15920598247588932
        ]
    )
}  # mean, std


class SingleDetectionMovingMNISTWrapper(Dataset):
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
        self.input_image_view_size = (0, 0)  # stub for compatibility with other datasets
        self.view_dropout_prob = -1  # stub for compatibility with other datasets

        self.indices = list(range(len(self.dataset)))
        if self.dataset_fraction < 1:
            self.indices = self.indices[:int(len(self.dataset) * self.dataset_fraction)]

        # Create a mapping from flat index to (video_idx, frame_idx)
        self.frame_mapping = []
        filtered_frames = 0
        print("Loading dataset and filtering empty frames...")
        for i, video_idx in tqdm(enumerate(self.indices), total=len(self.indices), desc="Processing videos"):
            video, targets = self.dataset[video_idx]
            num_frames = video.shape[0]  # Assuming video shape is [frames, channels, height, width]

            for frame_idx in range(num_frames):
                # Check if the frame has objects
                has_objects = True
                frame_target = targets[frame_idx]
                boxes_tensor = frame_target['boxes']
                if boxes_tensor.numel() == 0:
                    has_objects = False
                    filtered_frames += 1
                if has_objects:
                    self.frame_mapping.append((video_idx, frame_idx))
        print(
            f"Dataset loaded with {len(self.frame_mapping)} frames after filtering empty frames ({filtered_frames} frames were filtered out).")

    def step_epoch(self):
        pass

    def set_epoch(self, epoch):
        pass

    def __len__(self):
        return len(self.frame_mapping)

    def __getitem__(self, idx):
        # Get video index and frame index
        video_idx, frame_idx = self.frame_mapping[idx]

        # Get the video and targets
        video, targets = self.dataset[video_idx]

        # Extract the single frame
        frames = video[frame_idx:frame_idx + 1]  # Keep the time dimension with size 1

        frame_targets = [targets[frame_idx]]
        frames, frame_targets = self.transforms(frames, frame_targets)
        frames = self.norm_transforms(frames)

        return frames, frame_targets
