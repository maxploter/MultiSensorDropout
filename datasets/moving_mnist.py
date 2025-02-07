import copy
import math
import random
from functools import partial
from types import SimpleNamespace

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

mnist_stats    = {
    (1,2,3,4,5,6,7,8,9,10): ([0.0321], [0.1631]) # mean, std
}

def padding(img_size=64, mnist_size=28): return (img_size - mnist_size) // 2

def apply_n_times(tf, x, n=1):
    "Apply `tf` to `x` `n` times, return all values"
    sequence = [x]
    for n in range(n):
        sequence.append(tf(sequence[n]))
    return sequence

affine_params = SimpleNamespace(
    angle=(-4, 4),
    translate=((-5, 5), (-5, 5)),
    scale=(.8, 1.2),
    shear=(-3, 3),
)

def get_affine_transformed_coordinates(point, center, angle=0, translate=(0, 0), scale=1, shear=(0, 0)):
    # Convert degrees to radians for rotation and shear
    x0, y0 = point
    theta = math.radians(angle)
    shear_x = math.radians(shear[0]) if isinstance(shear, (list, tuple)) else math.radians(shear)
    shear_y = math.radians(shear[1]) if isinstance(shear, (list, tuple)) else 0.0

    # Calculate the affine transformation matrix components
    a = scale * math.cos(theta + shear_x)
    b = -scale * math.sin(theta + shear_y)
    c = scale * math.sin(theta + shear_x)
    d = scale * math.cos(theta + shear_y)

    # Adjust translation to keep the transformation centered
    tx, ty = translate
    cx, cy = center
    tx = tx + cx - (a * cx + b * cy)
    ty = ty + cy - (c * cx + d * cy)

    # Apply the affine transformation
    x1 = a * x0 + b * y0 + tx
    y1 = c * x0 + d * y0 + ty

    return x1, y1

assert get_affine_transformed_coordinates(point=(32, 32), angle=0, translate=(10, 10), scale=1, shear=(0, 0), center=(32,32)) == (42.0, 42.0)
assert get_affine_transformed_coordinates(point=(32, 32), angle=0, translate=(-2, 2), scale=1, shear=(0, 0), center=(32,32)) == (30, 34)
assert get_affine_transformed_coordinates(point=(32, 32), angle=30, translate=(10, 10), scale=1, shear=(0, 0), center=(32,32)) == (42.0, 42.0)
assert get_affine_transformed_coordinates(point=(32, 32), angle=30, translate=(10, 10), scale=2, shear=(0, 0), center=(32,32)) == (42.0, 42.0)
assert (p:=get_affine_transformed_coordinates(point=(10, 10), angle=30, translate=(0, 0), scale=1, shear=(0, 0), center=(32,32))) == (23.947441116742347, 1.9474411167423504), f'Output: {p}'

def check_boundary_collision(next_point, img_size, digit_size=28):
    """
    Check if digit hits boundary or would cross it in next step.
    Returns collision flags and adjusted next position.
    Args:
        point: Current (x, y) coordinates
        next_point: Next predicted (x, y) coordinates
        img_size: Size of the frame
        digit_size: Size of the MNIST digit (default 28)
    """
    next_x, next_y = next_point
    half_digit = digit_size // 2

    min_bound = half_digit
    max_bound = img_size - half_digit
    # Check if next position would cross boundary and adjust it
    if next_x <= min_bound:
        next_x = min_bound + (min_bound - next_x)
        collision_x = True
    elif next_x >= max_bound:
        next_x = max_bound - (next_x - max_bound)
        collision_x = True
    else:
        collision_x = False

    if next_y <= min_bound:
        next_y = min_bound + (min_bound - next_y)
        collision_y = True
    elif next_y >= max_bound:
        next_y = max_bound - (next_y - max_bound)
        collision_y = True
    else:
        collision_y = False
    return collision_x, collision_y, (next_x, next_y)


def reflect_trajectory(translate, collision_x, collision_y):
    """Reflect trajectory based on collision"""
    tx, ty = translate
    if collision_x:
        tx = -tx
    if collision_y:
        ty = -ty
    return (tx, ty)


class RandomTrajectory:
    def __init__(self, affine_params, center, n=5, bounce=True, **kwargs):
        self.angle = random.uniform(*affine_params.angle)
        self.translate = (random.uniform(*affine_params.translate[0]),
                         random.uniform(*affine_params.translate[1]))
        self.scale = random.uniform(*affine_params.scale)
        self.shear = random.uniform(*affine_params.shear)
        self.n = n
        self.center = center
        self.bounce = bounce
        self.img_size = center[0] * 2  # Assuming square image
        self.tf = partial(TF.affine, angle=self.angle, translate=self.translate,
                         scale=self.scale, shear=self.shear, **kwargs)
        self.tcf = partial(
            get_affine_transformed_coordinates,
            angle=self.angle,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            center=self.center
        )

    def __call__(self, img, target):
        sequence_frames = [img]
        sequence_center_points = [(round(target[0]), round(target[1]))]
        for i in range(self.n):
            current_translate = self.translate
            if self.bounce:
                # Get next position without actually moving
                next_tcf = partial(
                    get_affine_transformed_coordinates,
                    angle=self.angle,
                    translate=current_translate,
                    scale=self.scale,
                    shear=self.shear,
                    center=self.center
                )
                next_x, next_y = next_tcf(sequence_center_points[-1])
                # Check for boundary collision and get adjusted position
                collision_x, collision_y, adjusted_next_pos = check_boundary_collision(
                    (next_x, next_y),
                    self.img_size,
                )

                if collision_x or collision_y:
                    # Update translation for next frame
                    self.translate = reflect_trajectory(
                        self.translate, collision_x, collision_y
                    )
                    # Calculate transformation to reach adjusted position
                    dx = adjusted_next_pos[0] - sequence_center_points[-1][0]
                    dy = adjusted_next_pos[1] - sequence_center_points[-1][1]
                    current_translate = (dx, dy)

            # Apply transformation with potentially adjusted translation
            transform_frame = partial(TF.affine, angle=self.angle, translate=current_translate,
                        scale=self.scale, shear=self.shear)
            transform_center_point = partial(
                get_affine_transformed_coordinates,
                angle=self.angle,
                translate=current_translate,
                scale=self.scale,
                shear=self.shear,
                center=self.center
            )
            sequence_frames.append(transform_frame(sequence_frames[-1]))
            x1, y1 = transform_center_point(sequence_center_points[-1])
            sequence_center_points.append((round(x1), round(y1)))
        return sequence_frames, sequence_center_points


def translate_digits_overlap_free(canvas_width, canvas_height, num_objects, digit_size=28):
    placed_positions = []

    for _ in range(num_objects):
        max_attempts = 20  # Retry limit
        min_overlap_area = 0
        min_overlap_point = None
        for _ in range(max_attempts):
            # Randomly generate a position
            x = random.randint(0, canvas_width - digit_size)
            y = random.randint(0, canvas_height - digit_size)

            overlap_area = 0

            for px, py in placed_positions:
                horizontal_overlap = max(0, min(x + digit_size, px + digit_size) - max(x, px))
                vertical_overlap = max(0, min(y + digit_size, py + digit_size) - max(y, py))
                overlap_area += horizontal_overlap * vertical_overlap

            if overlap_area == 0:
                placed_positions.append((x, y))
                break
            elif min_overlap_point is None or min_overlap_area > overlap_area:
                min_overlap_point = (x, y)
        else:
            assert min_overlap_point is not None
            placed_positions.append(min_overlap_point)

    placed_position_translations = []
    for p in placed_positions:
        x, y = p
        cx, cy = x+digit_size//2, y+digit_size//2
        tx, ty = canvas_width//2 - cx, canvas_height//2 - cy
        placed_position_translations.append((tx, ty))
    return placed_position_translations


class MovingMNIST(Dataset):
    def __init__(self, path=".",  # path to store the MNIST dataset
                 affine_params: dict=affine_params, # affine transform parameters, refer to torchvision.transforms.functional.affine
                 num_digits: list[int]=[1,2], # how many digits to move, random choice between the value provided
                 num_frames: int=4, # how many frames to create
                 img_size=64, # the canvas size, the actual digits are always 28x28
                 concat=True, # if we concat the final results (frames, 1, 28, 28) or a list of frames.
                 normalize=False, # scale images in [0,1] and normalize them with MNIST stats. Applied at batch level. Have to take care of the canvas size that messes up the stats!
                 bounce=False,  # Enable/disable bouncing
                 frame_dropout_pattern = None,
                 sequences_path = None, #TODO: REMOVE
                 split_indices=None,
                 sampler_steps=[], # epochs at which assign coresponding frame dropout probability
                 frame_dropout_probs=[], # absolut frame drop probability values
                 dataset_fraction = 1,
                 overlap_free_initial_translation=False, # Initial digits translation in overlap free way
                 ):
        self.bounce = bounce
        self.sequences = None
        if sequences_path is not None:
          data = torch.load(sequences_path)
          self.sequences = list(zip(data['imgs'], data['targets']))

          num_frames = self.sequences[0][0].shape[0]
          img_size = self.sequences[0][0].shape[2]
          print(f'Num frames: {num_frames}')
          print(f'Img size: {img_size}')

        self.num_digits = num_digits
        if self.sequences is None:
          mnist = MNIST(path, download=True)
          self.mnist_dataset = mnist.data
          self.mnist_targets = mnist.targets

          if split_indices is not None:
            self.mnist_dataset = self.mnist_dataset[split_indices]
            self.mnist_targets = self.mnist_targets[split_indices]
          self.ids = [[random.randrange(0, len(self.mnist_dataset)) for _ in range(random.choice(self.num_digits))] for _ in range(len(self.mnist_dataset))]
          self.affine_params = affine_params

        self.num_frames = num_frames
        self.img_size = img_size
        self.pad = padding(img_size)
        self.concat = concat
        self.overlap_free_initial_translation = overlap_free_initial_translation

        self.keep_frame_mask = None
        self.frame_dropout_prob = 0.0
        self.dataset_fraction = dataset_fraction

        self.sampler_steps = sampler_steps
        self.frame_dropout_probs = frame_dropout_probs
        print("sampler_steps={} frame_dropout_probs={}".format(self.sampler_steps, self.frame_dropout_probs))

        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.frame_dropout_probs) > 0
            assert len(self.frame_dropout_probs) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.period_idx = 0
            self.frame_dropout_prob = self.frame_dropout_probs[0]
            self.current_epoch = 0

        if frame_dropout_pattern is not None:
          print('Disable probability based frame drops. Use frame drops based on fixed mask.')
          self.frame_dropout_prob = None
          self.sampler_steps = []
          drop_frame_mask = torch.tensor([int(char) for char in frame_dropout_pattern])
          self.keep_frame_mask = 1 - drop_frame_mask
          assert self.keep_frame_mask.size(0) == self.num_frames, f"Frame dropout pattern must have the same length as the number of frames. Num of frames {self.num_frames} and mask size {self.keep_frame_mask.size(0)}"
          print(f'Set frame keep mask: {self.keep_frame_mask}')

        # some computation to ensure normalizing correctly-ish
        batch_tfms = [T.ConvertImageDtype(torch.float32)]
        if normalize:
            # Default value on dataset with variable number of digits
            mean, std = mnist_stats.get(tuple(num_digits), ([0.0321], [0.1631]))
            print(f"New computed stats for MovingMNIST: {(mean, std)}")
            batch_tfms += [T.Normalize(mean=mean, std=std)] if normalize else []
        self.batch_tfms = T.Compose(batch_tfms)


    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.frame_dropout_probs is None or len(self.frame_dropout_probs) == 0:
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.frame_dropout_prob = self.frame_dropout_probs[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

        if self.dataset_fraction < 1:
            random.shuffle(self.ids)

    def random_place(self, img, initial_translation):
        "Randomly place the digit inside the canvas"
        x, y = initial_translation

        center_point = get_affine_transformed_coordinates(
            (self.img_size//2, self.img_size//2),
            translate=(x,y),
            center=(self.img_size//2, self.img_size//2),
        )
        return TF.affine(img, translate=(x,y), angle=0, scale=1, shear=(0,0)), center_point

    def translate_digit(self, digit_idx, initial_translation):
        "Get a MNIST digit randomly placed on the canvas"
        img = self.mnist_dataset[[digit_idx]]
        pimg = TF.pad(img, padding=self.pad)
        img, center_point = self.random_place(pimg, initial_translation)
        target = {
            'label': int(self.mnist_targets[digit_idx]),
            'center_point': center_point
        }
        return img, target

    def _one_moving_digit(self, _id, initial_translation):
        digit, target = self.translate_digit(digit_idx=_id, initial_translation=initial_translation)
        traj = RandomTrajectory(
            self.affine_params, center=(self.img_size//2, self.img_size//2),
            n=self.num_frames-1,
            bounce=self.bounce,
        )
        sequence, points = traj(digit, target['center_point'])

        targets = []
        for point in points:
            targets.append({
                'label': target['label'],
                'center_point': point
            })

        return torch.stack(sequence), targets

    def generate_sequence(self, idx):
      ids = self.ids[idx]
      initial_digit_translations = []
      if self.overlap_free_initial_translation:
          initial_digit_translations = translate_digits_overlap_free(self.img_size, self.img_size, len(ids))
      else:
          for _ in ids:
              initial_digit_translations.append(
                  (random.uniform(-self.pad, self.pad), random.uniform(-self.pad, self.pad)) #dx, dy
              )

      moving_digits_and_targets = [self._one_moving_digit(_id, initial_digit_translations[i]) for i, _id in enumerate(ids)]
      moving_digits = torch.stack([d[0] for d in moving_digits_and_targets])

      combined_digits = moving_digits.max(dim=0)[0]
      targets = []

      for frame_number in range(self.num_frames):
          target = {}

          labels = []
          center_points = []

          for i in range(len(self.ids[idx])):
              digit_target = moving_digits_and_targets[i][1][frame_number]

              if 'label' in digit_target and digit_target['center_point'][0] >= 0 and digit_target['center_point'][1] >= 0 and digit_target['center_point'][0] < self.img_size and digit_target['center_point'][1] < self.img_size:
                labels.append(digit_target['label'])
                center_points.append(digit_target['center_point'])

          target['labels'] = torch.tensor(labels, dtype=torch.int64)
          target['center_points'] = torch.tensor(center_points, dtype=torch.float32)
          targets.append(target)

      return combined_digits, targets


    def __getitem__(self, idx):
      if self.sequences is None:
          images, targets = self.generate_sequence(idx)
      else:
          images, targets = self.sequences[idx]

      targets = copy.deepcopy(targets) # targets dictionary is mutable

      if self.keep_frame_mask is not None:
        keep_frame_flags = self.keep_frame_mask
      else:
        num_potential_drop_frames = self.num_frames // 2
        frame_keep_probs = torch.rand(num_potential_drop_frames)
        keep_frame_flags = (frame_keep_probs > self.frame_dropout_prob).int()
        keep_frame_flags = torch.cat([torch.ones(self.num_frames - num_potential_drop_frames), keep_frame_flags])

      for frame_number, (img, target) in enumerate(zip(images, targets)):
        if target['center_points'].size(0) > 0:
          target['center_points'] /= torch.tensor([self.img_size, self.img_size], dtype=torch.float32)

          target['labels'] = target['labels'].to(torch.int64)

        target['keep_frame'] = keep_frame_flags[frame_number]
        target['orig_size'] = torch.as_tensor([int(self.img_size), int(self.img_size)])

      images = self.batch_tfms(images)

      return images, targets

    def __len__(self):
        if self.sequences is None:
          return int(len(self.ids) * self.dataset_fraction)
        else:
          return len(self.sequences)