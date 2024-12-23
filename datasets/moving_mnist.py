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

mnist_stats    = ([0.131], [0.308])

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

def check_boundary_collision(point, next_point, img_size, digit_size=28, use_center=True):
    """
    Check if digit hits boundary or would cross it in next step.
    Returns collision flags and adjusted next position.
    Args:
        point: Current (x, y) coordinates
        next_point: Next predicted (x, y) coordinates
        img_size: Size of the frame
        digit_size: Size of the MNIST digit (default 28)
        use_center: If True, bounce when center hits boundary. If False, bounce when digit boundary hits
    """
    x, y = point
    next_x, next_y = next_point
    half_digit = digit_size // 2
    # Determine boundaries based on configuration
    if use_center:
        min_bound = 0
        max_bound = img_size
    else:
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
    def __init__(self, affine_params, center, n=5, bounce=True, use_center_bounce=True, **kwargs):
        self.angle = random.uniform(*affine_params.angle)
        self.translate = (random.uniform(*affine_params.translate[0]),
                         random.uniform(*affine_params.translate[1]))
        self.scale = random.uniform(*affine_params.scale)
        self.shear = random.uniform(*affine_params.shear)
        self.n = n
        self.center = center
        self.bounce = bounce
        self.use_center_bounce = use_center_bounce
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
        sequence = [img]
        sequence_y = [(round(target[0]), round(target[1]))]
        current_translate = self.translate
        for _ in range(self.n):
            # Calculate next position before applying transformation
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
                next_x, next_y = next_tcf(sequence_y[-1])
                # Check for boundary collision and get adjusted position
                collision_x, collision_y, adjusted_next_pos = check_boundary_collision(
                    sequence_y[-1],
                    (next_x, next_y),
                    self.img_size,
                    use_center=self.use_center_bounce
                )
                if collision_x or collision_y:
                    # Update translation for next frame
                    current_translate = reflect_trajectory(
                        current_translate, collision_x, collision_y
                    )
                    # Calculate transformation to reach adjusted position
                    dx = adjusted_next_pos[0] - sequence_y[-1][0]
                    dy = adjusted_next_pos[1] - sequence_y[-1][1]
                    current_translate = (dx, dy)
            # Apply transformation with potentially adjusted translation
            tf = partial(TF.affine, angle=self.angle, translate=current_translate,
                        scale=self.scale, shear=self.shear)
            tcf = partial(
                get_affine_transformed_coordinates,
                angle=self.angle,
                translate=current_translate,
                scale=self.scale,
                shear=self.shear,
                center=self.center
            )
            sequence.append(tf(sequence[-1]))
            x1, y1 = tcf(sequence_y[-1])
            sequence_y.append((round(x1), round(y1)))
        return sequence, sequence_y


class MovingMNIST(Dataset):
    def __init__(self, path=".",  # path to store the MNIST dataset
                 affine_params: dict=affine_params, # affine transform parameters, refer to torchvision.transforms.functional.affine
                 num_digits: list[int]=[1,2], # how many digits to move, random choice between the value provided
                 num_frames: int=4, # how many frames to create
                 img_size=64, # the canvas size, the actual digits are always 28x28
                 concat=True, # if we concat the final results (frames, 1, 28, 28) or a list of frames.
                 normalize=False, # scale images in [0,1] and normalize them with MNIST stats. Applied at batch level. Have to take care of the canvas size that messes up the stats!
                 bounce=True,  # Enable/disable bouncing
                 use_center_bounce=True,  # Use center or boundary for bounce detection
                 frame_dropout_pattern = None,
                 sequences_path = None, #TODO: REMOVE
                 split_indices=None,
                 sampler_steps=[], # epochs at which assign coresponding frame dropout probability
                 frame_dropout_probs=[], # absolut frame drop probability values
                 ):
        self.bounce = bounce
        self.use_center_bounce = use_center_bounce
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

        self.keep_frame_mask = None
        self.frame_dropout_prob = 0.0

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
            ratio = (28/img_size)**2*max(num_digits)
            mean, std = mnist_stats
            scaled_mnist_stats = ([mean[0]*ratio], [std[0]*ratio])
            print(f"New computed stats for MovingMNIST: {scaled_mnist_stats}")
            batch_tfms += [T.Normalize(*scaled_mnist_stats)] if normalize else []
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

    def random_place(self, img):
        "Randomly place the digit inside the canvas"
        x = random.uniform(-self.pad, self.pad)
        y = random.uniform(-self.pad, self.pad)

        center_point = get_affine_transformed_coordinates(
            (self.img_size//2, self.img_size//2),
            translate=(x,y),
            center=(self.img_size//2, self.img_size//2),
        )
        return TF.affine(img, translate=(x,y), angle=0, scale=1, shear=(0,0)), center_point

    def get_digit(self, digit_idx):
        "Get a MNIST digit randomly placed on the canvas"
        img = self.mnist_dataset[[digit_idx]]
        pimg = TF.pad(img, padding=self.pad)
        img, center_point = self.random_place(pimg)
        target = {
            'label': int(self.mnist_targets[digit_idx]),
            'center_point': center_point
        }
        return img, target

    def _one_moving_digit(self, id):
        digit, target = self.get_digit(digit_idx=id)
        traj = RandomTrajectory(
            self.affine_params, center=(self.img_size//2, self.img_size//2),
            n=self.num_frames-1,
            bounce=self.bounce,
            use_center_bounce=self.use_center_bounce,
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
      moving_digits_and_targets = [self._one_moving_digit(id) for id in self.ids[idx]]
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
          return len(self.ids)
        else:
          return len(self.sequences)

    def save(self, fname="mmnist.pt"):
        data_imgs = []
        data_targets = []
        for i in progress_bar(range(len(self.ids))):
            imgs, targets = self.generate_sequence(i)
            data_imgs.append(imgs)

            data_targets.append(targets)

        print("Saving dataset")
        torch.save(
            {
                'imgs': torch.stack(data_imgs),
                'targets': data_targets,
            },
            f"{fname}"
            )
