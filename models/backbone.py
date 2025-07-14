import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Add YOLO backbone import (adjust as needed for your YOLO implementation)
try:
    from ultralytics.nn.tasks import DetectionModel as YOLOBackbone
except ImportError:
    YOLOBackbone = None  # Fallback if ultralytics is not installed


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool,
                 layers_used: set = None):
        super().__init__()
        if layers_used is None:
            layers_used = {'layer1', 'layer2', 'layer3', 'layer4'}
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(layer in name for layer in layers_used):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        for name, x in xs.items():
            return x

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 input_image_view_size: tuple[int, int],
                 dilation: bool = False,
                 layers_used: set[str] = None):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, layers_used)
        h, w = input_image_view_size
        self.output_size = (h // 32, w // 32)

    def forward(self, tensor):
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
        return super().forward(tensor)


class BackboneCnn(nn.Module):
    def __init__(self, input_image_view_size):
        super(BackboneCnn, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.num_channels = 32
        h, w = input_image_view_size
        self.output_size = self._output_size(h, w)

    # Keep get_output_size for cases where we need to calculate size later
    def _output_size(self, input_h, input_w):
        return input_h // 2, input_w // 2

    def forward(self, x):
        x = self.block1(x)
        return x


class BackboneCnnV2(nn.Module):

    def __init__(self):
        """
        Initializes the Backbone CNN.

        Args:
            input_image_view_size (tuple): A tuple (height, width) representing
                                           the expected input image dimensions.
        """
        super(BackboneCnnV2, self).__init__()

        # Block 1: Increase 1 -> 16, first downsampling (x2) << MODIFIED >>
        self.block1 = nn.Sequential(
            # Input: [N, 1, H, W]
            # Adjusted intermediate channels for smoother transition potentially
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: [N, 16, H/2, W/2]
            nn.ReLU(),
        )

        # Block 2: Increase 16 -> 32, second downsampling (x4) << MODIFIED >>
        self.block2 = nn.Sequential(
            # Input: [N, 16, H/2, W/2]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Output: [N, 32, H/4, W/4]
            nn.ReLU(),
        )

        # Block 3: Increase 32 -> 64, third downsampling (x8) << MODIFIED >>
        self.block3 = nn.Sequential(
            # Input: [N, 32, H/4, W/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: [N, 64, H/8, W/8]
            nn.ReLU(),
        )

        # Block 4: Increase 64 -> 128, fourth downsampling (x16) << MODIFIED >>
        self.block4 = nn.Sequential(
            # Input: [N, 64, H/8, W/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Output: [N, 128, H/16, W/16]
            nn.ReLU(),
        )

        # Set the final number of channels explicitly << MODIFIED >>
        self.num_channels = 128

        # Calculate the final output spatial size based on input size and downsampling
        h, w = (0, 0)  # Placeholder values
        self.output_size = self._output_size(h, w)

    def _output_size(self, input_h, input_w):
        """
				Calculates the output spatial dimensions after passing through all blocks.
				Total downsampling factor is 16 (2x from each of the 4 blocks).
				"""
        # Downsampled by 2 in block1, block2, block3, and block4
        return input_h // 16, input_w // 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
				Defines the forward pass of the CNN.

				Args:
						x (torch.Tensor): Input tensor of shape [N, 1, H, W].

				Returns:
						torch.Tensor: Output feature map tensor of shape [N, 128, H/16, W/16].
				"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class SimpleVGGBackbone(nn.Module):
    """
    A simple VGG-style backbone suitable for larger grayscale images like 320x320
    with moderately sized objects (e.g., 70x70 digits).
    Uses BatchNorm and MaxPool for downsampling.
    Outputs features with a total stride of 16.
    """
    def __init__(self, input_channels=1):
        super().__init__()
        # Input size reference: 320x320

        self.features = nn.Sequential(
            # Block 1: 320x320 -> 160x160 (Stride 2)
            # Output channels: 32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 160x160 -> 80x80 (Stride 4)
            # Output channels: 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 80x80 -> 40x40 (Stride 8)
            # Output channels: 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 40x40 -> 20x20 (Stride 16)
            # Output channels: 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Adding a couple more conv layers here for more depth at this stage
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Optional extra layer
            nn.BatchNorm2d(256),                          # Optional extra layer
            nn.ReLU(inplace=True),                        # Optional extra layer
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # The number of channels output by the last layer of self.features
        self.num_channels = 256
        # The output feature map size will be input_size / 16
        # For 320x320 input, this gives 20x20 features.

    def forward(self, x):
        # Input x shape: (Batch, Channels, Height, Width) e.g., (B, 1, 320, 320)
        x = self.features(x)
        # Output x shape: (Batch, self.num_channels, Height/16, Width/16) e.g., (B, 256, 20, 20)
        return x


class BackboneIdentity(nn.Module):
    def __init__(self):
        super(BackboneIdentity, self).__init__()
        self.num_channels = 1

    def forward(self, x):
        return x

# --- YOLO backbone feature map hook support ---
FEATURE_MAPS = None  # Global variable to store feature maps from the hook

def yolo_feature_hook(module, input, output):
    global FEATURE_MAPS
    FEATURE_MAPS = output

class YOLOBackboneWrapper(nn.Module):
    """
    Wrapper for Ultralytics YOLOv8 backbone for feature extraction using a forward hook.
    Thread/process safe: feature maps are stored as an instance attribute.
    """
    def __init__(self, model_path='yolov8n.pt', input_image_view_size=(320, 320), target_layer_index=None):
        super().__init__()
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed.")
        self.yolo = YOLO(model_path)
        if target_layer_index is None:
            target_layer_index = len(self.yolo.model.model) - 2
        self.target_layer_index = target_layer_index
        self.feature_maps = None  # Instance attribute for thread/process safety
        self._register_hook()
        self.num_channels = 256
        h, w = input_image_view_size
        self.output_size = (h // 32, w // 32)
        self.set_yolo_requires_grad()  # <-- Set requires_grad as requested

    def set_yolo_requires_grad(self):
        """
        Sets requires_grad=True for all parameters with index <= self.target_layer_index,
        and requires_grad=False for all others.
        """
        for idx, layer in enumerate(self.yolo.model.model):
            requires_grad = idx <= self.target_layer_index
            for param in layer.parameters(recurse=True):
                param.requires_grad = requires_grad

    def _feature_hook(self, module, input, output):
        self.feature_maps = output

    def _register_hook(self):
        target_layer = self.yolo.model.model[self.target_layer_index]
        target_layer.register_forward_hook(self._feature_hook)

    def train(self, mode=True):
        self.yolo.model.train()
        return self

    def forward(self, x):
        self.feature_maps = None  # Reset before forward
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        _ = self.yolo.model(x)
        out = self.feature_maps
        if out is None:
            raise RuntimeError("Feature maps not captured. Check target_layer_index and hook registration.")
        return out

def build_backbone(args, input_image_view_size):
    if args.backbone == 'yolo':
        if YOLO is None:
            raise ImportError("YOLO backbone requested but ultralytics is not installed.")
        print("Using YOLOv8 backbone for feature extraction")
        model_path = getattr(args, 'yolo_model_path', 'yolov8n.pt')
        # Allow target_layer_index to be set via args, default to second-to-last
        target_layer_index = getattr(args, 'yolo_target_layer_index', None)
        return YOLOBackboneWrapper(model_path=model_path, input_image_view_size=input_image_view_size,
                                   target_layer_index=21)

    if 'resnet' in args.backbone:
        return Backbone(args.backbone, train_backbone=True, return_interm_layers=False, input_image_view_size=input_image_view_size)

    if 'vgg' in args.backbone:
        print("Using VGG backbone")
        return SimpleVGGBackbone(input_channels=1)

    if args.backbone == 'cnn' and args.resize_frame:
        print(f"Using CNN v2 backbone with resized input size {args.resize_frame}x{args.resize_frame}")
        return BackboneCnnV2()

    backbone = BackboneCnn(input_image_view_size) if args.backbone == 'cnn' else BackboneIdentity()
    return backbone
