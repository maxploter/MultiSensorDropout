import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d


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

class BackboneIdentity(nn.Module):
    def __init__(self):
        super(BackboneIdentity, self).__init__()
        self.num_channels = 1

    def forward(self, x):
        return x

def build_backbone(args, input_image_view_size):
    if 'resnet' in args.backbone:
        return Backbone(args.backbone, train_backbone=True, return_interm_layers=False, input_image_view_size=input_image_view_size)

    backbone = BackboneCnn(input_image_view_size) if args.backbone == 'cnn' else BackboneIdentity()
    return backbone
