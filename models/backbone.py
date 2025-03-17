import torch.nn as nn


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


class BackboneCnnForLstm(nn.Module):
    def __init__(self, input_image_view_size):
        super(BackboneCnnForLstm, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.num_channels = 64
        h, w = input_image_view_size
        self.output_size = self._output_size(h, w)

    # Keep get_output_size for cases where we need to calculate size later
    def _output_size(self, input_h, input_w):
        return input_h // 2**4, input_w // 2**4

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

    if args.backbone == 'cnn' and args.model == 'lstm':
        return BackboneCnnForLstm(input_image_view_size)

    backbone = BackboneCnn(input_image_view_size) if args.backbone == 'cnn' else BackboneIdentity()
    return backbone
