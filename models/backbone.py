import torch.nn as nn


class BackboneCnn(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.block1(x)
        return x


class BackboneIdentity(nn.Module):
    def __init__(self):
        super(BackboneIdentity, self).__init__()
        self.num_channels = 1

    def forward(self, x):
        return x

def build_backbone(args):
    if args.backbone == 'cnn':
        return BackboneCnn()
    else:
        return BackboneIdentity()
