import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_lstm import ConvLSTM


class SimpleCenterNetWithLSTM(nn.Module):
    def __init__(self, num_classes=10, lstm_hidden_size=64, img_size=128):
        super(SimpleCenterNetWithLSTM, self).__init__()
        self.num_classes = num_classes + 1 # Account for background class
        self.lstm_hidden_size = lstm_hidden_size
        self.img_size = img_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # ConvLSTM layer
        self.convlstm = ConvLSTM(
            input_dim=64,
            hidden_dim=lstm_hidden_size,
            kernel_size=(3,3),
            num_layers=1
        )

        # Output layers for center points and class scores
        self.fc_center = nn.Linear(lstm_hidden_size, 2)  # Predicts (x, y) for each object
        self.fc_class = nn.Linear(lstm_hidden_size, self.num_classes)  # Predicts class scores for each object

    def forward(self, samples, targets):

        samples = samples.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        out_logits = []
        out_center_points = []

        # Temporal feature extraction
        temporal_features = []

        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(samples):
            keep_frame = targets[timestamp]['keep_frame'].bool().item()

            if keep_frame:
                # Forward pass through conv layers
                x = F.relu(self.conv1(batch))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            else:
                x = torch.zeros(1, 64, self.img_size//4, self.img_size//4).to(samples.device)

            temporal_features.append(x)

        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=0)  # TBCHW
        lstm_out_list, _ = self.convlstm(temporal_features) # List of layers outputs

        lstm_out = lstm_out_list[0]  # BTCHW

        for t in range(lstm_out.size(1)):
            x = lstm_out[:, t]  # BCHW

            B, C, H, W = x.shape
            x = x.view(B, C, H * W)  # B,C,H*W
            x = x.permute(0, 2, 1)  # B,H*W,C

            class_output = self.fc_class(x)  # B,H*W,num_classes
            center_output = torch.sigmoid(self.fc_center(x))  # B,H*W,2

            out_logits.append(class_output) # [BQC]
            out_center_points.append(center_output)

        return {
            'pred_logits': torch.cat(out_logits), #TQC
            'pred_center_points': torch.cat(out_center_points) #TQ2
        }, targets
