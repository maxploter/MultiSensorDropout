import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_lstm import ConvLSTM


class SimpleCenterNetWithLSTM(nn.Module):
    def __init__(self, num_objects=5, num_classes=10, lstm_hidden_size=64, img_size=128):
        super(SimpleCenterNetWithLSTM, self).__init__()
        self.num_objects = num_objects
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

        self.hidden_h0 = nn.Parameter(torch.randn(1, 1, self.lstm_hidden_size))
        self.hidden_c0 = nn.Parameter(torch.randn(1, 1, self.lstm_hidden_size))

        # Output layers after LSTM
        self.fc_temporal = nn.Linear(lstm_hidden_size*32*32, 128)

        # Output layers for center points and class scores
        self.fc_center = nn.Linear(128, 2 * self.num_objects)  # Predicts (x, y) for each object
        self.fc_class = nn.Linear(128, self.num_objects * self.num_classes)  # Predicts class scores for each object

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
            x = x.view(x.size(0), -1) # BC*H*W

            x = F.relu(self.fc_temporal(x))
            center_output = self.fc_center(x).view(-1, self.num_objects, 2)
            class_output = self.fc_class(x).view(-1, self.num_objects, self.num_classes)

            out_logits.append(class_output) # [BQC]
            out_center_points.append(center_output)

        return {
            'pred_logits': torch.cat(out_logits), #TQC
            'pred_center_points': torch.cat(out_center_points) #TQ2
        }, targets
