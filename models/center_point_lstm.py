import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCenterNetWithLSTM(nn.Module):
    def __init__(self, num_objects=5, num_classes=10, lstm_hidden_size=64):
        super(SimpleCenterNetWithLSTM, self).__init__()
        self.num_objects = num_objects
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for feature extraction
        self.fc1 = nn.Linear(64 * 8 * 8, 128)

        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,  # Input size from fc1 layer
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=False
        )

        self.hidden_h0 = nn.Parameter(torch.randn(1, 1, self.lstm_hidden_size))
        self.hidden_c0 = nn.Parameter(torch.randn(1, 1, self.lstm_hidden_size))

        # Output layers after LSTM
        self.fc_temporal = nn.Linear(lstm_hidden_size, 128)

        # Output layers for center points and class scores
        self.fc_center = nn.Linear(128, 2 * num_objects)  # Predicts (x, y) for each object
        self.fc_class = nn.Linear(128, num_objects * num_classes)  # Predicts class scores for each object

    def forward(self, samples, targets):

        samples = samples.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        out_logits = []
        out_center_points = []

        # Initialize LSTM hidden state
        batch_size = samples.size(1)
        h0 = self.hidden_h0.expand(1, batch_size, self.lstm_hidden_size).contiguous() # [num_layers, batch_size, hidden_size]
        c0 = self.hidden_c0.expand(1, batch_size, self.lstm_hidden_size).contiguous() # [num_layers, batch_size, hidden_size]

        # Temporal feature extraction
        temporal_features = []

        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(samples):
            keep_frame = targets[timestamp]['keep_frame'].item()

            if keep_frame:
                # Forward pass through conv layers
                x = F.relu(self.conv1(batch))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv3(x))
                x = F.max_pool2d(x, 2)

                # Flatten and apply fully connected layer
                x = x.view(x.size(0), -1)  # Flatten

                x = F.relu(self.fc1(x))

            else:
                x = torch.zeros(1, 128).to(samples.device)

            temporal_features.append(x)

        # Stack temporal features
        temporal_features = torch.stack(temporal_features)

        # Pass through LSTM
        lstm_out, _ = self.lstm(temporal_features, (h0, c0))

        # Process LSTM output for each time step
        for t in range(lstm_out.size(0)):
            # Apply temporal feature transformation
            x = F.relu(self.fc_temporal(lstm_out[t]))

            # Predict centers
            center_output = self.fc_center(x)  # Output shape: (batch_size, 2 * num_objects)
            center_output = center_output.view(-1, self.num_objects, 2)  # Reshape to (batch_size, num_objects, 2)

            # Predict class scores
            class_output = self.fc_class(x)  # Output shape: (batch_size, num_objects * num_classes)
            class_output = class_output.view(-1, self.num_objects, self.num_classes)  # Reshape to (batch_size, num_objects, num_classes)

            out_logits.append(class_output)
            out_center_points.append(center_output)

        return {
            'pred_logits': torch.cat(out_logits),
            'pred_center_points': torch.cat(out_center_points)
        }, targets
