"""
CNN model for stress detection
"""
import torch
import torch.nn as nn


class StressCNN(nn.Module):
    """
    1D CNN for stress detection from physiological signals.
    """

    def __init__(self, input_channels=8, num_classes=2, sequence_length=128):
        """
        Parameters:
        -----------
        input_channels : int
            Number of input signal channels
        num_classes : int
            Number of output classes
        sequence_length : int
            Length of input sequence
        """
        super(StressCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate size after convolutions and pooling
        self.feature_size = 256 * (sequence_length // 8)

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
