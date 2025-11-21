"""
LSTM model for stress detection
"""
import torch
import torch.nn as nn


class StressLSTM(nn.Module):
    """
    LSTM model for stress detection from physiological signals.
    """

    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=2):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features per timestep
        hidden_size : int
            Number of LSTM hidden units
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of output classes
        """
        super(StressLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last timestep
        out = out[:, -1, :]

        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
