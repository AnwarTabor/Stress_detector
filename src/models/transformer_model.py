"""
Transformer model for stress detection
"""
import torch
import torch.nn as nn
import math


class StressTransformer(nn.Module):
    """
    Transformer model for stress detection from physiological signals.
    """

    def __init__(self, input_size=8, d_model=128, nhead=8, num_layers=3,
                 num_classes=2, sequence_length=128, dropout=0.1):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features per timestep
        d_model : int
            Dimension of the model
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        num_classes : int
            Number of output classes
        sequence_length : int
            Length of input sequence
        dropout : float
            Dropout rate
        """
        super(StressTransformer, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)

        # Project input to d_model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
