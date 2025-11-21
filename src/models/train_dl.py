"""
Training script for deep learning models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_deep_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """
    Train a deep learning model.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : str
        Device to train on ('cpu' or 'cuda')

    Returns:
    --------
    dict : Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100. * train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(100. * val_correct / val_total)

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {history["train_loss"][-1]:.4f}, '
              f'Train Acc: {history["train_acc"][-1]:.2f}%, '
              f'Val Loss: {history["val_loss"][-1]:.4f}, '
              f'Val Acc: {history["val_acc"][-1]:.2f}%')

    return history


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch data loaders.

    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data and labels
    X_val, y_val : np.ndarray
        Validation data and labels
    batch_size : int
        Batch size

    Returns:
    --------
    tuple : (train_loader, val_loader)
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
