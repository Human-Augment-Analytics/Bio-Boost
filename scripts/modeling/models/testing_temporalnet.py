from typing import List

import torch
import torch.nn as nn


import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class TemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], dropout: float = 0.5):
        '''
        A simple MLP stack to serve as a starting point for our post-processing temporal network.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super(TemporalNet, self).__init__()
        self.__version__ = '0.0.1'

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        layers = []
        prev_dim = input_size

        for idx, hidden_dim in enumerate(hidden_sizes):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh()
            ]

            if idx == len(hidden_sizes):
                layers.append(nn.Dropout(p=dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass of the model defined in __init__.

        Input:
            x: a Tensor containing temporal data.

        Output:
            x: a Tensor containing the output of the final linear activation layer, after running the input x through the entire model.
        '''

        for layer in self.layers:
            x = layer(x)

        return x

# Read training data as df
training = pd.read_csv("train.csv")
training = training.drop(columns=["track_id"])
training = training.astype("float32")

# Read testing data as df
test = pd.read_csv("test.csv")
test = test.drop(columns=["track_id"])
test = test.astype("float32")

# Separate true labels from train/test
features = training.iloc[:, 0:-1].to_numpy()
labels = training.iloc[:, -1].to_numpy()
test_features = test.iloc[:, 0:-1].to_numpy()
test_labels = test.iloc[:, -1].to_numpy()

# Split training into train/val
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_features = scaler.transform(test_features)

# Recreating Charlie's initial model
model = TemporalNet(input_size=4, output_size=2, hidden_sizes=[64, 32, 16])


# Convert data to PyTorch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
train_labels = torch.tensor(y_train, dtype=torch.long)
val_features = torch.tensor(X_val, dtype=torch.float32)
val_labels = torch.tensor(y_val, dtype=torch.long)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_features, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}')

# Test evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Compute metrics
overall_accuracy = accuracy_score(all_labels, all_preds)
class_precision = precision_score(all_labels, all_preds, average=None)
class_recall = recall_score(all_labels, all_preds, average=None)
cm = confusion_matrix(all_labels, all_preds)
class_accuracy = cm.diagonal() / cm.sum(axis=1)

print(f'Overall Accuracy: {overall_accuracy:.4f}')
for i in range(len(class_accuracy)):
    print(f'Class {i} Accuracy: {class_accuracy[i]:.4f}\nClass {i} Precision: {class_precision[i]:.4f}\nClass {i} Recall: {class_recall[i]:.4f}')