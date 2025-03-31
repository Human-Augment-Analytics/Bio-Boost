from typing import List

import numpy as np
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
                nn.Linear(prev_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.GLU(dim=1)
            ]

            if idx == len(hidden_sizes) - 1:
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
            # residual = x[:]
            x = layer(x)
        return x
    
# --- Configuration ---
BEST_WEIGHTS_PATH = "best_temporalNet_weights.pth"
TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "test.csv"
OUTPUT_CSV_PATH = "temporalNet_predictions_val.csv"

# Model parameters (match model Charlie and I configured)
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZES = [128, 64]
DROPOUT = 0.5

# --- Data Loading and Preprocessing ---
print("Loading data...")
try:
    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    track_ids = train_df["track_id"].copy()
    train_df = train_df.drop(columns=["track_id"])
    train_df = train_df.astype("float32")
    train_features = train_df.iloc[:, 0:-1].to_numpy()

    # Generate same test/val split used to train model
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        train_df.iloc[:, 0:-1], train_df.iloc[:, -1], test_size=0.1, random_state=42
    )
    val_track_ids = track_ids.iloc[X_val_df.index].copy()
    X_train = X_train_df.to_numpy()
    X_val = X_val_df.to_numpy()
    y_train = y_train.to_numpy(dtype=int)
    y_val = y_val.to_numpy(dtype=int)

    # Load test data (keeping track_id this time)
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_track_ids = test_df["track_id"].copy()
    test_df_processed = test_df.drop(columns=["track_id"])
    test_df_processed = test_df_processed.astype("float32")
    test_features = test_df_processed.iloc[:, 0:-1].to_numpy()
    test_ground_truth = test_df_processed.iloc[:, -1].to_numpy(dtype=int)

except FileNotFoundError as e:
    print(f"Error: Could not find required CSV file: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


print("Preprocessing data...")
scaler = StandardScaler()
scaler.fit(train_features)

# Transform test & val features using the fitted scaler
test_features_scaled = scaler.transform(test_features)
val_features_scaled = scaler.transform(X_val)

# Convert test & val features to PyTorch tensor
test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features_scaled, dtype=torch.float32)

# Load best weights with correct configuration
print(f"Loading model weights from {BEST_WEIGHTS_PATH}...")
model = TemporalNet(input_size=4, output_size=2, hidden_sizes=[128, 64], dropout=0.5)
try:
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH))
except FileNotFoundError:
    print(f"Error: Weights file not found at '{BEST_WEIGHTS_PATH}'.")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# Set the model to evaluation mode
model.eval()

# --- Prediction ---
print("Generating predictions...")
all_preds = []
with torch.no_grad():
    # Create a DataLoader for batch processing
    test_dataset = TensorDataset(val_features_tensor) # swap with test_features_tensor to test on testing dataset
    test_loader = DataLoader(test_dataset, batch_size=64)

    for (inputs,) in test_loader:
        outputs = model(inputs)
        _, predicted_indices = torch.max(outputs, 1)
        all_preds.extend(predicted_indices.numpy())

# Ensure the number of predictions matches the number of samples
if len(all_preds) != len(val_track_ids): # swap with test_features_tensor to test on testing dataset
    print(f"Number of predictions ({len(all_preds)}) does not match number of track IDs ({len(val_track_ids)}).")
    exit()


# --- Output Results ---
print(f"Saving predictions to {OUTPUT_CSV_PATH}...")
results_df = pd.DataFrame({
    'track_id': val_track_ids, # swap with test_track_ids to test on testing dataset
    'prediction': all_preds,
    'ground_truth': y_val # swap with test_ground_truth to test on testing dataset
})

# Save the DataFrame to a CSV file
try:
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Successfully saved predictions.")
except Exception as e:
    print(f"Error saving predictions to CSV: {e}")