from typing import List

import torch.nn as nn
import torch

class TemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [128, 64], dropout: float = 0.5):
        '''
        A simple MLP stack to serve as a starting point for our post-processing temporal network.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super(TemporalNet, self).__init__()
        
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
        Performs a forward pass through the TNet architecture.

        Input:
            x: a Tensor containing temporal data.

        Output:
            x: a Tensor containing the output of the final linear layer.
        '''

        for layer in self.layers:
            # residual = x[:]
            x = layer(x)

        return x