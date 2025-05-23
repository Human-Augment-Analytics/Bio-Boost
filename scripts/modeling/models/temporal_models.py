from typing import List

import torch.nn as nn
import torch

class OGTemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        '''
        A simple MLP stack to serve as a starting point for our post-processing temporal network.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super().__init__()
        self.__version__ = '0.0.1'

        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

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
    
class TemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], dropout: float = 0.5):
        '''
        A simple MLP stack to serve as a starting point for our post-processing temporal network.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
            dropout: dropout probability to be used for regularization.
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

            # x = residual + x if residual.shape == x.shape else x

        return x



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

class SigmoidTemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        '''
        An MLP stack with sigmoid activation on the output.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super(SigmoidTemporalNet, self).__init__()
        self.__version__ = '0.0.1'

        self.model = TemporalNet(input_size, output_size, hidden_sizes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass of the model defined in __init__.

        Input:
            x: a Tensor containing temporal data.

        Output:
            x: a Tensor containing the output logits of the final sigmoid activation layer, after running the input x through the entire model.
        '''

        x = self.model(x)
        x = self.sigmoid(x)

        return x

class TanhTemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        '''
        An MLP stack with tanh activation on the output.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super(TanhTemporalNet, self).__init__()
        self.__version__ = '0.0.1'

        self.model = TemporalNet(input_size, output_size, hidden_sizes)
        self.tanh = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass of the model defined in __init__.

        Input:
            x: a Tensor containing temporal data.

        Output:
            x: a Tensor containing the output of the final tanh activation layer, after running the input x through the entire model.
        '''

        x = self.model(x)
        x = self.tanh(x)

        return x

class ReluTemporalNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        '''
        An MLP stack with ReLU activation on the output.

        Inputs:
            input_size: the size of the first input layer.
            output_size: the size of the final output layer.
            hidden_sizes: a list of sizes for the intermediate hidden layers.
        '''

        super(ReluTemporalNet, self).__init__()
        self.__version__ = '0.0.1'

        self.model = TemporalNet(input_size, output_size, hidden_sizes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass of the model defined in __init__.

        Input:
            x: a Tensor containing temporal data.

        Output:
            x: a Tensor containing the output of the final ReLU activation layer, after running the input x through the entire model.
        '''

        x = self.model(x)
        x = self.relu(x)

        return x
