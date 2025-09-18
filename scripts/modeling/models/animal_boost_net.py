'''
This module implements the AnimalBoostNet post-processing model, combining the TNet and the YOLO11 models.
'''

from typing import List, Tuple, Any

from tnet_model import TemporalNet as TNet
from ultralytics import YOLO
import numpy as np
import torch.nn as nn
import torch

class AnimalBoostNet(nn.Module):
    '''
    This is our multi-modal post-processing model, as proposed and discussed in our paper. Extends the PyTorch Module class.
    '''

    def __init__(self, yolo_weights: str, tnet_input_size: int = 7, tnet_output_size: int = 2, tnet_hidden_sizes: List[int] = [128, 64], tnet_dropout: float = 0.5, device: str = 'cpu'):
        '''
        Initializes an instance of the AnimalBoostNet model class.

        Input:
            yolo_weights: a path to the YOLO11-cls weights to be used for image processing.
            tnet_input_size: the number of temporal features to use in TNet post-processing (defaults to 7).
            tnet_output_size: the number of unique target classes/labels (defaults to 2).
            tnet_hidden_sizes: a list containing the hidden dimension sizes to be used in the TNet post-processing model (defaults to [128, 64]).
            tnet_dropout: the probability of dropout applied to the last hidden layer's output in the TNet (defaults to 0.5).
            device: the device on which the model should be stored (defaults to 'cpu').
        '''

        super(AnimalBoostNet, self).__init__()

        self.tnet_input_size = tnet_input_size
        self.tnet_output_size = tnet_output_size
        self.tnet_hidden_sizes = tnet_hidden_sizes
        self.tnet_dropout = tnet_dropout

        self.device = device

        self.yolo = YOLO(yolo_weights).to(device)
        self.yolo.eval()

        for param in self.yolo.parameters():
            param.requires_grad = False

        self.tnet = TNet(tnet_input_size, tnet_output_size, tnet_hidden_sizes, tnet_dropout).to(device)

    def silence_yolo_verbosity(self, img: str) -> None:
        '''
        Turns off verbose output from the YOLO11 backbone.

        Input:
            img: the filepath to an image of equal dimension to those in the intended dataset.
        Output: None.
        '''

        self.yolo(img)
        self.yolo.predictor.args.verbose = False

    def forward(self, img: np.ndarray, temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Performs a forward pass over the sub-modules in the AnimalBoostNet model.

        Input:
            img: a batch of image file paths to pass through the YOLO image processing backbone.
            temporal: a batch of temporal features to pass through the TNet post-processing model.

        Output:
            probs: the softmax class probabilities for each record in the input image/temporal batches.
            preds: the argmax class predictions for each record in the input image/temporal batches, based on the calculated class probabilities.
        '''
        
        yolo_results = self.yolo(img)
        
        yolo_logits = torch.tensor(np.array([yolo_results[idx].probs.data.cpu().numpy() for idx in range(len(yolo_results))]), dtype=torch.float32).to(self.device)
        tnet_logits = self.tnet(temporal)

        logits = yolo_logits + tnet_logits
        probs = logits.softmax(1)
        preds = probs.argmax(1)

        return probs, preds