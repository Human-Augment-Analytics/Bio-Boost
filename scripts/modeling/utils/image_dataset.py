'''
This module implements the ImageDataset used to train our combined model. This Dataset should be wrapped inside a dataloader.
'''

from torch.utils.data import Dataset
import torch 

import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    '''
    This is a PyTorch Dataset class that wraps around one of our CSV datasets (stored as a pandas DataFrame). It takes a path to the base
    directory where the image dataset is stored in the file system, as well as a device string to indicate whether the returned Tensors
    should be on CPU or GPU (useful for CUDA acceleration).
    '''

    def __init__(self, df: pd.DataFrame, base_path: str, device: str = 'cpu'):
        '''
        This initializes an instance of the ImageDatset class.

        Input:
            df: a pandas DataFrame storing one of our CSV datasets.
            base_path: the path to the base directory where the image dataset is stored in the file system.
            device: a string indicating which device the returned Tensors should be stored on (defaults to 'cpu').
        '''
        
        self.df = df
        self.base_path = base_path
        self.device = device
        
    def __len__(self) -> int:
        '''
        This returns the number of records in an instance of the ImageDataset class.

        Input: None.
        Output: an integer representing the number of records (number of rows in the pandas DataFrame).
        '''

        return self.df.shape[0]
    
    def __getitem__(self, idx: int):
        '''
        This returns the row in the stored pandas DataFrame indicated by the passed integer index.

        Input:
            idx: an integer row index.

        Output: the idx-th row in self.df.
        '''
        
        row = self.df.iloc[idx]
        
        if not self.expanded:
            img = self.base_path + row['filepath']
        else:
            img = self.base_path + row['split'] + '/' + row['filepath']
        
        distance = row['distance_traveled']
        speed = row['speed']
        acceleration = row['mean_acceleration']
        norm_max_displacement = row['norm_max_displacement']
        mean_autocorr = row['mean_autocorrelation']
        cross_corr_smooth = row['cross_correlation_with_median_smoothing']
        res_patches = row['number_of_residence_patches']

        temp_features = torch.tensor(np.array([distance, speed, acceleration, norm_max_displacement, mean_autocorr, cross_corr_smooth, res_patches]), dtype=torch.float32).to(self.device)
        is_male = torch.tensor(row['is_male'], dtype=torch.long).to(self.device)
            
        return img, temp_features, is_male