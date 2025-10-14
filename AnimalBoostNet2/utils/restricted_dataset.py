'''
Here we implement a simple extension of the PyTorch Dataset class for our restricted image datasets.
'''

# import necessary packages
from typing import Tuple, Any

from PIL import Image
from torch.utils.data import Dataset
import torch

import pandas as pd

class RestrictedDataset(Dataset):
    '''
    This is an extension of the PyTorch Dataset class that we'll use to wrap around our restricted datasets.
    '''

    def __init__(self, df: pd.DataFrame, base_dir: str, device: str, transforms):
        '''
        Initializes an instance of the RestrictedDatset class.

        Inputs:
            df: a pandas DataFrame containing the image file names and the associated temporal data.
            base_dir: an absolute string path to the base directory where the images in self.df are stored.
            device: the device to which the data should be moved upon loading.
            transforms: the data transforms/augmentations to be used, as created using timm.
        '''

        # store all passed arguments
        self.df: pd.DataFrame = df
        self.base_dir: str = base_dir
        self.transforms = transforms
        self.device: str = device

    def __len__(self) -> int:
        '''
        Returns the number of records in the currently-stored dataset.
        '''

        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns the image Tensor, temporal features, and label for the image stored at the passed index (label is derived from the filepath prefix).
        '''

        # get the row at the passed idx
        row: pd.Series[Any] = self.df.iloc[idx, :]

        # generate the absolute filepath for the image in the row
        file_path: str = f'/{self.base_dir.strip("/")}/{row["filepath"]}'
        
        # load and transform the image stored at file_path into a preprocessed Tensor
        img: Image = Image.open(file_path).convert('RGB')
        img: torch.Tensor = self.transforms(img)

        # extract temporal features and wrap in a Tensor
        distance: Any = row['distance_traveled']
        speed: Any = row['speed']
        acceleration: Any = row['mean_acceleration']
        norm_max_displacement: Any = row['norm_max_displacement']
        mean_autocorr: Any = row['mean_autocorrelation']
        cross_corr_smooth: Any = row['cross_correlation_with_median_smoothing']
        res_patches: Any = row['number_of_residence_patches']

        temp_features: torch.Tensor = torch.tensor(
            [distance, speed, acceleration, norm_max_displacement, mean_autocorr, cross_corr_smooth, res_patches], 
            dtype=torch.float32
        )
        
        # derive the label for the image (1 for "Male", 0 for "Female")
        label: int = torch.tensor(1 if '/Male/' in file_path else 0, dtype=torch.long)
        
        # move the image, temporal_features, and label to the correct device (GPU or CPU) and return them
        return img.to(self.device), temp_features.to(self.device), label.to(self.device)
