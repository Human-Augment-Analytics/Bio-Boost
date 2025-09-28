'''
This module implements the ImageDataset used to train our combined model. This Dataset should be wrapped inside a dataloader.
'''

from torch.utils.data import Dataset
import torch 
import torch.nn.functional as F

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    '''
    Optimized PyTorch Dataset class that preprocesses images to tensors for maximum efficiency.
    Eliminates file I/O overhead during training by loading images as preprocessed tensors.
    '''

    def __init__(self, df: pd.DataFrame, base_path: str, device: str = 'cpu', preprocess_images: bool = True):
        '''
        Initializes an optimized ImageDataset instance.

        Input:
            df: a pandas DataFrame storing one of our CSV datasets.
            base_path: the path to the base directory where the image dataset is stored.
            device: device for tensor storage ('cpu' or 'cuda').
            preprocess_images: if True, preprocesses images to match YOLO input format.
        '''
        
        self.df = df
        self.base_path = base_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.preprocess_images = preprocess_images
        
        # YOLO preprocessing pipeline - must exactly match YOLO's LetterBox preprocessing!
        if preprocess_images:
            # Import LetterBox from ultralytics for exact matching
            try:
                from ultralytics.data.augment import LetterBox
                
                # Use the same LetterBox that YOLO uses internally
                self.letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
                self.transform = transforms.ToTensor()  # Just convert to tensor, LetterBox handles resizing
            except ImportError:
                # Fallback to basic transforms if ultralytics not available
                self.letterbox = None
                self.transform = transforms.Compose([
                    transforms.Resize((640, 640)),  
                    transforms.ToTensor(),
                ])
        else:
            self.transform = None
        
    def __len__(self) -> int:
        '''
        This returns the number of records in an instance of the ImageDataset class.

        Input: None.
        Output: an integer representing the number of records (number of rows in the pandas DataFrame).
        '''

        return self.df.shape[0]
    
    def __getitem__(self, idx: int):
        '''
        Returns preprocessed data for training - either image tensors or file paths.

        Input:
            idx: an integer row index.

        Output: 
            img: either preprocessed image tensor (if preprocess_images=True) or filepath string
            temp_features: temporal features tensor
            is_male: classification label tensor
        '''
        
        row = self.df.iloc[idx]
        
        # Construct image path
        img_path = self.base_path + row['split'] + '/' + row['filepath']
        
        if self.preprocess_images:
            # Load and preprocess image to tensor for direct model inference
            try:
                image = Image.open(img_path).convert('RGB')
                
                if self.letterbox is not None:
                    # Use exact YOLO LetterBox preprocessing
                    image_np = np.array(image)
                    letterboxed = self.letterbox(image=image_np)
                    img_tensor = self.transform(Image.fromarray(letterboxed))
                else:
                    # Fallback preprocessing
                    img_tensor = self.transform(image)
                
                # Keep on CPU initially - DataLoader will move to GPU if needed
                img = img_tensor
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image tensor as fallback
                img = torch.zeros((3, 640, 640), dtype=torch.float32)
        else:
            # Return filepath for YOLO wrapper (backward compatibility)
            img = img_path
        
        # Extract temporal features
        distance = row['distance_traveled']
        speed = row['speed']
        acceleration = row['mean_acceleration']
        norm_max_displacement = row['norm_max_displacement']
        mean_autocorr = row['mean_autocorrelation']
        cross_corr_smooth = row['cross_correlation_with_median_smoothing']
        res_patches = row['number_of_residence_patches']

        temp_features = torch.tensor(
            [distance, speed, acceleration, norm_max_displacement, mean_autocorr, cross_corr_smooth, res_patches], 
            dtype=torch.float32
        )
        is_male = torch.tensor(row['is_male'], dtype=torch.long)
            
        return img, temp_features, is_male


# Backward compatibility alias
class OptimizedImageDataset(ImageDataset):
    '''Alias for the optimized ImageDataset with preprocessing enabled by default.'''
    
    def __init__(self, df: pd.DataFrame, base_path: str, device: str = 'cpu'):
        super().__init__(df, base_path, device, preprocess_images=True)


# Legacy compatibility alias  
class LegacyImageDataset(ImageDataset):
    '''Alias for the original ImageDataset behavior (filepath strings).'''
    
    def __init__(self, df: pd.DataFrame, base_path: str, device: str = 'cpu'):
        super().__init__(df, base_path, device, preprocess_images=False)