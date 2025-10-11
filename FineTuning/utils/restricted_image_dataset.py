from typing import Tuple, Any

from PIL import Image
from torch.utils.data import Dataset
import torch
import timm

import pandas as pd

class RestrictedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: str, device: torch.device, transforms):
        self.df: pd.DataFrame = df
        self.base_dir: str = base_dir
        self.transforms = transforms
        self.device: torch.device = device

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row: pd.Series[Any] = self.df.iloc[idx, :]

        file_path: str = f'/{self.base_dir.strip("/")}/{row["filepath"]}'
        
        img: Image = Image.open(file_path).convert('RGB')
        img: torch.Tensor = self.transforms(img)
        
        label: int = torch.tensor(1 if '/Male/' in file_path else 0, dtype=torch.long)
        
        return img.to(self.device), label.to(self.device)
