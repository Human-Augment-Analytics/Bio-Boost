from typing import Dict, List
import argparse

from utils.restricted_image_dataset import RestrictedImageDataset
from utils.visualize import generate_plots
from utils.loops import main_loop

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import timm

import pandas as pd

parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Takes arguments to run MobileNet fine-tuning/validation.')

parser.add_argument('trainfiles', type=str, help='The absolute path to the CSV file containing the image filepaths for fine-tuning.')
parser.add_argument('validfiles', type=str, help='The absolute path to the CSV file containing the image filepaths for validation.')
parser.add_argument('trainbasedir', type=str, help='The absolute path to the fine-tuning image dataset\'s root folder.')
parser.add_argument('validbasedir', type=str, help='The absolute path to the validation image dataset\'s root folder.')
parser.add_argument('checkpointdir', type=str, help='The path to the directory to store the checkpoint files.')

parser.add_argument('--variant', type=str, default='mobilenetv2_100',
                    choices=['mobilenetv2_035', 'mobilenetv2_050', 'mobilenetv2_075', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140'],
                    help='The MobileNetv2 model variant to be used.')
parser.add_argument('--numclasses', type=int, default=2, help='The number of classes in the fine-tuning dataset.')
parser.add_argument('--batchsize', type=int, default=128, help='The batch size to be used during fine-tuning.')
parser.add_argument('--numworkers', type=int, default=0, help='The number of data loading workers to use during fine-tuning.')
parser.add_argument('--numepochs', type=int, default=100, help='The number of epochs to run fine-tuning and validation over.')
parser.add_argument('--run', type=int, default=0, help='An integer label for the run, used for file saving.')

parser.add_argument('--pretrained', action='store_true', default=False, help='Instructs the program to use a pre-trained MobileNetv2.')
parser.add_argument('--gpu', action='store_true', default=False, help='Instructs the program to use GPU acceleration (if available).')

parser.add_argument('--visualize', action='store_true', default=False, help='Instructs the program to visualize the results.')
parser.add_argument('--grid', action='store_true', default=False, help='Instructs the visualizer to add a grid to its generated plots.')
parser.add_argument('--markers', action='store_true', default=False, help='Instructs the visualizer to add markers to each point in the generated plots.')
parser.add_argument('--extrema', action='store_true', default=False, help='Instructs the visualizer to label extrema in the plots (min validation loss, max validation accuracy).')

args: argparse.Namespace = parser.parse_args()

train_files: str = args.trainfiles
train_base_dir: str = args.trainbasedir

valid_files: str = args.validfiles
valid_base_dir: str = args.validbasedir

save_dir: str = args.checkpointdir

variant: str = args.model
num_classes: int = args.numclasses
batch_size: int = args.batchsize
num_workers: int = args.numworkers
num_epochs: int = args.num_epochs
run: int = args.run

pretrained: bool = args.pretrained
gpu: bool = args.gpu and torch.cuda.is_available()

visualize: bool = args.visualize
grid: bool = args.grid and visualize
markers: bool = args.markers and visualize
extrema: bool = args.extrema and visualize

device: torch.device = torch.device('cuda' if gpu else 'cpu')
model: nn.Module = timm.create_model(
    model_name=variant,
    pretrained=pretrained,
    num_classes=num_classes
).to(device=device)

data_config = timm.data.resolve_model_config(model)
train_transforms = timm.data.create_transform(**data_config, is_training=True)
valid_transforms = timm.data.create_transform(**data_config, is_training=False)

train_df: pd.DataFrame = pd.read_csv(train_files)
valid_df: pd.DataFrame = pd.read_csv(valid_files)

train_dataset: RestrictedImageDataset = RestrictedImageDataset(df=train_df, base_dir=train_base_dir,
                                                               device=device, transforms=train_transforms)
valid_dataset: RestrictedImageDataset = RestrictedImageDataset(df=valid_df, base_dir=valid_base_dir,
                                                               device=device, transforms=valid_transforms)

train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
valid_dataloader: DataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)


optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn: nn.Module = nn.CrossEntropyLoss()

def main() -> None:
    data: Dict[str, List] = main_loop(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                                      model=model, loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_epochs,
                                      run=run, save_dir=save_dir)
    
    if visualize:
        generate_plots(data=data, grid=grid, markers=markers, extrema=extrema)

if __name__ == '__main__':
    results = main()