from typing import List, Dict
import argparse

from models.abnet import AnimalBoostNet as ABNet
from utils.restricted_dataset import RestrictedDataset
from utils.visualize import generate_plots
from utils.loops import main_loop

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import timm

import pandas as pd

parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Script for running AnimalBoostNet tasks.')

parser.add_argument('task', type=str, choices=['train', 'eval', 'infer'], help='The task to use AnimalBoostNet for.')
parser.add_argument('variant', type=str, help='The model variant to use for the AnimalBoostNet backbone.')
parser.add_argument('weights', type=str, help='The absolute path to the pre-trained backbone model weights.')
parser.add_argument('trainfiles', type=str, help='The absolute path to the CSV containing the restricted training dataset.')
parser.add_argument('validfiles', type=str, help='The absolute path to the CSV containing the restricted validation dataset.')
parser.add_argument('trainbasedir', type=str, help='The absolute path to the base directory where all the training image files are stored.')
parser.add_argument('validbasedir', type=str, help='The absolute path to the base directory where all the validation images files are stored.')
parser.add_argument('checkpointdir', type=str, help='The absolute path to the directory where checkpoint files will be stored.')

parser.add_argument('--inputsize', type=int, default=7, help='The number of temporal features for the TemporalNet.')
parser.add_argument('--hiddensizes', type=List[int], nargs='+', default=[128, 64], help='A list of hidden layer sizes for the TemporalNet.')
parser.add_argument('--numclasses', type=int, default=2, help='The number of classes in the dataset.')
parser.add_argument('--dropout', type=float, default=0.5, help='The dropout probability for the TemporalNet to use.')
parser.add_argument('--numepochs', type=int, default=25, help='The number of epochs to train over.')
parser.add_argument('--batchsize', type=int, default=16, help='The batch size to use.')
parser.add_argument('--numworkers', type=int, default=0, help='The number of workers to use.')
parser.add_argument('--run', type=int, default=0, help='The integer identifier to use in distinguishing saved data.')

parser.add_argument('--gpu', action='store_true', default=False, help='Instructs the script to use GPU acceleration (if possible).')
parser.add_argument('--visualize', action='store_true', default=False, help='Instructs the script to create visualizations for the results.')
parser.add_argument('--grid', action='store_true', default=False, help='Instructs the script to add a grid to the visualizations.')
parser.add_argument('--markers', action='store_true', default=False, help='Instructs the script to add markers for each point in the visualizations.')
parser.add_argument('--extrema', action='store_true', default=False, help='Instructs the script to label the relevant extrema in the visualizations.')

args: argparse.Namespace = parser.parse_args()

task: str = args.task
variant: str = args.variant
weights: str = args.weights
train_files: str = args.trainfiles
valid_files: str = args.validfiles
train_base_dir: str = args.trainbasedir
valid_base_dir: str = args.validbasedir
checkpoint_dir: str = args.checkpointdir

input_size: int = args.inputsize
hidden_sizes: List[int] = args.hiddensizes
num_classes: int = args.numclasses
dropout: float = args.dropout
num_epochs: int = args.numepochs
batch_size: int = args.batchsize
num_workers: int = args.numworkers
run: int = args.run

gpu: bool = args.gpu and torch.cuda.is_available()
if not gpu and args.gpu:
    print('Warning: could not use CPU acceleration, defaulting to CPU.')

visualize: bool = args.visualize
grid: bool = args.grid
markers: bool = args.markers
extrema: bool = args.extrema

abnet: ABNet = ABNet(
    backbone_name=variant,
    backbone_weights=weights,
    tnet_input_size=input_size,
    tnet_output_size=num_classes,
    tnet_hidden_sizes=hidden_sizes,
    tnet_dropout=dropout,
    device='cuda' if gpu else 'cpu'
)

data_config = timm.data.resolve_data_config(abnet.backbone.pretrained_cfg)
train_transforms = timm.data.create_transform(**data_config, is_training=True)
valid_transforms = timm.data.create_transform(**data_config, is_training=False)

train_df: pd.DataFrame = pd.read_csv(train_files)
valid_df: pd.DataFrame = pd.read_csv(valid_files)

train_dataset: RestrictedDataset = RestrictedDataset(df=train_df, base_dir=train_base_dir,
                                                               device='cuda' if gpu else 'cpu',
                                                               transforms=train_transforms)
valid_dataset: RestrictedDataset = RestrictedDataset(df=valid_df, base_dir=valid_base_dir,
                                                               device='cuda' if gpu else 'cpu',
                                                               transforms=valid_transforms)

train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
valid_dataloader: DataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

optimizer: optim.Optimizer = optim.Adam(abnet.head.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn: nn.Module = nn.CrossEntropyLoss()

def main() -> Dict[str, List]:
    '''
    This defines the main functionality of the CLI tool.
    '''

    # run the main fine-tuning loop
    data: Dict[str, List] = main_loop(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                                      model=abnet, loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_epochs,
                                      run=run, save_dir=checkpoint_dir)
    
    # if the user wants to visualize the results, generate the plots
    if visualize:
        generate_plots(data=data, grid=grid, markers=markers, extrema=extrema)

    return data

# run the script
if __name__ == '__main__':
    results: Dict[str, List] = main()

    # print the results
    for key, value in results.items():
        print(f'{key}: {value}')