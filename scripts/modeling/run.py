from typing import List

from utils.engine import Engine
import argparse

parser = argparse.ArgumentParser(description='Takes arguments to train/validate, evaluate, or run inference.')

# AnimalBoostNet-specific arguments...
parser.add_argument('yolo_weights', type=str, help='The filepath to the YOLO11-cls weights to initialize YOLO with.')
parser.add_argument('--tnet_in', type=int, default=7, help='The number of temporal features the TemporalNet (TNet) will use (default: 7).')
parser.add_argument('--tnet_out', type=int, default=2, help='The number of classes the TemporalNet (TNet) will output (default: 2).')
parser.add_argument('--hidden_dims', nargs='+', type=List[int], default=[128, 64], help='The hidden dimension sizes to use in the TemporalNet (TNet) (default: [128, 64]).')
parser.add_argument('--dropout', type=float, default=0.5, help='The dropout probability to use on the last hidden TemporalNet (TNet) output (default: 0.5).')

# data loading arguments...
parser.add_argument('base_path', type=str, help='The path to the root directory where the dataset(s) is/are stored.')
parser.add_argument('--nworkers', type=int, default=1, help='The number of workers to use in the dataloader (default: 1).')

# training arguments...
parser.add_argument('--train_data_path', type=str, help='The filepath to the training dataset to be used (REQUIRED if task is "train_val").')
parser.add_argument('--valid_data_path', type=str, help='The path to the root directory where the dataset is stored (REQUIRED if task is "train_val").')
parser.add_argument('--nepochs', type=int, default=5, help='The number of epochs to train/validate over (default: 5).')
parser.add_argument('--start_epoch', type=int, default=0, help='The starting epoch of the training/validation run (default: 0).')
parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate the optimizer will use (default: 0.0001).')
parser.add_argument('--decay', type=float, default=1e-5, help='The weight decay the optimizer will use (default: 1e-5).')
parser.add_argument('--save_dir', type=str, help='The path to the directory where checkpoints will be saved (REQUIRED if task is "train_val").')

# general arguments...
parser.add_argument('task', type=str, choices=['train_val', 'eval', 'infer'], help='Indicates the task to be performed by the engine.')
parser.add_argument('--gpu', action='store_true', default=False, help='Indicates that GPU acceleration should be utilized (if possible... default: False).')
parser.add_argument('--batch_size', type=int, default=640, help='The batch size to use (default: 640).')

# miscellaneous options...
parser.add_argument('--visualize', action='store_true', default=False, help='Tells the engine to take the results from the ran task and create visualization(s) if possible (default: False).')

args = parser.parse_args()

engine = Engine(args=args)

if __name__ == '__main__':
    engine.run_task()
