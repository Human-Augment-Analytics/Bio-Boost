from models.animal_boost_net import AnimalBoostNet as ABNet
from utils.image_dataset import OptimizedImageDataset
from utils.preprocessor import Preprocessor
from loops import abnet_main_loop

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from argparse import Namespace
from typing import Any, List
import pandas as pd
import numpy as np
import os, sys

class Engine:
    '''
    The class which allows for the use of AnimalBoostNet in a pre-determined set of tasks. Appropriately named,
    as it essentially runs the main functionalities of our work.
    '''

    def __init__(self, args: Namespace):
        '''
        Initializes the engine.

        Input:
            args: the arguments Namespace parsed by the argparse.ArgumentParser in the run.py file.
        '''

        self.args = args

        if args.gpu and not torch.cuda.is_available():
            print(f'Warning: could not use GPU acceleration (CUDA not available).\n')

        # ABNet initialization
        self.abnet = ABNet(yolo_weights=args.yolo_weights,
                           tnet_input_size=args.tnet_in,
                           tnet_output_size=args.tnet_out,
                           tnet_hidden_sizes=args.hidden_dims,
                           tnet_dropout=args.dropout,
                           device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

        # Optimization initialization
        self.optimizer = optim.Adam(self.abnet.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.decay)
        
        self.loss_fn = nn.CrossEntropyLoss()

        # Data initialization
        self.preprocessor = Preprocessor()

        if args.task == 'train_val':
            try:
                train_df = pd.read_csv(args.train_data_path)
                valid_df = pd.read_csv(args.valid_data_path)
            except Exception:
                print(f'Error: could not load in training file "{args.train_data_path}" and/or "{args.valid_data_path}"...')
                print(f'Note: when task is set to "train_val", "--train_data_path" and "--valid_data_path" are REQUIRED!\n')

                sys.exit(1)

            try:
                assert args.save_dir is not None and os.path.exists(args.save_dir)
            except AssertionError:
                print(f'Error: invalid input to "--save_dir" argument.')
                print(f'Note: when task is set to "train_val", "--savae_dir" is REQUIRED!\n')

                sys.exit(1)

            train_dataset = OptimizedImageDataset(df=train_df,
                                                  base_path=args.base_path,
                                                  device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
            valid_dataset = OptimizedImageDataset(df=valid_df,
                                                  base_path=args.base_path,
                                                  device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

            self.train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               num_workers=args.nworkers,
                                               persistent_workers=True)
            self.valid_dataloader = DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               num_workers=args.nworkers,
                                               persistent_workers=True)
            
        elif args.task == 'eval':
            print(f'TODO: implement "eval" task data initialization...\n')
        elif args.task == 'infer':
            print(f'TODO: implement "infer" task data initialization...\n')
        else:
            print(f'This should not be printing :|\n')

    def _visualize(self, data: Any) -> None:
        '''
        Generates results visualizations, depending on the task type.

        Input:
            data: the output results from running the task.
        
        Output: nothing (displays the generated visualization, allowing the user to save manually).
        '''

        if self.args.task == 'train_val':
            import matplotlib.pyplot as plt

            plt.figure(figsize=(14, 8))

            train_losses: List[float] = data[0]
            valid_losses: List[float] = data[1]

            min_valid_loss: float = np.min(valid_losses)
            min_valid_loss_epoch: int = np.argmin(valid_losses) + 1

            xx: List[int] = [epoch for epoch in range(1, len(train_losses) + 1)]

            plt.subplot(1, 2, 1)

            plt.plot(xx, train_losses, color='tab:blue', marker='o', label='Training Loss')
            plt.plot(xx, valid_losses, color='tab:orange', marker='o', label='Validation Loss')
            plt.vlines(min_valid_loss_epoch,
                       ymin=min(np.min(train_losses), min_valid_loss),
                       ymax=max(np.max(train_losses), np.max(valid_losses)),
                       colors='tab:green',
                       linestyles='--',
                       label=f'Min Validation Loss: {min_valid_loss:.2f}\n(Epoch {min_valid_loss_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Cross-Entropy Loss')
            plt.legend()

            train_accs: List[float] = data[2]
            valid_accs: List[float] = data[3]

            max_valid_acc: float = np.max(valid_accs)
            max_valid_acc_epoch: int = np.argmax(valid_accs) + 1

            plt.subplot(1, 2, 2)

            plt.plot(xx, train_accs, color='tab:blue', marker='o', label='Training Accuracy')
            plt.plot(xx, valid_accs, color='tab:orange', marker='o', label='Validation Accuracy')
            plt.vlines(max_valid_acc_epoch,
                       ymin=min(np.min(train_accs), np.min(valid_accs)),
                       ymax=max(np.max(train_accs), max_valid_acc),
                       colors='tab:green',
                       linestyles='--',
                       label=f'Max Validation Accuracy: {max_valid_acc:.2f}\n(Epoch {max_valid_acc_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.show()

        elif self.args.task == 'eval':
            print(f'TODO: implement "eval" task visualization...\n')
        elif self.args.task == 'infer':
            print('Warning: no visualizations to generate for "infer" task, skipping...\n')
        else:
            print(f'This should not be printing either :|\n')

    def run_task(self) -> None:
        '''
        Runs the task indicated by the user to the CLI.

        Inputs: None.
        Output: nothing. (all results are written to files).
        '''

        if self.args.task == 'train_val':
            data: Any = abnet_main_loop(train_dataloader=self.train_dataloader, valid_dataloader=self.valid_dataloader, abnet=self.abnet, optimizer=self.optimizer, loss_fn=self.loss_fn, save_dir=self.args.save_dir, epochs=self.args.nepochs, start_epoch=self.args.start_epoch)
        elif self.args.task == 'eval':
            print(f'TODO: implement "eval" task running...')

        if self.args.visualize:
            try:
                self._visualize(data=data)
            except Exception:
                print(f'Error: could not successfully run visualzation task.')