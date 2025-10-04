from models.animal_boost_net import AnimalBoostNet as ABNet

from utils.image_dataset import OptimizedImageDataset, LegacyImageDataset
from utils.preprocessor import Preprocessor
from loops import abnet_main_loop, evaluate_abnet

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from argparse import Namespace
from typing import Any, List, Dict
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

        self.args: Namespace = args

        if args.gpu and not torch.cuda.is_available():
            print(f'Warning: could not use GPU acceleration (CUDA not available).\n')

        # ABNet initialization
        if not os.path.exists(args.yolo_weights):
            print(f'Error: invalid input filepath to "yolo_weights" argument (does not exist).')

            sys.exit(1)
        elif args.task == 'eval' and args.tnet_weights is None:
            print(f'Error: invalid input filepath to "tnet_weights" argument (does not exist).')
            print(f'Note: when task is set to "eval", "--tnet_weights" is REQUIRED!')

            sys.exit(1)
    
        self.abnet: ABNet = ABNet(yolo_weights=args.yolo_weights,
                           tnet_weights=args.tnet_weights,
                           tnet_input_size=args.tnet_in,
                           tnet_output_size=args.tnet_out,
                           tnet_hidden_sizes=args.hidden_dims,
                           tnet_dropout=args.dropout,
                           device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

        # Optimization initialization
        self.optimizer: optim.Optimizer = optim.Adam(self.abnet.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.decay)
        
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()

        # Data initialization
        self.preprocessor: Preprocessor = Preprocessor()

        if args.task == 'train_val':
            try:
                # read temporal data
                train_df: pd.DataFrame = pd.read_csv(args.train_data_path)
                valid_df: pd.DataFrame = pd.read_csv(args.valid_data_path)

                # preprocess temporal data
                train_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ] = self.preprocessor.preprocess(train_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ], fit=True)

                valid_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ] = self.preprocessor.preprocess(valid_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ], fit=False)

            except Exception:
                print(f'Error: could not load in training file "{args.train_data_path}" and/or "{args.valid_data_path}"...')
                print(f'Note: when task is set to "train_val", "--valid_data_path" is REQUIRED!\n')

                sys.exit(1)

            if args.save_dir is not None and not os.path.exists(args.save_dir):
                print(f'Error: invalid input to "--save_dir" argument.')
                print(f'Note: when task is set to "train_val", "--save_dir" is REQUIRED!\n')

                sys.exit(1)

            train_dataset: OptimizedImageDataset = OptimizedImageDataset(df=train_df,
                                                  base_path=args.base_path,
                                                  device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
            valid_dataset: OptimizedImageDataset = OptimizedImageDataset(df=valid_df,
                                                  base_path=args.base_path,
                                                  device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

            self.train_dataloader: DataLoader = DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               num_workers=args.nworkers,
                                               persistent_workers=True)
            self.valid_dataloader: DataLoader = DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               num_workers=args.nworkers,
                                               persistent_workers=True)
            
        elif args.task == 'eval':
            try:
                train_df: pd.DataFrame = pd.read_csv(args.train_data_path)
                eval_df: pd.DataFrame = pd.read_csv(args.eval_data_path)

                _: pd.DataFrame = self.preprocessor.preprocess(train_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ], fit=True)

                eval_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ] = self.preprocessor.preprocess(eval_df[
                    ['distance_traveled', 'speed', 'mean_acceleration', 'norm_max_displacement', 'mean_autocorrelation', 'cross_correlation_with_median_smoothing', 'number_of_residence_patches']
                ], fit=False)
            except Exception:
                print(f'Error: could not load in training file "{args.train_data_path}" and/or "{args.eval_data_path}"...')
                print(f'Note: when task is set to "eval", "--eval_data_path" is REQUIRED!\n')

                sys.exit(1)

            
            opt_dataset: OptimizedImageDataset = OptimizedImageDataset(df=eval_df,
                                                                       base_path=args.base_path,
                                                                       device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
            leg_dataset: LegacyImageDataset = LegacyImageDataset(df=eval_df,
                                                                 base_path=args.base_path,
                                                                 device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

            self.opt_dataloader = DataLoader(dataset=opt_dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=args.nworkers,
                                             persistent_workers=True)
            self.leg_dataloader = DataLoader(dataset=leg_dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=args.nworkers,
                                             persistent_workers=True)
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
            results: Dict[str, List] = evaluate_abnet(opt_dataloader=self.opt_dataloader, leg_dataloader=self.leg_dataloader, abnet=self.abnet, loss_fn=self.loss_fn)

            try:
                results_df: pd.DataFrame = pd.DataFrame.from_dict(results)
                results_df.to_csv(self.args.results_path, index=False)
            except Exception:
                print(f'Warning: could not save "eval" task results to path "{self.args.results_path}".')
        else:
            print(f'This should not be printing either :|\n')

        if self.args.visualize and self.task == 'train_val':
            try:
                self._visualize(data=data)
            except Exception:
                print(f'Error: could not successfully run visualzation task.')
        else:
            print('Warning: visualization is not currently supported for the "eval" task.')