'''
Here we define the checkpointing functionality for fine-tuning. This is solely for preservign results: the dataset is small enough 
that we don't really need to worry about timing out or other interruptions.
'''

# import necessary libraries
from typing import List

import torch.nn as nn
import torch

def save_checkpoint(model: nn.Module, train_losses: List[float], valid_losses: List[float], train_accs: List[float],
                    valid_accs: List[float], epoch: int, run: int, save_dir: str) -> None:
    '''
    This function defines how all the relevant fine-tuning information is saved after each epoch.

    Inputs:
        model: the PyTorch module who's weights are to be saved.
        train_losses: a list containing the training losses up to and including the epoch being saved.
        valid_losses: a list containing the validation losses up to and including the epoch being saved.
        train_accs: a list containing the training accuracies up to and including the epoch being saved.
        valid_accs: a list containing the validation accuracies up to and including the epoch being saved.
        epoch: the number of the epoch being saved.
        run: a user-defined integer run number, used to identify information from different runs.
        save_dir: the string filepath to the directory where the checkpoints are to be saved.
    '''    

    # define dictionary containing all info to be saved
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs
    }

    # save the checkpoint dictionary to a .pt file and notify the user
    checkpoint_path = save_dir + f'/run{run}_epoch{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    print(f'Epoch {epoch} Checkpoint Saved!')
