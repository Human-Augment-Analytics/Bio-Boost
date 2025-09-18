'''
This module implements our checkpointing utility functions.
'''

from typing import Tuple, List
import os

from models.animal_boost_net import AnimalBoostNet as ABNet
import torch.optim as optim
import torch

def save_checkpoint(abnet: ABNet, optimizer: optim.Optimizer, t_losses: List[float], t_accs: List[float], v_losses: List[float], v_accs: List[float], epoch: int, save_dir: str) -> None:
    '''
    This function saves state information in a checkpoint file.

    Input:
        abnet: the AnimalBoostNet instance whose TNet state is to be stored.
        optimizer: the optimizer used during training whose state is to be stored.
        t_losses: the list of stored per-epoch training losses, from start to the passed (presumably current) epoch.
        t_accs: the list of stored per-epoch training accuracies, from start to the passed (presumably current) epoch.
        v_losses: the list of stored per-epoch validation losses, from start to the passed (presumably current) epoch.
        v_accs: the list of stored per-epoch validation accuracies, from start to the passed (presumably current) epoch.
        epoch: the (current) epoch number.
        save_dir: the string path to the directory where checkpoints should be stored.

    Output: None (.pt file saved to passed save_dir).
    '''
    
    checkpoint = {
        'epoch': epoch,
        'model_state': abnet.tnet.state_dict(),
        'optim_state': optimizer.state_dict(),
        'train_losses': t_losses,
        'train_accs': t_accs,
        'valid_losses': v_losses,
        'valid_accs': v_accs
    }

    checkpoint_path = save_dir + f'/checkpoint_new4_epoch{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    print(f'Epoch {epoch} Checkpoint Saved!')

def load_checkpoint(abnet: ABNet, optimizer: optim.Optimizer, epoch: int, save_dir: str, device: str = 'cpu') -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
    This function loads the state information from the *previous* epoch.

    Input:
        abnet: the AnimalBoostNet instance whose TNet will have a state loaded into.
        optimizer: the optimizer used during training that will have a state loaded into.
        epoch: the *current* epoch number.
        save_dir: the string path to the directory where checkpoints are stored.
        device: the device that the states should be mapped to during loading (defaults to 'cpu').

    Output:
        t_losses: the stored list of training losses, from start until the *previous* epoch.
        t_accs: the stored list of training accuracies, from strat until the *previous* epoch.
        v_losses: the stored list of validation losses, from start until the *previous* epoch.
        v_accs: the stored list of validation accuracies, from start until the *previous* epoch.
    '''
    
    if epoch > 0:
        checkpoint_path = save_dir + f'/best_checkpoint_new4_epoch{epoch - 1}.pt'
        assert os.path.exists(checkpoint_path), f'Loading Error: Checkpoint file "{checkpoint_path}" does not exist.'

        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
            abnet.tnet.load_state_dict(checkpoint['model_state'])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            abnet.tnet.load_state_dict(checkpoint['model_state'])

        optimizer.load_state_dict(checkpoint['optim_state'])
        
        t_losses = checkpoint['train_losses']
        t_accs = checkpoint['train_accs']
        
        v_losses = checkpoint['valid_losses']
        v_accs = checkpoint['valid_accs']

        print(f'Epoch {epoch - 1} Checkpoint Loaded!')
        
        return t_losses, t_accs, v_losses, v_accs
    else:
        return [], [], [], []