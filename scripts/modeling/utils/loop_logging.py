import torch
import os

def load_checkpoint(model, optimizer, epoch: int, save_dir: str, model_type: str, device='cuda'):
    '''
    Loads a saved PyTorch checkpoint file into the input model and optimizer.

    Inputs:
        model: The model to have its state loaded from the checkpoint file.
        optimizer: The PyTorch optimizer to have its state loaded from the checkpoint file.
        epoch: Integer representing the epoch from which data is being loaded from.
        save_dir: String indicating the directory where checkpoint files should be loaded from.
        model_type: String representing the type of model being loaded into (should be "tnet" or "head").
        device: String representing the device used in defining the map_location for the torch.load function (defaults to "cuda").

    Returns:
        List of averaged training losses stored in the loaded checkpoint file.
        List of averaged training accuracies stored in the loaded checkpoint file.
        List of averaged validation losses stored in the loaded checkpoint file.
        List of averaged validation accuracies stored in the loaded checkpoint file.
    '''

    if epoch > 0:
        checkpoint_path = save_dir + f'/checkpoint_{model_type}_epoch{epoch - 1}.pth'
        assert os.path.exists(checkpoint_path), f'Loading Error: Checkpoint file "{checkpoint_path}" does not exist.'

        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
            model.load_state_dict(checkpoint['model_state'])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state'])

        optimizer.load_state_dict(checkpoint['optim_state'])
        
        t_losses = checkpoint['train_losses']
        t_accs = checkpoint['train_accs']
        
        v_losses = checkpoint['valid_losses']
        v_accs = checkpoint['valid_accs']

        print(f'Epoch {epoch - 1} Checkpoint Loaded!')
        
        return t_losses, t_accs, v_losses, v_accs
    else:
        return [], [], [], []
    
def save_checkpoint(model, optimizer, t_losses, t_accs, v_losses, v_accs, epoch: int, save_dir: str, model_type: str) -> None:
    '''
    Saves the model and optimizer states, as well as passed training/validation losses/accuracies into a PyTorch checkpoint file.

    Inputs:
        model: The model to have its state saved to the checkpoint file.
        optimizer: The PyTorch optimizer to have its state saved to the checkpoint file.
        t_losses: List of averaged training losses to be stored in the loaded checkpoint file.
        t_accs: List of averaged training accuracies to be stored in the loaded checkpoint file.
        v_losses: List of averaged validation losses to be stored in the loaded checkpoint file.
        v_accs: List of averaged validation accuracies to be stored in the loaded checkpoint file.
        epoch: Integer representing the epoch from which data is being saved for.
        save_dir: String indicating the directory where checkpoint files should be saved to.
        model_type: String representing the type of model being saved from (should be "tnet" or "head").

    Returns: Nothing.
    '''    
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'train_losses': t_losses,
        'train_accs': t_accs,
        'valid_losses': v_losses,
        'valid_accs': v_accs
    }

    checkpoint_path = save_dir + f'/checkpoint_{model_type}_epoch{epoch - 1}.pth'
    torch.save(checkpoint, checkpoint_path)

    print(f'Epoch {epoch} Checkpoint Saved!')
    
