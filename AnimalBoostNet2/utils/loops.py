'''
Here we define the training, validation, and main epoch loops needed for training the AnimalBoostNet model.
'''

# import necessary libraries
from typing import Tuple, Any, List, Dict

from models.abnet import AnimalBoostNet as ABNet
from .checkpointing import save_checkpoint

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm

def train_loop(train_dataloader: DataLoader, model: ABNet, loss_fn: nn.Module, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, float]:
    '''
    This is the training loop to be used.

    Inputs:
        train_dataloader: a PyTorch dataloader wrapping around the restricted image training dataset.
        model: the AnimalBoostNet PyTorch module to be trained.
        loss_fn: the loss function to be used (Cross Entropy Loss hard-coded).
        optimizer: the optimizer to use during gradient descent (Adam hard-coded).
        epoch: the number for the current epoch.
    '''

    # initialize sum stats
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    # initialize tqdm loop
    num_batches: int = len(train_dataloader)
    train_loop: tqdm[Any] = tqdm(train_dataloader, total=num_batches)
    
    # put model head in train mode (keep backbone in eval mode) and iterate through batches
    model.backbone.eval()
    model.head.train()
    for batch, (imgs, temp_features, labels) in enumerate(train_loop):
        optimizer.zero_grad()

        # get logits from forward pass
        logits: torch.Tensor = model(imgs, temp_features)

        # convert to probabilities (softmax) and predictions (argmax)
        probs: torch.Tensor = logits.softmax(dim=1)
        preds: torch.Tensor = probs.argmax(dim=1)

        # calculate more stats and increment sum stats
        num_samples: int = imgs.shape[0]
        num_correct: int = (preds == labels).sum().item()

        total_samples += num_samples
        total_correct += num_correct

        # calculate loss, backpropagate, and update optimizer
        batch_loss: torch.Tensor = loss_fn(probs, labels)
        batch_loss.backward()
        optimizer.step()

        # calculate loss and accuracy stats
        total_batch_loss: float = batch_loss.item() * num_samples
        total_loss += total_batch_loss

        batch_acc: float = num_correct / num_samples
        
        avg_loss: float = total_loss / total_samples
        avg_acc: float = total_correct / total_samples

        # update tqdm loop displayed info
        train_loop.set_description(f'Epoch {epoch} Train, Batch [{batch + 1}/{num_batches}]')
        train_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                         'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    # return avg loss and avg accuracy
    return (total_loss / total_samples), (total_correct / total_samples)

def valid_loop(valid_dataloader: DataLoader, model: ABNet, loss_fn: nn.Module, epoch: int) -> Tuple[float, float]:
    '''
    This is the validation loop to be used.

    Inputs:
        valid_dataloader: the PyTorch DataLoader to wrap around the restricted image validation dataset.
        model: the AnimalBoostNet PyTorch module to be validated.
        loss_fn: the loss function to be used (Cross Entropy Loss hard-coded).
        epoch: the number of the current epoch.
    '''

    # initialize sum stats
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    # setup tqdm loop around dataloader
    num_batches: int = len(valid_dataloader)
    valid_loop: tqdm[Any] = tqdm(valid_dataloader, total=num_batches)

    # put model (backbone and head) in eval mode and start loop through batches (without weight updates)
    model.backbone.eval()
    model.head.eval()
    with torch.no_grad():
        for batch, (imgs, temp_features, labels) in enumerate(valid_loop):
            # get logits from forward pass
            logits: torch.Tensor = model(imgs, temp_features)

            # calculate probabilities (softmax) and predictions (argmax)
            probs: torch.Tensor = logits.softmax(dim=1)
            preds: torch.Tensor = probs.argmax(dim=1)

            # calculate stats and increment sum stats
            num_samples: int = imgs.shape[0]
            num_correct: int = (preds == labels).sum().item()

            total_samples += num_samples
            total_correct += num_correct

            # calculate loss and accuracy stats
            batch_loss: float = loss_fn(probs, labels).item()
            total_batch_loss: float = batch_loss * num_samples
            total_loss += total_batch_loss

            batch_acc: float = num_correct / num_samples

            avg_loss: float = total_loss / total_samples
            avg_acc: float = total_correct / total_samples

            # update tqdm loop displayed info
            valid_loop.set_description(f'Epoch {epoch} Valid, Batch [{batch + 1}/{num_batches}]')
            valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
                            'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
            
    # return avg loss and avg accuracy
    return (total_loss / total_samples), (total_correct / total_samples)

def main_loop(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: ABNet, loss_fn: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, num_epochs: int, run: int, save_dir: str) -> Dict[str, List]:
    '''
    This is the main loop over all the epochs.

    Inputs:
        train_dataloader: the PyTorch DataLoader for the restricted training dataset.
        valid_dataloader: the PyTorch DataLoader for the restricted validation dataset.
        model: the AnimalBoostNet PyTorch module from timm to be fine-tuned.
        loss_fn: the loss function to be used (Cross Entropy Loss hard-coded).
        optimizer: the optimizer to be used during the training loops (Adam hard-coded).
        num_epochs: the total number of epochs to run over (training and validation).
        run: the user-defined run identifier, used for distinguishing results from separate trials and configurations.
        save_dir: the absolute string filepath where results and checkpoints should be stored.
    '''

    # initialize the results dictionary
    results: Dict[str, List] = {
        'train_losses': [],
        'valid_losses': [],
        'train_accs': [],
        'valid_accs': []
    }

    # iterate through each epoch
    for epoch in range(num_epochs):
        # get the avg training loss and avg training accuracy for the current epoch via the training loop
        train_loss, train_acc = train_loop(train_dataloader=train_dataloader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           epoch=epoch)
        
        # get the avg validation loss and avg validation accuracy for the current epoch via the validation loop
        valid_loss, valid_acc = valid_loop(valid_dataloader=valid_dataloader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           epoch=epoch)
        
        # store the avg losses and avg accuracies in the results dictionary
        results['train_losses'].append(train_loss)
        results['valid_losses'].append(valid_loss)

        results['train_accs'].append(train_acc)
        results['valid_accs'].append(valid_acc)

        # step scheduler
        scheduler.step(valid_loss)

        # save the model state and the results from the current epoch
        save_checkpoint(model=model, train_losses=results['train_losses'], train_accs=results['train_accs'],
                        valid_losses=results['valid_losses'], valid_accs=results['valid_accs'], epoch=epoch,
                        run=run, save_dir=save_dir)

    # return the results dictionary
    return results