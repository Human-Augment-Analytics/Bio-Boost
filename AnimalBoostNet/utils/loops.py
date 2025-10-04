from typing import Any, Dict, List
from tqdm import tqdm

from models.animal_boost_net import AnimalBoostNet as ABNet
from checkpointing import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

# AnimalBoostNet training/validation/evaluation loops...
def train_abnet(dataloader: DataLoader, abnet: ABNet, optimizer: optim.Optimizer, loss_fn: nn.Module, epoch: int):
    '''
    Training loop for the AnimalBoostNet approach.

    Inputs:
        dataloader: PyTorch DataLoader containing the training data.
        abnet: an AnimalBoostNet instance with pre-trained (and frozen) YOLO11 backbone, and an untrained TNet temporal post-processing model.
        optimizer: PyTorch optimizer instance (e.g., Adam).
        loss_fn: PyTorch loss function instance (e.g., CELoss).
        epoch: Integer representing the current epoch number.

    Returns:
        Average loss, computed across all batches.
        Average accuracy, computed across all batches.
    '''

    total_loss: int = 0
    total_correct: int = 0
    total_samples: int = 0
    
    num_batches: int = len(dataloader)
    train_loop: tqdm[Any] = tqdm(dataloader, total=num_batches)
    
    for batch, (img, temp_features, is_male) in enumerate(train_loop):
        # forward pass
        probs, preds = abnet(img, temp_features)
        
        num_samples: int = temp_features.shape[0]
        num_correct: int = (preds == is_male).sum().item()
        
        total_samples += num_samples
        total_correct += num_correct
        
        # compute loss, backward pass
        batch_loss: torch.Tensor = loss_fn(probs, is_male)
        
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses_sum: float = batch_loss.item() * num_samples
        total_loss += losses_sum
        
        # compute metrics
        batch_acc: float = num_correct / num_samples
        
        avg_loss: float = total_loss / total_samples
        avg_acc: float = total_correct / total_samples
        
        train_loop.set_description(f'Epoch {epoch} Train, Batch [{batch + 1}/{num_batches}]')
        train_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                         'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return avg_loss, avg_acc

def validate_abnet(dataloader: DataLoader, abnet: ABNet, loss_fn: nn.Module, epoch: int):
    '''
    Validation loop for the AnimalBoostNet approach.

    Inputs:
        dataloader: PyTorch DataLoader containing the validation data.
        abnet: an AnimalBoostNet instance with pre-trained (and frozen) YOLO11 backbone, and an actively training TNet temporal post-processing model.
        loss_fn: PyTorch loss function instance (e.g., CELoss).
        epoch: Integer representing the current epoch number.

    Returns:
        Average loss, computed across all batches.
        Average accuracy, computed across all batches.
    '''

    total_loss: int = 0
    total_correct: int = 0
    total_samples: int = 0
    
    num_batches: int = len(dataloader)
    valid_loop: tqdm[Any] = tqdm(dataloader, total=num_batches)
    
    abnet.eval()
    abnet.tnet.eval()
    with torch.no_grad():
        for batch, (img, temp_features, is_male) in enumerate(valid_loop):
            
            # forward pass
            probs, preds = abnet(img, temp_features)

            num_samples: int = temp_features.shape[0]
            num_correct: int = (preds == is_male).sum().item()

            total_samples += num_samples
            total_correct += num_correct

            # compute loss, backward pass
            batch_loss: torch.Tensor = loss_fn(probs, is_male)

            losses_sum: float = batch_loss.item() * num_samples
            total_loss += losses_sum

            # compute metrics
            batch_acc: float = num_correct / num_samples

            avg_loss: float = total_loss / total_samples
            avg_acc: float = total_correct / total_samples

            valid_loop.set_description(f'Epoch {epoch} Validate, Batch [{batch + 1}/{num_batches}]')
            valid_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                             'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return avg_loss, avg_acc

def abnet_main_loop(train_dataloader: DataLoader, valid_dataloader: DataLoader, abnet: ABNet, optimizer: optim.Optimizer, loss_fn: nn.Module, save_dir: str, epochs: int, start_epoch: int = 0):
    '''
    Main training/validation outer loop for the trained YOLO11 + untrained TemporalNet (TNet) approach.

    Inputs:
        train_dataloader: PyTorch DataLoader containing the training data.
        valid_dataloader: PyTorch DataLoader containing the validation data.
        abnet: an AnimalBoostNet instance with pre-trained (and frozen) YOLO11 backbone, and an untrained TNet temporal post-processing model.
        optimizer: PyTorch optimizer instance (e.g., Adam).
        loss_fn: PyTorch loss function instance (e.g., CELoss).
        save_dir: String indicating the directory where checkpoint files should be saved to and loaded from.
        epochs: Integer representing the total number of epochs to train/validate over.
        start_epoch: Integer indicating which epoch to begin training from, useful for resuming previous training which was interrupted (defaults to 0).

    Returns:
        train_losses: List of averaged training losses for each epoch.
        train_accs: List of averaged training accuracies for each epoch.
        valid_losses: List of averaged validation losses for each epoch.
        valid_accs: List of averaged validation accuracies for each epoch.
    '''

    for epoch in range(start_epoch, epochs):
        # load previous checkpoint
        train_losses, train_accs, valid_losses, valid_accs = load_checkpoint(abnet, optimizer, epoch=epoch, save_dir=save_dir, device=abnet.device)
        
        # train
        train_loss, train_acc = train_abnet(train_dataloader, abnet, optimizer, loss_fn, epoch=epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # validate
        valid_loss, valid_acc = validate_abnet(valid_dataloader, abnet, loss_fn, epoch=epoch)
        
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # save current checkpoint
        save_checkpoint(abnet, optimizer, train_losses, train_accs, valid_losses, valid_accs, epoch=epoch, save_dir=save_dir)
        
    return train_losses, train_accs, valid_losses, valid_accs

def evaluate_abnet(opt_dataloader: DataLoader, leg_dataloader: DataLoader, abnet: ABNet, loss_fn: nn.Module):
    '''
    Evaluation loop for the trained YOLO11 + trained TemporalNet (TNet) approach.

    Inputs:
        opt_dataloader: PyTorch DataLoader containing the evaluation data, wrapping around an OptimizedImageDataset instance.
        leg_dataloader: PyTorch DataLoader containing the evaluation data, wrapping around a LegacyImageDataset instance.
        abnet: an AnimalBoostNet instance with pre-trained (and frozen) YOLO11 backbone, and a trained TNet temporal post-processing model.
        loss_fn: PyTorch loss function instance (e.g., CELoss).

    Returns:
        Average loss, computed across all batches.
        Average overall accuracy, computed across all batches.
        Average accuracies for male and female classes, computed across all batches.
        Average recall for male and female classes, computed across all batches.
        Average precision for male and female classes, computed across all batches.
    '''

    total_loss: int = 0
    total_correct: int = 0
    total_samples: int = 0
    
    num_batches: int = len(opt_dataloader)
    eval_loop: tqdm[Any] = tqdm(zip(opt_dataloader, total=num_batches), total=num_batches)

    eval_results: Dict[str, List] = {
        'prob_class0': [],
        'prob_class1': [],
        'predicted_class': [],
        'true_class': [],
        'filename': []
    }
    
    abnet.eval()
    abnet.tnet.eval()
    with torch.no_grad():
        for batch_idx, (opt_batch, leg_batch) in enumerate(eval_loop):
            img, temp_features, is_male = opt_batch
            img_file, _, _ = leg_batch
            
            # forward pass
            probs, preds = abnet(img, temp_features)

            # save records
            eval_results['prob_class0'] += probs[:, 0].squeeze().tolist()
            eval_results['prob_class1'] += probs[:, 1].squeeze().tolist()
            eval_results['predicted_class'] += preds.squeeze().tolist()
            eval_results['true_class'] += is_male.squeeze().tolist()
            eval_results['filename'] += list(img_file)

            # metric tracking
            num_samples: int = temp_features.shape[0]
            num_correct: int = (preds == is_male).sum().item()

            total_samples += num_samples
            total_correct += num_correct

            # compute loss
            batch_loss: torch.Tensor = loss_fn(probs, is_male)

            losses_sum: float = batch_loss.item() * num_samples
            total_loss += losses_sum

            # compute metrics
            batch_acc: float = num_correct / num_samples

            avg_loss: float = total_loss / total_samples
            avg_acc: float = total_correct / total_samples

            eval_loop.set_description(f'Evaluate, Batch [{batch_idx + 1}/{num_batches}]')
            eval_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                             'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return eval_results