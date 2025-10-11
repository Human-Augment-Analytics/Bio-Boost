from typing import Tuple, Any, List, Dict

from .checkpointing import save_checkpoint

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm

def train_loop(train_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, float]:
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    num_batches: int = len(train_dataloader)
    train_loop: tqdm[Any] = tqdm(train_dataloader, total=num_batches)
    
    model.train()
    for batch, (imgs, labels) in enumerate(train_loop):
        out: torch.Tensor = model(imgs)

        probs: torch.Tensor = out.softmax(dim=1)
        preds: torch.Tensor = probs.argmax(dim=1)

        num_samples: int = imgs.shape[0]
        num_correct: int = (preds == labels).sum().item()

        total_samples += num_samples
        total_correct += num_correct

        batch_loss: torch.Tensor = loss_fn(probs, labels)
        batch_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_batch_loss: float = batch_loss.item() * num_samples
        total_loss += total_batch_loss

        batch_acc: float = num_correct / num_samples
        
        avg_loss: float = total_loss / total_samples
        avg_acc: float = total_correct / total_samples

        train_loop.set_description(f'Epoch {epoch} Train, Batch [{batch + 1}/{num_batches}]')
        train_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                         'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return (total_loss / total_samples), (total_correct / total_samples)

def valid_loop(valid_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, float]:
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    num_batches: int = len(valid_dataloader)
    valid_loop: tqdm[Any] = tqdm(valid_dataloader, total=num_batches)

    model.eval()
    with torch.no_grad():
        for batch, (imgs, labels) in enumerate(valid_loop):
            out: torch.Tensor = model(imgs)

            probs: torch.Tensor = out.softmax(dim=1)
            preds: torch.Tensor = probs.argmax(dim=1)

            num_samples: int = imgs.shape[0]
            num_correct: int = (preds == labels).sum().item()

            total_samples += num_samples
            total_correct += num_correct

            batch_loss: float = loss_fn(probs, labels).item()
            total_batch_loss: float = batch_loss * num_samples
            total_loss += total_batch_loss

            batch_acc: float = num_correct / num_samples

            avg_loss: float = total_loss / total_samples
            avg_acc: float = total_correct / total_samples

            valid_loop.set_description(f'Epoch {epoch} Valid, Batch [{batch + 1}/{num_batches}]')
            valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
                            'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
            
    return (total_loss / total_samples), (total_correct / total_samples)

def main_loop(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, num_epochs: int, run: int, save_dir: str) -> Dict[str, List]:
    results: Dict[str, List] = {
        'train_losses': [],
        'valid_losses': [],
        'train_accs': [],
        'valid_accs': []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_loop(train_dataloader=train_dataloader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           epoch=epoch)
        
        valid_loss, valid_acc = valid_loop(valid_dataloader=valid_dataloader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           epoch=epoch)
        
        results['train_losses'].append(train_loss)
        results['valid_losses'].append(valid_loss)

        results['train_accs'].append(train_acc)
        results['valid_accs'].append(valid_acc)

        save_checkpoint(model=model, train_losses=results['train_losses'], train_accs=results['train_accs'],
                        valid_losses=results['valid_losses'], valid_accs=results['valid_accs'], epoch=epoch,
                        run=run, save_dir=save_dir)

    # compare_checkpoints(save_dir=save_dir)

    return results