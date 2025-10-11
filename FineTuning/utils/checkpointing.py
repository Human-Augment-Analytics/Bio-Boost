from typing import List, Any
import os, sys

import torch.nn as nn
import torch

def save_checkpoint(model: nn, train_losses: List[float], valid_losses: List[float], train_accs: List[float],
                    valid_accs: List[float], epoch: int, run: int, save_dir: str) -> None:    
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs
    }

    checkpoint_path = save_dir + f'/run{run}_epoch{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    print(f'Epoch {epoch} Checkpoint Saved!')

# def compare_checkpoints(save_dir: str) -> None:
#     all_checkpoints: List[str] = os.listdir(save_dir)

#     max_acc: float = 0.0
#     max_acc_checkpoint_path: str = None

#     for checkpoint_file in all_checkpoints:
#         checkpoint_path: str = f'{save_dir}{checkpoint_file}'
#         if not os.path.exists(checkpoint_path):
#             print(f'No checkpoint found with absolute path "{checkpoint_path}"!')

#             sys.exit(1)

#         checkpoint: Any = torch.load(checkpoint_path)
        
#         last_acc: float = checkpoint['valid_accs'][-1]
#         if last_acc > max_acc:
#             max_acc = last_acc
#             max_acc_checkpoint_path = checkpoint_path

#     print(f'Checkpoint with maximum validation accuracy @ "{max_acc_checkpoint_path}"')
