from typing import Any
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


        batch_loss_sum: float = batch_loss.item() * num_samples
        total_loss += batch_loss_sum
        
        # compute metrics
        batch_acc: float = num_correct / num_samples
        
        avg_loss: float = total_loss / total_samples
        avg_acc: float = total_correct / total_samples
        
        train_loop.set_description(f'Epoch {epoch} Train, Batch [{batch + 1}/{num_batches}]')
        train_loop.set_postfix({'Loss': f'{batch_loss.item():.4f} [{avg_loss:.4f}]',
                         'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return avg_loss.item(), avg_acc

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

            losses_sum: Any | float = batch_loss * num_samples
            total_loss += losses_sum

            # compute metrics
            batch_acc = num_correct / num_samples

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples

            valid_loop.set_description(f'Epoch {epoch} Validate, Batch [{batch + 1}/{num_batches}]')
            valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
                             'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return avg_loss.cpu().item(), avg_acc

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

def evaluate_abnet(dataloader: DataLoader, abnet: ABNet, loss_fn: nn.Module):
    '''
    Evaluation loop for the trained YOLO11 + trained TemporalNet (TNet) approach.

    Inputs:
        dataloader: PyTorch DataLoader containing the evaluation data.
        abnet: an AnimalBoostNet instance with pre-trained (and frozen) YOLO11 backbone, and a trained TNet temporal post-processing model.
        loss_fn: PyTorch loss function instance (e.g., CELoss).

    Returns:
        Average loss, computed across all batches.
        Average overall accuracy, computed across all batches.
        Average accuracies for male and female classes, computed across all batches.
        Average recall for male and female classes, computed across all batches.
        Average precision for male and female classes, computed across all batches.
    '''

    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # total_males_correct = 0
    # total_male_samples = 0
    
    # total_females_correct = 0
    # total_female_samples = 0

    # total_male_tp = 0
    # total_male_fp = 0
    # total_male_fn = 0

    # total_female_tp = 0
    # total_female_fp = 0
    # total_female_fn = 0
    
    num_batches = len(dataloader)
    valid_loop = tqdm(dataloader, total=num_batches)

    eval_results = {
        'prob_class0': [],
        'prob_class1': [],
        'predicted_class': [],
        'true_class': [],
        'filename': []
    }
    
    abnet.eval()
    abnet.tnet.eval()
    with torch.no_grad():
        for batch, (img, temp_features, is_male) in enumerate(valid_loop):
            
            # forward pass
            probs, preds = abnet(img, temp_features)

            # save records
            eval_results['prob_class0'] += probs[:, 0].squeeze().tolist()
            eval_results['prob_class1'] += probs[:, 1].squeeze().tolist()
            eval_results['predicted_class'] += preds.squeeze().tolist()
            eval_results['true_class'] += is_male.squeeze().tolist()
            eval_results['filename'] += list(img)

            # metric tracking
            num_samples = temp_features.shape[0]
            num_correct = (preds == is_male).sum().item()

            total_samples += num_samples
            total_correct += num_correct
            
            # num_males = is_male.sum().item()
            # num_correct_males = (is_male & (preds == is_male)).sum().item()
            
            # is_female = 1 - is_male
            # inv_preds = 1 - preds
            
            # num_females = is_female.sum().item()
            # num_correct_females = (is_female & (inv_preds == is_female)).sum().item()
            
            # total_males_correct += num_correct_males
            # total_male_samples += num_males

            # total_females_correct += num_correct_females
            # total_female_samples += num_females

            # male_tp = (is_male & preds).sum().item()
            # male_fp = (is_female & preds).sum().item()
            # male_fn = (is_male & inv_preds).sum().item()

            # female_tp = (is_female & inv_preds).sum().item()
            # female_fp = (is_male & inv_preds).sum().item()
            # female_fn = (is_female & preds).sum().item()

            # total_male_tp += male_tp
            # total_male_fp += male_fp
            # total_male_fn += male_fn

            # total_female_tp += female_tp
            # total_female_fp += female_fp
            # total_female_fn += female_fn

            # compute loss, backward pass
            batch_loss = loss_fn(probs, is_male)

            losses_sum = batch_loss * num_samples
            total_loss += losses_sum

            # compute metrics
            batch_acc = num_correct / num_samples

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            
            # avg_male_acc = total_males_correct / total_male_samples
            # avg_female_acc = total_females_correct / total_female_samples

            # avg_male_precision = total_male_tp / (total_male_tp + total_male_fp)
            # avg_male_recall = total_male_tp / (total_male_tp + total_male_fn)

            # avg_female_precision = total_female_tp / (total_female_tp + total_female_fp)
            # avg_female_recall = total_female_tp / (total_female_tp + total_female_fn)

            valid_loop.set_description(f'Evaluate, Batch [{batch + 1}/{num_batches}]')
            valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
                             'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
    return eval_results # avg_loss.cpu().item(), avg_acc, avg_male_acc, avg_female_acc, avg_male_precision, avg_male_recall, avg_female_precision, avg_female_recall

# # YOLO11 + TemporalNet + Classification head training/validation/evaluation loops...

# def train_head(dataloader, yolo, tnet, head, optimizer, loss_fn, epoch: int):
#     '''
#     Training loop for the trained YOLO11 + trained TemporalNet (TNet) + untrained Classification head approach.

#     Inputs:
#         dataloader: PyTorch DataLoader containing the training data.
#         yolo: Pre-trained YOLO11 module with frozen layers.
#         tnet: Pre-trained TemporalNet module with frozen layers.
#         head: Untrained Classification head module.
#         optimizer: PyTorch optimizer instance (e.g., Adam).
#         loss_fn: PyTorch loss function instance (e.g., CELoss).
#         epoch: Integer representing the current epoch number.

#     Returns:
#         Average loss, computed across all batches.
#         Average accuracy, computed across all batches.
#     '''

#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
    
#     num_batches = len(dataloader)
#     train_loop = tqdm(dataloader, total=num_batches)
    
#     head.train()
#     yolo.eval()
#     tnet.eval()
#     for batch, (img, temp_features, is_male) in enumerate(train_loop):
        
#         # forward pass (YOLO11 + TemporalNet)
#         yolo_results = yolo(img)

#         yolo_logits = torch.tensor(np.array([yolo_results[idx].probs.data.cpu().numpy() for idx in range(len(yolo_results))]), dtype=torch.float32).cuda()
#         tnet_logits = tnet(temp_features)
        
#         # additively combine logits...
#         intermediate_logits = yolo_logits + tnet_logits
        
#         # OR, concatenate logits along dim=1 (uncomment next line, comment-out previous line)
#         # intermediate_logits = torch.cat((yolo_logits, tnet_logits), dim=1)

#         final_logits = head(intermediate_logits)

#         # get probs and predictions...
#         probs = final_logits.softmax(1)
#         preds = probs.argmax(1)
        
#         num_samples = temp_features.shape[0]
#         num_correct = (preds == is_male).sum().item()
        
#         total_samples += num_samples
#         total_correct += num_correct
        
#         # compute loss, backward pass
#         batch_loss = loss_fn(probs, is_male)
        
#         batch_loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         losses_sum = batch_loss * num_samples
#         total_loss += losses_sum
        
#         # compute metrics
#         batch_acc = num_correct / num_samples
        
#         avg_loss = total_loss / total_samples
#         avg_acc = total_correct / total_samples
        
#         train_loop.set_description(f'Epoch {epoch} Train, Batch [{batch + 1}/{num_batches}]')
#         train_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
#                          'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
#     return avg_loss.cpu().detach().item(), avg_acc

# def validate_head(dataloader, yolo, tnet, head, loss_fn, epoch: int):
#     '''
#     Validation loop for the trained YOLO11 + trained TemporalNet (TNet) + untrained Classification head approach.

#     Inputs:
#         dataloader: PyTorch DataLoader containing the validation data.
#         yolo: Pre-trained YOLO11 module with frozen layers.
#         tnet: Pre-trained TemporalNet module with frozen layers.
#         head: Untrained Classification head module.
#         loss_fn: PyTorch loss function instance (e.g., CELoss).
#         epoch: Integer representing the current epoch number.

#     Returns:
#         Average loss, computed across all batches.
#         Average accuracy, computed across all batches.
#     '''

#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
    
#     num_batches = len(dataloader)
#     valid_loop = tqdm(dataloader, total=num_batches)
    
#     yolo.eval()
#     tnet.eval()
#     head.eval()
#     with torch.no_grad():
#         for batch, (img, temp_features, is_male) in enumerate(valid_loop):
            
#             # forward pass (YOLO11 + TemporalNet)
#             yolo_results = yolo(img)

#             yolo_logits = torch.tensor(np.array([yolo_results[idx].probs.data.cpu().numpy() for idx in range(len(yolo_results))]), dtype=torch.float32).cuda()
#             tnet_logits = tnet(temp_features)

#             # additively combine logits...
#             intermediate_logits = yolo_logits + tnet_logits
            
#             # OR, concatenate logits along dim=1 (uncomment next line, comment-out previous line)
#             # intermediate_logits = torch.cat((yolo_logits, tnet_logits), dim=1)

#             final_logits = head(intermediate_logits)

#             # get probs and predictions...
#             probs = final_logits.softmax(1)
#             preds = probs.argmax(1)

#             num_samples = temp_features.shape[0]
#             num_correct = (preds == is_male).sum().item()

#             total_samples += num_samples
#             total_correct += num_correct

#             # compute loss, backward pass
#             batch_loss = loss_fn(probs, is_male)

#             losses_sum = batch_loss * num_samples
#             total_loss += losses_sum

#             # compute metrics
#             batch_acc = num_correct / num_samples

#             avg_loss = total_loss / total_samples
#             avg_acc = total_correct / total_samples

#             valid_loop.set_description(f'Epoch {epoch} Validate, Batch [{batch + 1}/{num_batches}]')
#             valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
#                              'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
#     return avg_loss.cpu().item(), avg_acc

# def head_main_loop(train_dataloader, valid_dataloader, yolo, tnet, head, optimizer, loss_fn, save_dir: str, epochs: int, start_epoch=0):
#     '''
#     Main training/validation outer loop for the trained YOLO11 + trained TemporalNet (TNet) + untrained Classification head approach.

#     Inputs:
#         train_dataloader: PyTorch DataLoader containing the training data.
#         valid_dataloader: PyTorch DataLoader containing the validation data.
#         yolo: Pre-trained YOLO11 module with frozen layers.
#         tnet: Pre-trained TemporalNet module with frozen layers.
#         head: Untrained Classification head module.
#         optimizer: PyTorch optimizer instance (e.g., Adam).
#         loss_fn: PyTorch loss function instance (e.g., CELoss).
#         save_dir: String indicating the directory where checkpoint files should be saved to and loaded from.
#         epochs: Integer representing the total number of epochs to train/validate over.
#         start_epoch: Integer indicating which epoch to begin training from, useful for resuming previous training which was interrupted (defaults to 0).

#     Returns:
#         List of averaged training losses for each epoch.
#         List of averaged training accuracies for each epoch.
#         List of averaged validation losses for each epoch.
#         List of averaged validation accuracies for each epoch.
#     '''

#     for epoch in range(start_epoch, epochs):
        
#         # load previous checkpoint
#         train_losses, train_accs, valid_losses, valid_accs = load_checkpoint(head, optimizer, epoch=epoch, save_dir=save_dir, model_type='head')
        
#         # train
#         train_loss, train_acc = train_head(train_dataloader, yolo, tnet, head, optimizer, loss_fn, epoch=epoch)
        
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
        
#         # validate
#         valid_loss, valid_acc = validate_head(valid_dataloader, yolo, tnet, head, loss_fn, epoch=epoch)
        
#         valid_losses.append(valid_loss)
#         valid_accs.append(valid_acc)
        
#         # save current checkpoint
#         save_checkpoint(head, optimizer, train_losses, train_accs, valid_losses, valid_accs, epoch=epoch, save_dir=save_dir, model_type='head')
        
#     return train_losses, train_accs, valid_losses, valid_accs

# def evaluate_head(dataloader, yolo, tnet, head, loss_fn):
#     '''
#     Evaluation loop for the trained YOLO11 + trained TemporalNet (TNet) + trained Classification head approach.

#     Inputs:
#         dataloader: PyTorch DataLoader containing the evaluation data.
#         yolo: Pre-trained YOLO11 module with frozen layers.
#         tnet: Trained TemporalNet module with frozen layers.
#         head: Trained Classification head module.
#         loss_fn: PyTorch loss function instance (e.g., CELoss).

#     Returns:
#         Average loss, computed across all batches.
#         Average overall accuracy, computed across all batches.
#         Average accuracies for male and female classes, computed across all batches.
#         Average recall for male and female classes, computed across all batches.
#         Average precision for male and female classes, computed across all batches.
#     '''

#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
    
#     total_males_correct = 0
#     total_male_samples = 0
    
#     total_females_correct = 0
#     total_female_samples = 0

#     total_male_tp = 0
#     total_male_fp = 0
#     total_male_fn = 0

#     total_female_tp = 0
#     total_female_fp = 0
#     total_female_fn = 0
    
#     num_batches = len(dataloader)
#     valid_loop = tqdm(dataloader, total=num_batches)
    
#     tnet.eval()
#     yolo.eval()
#     with torch.no_grad():
#         for batch, (img, temp_features, is_male) in enumerate(valid_loop):
            
#             # forward pass (YOLO11 + TemporalNet)
#             yolo_results = yolo(img)

#             yolo_logits = torch.tensor(np.array([yolo_results[idx].probs.data.cpu().numpy() for idx in range(len(yolo_results))]), dtype=torch.float32).cuda()
#             tnet_logits = tnet(temp_features)

#             # additively combine logits...
#             intermediate_logits = yolo_logits + tnet_logits
            
#             # OR, concatenate logits along dim=1 (uncomment next line, comment-out previous line)
#             # intermediate_logits = torch.cat((yolo_logits, tnet_logits), dim=1)

#             final_logits = head(intermediate_logits)

#             # get probs and predictions...
#             probs = final_logits.softmax(1)
#             preds = probs.argmax(1)

#             num_samples = temp_features.shape[0]
#             num_correct = (preds == is_male).sum().item()

#             total_samples += num_samples
#             total_correct += num_correct
            
#             num_males = is_male.sum().item()
#             num_correct_males = (is_male & (preds == is_male)).sum().item()
            
#             is_female = 1 - is_male
#             inv_preds = 1 - preds
            
#             num_females = is_female.sum().item()
#             num_correct_females = (is_female & (inv_preds == is_female)).sum().item()
            
#             total_males_correct += num_correct_males
#             total_male_samples += num_males

#             total_females_correct += num_correct_females
#             total_female_samples += num_females

#             male_tp = (is_male & preds).sum().item()
#             male_fp = (is_female & preds).sum().item()
#             male_fn = (is_male & inv_preds).sum().item()

#             female_tp = (is_female & inv_preds).sum().item()
#             female_fp = (is_male & inv_preds).sum().item()
#             female_fn = (is_female & preds).sum().item()

#             total_male_tp += male_tp
#             total_male_fp += male_fp
#             total_male_fn += male_fn

#             total_female_tp += female_tp
#             total_female_fp += female_fp
#             total_female_fn += female_fn

#             # compute loss, backward pass
#             batch_loss = loss_fn(probs, is_male)

#             losses_sum = batch_loss * num_samples
#             total_loss += losses_sum

#             # compute metrics
#             batch_acc = num_correct / num_samples

#             avg_loss = total_loss / total_samples
#             avg_acc = total_correct / total_samples
            
#             avg_male_acc = total_males_correct / total_male_samples
#             avg_female_acc = total_females_correct / total_female_samples

#             avg_male_precision = total_male_tp / (total_male_tp + total_male_fp)
#             avg_male_recall = total_male_tp / (total_male_tp + total_male_fn)

#             avg_female_precision = total_female_tp / (total_female_tp + total_female_fp)
#             avg_female_recall = total_female_tp / (total_female_tp + total_female_fn)

#             valid_loop.set_description(f'Batch [{batch + 1}/{num_batches}]')
#             valid_loop.set_postfix({'Loss': f'{batch_loss:.4f} [{avg_loss:.4f}]',
#                              'Acc': f'{(batch_acc * 100):.2f}% [{(avg_acc * 100):.2f}%]'})
        
#     return avg_loss.cpu().item(), avg_acc, avg_male_acc, avg_female_acc, avg_male_precision, avg_male_recall, avg_female_precision, avg_female_recall
