from typing import Dict, List
import os, sys

import matplotlib.pyplot as plt
import numpy as np

def generate_loss_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    if grid:
        plt.grid(visible=True)

    train_losses: List[float] = data['train_losses']
    valid_losses: List[float] = data['valid_losses']

    xx: List[int] = [x for x in range(1, len(train_losses) + 1)]

    if markers:
        plt.plot(xx, train_losses, marker='o', label='Training Loss')
        plt.plot(xx, valid_losses, marker='o', label='Validation Loss')
    else:
        plt.plot(xx, train_losses, label='Training Loss')
        plt.plot(xx, valid_losses, label='Validation Loss')

    if extrema:
        plt.vlines(np.argmin(valid_losses) + 1, min(min(valid_losses), min(train_losses)), max(max(valid_losses), max(train_losses)), colors='tab:green', linestyles='--')
        
    plt.title('Average Loss per Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')

    plt.legend()

def generate_accuracy_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    if grid:
        plt.grid(visible=True)

    train_accs: List[float] = data['train_accs']
    valid_accs: List[float] = data['valid_accs']

    xx: List[int] = [x for x in range(1, len(train_accs) + 1)]
    
    if markers:
        plt.plot(xx, train_accs, marker='o', label='Training Accuracy')
        plt.plot(xx, valid_accs, marker='o', label='Validation Accuracy')
    else:
        plt.plot(xx, train_accs, label='Training Accuracy')
        plt.plot(xx, valid_accs, label='Validation Accuracy')

    if extrema:
        plt.vlines(np.argmax(valid_accs) + 1, min(min(valid_accs), min(train_accs)), max(max(valid_accs), max(train_accs)), colors='tab:green', linestyles='--')

    plt.title('Average Accuracy per Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

def generate_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    generate_loss_plots(data, grid, markers, extrema)

    plt.subplot(1, 2, 2)
    generate_accuracy_plots(data, grid, markers, extrema)

    plt.show()