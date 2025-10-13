'''
Here we implement the core visualization functionalities.
'''

# import necessary packages
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

def generate_loss_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    '''
    This function generates the loss plots using the passed data dictionary, as well as the visualization flags.

    Inputs:
        data: a dictionary containing all the training and validation results.
        grid: a Boolean flag indicating whether a grid should be added to the plot.
        markers: a Boolean flag indicating whether circle markers should be added to each point in the plot.
        extrema: a Boolean flag indicating whether the extrema (min validation loss, max validation accuracy) should be labeled in the plot.
    '''
    
    # add a grid if the Boolean flag indicates True
    if grid:
        plt.grid(visible=True)

    # extract the training and validation losses
    train_losses: List[float] = data['train_losses']
    valid_losses: List[float] = data['valid_losses']

    # set the x-axis
    xx: List[int] = [x for x in range(len(train_losses))]

    # if the Boolean flag indicates circle markers should be used, use marker='o'; otherwise, don't specify any marker
    if markers:
        plt.plot(xx, train_losses, marker='o', label='Training Loss')
        plt.plot(xx, valid_losses, marker='o', label='Validation Loss')
    else:
        plt.plot(xx, train_losses, label='Training Loss')
        plt.plot(xx, valid_losses, label='Validation Loss')

    # if the Boolean flag indicates extrema should be labeled, add in a vertical dashed line at the min validation loss
    if extrema:
        plt.vlines(np.argmin(valid_losses), min(min(valid_losses), min(train_losses)), max(max(valid_losses), max(train_losses)),
                   colors='tab:green', linestyles='--', label=f'Best Loss @ Epoch {np.argmin(valid_losses)}')
    
    # set labels and legend
    plt.title('Average Loss per Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')

    plt.legend()

def generate_accuracy_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    '''
    This function generates the accuracy plots using the passed data dictionary, as well as the visualization flags.

    Inputs:
        data: a dictionary containing all the training and validation results.
        grid: a Boolean flag indicating whether a grid should be added to the plot.
        markers: a Boolean flag indicating whether circle markers should be added to each point in the plot.
        extrema: a Boolean flag indicating whether the extrema (min validation loss, max validation accuracy) should be labeled in the plot.
    '''

    # add a grid if the Boolean flag indicates True
    if grid:
        plt.grid(visible=True)

    # extract the training and validation accuracies
    train_accs: List[float] = data['train_accs']
    valid_accs: List[float] = data['valid_accs']

    # set the x-axis
    xx: List[int] = [x for x in range(len(train_accs))]
    
    # if the Boolean flag indicates circle markers should be used, plot with marker='o'; otherwise, don't set any markers
    if markers:
        plt.plot(xx, train_accs, marker='o', label='Training Accuracy')
        plt.plot(xx, valid_accs, marker='o', label='Validation Accuracy')
    else:
        plt.plot(xx, train_accs, label='Training Accuracy')
        plt.plot(xx, valid_accs, label='Validation Accuracy')

    # if the Boolean flag indicates extrema should be labeled, draw a dashed line at the max validation accuracy
    if extrema:
        plt.vlines(np.argmax(valid_accs), min(min(valid_accs), min(train_accs)), max(max(valid_accs), max(train_accs)),
                   colors='tab:green', linestyles='--', label=f'Best Accuracy @ Epoch {np.argmax(valid_accs)}')

    # set labels and legend
    plt.title('Average Accuracy per Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

def generate_plots(data: Dict[str, List], grid: bool = False, markers: bool = False, extrema: bool = False) -> None:
    '''
    This implements the main visualization functionality.

    Inputs:
        data: a dictionary containing all the training and validation results.
        grid: a Boolean flag indicating whether a grid should be added to the plot.
        markers: a Boolean flag indicating whether circle markers should be added to each point in the plot.
        extrema: a Boolean flag indicating whether the extrema (min validation loss, max validation accuracy) should be labeled in the plot.
    '''

    # set the figure size
    plt.figure(figsize=(18, 8))

    # generate the first subplot (losses)
    plt.subplot(1, 2, 1)
    generate_loss_plots(data, grid, markers, extrema)

    # generate the second subplot (accuracies)
    plt.subplot(1, 2, 2)
    generate_accuracy_plots(data, grid, markers, extrema)

    # show the visualization to the user (they can save where ever they'd like, if they'd like to)
    plt.show()