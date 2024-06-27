#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:01:44 2024

@author: breannashi
"""

## Metrics Calculation
import numpy as np

class Metrics:
    """
    A class to calculate various metrics for binary classification.

    Attributes:
        label (array-like): Ground truth binary labels.
        predict (array-like): Predicted binary labels.
        metrics (dict): Dictionary containing various calculated metrics.
    """

    def __init__(self, label, predict):
        """
        Initialize the Metrics class with ground truth labels and predictions.

        Parameters:
            label (array-like): Ground truth binary labels.
            predict (array-like): Predicted binary labels.
        """
        print('label=1 is positive, label=0 is negative')
        self.label=label
        self.predict=predict
        
        tp = np.sum((label == 1) & (predict == 1))
        tn = np.sum((label == 0) & (predict == 0))
        fp = np.sum((label == 0) & (predict == 1))
        fn = np.sum((label == 1) & (predict == 0))
        
        total = tp + tn + fp + fn
        pp = (tp + fn) / total
        pn = (tn + fp) / total
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) != 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) != 0 else 0
        self.metrics = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total,
            'pp': pp,
            'pn': pn,
            'tpr': tpr,
            'tnr': tnr,
            'fpr': fpr,
            'fnr': fnr,
            'acc': (tp + tn) / total,
            'pprecision': tp / (tp + fp) if (tp + fp) != 0 else 0,
            'nprecision': tn / (tn + fn) if (tn + fn) != 0 else 0,
            'precall': tpr,
            'nrecall': tnr,
            'pFm': 2 * tp / (2 * tp + tn + fp) if (2 * tp + tn + fp) != 0 else 0,
            'nFm': 2 * tn / (2 * tn + tp + fn) if (2 * tn + tp + fn) != 0 else 0,
            'bacc': 0.5 * (tpr + tnr),
            'mcc': (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) != 0 else 0
        }
        
    def get_metrics(self):
        """
        Get the calculated metrics.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        return self.metrics
    
## Data Processing
import pandas as pd
import os
import shutil

# Load data from CSV files
r1_1=pd.read_csv('/home/bshi/Documents/results/Results1_1.csv')
r1_2=pd.read_csv('/home/bshi/Documents/results/Results1_2_yolo.csv')
r1_3=pd.read_csv('/home/bshi/Documents/results/Results1_3_yolo.csv')
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('')

# Calculate metrics for the first dataset
a1_1=Metrics(r1_1['label'], r1_1['prediction']).get_metrics()

# Add trial information to the first dataset
r1_1['trial']=['_'.join(i.split('_')[:5]) for i in r1_1.image]

# Assign experiment labels based on trial
r1_1['exp']=[exp_dict[i] for i in r1_1.trial]
r1_2['exp']=[exp_dict[i] for i in r1_2.trial]
r2_1['exp']=[exp_dict[i] for i in r2_1.trial]
r2_2['exp']=[exp_dict[i] for i in r2_2.trial]

# Encode labels as binary (0 for male, 1 for female)
r2_2['gt']=[0 if i=='male' else 1 for i in r2_2.label]
r2_2['pred']=[0 if i=='male' else 1 for i in r2_2.prediction]

# Calculate accuracy for each trial
acc_1={}
for name, group in r2_2.groupby('trial'):
    metric=Metrics(group['gt'], group['pred']).get_metrics()['pn']
    acc1_1[name]=metric

# Calculate accuracy for female and male groups in dataset r2_1
facc1_1={}
for name, group in r2_1[r2_1.label==1].groupby('exp'):
    metric=Metrics(group['label'], group['class_id']).get_metrics()['acc']
    facc1_1[name]=metric
macc1_1={}
for name, group in r2_1[r2_1.label==0].groupby('exp'):
    metric=Metrics(group['label'], group['class_id']).get_metrics()['acc']
    macc1_1[name]=metric

# Calculate accuracy for female and male groups in dataset r2_2
facc1_2={}
for name, group in r2_2[r2_2['gt']==1].groupby('exp'):
    metric=Metrics(group['gt'], group['pred']).get_metrics()['acc']
    facc1_2[name]=metric
macc1_2={}
for name, group in r2_2[r2_2['gt']==0].groupby('exp'):
    metric=Metrics(group['gt'], group['pred']).get_metrics()['acc']
    macc1_2[name]=metric

# Print the difference in accuracy between datasets for each experiment
for key in facc1_2.keys():
    print(key)
    print(facc1_2[key]-facc1_1[key])
    input(facc1_2[key])

# Recalculate metrics for dataset r1_1
a1_1=Metrics(r1_1['label'], r1_1['prediction']).get_metrics()