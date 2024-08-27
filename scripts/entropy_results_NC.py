"""
Results

The following script was used to calculate entropy for the results.
"""

# Encoding, etc.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Entropy Calculation
def entropy(predict, base=2):
    """
    Calculate the entropy of a prediction array.

    Parameters:
        predict (array-like): Array of predictions (binary: 0 and 1).
        base (int): Logarithmic base to use. Default is 2.

    Returns:
        float: Entropy value.
    """
    predict = np.array(predict)
    
    total = len(predict)
    num_zeros = np.sum(predict == 0)
    num_ones = np.sum(predict == 1)
    if num_ones==0:
        entropy=-((num_zeros/total) * np.log2(num_zeros/total)+ (num_ones/total))
    elif num_zeros==0:
        entropy=-((num_zeros/total) + (num_ones/total) * np.log2(num_ones/total))
    else:
        entropy=-((num_zeros/total) * np.log2(num_zeros/total)+ (num_ones/total) * np.log2(num_ones/total))
    return abs(entropy)

# Load Data
r2_1=pd.read_csv('/home/path_to_file/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)
r2_2=pd.read_csv('/home/path_to_file/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/path_to_file/Documents/results/Results2_3.csv')

# Create Ground Truth and Prediction Columns
r2_2['gt'] = [0 if i == 'male' else 1 for i in r2_2.label]
r2_2['pred'] = [0 if i == 'male' else 1 for i in r2_2.prediction]
r2_2['id'] = r2_2['trial'] + '__' + r2_2['base_name'] + '__' + r2_2['track_id'].astype(str)
r2_3['gt'] = [0 if i == 'male' else 1 for i in r2_3.label]
r2_3['pred'] = [0 if i == 'male' else 1 for i in r2_3.prediction]
r2_3['id'] = r2_3['trial'] + '__' + r2_3['base_name'] + '__' + r2_3['track_id'].astype(str)
r2_1['id'] = r2_1['trial'] + '__' + r2_1['base_name'] + '__' + r2_1['a2_track_id'].astype(str)

# Group by ID
track2_1 = r2_1.groupby(['id'])['class_id']
track2_2 = r2_2.groupby(['id'])['pred']
track2_3 = r2_3.groupby(['id'])['pred']

# Calculate Entropy for Each Group
e2_1 = track2_1.apply(lambda x: entropy(x))
e2_2 = track2_2.apply(lambda x: entropy(x))
e2_3 = track2_3.apply(lambda x: entropy(x))

# Define Percentiles and Accuracy Lists
precent=[0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]
acc2_1=[]
acc2_2=[]
acc2_3=[]

# Calculate Accuracy using MCC
def get_acc(df, label, predict):
    """
    Parameters:
        df (pd.DataFrame): DataFrame containing label and prediction columns.
        label (str): Column name of the ground truth labels.
        predict (str): Column name of the predictions.

    Returns:
        float: MCC accuracy.
    """
    metrics = Metrics(df[label].values, df[predict].values)
    acc=metrics.get_metrics()['mcc']
    return acc

# Calculate Accuracy for Each Percentile
for p in precent:
    q_e2_1 = e2_1.quantile(p)
    entropy_mask = e2_1 >= q_e2_1
    fe2_1= e2_1[entropy_mask]
    df2_1 = r2_1[r2_1.id.isin(list(fe2_1.index))]
    acc1=get_acc(df2_1, 'label', 'class_id')
    acc2_1.append(acc1)
    q_e2_2 = e2_2.quantile(p)
    entropy_mask = e2_2 >= q_e2_2
    fe2_2= e2_2[entropy_mask]
    df2_2 = r2_2[r2_2.id.isin(list(fe2_2.index))]
    acc2=get_acc(df2_2, 'gt', 'pred')
    acc2_2.append(acc2)
    q_e2_3 = e2_3.quantile(p)
    entropy_mask = e2_3 > q_e2_3
    fe2_3= e2_3[entropy_mask]
    df2_3 = r2_3[r2_3.id.isin(list(fe2_3.index))]
    acc3=get_acc(df2_3, 'gt', 'pred')
    acc2_3.append(acc3)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(precent, acc2_1, label='Manual frame', marker='o')
plt.plot(precent, acc2_2, label='Manual video', marker='s')
plt.plot(precent, acc2_3, label='Automatic', marker='^')
plt.xlabel('Percentile')
plt.ylabel('mcc')
plt.title('Accuracy by entropy percentile')
plt.legend()
plt.grid(True)
plt.xticks(precent)
plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
plt.savefig('./figures/fig7_title.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot without Labels and Titles
plt.figure(figsize=(10, 6))
plt.plot(precent, acc2_1, marker='o')
plt.plot(precent, acc2_2, marker='s')
plt.plot(precent, acc2_3,  marker='^')
plt.legend()
plt.grid(True)
plt.xticks(precent)
plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
plt.savefig('./figures/fig7.png', dpi=300, bbox_inches='tight')
plt.show()
