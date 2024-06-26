#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:56:08 2024

@author: bshi
"""
import pandas
import numpy as np


def entropy(predict,  base=2):
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


r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['gt'] = [0 if i == 'male' else 1 for i in r2_2.label]
r2_2['pred'] = [0 if i == 'male' else 1 for i in r2_2.prediction]
r2_2['id'] = r2_2['trial'] + '__' + r2_2['base_name'] + '__' + r2_2['track_id'].astype(str)

r2_3['gt'] = [0 if i == 'male' else 1 for i in r2_3.label]
r2_3['pred'] = [0 if i == 'male' else 1 for i in r2_3.prediction]
r2_3['id'] = r2_3['trial'] + '__' + r2_3['base_name'] + '__' + r2_3['track_id'].astype(str)

r2_1['id'] = r2_1['trial'] + '__' + r2_1['base_name'] + '__' + r2_1['a2_track_id'].astype(str)

track2_1 = r2_1.groupby(['id'])['class_id']
track2_2 = r2_2.groupby(['id'])['pred']
track2_3 = r2_3.groupby(['id'])['pred']


e2_1 = track2_1.apply(lambda x: entropy(x))
e2_2 = track2_2.apply(lambda x: entropy(x))
e2_3 = track2_3.apply(lambda x: entropy(x))
precent=[0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]
acc2_1=[]
acc2_2=[]
acc2_3=[]
def get_acc(df, label, predict):
    metrics = Metrics(df[label].values, df[predict].values)
    acc=metrics.get_metrics()['mcc']
    return acc


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



import matplotlib.pyplot as plt

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
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(precent, acc2_1, marker='o')
plt.plot(precent, acc2_2, marker='s')
plt.plot(precent, acc2_3,  marker='^')

#plt.xlabel('Percentile')
#plt.ylabel('acc')
#plt.title('Accuracy by entropy percentile')
plt.legend()
plt.grid(True)

plt.xticks(precent)
plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
plt.savefig('./figures/fig7.png', dpi=300, bbox_inches='tight')
plt.show()
