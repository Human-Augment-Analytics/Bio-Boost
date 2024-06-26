#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:32:01 2024

@author: bshi
"""


import os
import pdb
import pandas as pd
import numpy as np
from collections import Counter


class Prediction_Cleaning:
    def __init__(self, predict_file):
        self.predict_file = predict_file
        self.predict={}
        with open (predict_file, 'r') as file:
            data = [line.strip().split(' ') for line in file]
            for sex in data:
                if sex[1]=='Female':
                    female=float(sex[0])
                else:
                    male=float(sex[0])
            if male> female:
                label='male'
            else:
                label='female'
            title=['_'.join(predict_file.split('.txt')[0].split('/')[-1].split('_')[:5]), '_'.join(predict_file.split('.txt')[0].split('/')[-1].split('_')[5:7]), predict_file.split('.txt')[0].split('_')[-2], predict_file.split('.txt')[0].split('_')[-1]]
            self.row= title+ [female, male, label]


fyolo2_0='/home/bshi/test/predict_results/yolo/1_2_yolo_female_eval/labels/'
myolo2_0='/home/bshi/test/predict_results/yolo/1_2_yolo_male_eval/labels/'
fyolo3_0='/home/bshi/test/predict_results/yolo/1_3_yolo_female_eval/labels/'
myolo3_0='/home/bshi/test/predict_results/yolo/1_3_yolo_male_eval/labels/'

rm_images=[]
import os
#2_0 analysis
df2_0=pd.DataFrame(columns=['trial', 'base_name', 'frame', 'detection','c_female', 'c_male', 'prediction', 'label'])
fprediction_files = os.listdir(fyolo2_0)
t_label='female'
for predict_file in fprediction_files:
    row=Prediction_Cleaning(fyolo2_0+predict_file).row+[t_label]
    df2_0.loc[len(df2_0.index)]=row


mprediction_files = os.listdir(myolo2_0)
t_label='male'
for predict_file in mprediction_files:
    row=Prediction_Cleaning(myolo2_0+predict_file).row+[t_label]
    df2_0.loc[len(df2_0.index)]=row
    

#3_0 analysis
df3_0=pd.DataFrame(columns=['trial', 'base_name', 'frame', 'detection', 'c_female', 'c_male', 'prediction', 'label'])
prediction_files = os.listdir(fyolo3_0)

for predict_file in prediction_files:
    if predict_file in fprediction_files:
        t_label='female'
    elif predict_file in mprediction_files:
        t_label='male'
    else: 
        print('scream')
    row=Prediction_Cleaning(fyolo3_0+predict_file).row+[t_label]
    df3_0.loc[len(df3_0.index)]=row


prediction_files = os.listdir(myolo3_0)

for predict_file in prediction_files:
    if predict_file in fprediction_files:
        t_label='female'
    elif predict_file in mprediction_files:
        t_label='male'
    else: 
        print('scream')
    row=Prediction_Cleaning(myolo3_0+predict_file).row+[t_label]
    df3_0.loc[len(df3_0.index)]=row

df2_0['acc']=df2_0.label==df2_0.prediction
df3_0['acc']=df3_0.label==df3_0.prediction
df2_0['gt']=[0 if i=='male' else 1 for i in df2_0.label]
df2_0['pred']=[0 if i=='male' else 1 for i in df2_0.prediction]
df3_0['gt']=[0 if i=='male' else 1 for i in df3_0.label]
df3_0['pred']=[0 if i=='male' else 1 for i in df3_0.prediction]
df2_0.to_csv('./Results1_2_yolo.csv')
df3_0.to_csv('./Results1_3_yolo.csv')

