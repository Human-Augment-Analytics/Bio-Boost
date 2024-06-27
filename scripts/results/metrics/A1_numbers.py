#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:24:59 2024

@author: bshi
"""
## Getting All Numbers
import os

# List Files and Directories
path = "/home/bshi/Documents/Annotation1_0/images/train"
train_files = os.listdir(path)
path = "/home/bshi/Documents/Annotation1_0/images/val"
val_files = os.listdir(path)

# Get Trial Info
val_trial=['_'.join(i.split('_')[:5]) for i in val_files]
train_trial=['_'.join(i.split('_')[:5]) for i in train_files]

# Count Trials
from collections import Counter
val=Counter(val_trial)
train=Counter(train_trial)

# Filter Trials with Specific Counts
keys_with_value_21 = [key for key, value in train.items() if value == 21]
keys_with_value_22 = [key for key, value in train.items() if value == 22]

# Read and Filter Detection Data
import pandas as pd
detections=pd.read_csv('/home/bshi/Documents/Annotation1_0/Annotation1.0_detections.csv')
filter_detections=detections[detections.Project_ID.isin(list(val.keys()))]
all_males=filter_detections[filter_detections.class_id==0]
all_females=filter_detections[filter_detections.class_id==1]

# Categorize Detections by Image Type
val_all=filter_detections[filter_detections.img_type=='val']
train_all=filter_detections[filter_detections.img_type=='train']

val_f=all_females[all_females.img_type=='val']
train_f=all_females[all_females.img_type=='train']

train_m=all_males[all_males.img_type == 'train']
val_m=all_males[all_males.img_type == 'val']