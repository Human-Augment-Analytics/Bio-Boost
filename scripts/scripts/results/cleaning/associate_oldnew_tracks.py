#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:42:01 2024

@author: bshi
"""
import os
import pandas as pd
# Set the directory path
path1 = "/home/bshi/Documents/Annotation2_0/final_resnet/train/Female"
path2 = "/home/bshi/Documents/Annotation2_0/final_resnet/train/Male"
path3 = "/home/bshi/Documents/Annotation2_0/final_resnet/validation/Female"
path4 = "/home/bshi/Documents/Annotation2_0/final_resnet/validation/Male"

# Get a list of all files in the directory
all_2 = os.listdir(path1)+os.listdir(path2)+os.listdir(path3)+os.listdir(path4)
all_files=[i.split('.jpg')[0].split('__') for i in all_2]

df=pd.DataFrame(all_files, columns=['trial', 'base_name', 'track_id', 'frame'])

for trial, trial_group in df.groupby('trial'):
        for base_name, base_name_group in trial_group.groupby('base_name'):
            for track_id in base_name_group['track_id']:
                