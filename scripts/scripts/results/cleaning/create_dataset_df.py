#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:49:38 2024

@author: bshi
"""
''
import pandas as pd 
import os

auto_path='/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/raw_data/training_data/YOLOV5_Cls_Automatic_Videos/'
man_path='/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/raw_data/training_data/YOLOV5_Cls_Manual_Videos/'


class Meta_ClsModel_Analysis:
    def __init__(self, base_path):
        self.img_type=['val/','train/']
        self.img_cls=['Male/','Female/']
        self.train_files=[]
        for clsx in self.img_cls:
            self.train_files.append(os.listdir(base_path+self.img_type[1]+clsx))
        self.val_files=[]
        for clsx in self.img_cls:
            self.val_files.append(os.listdir(base_path+self.img_type[0]+clsx))
        self.exp={'MC_singlenuc55_2_Tk47_051220': 'exp1','MC_singlenuc91_b1_Tk9_081120': 'exp2',	
         'MC_singlenuc86_b1_Tk47_073020': 'exp3',
        'MC_singlenuc24_4_Tk47_030320': 'exp4',
        'MC_singlenuc37_2_Tk17_030320': 'exp5',
        'MC_singlenuc40_2_Tk3_030920': 'exp6',
        'MC_singlenuc59_4_Tk61_060220': 'exp7',
        'MC_singlenuc90_b1_Tk3_081120': 'exp8',
        'MC_singlenuc65_4_Tk9_072920': 'exp9',
        'MC_singlenuc94_b1_Tk31_081120': 'exp10',
        'MC_singlenuc41_2_Tk9_030920': 'exp11',
        'MC_singlenuc96_b1_Tk41_081120': 'exp12',
        'MC_singlenuc23_1_Tk33_021220': 'exp13',
        'MC_singlenuc76_3_Tk47_072920': 'exp14',
        'MC_singlenuc23_8_Tk33_031720': 'exp15',
        'MC_singlenuc62_3_Tk65_060220': 'exp16',
        'MC_singlenuc36_2_Tk3_030320': 'exp17',
        'MC_singlenuc28_1_Tk3_022520': 'exp18',
        'MC_singlenuc43_11_Tk41_060220': 'exp19',
        'MC_singlenuc45_7_Tk47_050720': 'exp20',
        'MC_singlenuc34_3_Tk43_030320': 'exp21',
        'MC_singlenuc56_2_Tk65_051220': 'exp22',
        'MC_singlenuc63_1_Tk9_060220': 'exp23',
        'MC_singlenuc81_1_Tk51_072920': 'exp24',
        'MC_singlenuc82_b2_Tk63_073020': 'exp25',
        'MC_singlenuc35_11_Tk61_051220': 'exp26',
        'MC_singlenuc64_1_Tk51_060220': 'exp27' }
    def create_df(self):
        df=pd.DataFrame(columns=[])
        male_train=[i.split('.jpg')[0].split('__')+['train', 'male'] for i in self.train_files[0]]
        female_train=[i.split('.jpg')[0].split('__')+['train','female'] for i in self.train_files[1]]
        male_val=[i.split('.jpg')[0].split('__')+['val', 'male'] for i in self.val_files[0]]
        female_val=[i.split('.jpg')[0].split('__')+[ 'val', 'female'] for i in self.val_files[1]]
        data=male_train+female_train+male_val+female_val
        df=pd.DataFrame(data, columns=['trial', 'base_name', 'track_id', 'count', 'image_type','label'])
        df['uid']=df['trial']+'__'+df['base_name']+'__'+df['track_id']
        df['exp']=[self.exp[i] for i in df.trial]
        return df

auto_df=Meta_ClsModel_Analysis(auto_path).create_df()
mv_df=Meta_ClsModel_Analysis(man_path).create_df()


auto_df.to_csv('/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/data_frames/Meta_Data/YOLOV5_Cls_Automatic_Videos_Labels.csv')
mv_df.to_csv('/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/data_frames/Meta_Data/YOLOV5_Cls_Manual_Videos_Labels.csv')


def cls_p(df):
    total_train = df[df.image_type == 'train'].groupby('exp')['label'].count()
    female_pt = df[(df.image_type == 'train') & (df.label == 'female')].groupby('exp')['label'].count() / total_train
    male_pt = df[(df.image_type == 'train') & (df.label == 'male')].groupby('exp')['label'].count() / total_train
    female_t = df[(df.image_type == 'train') & (df.label == 'female')].groupby('exp')['label'].count()
    male_t = df[(df.image_type == 'train') & (df.label == 'male')].groupby('exp')['label'].count()
    total_val = df[df.image_type == 'val'].groupby('exp')['label'].count()
    female_v = df[(df.image_type == 'val') & (df.label == 'female')].groupby('exp')['label'].count()
    male_v = df[(df.image_type == 'val') & (df.label == 'male')].groupby('exp')['label'].count()
    female_pv = df[(df.image_type == 'val') & (df.label == 'female')].groupby('exp')['label'].count() / total_val
    male_pv = df[(df.image_type == 'val') & (df.label == 'male')].groupby('exp')['label'].count() / total_val
    
    result = pd.concat([
        total_train.rename('Total_Train'),
        female_t.rename('Total_Female_Train'),
        male_t.rename('Total_Male_Train'),
        female_pt.rename('Female_Proportion_Train'),
        male_pt.rename('Male_Proportion_Train'),
        total_val.rename('Total_Val'),
        female_v.rename('Total_Female_Val'),
        male_v.rename('Total_Male_Val'),
        female_pv.rename('Female_Proportion_Val'),
        male_pv.rename('Male_Proportion_Val')
    ], axis=1, join='outer')
    
    # Reset index to make 'exp' a column
    result = result.reset_index()
    
    # Fill all NaN values with 0
    result = result.fillna(0)
    
    return result
auto_metrics=cls_p(auto_df)
mv_metrics=cls_p(mv_df)

auto_metrics.to_csv('/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/data_frames/Meta_Data/YOLOV5_Cls_Automatic_Videos_exp.csv')
mv_metrics.to_csv('/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/data_frames/Meta_Data/YOLOV5_Cls_Manual_Videos_exp.csv')
