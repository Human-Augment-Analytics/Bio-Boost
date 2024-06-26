#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:01:44 2024

@author: breannashi
"""

##metrics
import numpy as np

class Metrics:
    def __init__(self, label, predict):
        print('label=1 is positive, label=0 is negative')
        self.label=label
        self.predict=predict
        
        self.tp = np.sum((self.label == 1) & (self.predict == 1))
        self.tn = np.sum((self.label == 0) & (self.predict == 0))
        self.fp = np.sum((self.label == 0) & (self.predict == 1))
        self.fn = np.sum((self.label == 1) & (self.predict == 0))
        #rates
        self.total=self.tp+self.tn+self.fp+self.fn
        self.pp=(self.tp+self.fn)/self.total
        self.pn=(self.tn+self.fp)/self.total
        self.tpr=self.tp/(self.tp+self.fn)
        self.tnr=self.tn/(self.tn+self.fp)
        self.fpr=self.fp/(self.tn+self.fp)
        self.fnr=self.fn/(self.tp+self.fn)
        #general
        self.acc=(self.tp+self.tn)/self.total
        self.pprecision=self.tp/(self.tp+self.fp)
        self.nprecision=self.tn/(self.tn+self.fn)
        self.precall=self.tpr
        self.nrecall=self.tnr
        self.pFm=2*(self.tp)/(2*self.tp+self.tn+self.fp)
        self.nFm=2*(self.tn)/(2*self.tn+self.tp+self.fn)
        
        #for unbalanced set
        self.bacc=0.5*(self.tpr+self.tnr)
        self.mcc=(self.tp*self.tn-self.fp*self.fn)/np.sqrt((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))
 
import pandas as pd
import os
import shutil

r1=pd.read_csv('/Users/breannashi/Desktop/aim1_results/Results1_0_fix.csv')
r1_1=pd.read_csv('/Users/breannashi/Desktop/aim1_results/Result_1_1.csv')
r2=pd.read_csv('/Users/breannashi/Desktop/aim1_results/Result_2_0.csv')
r2_1=pd.read_csv('/Users/breannashi/Desktop/aim1_results/Result_2_1.csv')

track_sex=r2.groupby(['trial','base_name', 'track_id'])['val_predictions'].sum()/r2.groupby(['trial','base_name', 'track_id'])['track_id'].count()
track_l=r2.groupby(['trial','base_name', 'track_id'])['err'].sum()/r2.groupby(['trial','base_name', 'track_id'])['track_id'].count()
mask = (track_l >0.5)

# Filter track_l using the mask
filtered_track_l = track_l[mask]






      