#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:38:05 2024

@author: bshi
"""

exp_dict={'MC_singlenuc55_2_Tk47_051220': 'exp1','MC_singlenuc91_b1_Tk9_081120': 'exp2',	
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

trial_dict = {
    'exp1': 'MC_singlenuc55_2_Tk47_051220',
    'exp2': 'MC_singlenuc91_b1_Tk9_081120',
    'exp3': 'MC_singlenuc86_b1_Tk47_073020',
    'exp4': 'MC_singlenuc24_4_Tk47_030320',
    'exp5': 'MC_singlenuc37_2_Tk17_030320',
    'exp6': 'MC_singlenuc40_2_Tk3_030920',
    'exp7': 'MC_singlenuc59_4_Tk61_060220',
    'exp8': 'MC_singlenuc90_b1_Tk3_081120',
    'exp9': 'MC_singlenuc65_4_Tk9_072920',
    'exp10': 'MC_singlenuc94_b1_Tk31_081120',
    'exp11': 'MC_singlenuc41_2_Tk9_030920',
    'exp12': 'MC_singlenuc96_b1_Tk41_081120',
    'exp13': 'MC_singlenuc23_1_Tk33_021220',
    'exp14': 'MC_singlenuc76_3_Tk47_072920',
    'exp15': 'MC_singlenuc23_8_Tk33_031720',
    'exp16': 'MC_singlenuc62_3_Tk65_060220',
    'exp17': 'MC_singlenuc36_2_Tk3_030320',
    'exp18': 'MC_singlenuc28_1_Tk3_022520',
    'exp19': 'MC_singlenuc43_11_Tk41_060220',
    'exp20': 'MC_singlenuc45_7_Tk47_050720',
    'exp21': 'MC_singlenuc34_3_Tk43_030320',
    'exp22': 'MC_singlenuc56_2_Tk65_051220',
    'exp23': 'MC_singlenuc63_1_Tk9_060220',
    'exp24': 'MC_singlenuc81_1_Tk51_072920',
    'exp25': 'MC_singlenuc82_b2_Tk63_073020',
    'exp26': 'MC_singlenuc35_11_Tk61_051220',
    'exp27': 'MC_singlenuc64_1_Tk51_060220'}