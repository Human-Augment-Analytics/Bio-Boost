#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:47:03 2024

@author: bshi
"""
import numpy as np
import pandas as pd

def entropy(predict, base=2):
    predict = np.array(predict)
    total = len(predict)
    num_zeros = np.sum(predict == 0)
    num_ones = np.sum(predict == 1)
    if num_ones == 0:
        entropy = -((num_zeros / total) * np.log2(num_zeros / total) + (num_ones / total))
    elif num_zeros == 0:
        entropy = -((num_zeros / total) + (num_ones / total) * np.log2(num_ones / total))
    else:
        entropy = -((num_zeros / total) * np.log2(num_zeros / total) + (num_ones / total) * np.log2(num_ones / total))
    return abs(entropy)

df2_2 = pd.read_csv('./Results2_2.csv')
df2_2['gt'] = [0 if i == 'male' else 1 for i in df2_2.label]
df2_2['pred'] = [0 if i == 'male' else 1 for i in df2_2.prediction]
df2_2['id'] = df2_2['trial'] + '__' + df2_2['base_name'] + '__' + df2_2['track_id'].astype(str)

track2_2 = df2_2.groupby(['id'])['pred']
group_entropies = track2_2.apply(lambda x: entropy(x))
gt_mean = df2_2.groupby(['id'])['pred'].mean()
male_mask = gt_mean >= 0.5
#male_mask = gt_mean > -0.5
female_entropies = group_entropies[male_mask]

q3_entropy = group_entropies.quantile(0.90)
entropy_mask = female_entropies > q3_entropy
#entropy_mask = female_entropies > -1
high_male_entropies = female_entropies[entropy_mask]

temp_met = pd.read_csv('./track_decision_trees.csv')
temp_fix = temp_met[temp_met.id.isin(list(high_male_entropies.index))]
fix_df2_2 = df2_2[df2_2.id.isin(list(temp_fix.id))]

temp_fix['test1'] = temp_fix.mean_acceleration <= 0.384
temp_fix['test2'] = temp_fix.distance_traveled <= 4332.171
temp_fix['test3'] = temp_fix.distance_traveled <= 5456.663
temp_fix['test4'] = temp_fix.mean_acceleration <= 0.27
temp_fix['test5'] = temp_fix.speed <= 3.501

test2 = temp_fix[temp_fix.test1 == True]
test4 = test2[test2.test2 == True]
results1 = test4[test4.test4 == True]
results1['p_m'] = 5 / 106
results1['p_f'] = 101 / 106

results2 = test4[test4.test4 == False]
results2['p_m'] = 13 / 72
results2['p_f'] = 59 / 72

test5 = test2[test2.test2 == False]
results3 = test5[test5.test5 == True]
results3['p_m'] = 49 / (135 + 49)
results3['p_f'] = 135 / (135 + 49)

results4 = test5[test5.test5 == False]
results4['p_m'] = 43 / (19 + 43)
results4['p_f'] = 19 / (19 + 43)

test3 = temp_fix[temp_fix.test1 == False]
results5 = test3[test3.test3 == True]
results5['p_m'] = 41 / (41 + 18)
results5['p_f'] = 18 / (41 + 18)

results6 = test3[test3.test3 == False]
results6['p_m'] = 48 / (48 + 5)
results6['p_f'] = 5 / (5 + 48)

results_list = [results1, results2, results3, results4, results5, results6]
combined_results = pd.concat(results_list, ignore_index=True)
classifier = fix_df2_2.groupby('id')[['c_female', 'c_male']].mean()
c_female_dict = classifier['c_female'].to_dict()
c_male_dict = classifier['c_male'].to_dict()

# Add new columns to combined_results using map
combined_results['c_female'] = combined_results['id'].map(c_female_dict)
combined_results['c_male'] = combined_results['id'].map(c_male_dict)
fcw = combined_results.c_female
mcw = combined_results.c_male
ftw = combined_results.p_f
mtw = combined_results.p_m

# Subset accuracy calculation
print("Subset Accuracy Calculation:")
combined_results['fx'] = 0.5 * (fcw) + 0.5 * (ftw)
combined_results['mx'] = 0.5 * (mcw) + 0.5 * (mtw)
combined_results['male_pred'] = np.where(combined_results['mx'] > combined_results['fx'], 1, 0)

############## VAR RENAMED ONLY ##############
subset_acc = (combined_results['male_pred'] == combined_results['is_male']).sum() / len(combined_results.index)
print("Subset Accuracy:", subset_acc)

combined_results['fx'] = 1 * (fcw) + 0 * (ftw)
combined_results['mx'] = 1 * (mcw) + 0 * (mtw)
combined_results['male_pred'] = np.where(combined_results['mx'] > combined_results['fx'], 1, 0)
############## VAR RENAMED ONLY ##############
subset_acc = (combined_results['male_pred'] == combined_results['is_male']).sum() / len(combined_results.index)
print("Subset Accuracy with (mcw, fcw) only:", subset_acc)

combined_results['fx'] = 0 * (fcw) + 1 * (ftw)
combined_results['mx'] = 0 * (mcw) + 1 * (mtw)
combined_results['male_pred'] = np.where(combined_results['mx'] > combined_results['fx'], 1, 0)
############## VAR RENAMED ONLY ##############
subset_acc = (combined_results['male_pred'] == combined_results['is_male']).sum() / len(combined_results.index)
print("Subset Accuracy with (mtw, ftw) only:", subset_acc)

combined_results['female_pred'] = np.where(combined_results['fx'] > combined_results['mx'], 1, 0)

# Update track2_2['pred'] using combined_results
track2_2 = df2_2.groupby(['id'])[['gt', 'pred']].mean()

track2_2.loc[track2_2.index.isin(combined_results['id']), 'pred'] = combined_results.set_index('id')['female_pred']

# Calculate overall accuracy
pred_values = np.where(np.array(track2_2['pred'].values) > 0.5, 1, 0)
total_acc = (track2_2['gt'].values == pred_values).sum() / len(pred_values)
print("Total Accuracy:", total_acc)
print("Number of replaced values:", track2_2.index.isin(combined_results['id']).sum())

