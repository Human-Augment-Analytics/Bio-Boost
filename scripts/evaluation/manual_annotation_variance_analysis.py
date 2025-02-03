"""
Manual Annotation, Data Filtering

The following script was used to evaluate the consistency of the annotations
made by two different annotators. The annotations were first compared with 
metrics and then visualized to identify differences.
"""

# Encoding, etc.
# -*- coding: utf-8 -*-

# Imports
from typing import List, Iterable
import pycoral.adapters.detect as pc
import fiftyone as fo
import os
import pdb
import pandas as pd
import numpy as np
from fiftyone import ViewField as F
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Annotation File Paths and File Reading
root_1='/path_to_dataset/blind_annotation_testing/Annotator1_CVAT_dataset/'
root_2='/path_to_dataset/blind_annotation_testing/Annotator2_CVAT_dataset/'
labels_start='obj_train_data/obj_train_data/'
label_end='.txt'
annotator1='Annotator1'
annotator2='Annotator2'
text1 = glob.glob(root_1+labels_start+'*.txt')
text2= glob.glob(root_2+labels_start+'*.txt')

def convert(yolo_list: List, flag: str = 'off') -> List:
    """
    Transform Bounding Box Coords from Yolo Format to Calculation Format

    Parameters:
        yolo_list (List): Bounding box coordinates in YOLO format
        flag (str): Flag determining if first two indices are kept the same.
    
    Returns:
        List: Bounding box coordinates in Calculation format
    """

    x1=yolo_list[0]-(yolo_list[2]/2)
    y1=yolo_list[1]-(yolo_list[3]/2)
    x2=yolo_list[0]+(yolo_list[2]/2)
    y2=yolo_list[1]+(yolo_list[3]/2)
    if flag=='on':
        x1=yolo_list[0]
        y1=yolo_list[1]
        x2=yolo_list[0]+(yolo_list[2])
        y2=yolo_list[1]+(yolo_list[3])
    return [x1,y1,x2,y2]

def find_bad(location: List[List], label: List[List], bbox: Iterable) -> tuple:
    """
    Find and Remove Inconsisten Bounding Box Annotations

    Parameters:
        location (list of lists): List of lists of bounding box coordinates
        label (list of lists): List of lists of class labels for bounding boxes
        bbox (array-like): Bounding box to compare against others.
    
    Returns:
        tuple: Tuple containing locations and labels where bad annotations are removed.
    """
    bbox=convert(bbox, flag='on')
    bad_box=pc.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
    iou_map=[]
    for a in location:
        score=[]
        for i in a:
            i=convert(i)
            iou=pc.BBox.iou(bad_box, pc.BBox(i[0], i[1], i[2], i[3]))
            score.append(iou)
        iou_map.append(np.argmax(score))

    if iou_map[0]==iou_map[1]:
        location[0]=location[0][:iou_map[0]]+location[0][iou_map[0]+1:]
        location[1]=location[1][:iou_map[0]]+location[1][iou_map[0]+1:]
        label[0]=label[0][:iou_map[0]]+label[0][iou_map[0]+1:]
        label[1]=label[1][:iou_map[0]]+label[1][iou_map[0]+1:]
    else:
        pdb.set_trace()
    return location, label

def crop_annotation(location: List[List], label: List[List], imageid: str) -> tuple:
    """
    Adjust Bounding Box Annotations

    Parameters:
        location (list of lists): List of lists of bounding box coordinates
        label (list of lists): List of lists of class labels.
        imageid (str): Unique image identifier. 
    
    Returns:
        tuple: Contains the adjusted bounding box location and label.
    """
    dataset = fo.load_dataset("master_dataset")
    sub_ds=dataset.match(F('image_uid').ends_with(imageid))
    for sample in sub_ds:
        for detection in sample.gt_MF.detections:
            if detection.truncated==True:
                new_location, label=find_bad(location, label, detection.bounding_box)
            elif detection.reflected==True:
                new_location, label=find_bad(location, label, detection.bounding_box)
            else:
                new_location=location
    return new_location, label

def iou_mapping(location: List[List], label: List[List]) -> tuple:
    """
    Align and Match Bounding Boxes of the Two Annotators Based on IoU Scores

    Parameters:
        location (list of list of floats): Contains data representing bounding boxes 
        label (list of lists): Contains labels for bounding boxes
    
    Returns:
        tuple: Contains the aligned/matched bounding boxes, and laebls for the bounding boxes
    """

    a=location[0]
    b=location[1]
    print(3)
    new_a=[]
    new_b=[]
    for i in a:
        i=convert(i)
        new_a.append(pc.BBox(i[0], i[1], i[2], i[3]))
    for i in b:
        i=convert(i)
        new_b.append(pc.BBox(i[0], i[1], i[2], i[3]))
    iou_map=[]
    for i in new_a:
        score=[]
        for j in new_b:
            iou=pc.BBox.iou(i, j)
            score.append(iou)
        iou_map.append(np.argmax(score))
    
    if set(iou_map)==set(range(len(score))):
        c=[]
        label_edit=[]
        for i in iou_map:
            c.append(b[i])
            label_edit.append(label[1][i])
        return [a,c], [label[0], label_edit]
    else:
        pdb.set_trace()

def associate_files(paths: List[str], uid: str) -> List:
    """
    Load and Process Annotation Data from Files

    Parameters:
        paths (List[str]): List of file paths with annotation data.
        uid (str): Unique identifier used for further processing
    
    Returns:
        List: Processed labels from inputted files.
    """
    label=[]
    location=[]
    for i in paths:
        file = open(i, 'r')
        data1=[]
        data2=[]
        for line in file:
            data=[float(x) for x in line.strip().split(' ')]
            data1.append(int(data[0]))
            data2.append(data[1:])
        label.append(data1)
        location.append(data2)
        file.close()
    print(2)
    location, label=iou_mapping(location,label)
    location, label=crop_annotation(location, label, uid)
    if np.sum(np.abs(np.array(location[0])-np.array(location[1]))) < 0.05:
        return label
    else:
        pdb.set_trace()

# Load Dataset from FiftyOne
dataset = fo.load_dataset("master_dataset")

# Load and Processes Data, Aggregates and Analyzes Data
out_count=0
a_df = pd.DataFrame(columns=['trialid', 'imageid', 'detectionid', annotator1, annotator2])
for sample in dataset:
    if os.path.exists(root_1+labels_start+sample.image_uid+label_end):
        if os.path.exists(root_2+labels_start+sample.image_uid+label_end):
            print(1)
            path1=root_1+labels_start+sample.image_uid+label_end
            path2=root_2+labels_start+sample.image_uid+label_end
            paths=[path1,path2]
            data=associate_files(paths, sample.image_uid)
        else:
            print('something is wrong here 1')
            print(sample)
    else:
        out_count+=1
        data=None
    if data!=None:
        detection_num=len(data[0])
        label1=data[0]
        label2=data[1]
        for i in range(detection_num):
            new_row = {'trialid': sample.projectID, 'imageid': sample.image_uid, 'detectionid': i, annotator1: label1[i], annotator2: label2[i]}
            a_df = pd.concat([a_df, pd.DataFrame([new_row])], ignore_index=True)
a_df[annotator1] = a_df[annotator1].astype(int)
a_df[annotator2] = a_df[annotator2].astype(int)
a_df['difference']=abs(a_df[annotator1]-a_df[annotator2])
df1 = a_df.groupby('trialid')['difference'].sum()
df2 = a_df.groupby('trialid')['trialid'].count()
acc=[1]*len(df1.values)-(df1.values/df2.values)
df=pd.DataFrame(columns=['trial', 'acc'])
df['trial']=list(df1.index)
df['acc']=acc

# Compare the Proportion of Male Detections in Annotations Made by the Annotators
males=a_df[a_df[annotator2]==0].groupby('trialid')['trialid'].count()
total=a_df.groupby('trialid')['trialid'].count()
missing=list(set(total.index)-set(males.index))
for i in missing:
    males[i]=0
males=males.reindex(total.index)
bt_male=males.values/total.values
print(males.index==df.trial)
males=a_df[a_df[annotator1]==0].groupby('trialid')['trialid'].count()
total=a_df.groupby('trialid')['trialid'].count()
missing=list(set(total.index)-set(males.index))
for i in missing:
    males[i]=0
males=males.reindex(total.index)
p_male=males.values/total.values
print(males.index==df.trial)

# Create DF to Summarize and Store Results of Detections for Different Trials
final_df=pd.DataFrame(columns=['trial', 'PBT_acc', 'P_pmales', 'BT_pmales'])
final_df['trial']=df.trial.copy()
final_df['PBT_acc']=df.acc.copy()
final_df['P_pmales']=p_male.copy()
final_df['BT_pmales']=bt_male.copy()
final_df.to_csv('./PBT_acc_compare_final.csv')

# Perform Calculations to Compare Proportions of Male Detections
PBT_Amales=(bt_male.copy()+p_male.copy())/2
PBT_Imales=(np.array([1]*len(PBT_Amales))-PBT_Amales)
PBT_racc=[]
for i in range(len(PBT_Amales)):
    val=max(PBT_Amales[i], PBT_Imales[i])
    PBT_racc.append(val)
PBT_racc=np.array(PBT_racc)
PBT_acc=df.acc.copy()

# Generate Sample Data
x = PBT_acc
y =  PBT_racc

# Create Scatter Plot, Add Labels, and Show Plot
plt.scatter(x, y)
#plt.plot( [0, 0.6], [0.4, 1])
#plt.annotate(trial[1].split('c')[1].split('T')[0], (x[1], y[1]))
#plt.annotate(trial[0].split('c')[1].split('T')[0], (x[0], y[0])) 
#plt.annotate(trial[22].split('c')[1].split('T')[0], (x[22], y[22]))
#plt.annotate(trial[2].split('c')[1].split('T')[0], (x[2], y[2])) 
plt.xlabel('PBT_acc') 
plt.ylabel('PBT_racc')
plt.show()

# Store and Organize Metrics Related to the Trials
acc_df=pd.DataFrame(columns=['trial', 'PBT_acc', 'PBT_racc'])
acc_df['trial']=df.trial.copy()
acc_df['PBT_acc']=PBT_acc
acc_df['PBT_racc']=PBT_racc

# Categorize Trials Based on Accuracy Metrics and Thresholds
bad_thres=0.15
unknown_thres=0.9
PBT_bad=set(acc_df[PBT_racc>=(PBT_acc-bad_thres)].trial)
PBT_unknown=set(acc_df[PBT_racc>=(unknown_thres)].trial)
PBT_exclude=PBT_bad- PBT_unknown

# Collect and Save McGraph Scores from Each Trial
mcgrath_score=[]
for i in df.trial:
    print(i)
    val=int(input('What is mcgrath score'))
    mcgrath_score.append(val)
acc_df['mcgrath_score']=mcgrath_score
acc_df.to_csv('final_PBT_acc.csv')

# Categorize Trials Based on McGrath Scores
mcgrath_bad=set(acc_df[acc_df.mcgrath_score<=5].trial)
mcgrath_good=set(acc_df[acc_df.mcgrath_score>5].trial)

# Print Trial Scores and Percentages
print('trial scores less than or equal to 5 and in at least meets threshold for inconclusive trial ')
print(PBT_unknown & mcgrath_bad)
print('trial scores less than or equal to 5 and in at least meets threshold for bad trial ')
print(PBT_exclude & mcgrath_bad)
print('trial scores greater than 5 and in at least meets threshold for inconclusive trial ')
print(PBT_unknown & mcgrath_good)
print('trial scores greater than 5 and in at least meets threshold for bad trial')
print(PBT_exclude & mcgrath_good)
print('union')
bpercent_bad=len(PBT_exclude & mcgrath_bad)/len(mcgrath_bad)
bpercent_unknown=len(PBT_unknown & mcgrath_bad)/len(mcgrath_bad)
gpercent_bad=len(PBT_exclude & mcgrath_good)/len(mcgrath_good)
gpercent_unknown=len(PBT_unknown & mcgrath_good)/len(mcgrath_good)
print('percent of scored bad deemed threshold bad')
print(bpercent_bad)
print('percent of scored bad deemed threshold inconclusive')
print(bpercent_unknown)
print('percent of scored good deemed threshold bad')
print(gpercent_bad)
print('percent of scored good deemed threshold inconclusive')
print(gpercent_unknown)

# Thresholding Analysis
x_values = np.arange(0.005, 0.5, 0.005)
scores = np.empty((x_values.shape[0], 2))
gscores=np.empty((x_values.shape[0], 1))
for i, threshold in enumerate(x_values):
    bad_thres=threshold
    PBT_bad=set(acc_df[acc_df.PBT_racc>=(acc_df.PBT_acc-bad_thres)].trial)
    bpercent_bad=len(PBT_bad & mcgrath_bad)/len(mcgrath_bad)
    gpercent_bad=(len(PBT_bad & mcgrath_good)/len(mcgrath_good))
    scores[i, 0]=bpercent_bad
    scores[i, 1]=1-gpercent_bad
    
    gscores[i]=gpercent_bad

# Line Plot for Threshold Values and Percentage Categorized as Bad or Good
plt.plot(x_values, scores[:,0], label='Percent Bad')
plt.plot(x_values, gscores, label='Percent Good')
plt.xlabel('Threshold')
plt.ylabel('Percent')
plt.title('Accuracy vs Threshold')
plt.legend()
plt.show()

# Line Plot for How Accuracy Changes Based on Threshold Value
plt.plot(x_values, scores.mean(axis=1), label='Average acc')
plt.xlabel('Threshold')
plt.ylabel('Percent')
plt.title('ave Accuracy vs Threshold')
plt.legend()
plt.show()

# Find the Best Threshold and Get Final Scores
final_thres=x_values[np.argmax(scores.mean(axis=1))]
final_b=scores[np.argmax(scores.mean(axis=1)), 0]
final_g=gscores[np.argmax(scores.mean(axis=1))]
marker=[]

# Assign Markers
for i in df.trial:
    if i in mcgrath_good:
        marker.append('^')
    else:
        marker.append('v')

# Scatter Plot
for i in range(len(PBT_acc)):
    if PBT_racc[i]>=PBT_acc[i]-0.125:
        plt.scatter(PBT_acc[i], PBT_racc[i], c='r', marker=marker[i])
    else:
        plt.scatter(PBT_acc[i], PBT_racc[i], c='g', marker=marker[i])

# Generate Plot to Visualize Accuracy and Rate Accuracy
xmin, xmax = plt.xlim()
plt.plot([xmin, xmax], [xmin-0.125, xmax-0.125], 'k--')
plt.xlabel('acc')
plt.ylabel('racc')
plt.title('Exlcuded versus Included Trials')
plt.show()

# Generate Plot to Visualize Metrics of Male Percentages across Trials
x=final_df['P_pmales']
y=final_df['BT_pmales']
plt.scatter(x, y)
xmin, xmax = plt.xlim()
plt.plot([xmin, xmax], [xmin, xmax], 'k--')
plt.xlabel('BT_pmales') 
plt.ylabel('P_pmales')
plt.show()

# Categorize Trial Based on Accuracy and Rate Accuracy
exclude_trial=[]
include_trial=[]
for i in range(len(df.trial.copy())):
    if PBT_racc[i]>=PBT_acc[i]-0.125:
        exclude_trial.append(list(df.trial)[i])
    else:
        include_trial.append(list(df.trial)[i])
        if mcgrath_score[i]<3:
            print(list(df.trial)[i])