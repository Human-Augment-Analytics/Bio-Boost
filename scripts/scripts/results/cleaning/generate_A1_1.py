#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:08:46 2024

@author: bshi
"""
import os
import cv2

label_directory = "/home/bshi/Documents/Annotation1_0/labels/train/"
image_directory = "/home/bshi/Documents/Annotation1_0/images/train/"
out_directory='/home/bshi/Documents/Annotation1_1_fixed/train/'
width=1296
height=972
for filename in os.listdir(label_directory):
    file_path = os.path.join(label_directory, filename)
    with open(file_path, 'r') as file:
        data = [line.strip().split(' ') for line in file]
        locations=[[float(y) for y in x[1:]] for x in data]
        classes=[int(x[0]) for x in data]
        img_name=filename.split('.txt')[0]+'.jpg'
        img = cv2.imread(image_directory+img_name)
        mcount=0
        fcount=0
        for idx in range(len(locations)):
            if classes[idx]==0:
                outFile=out_directory+'Male/'+img_name.split('.jpg')[0]+'_M'+str(mcount)+'.jpg'
                mcount+=1
            else:
                outFile=out_directory+'Female/'+img_name.split('.jpg')[0]+'_F'+str(fcount)+'.jpg'
                fcount+=1
            w=int(width*locations[idx][2])
            h=int(height*locations[idx][3])
            xc=int(width*locations[idx][0])
            yc=int(height*locations[idx][1])
            delta_xy=(int((1/2)*max(int(w)+1, int(h)+1)))+10
            frame = img[int(max(0, yc - delta_xy)):int(min(yc + delta_xy,height)) , int(max(0, xc - delta_xy)):int(min(xc + delta_xy, width))]
            frame=cv2.resize(frame, (100, 100))
            cv2.imwrite(outFile, frame) 
filter_images=['MC_singlenuc23_1_Tk33_021220_0003_vid_626994_M0.jpg',
 'MC_singlenuc81_1_Tk51_072920_0001_vid_28392_F0.jpg',
 'MC_singlenuc81_1_Tk51_072920_0001_vid_16997_M0.jpg',
 'MC_singlenuc40_2_Tk3_030920_0001_vid_17689_F0.jpg',
 'MC_singlenuc35_11_Tk61_051220_0001_vid_213884_F0.jpg',
 'MC_singlenuc35_11_Tk61_051220_0001_vid_213884_F1.jpg',
 'MC_singlenuc24_4_Tk47_030320_0002_vid_165181_F0.jpg',
 'MC_singlenuc24_4_Tk47_030320_0002_vid_165181_F1.jpg',
 'MC_singlenuc34_3_Tk43_030320_0001_vid_375978_F0.jpg',
 'MC_singlenuc36_2_Tk3_030320_0001_vid_540706_F0.jpg',
 'MC_singlenuc86_b1_Tk47_073020_0001_vid_347442_F0.jpg',
 'MC_singlenuc64_1_Tk51_060220_0002_vid_63240_F0.jpg',
 'MC_singlenuc94_b1_Tk31_081120_0001_vid_308168_F0.jpg']

label_directory = "/home/bshi/Documents/Annotation1_0/labels/val/"
image_directory = "/home/bshi/Documents/Annotation1_0/images/val/"
out_directory='/home/bshi/Documents/Annotation1_1/validation/'
width=1296
height=972
for filename in os.listdir(label_directory):
    file_path = os.path.join(label_directory, filename)
    with open(file_path, 'r') as file:
        data = [line.strip().split(' ') for line in file]
        locations=[[float(y) for y in x[1:]] for x in data]
        classes=[int(x[0]) for x in data]
        img_name=filename.split('.txt')[0]+'.jpg'
        img = cv2.imread(image_directory+img_name)
        mcount=0
        fcount=0
        for idx in range(len(locations)):
            if classes[idx]==0:
                outFile=out_directory+'Male/'+img_name.split('.jpg')[0]+'_M'+str(mcount)+'.jpg'
                mcount+=1
            else:
                outFile=out_directory+'Female/'+img_name.split('.jpg')[0]+'_F'+str(fcount)+'.jpg'
                fcount+=1
            w=int(width*locations[idx][2])
            h=int(height*locations[idx][3])
            xc=int(width*locations[idx][0])
            yc=int(height*locations[idx][1])
            delta_xy=(int((1/2)*max(int(w)+1, int(h)+1)))+10
            frame = img[int(max(0, yc - delta_xy)):int(min(yc + delta_xy,height)) , int(max(0, xc - delta_xy)):int(min(xc + delta_xy, width))]
            frame=cv2.resize(frame, (100, 100))
            exclude=0
            if outFile.split('/')[-1] in filter_images:
                exclude+=1
            else:
                cv2.imwrite(outFile, frame) 