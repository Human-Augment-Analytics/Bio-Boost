#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:59:29 2024

@author: bshi
"""

import numpy as np
#import pytorch
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
import shutil
import os
import subprocess
import pdb

#add log that save which model is being predicted on which data

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    

class YOLOModel:
    def __init__(self, epoch, run_name):
        print(1)
        self.train_script = '/home/bshi/yolov5/train.py'
        self.detect_path = '/home/bshi/yolov5/detect.py'
        self.predict_run = '/home/bshi/yolov5/classify/predict.py'
        self.train_c = '/home/bshi/yolov5/classify/train.py'
        #result folder
        self.run_name=run_name
        self.train_path='/home/bshi/test/train_results/yolo'
        self.predict_path='/home/bshi/test/predict_results/yolo'
        self.train_log='/home/bshi/test/train_results/yolo/train_log_'+run_name +'.txt'
        self.predict_log='/home/bshi/test/train_results/yolo/predict_log_'+run_name+'.txt'
        self.epoch=epoch
        
    def train(self, data_yaml_path,batch_size=-1, imgsz=640, device=0, weights="yolov5s.pt"):
        epochs=self.epoch
        cmd = [
            'python', self.train_script,
            '--data', data_yaml_path,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img', str(imgsz),
            '--weights', weights,
            '--device', str(device),
            '--project', self.train_path,
            '--name', self.run_name
        ]
        with open(self.train_log, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
        model_path=self.train_path+'/'+self.run_name+'/weights/best.pt'
        shutil.move(self.train_log, self.train_path+'/'+self.run_name+'/'+'train_log_'+self.run_name +'.txt')
        return model_path

    def predict(self, input_path, model_path, device=0):
        cmd = [
            'python', self.detect_path,
            '--weights', model_path,
            '--iou-thres', str(0.45),
            '--conf-thres', str(0.25),
            '--source', str(input_path),
            '--device', str(device),
            '--project', self.predict_path,
            '--name', self.run_name+'_eval',
            "--save-txt","--save-csv",
            "--save-conf","--agnostic-nms"
        ]
        with open(self.predict_log, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
        shutil.move(self.predict_log, self.predict_path+'/'+self.run_name+'_eval/'+'predict_log_'+self.run_name +'.txt')
    
    def train_class(self, data_yaml_path,batch_size=800, imgsz=75, device=0, model="yolov5s-cls.pt"):
        print(1)
        epochs=self.epoch
        cmd = [
            'python3', self.train_c,
            '--data', data_yaml_path,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img', str(imgsz),
            '--model', model,
            '--device', str(device),
            '--project', self.train_path,
            '--name', self.run_name
        ]
        with open(self.train_log, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
        model_path=self.train_path+'/'+self.run_name+'/weights/best.pt'
        shutil.move(self.train_log, self.train_path+'/'+self.run_name+'/'+'train_log_'+self.run_name +'.txt')
        return model_path
    def predict_class(self, input_path, model_path, device=0):
        cmd = [
            'python', self.predict_run,
            '--weights', model_path,
            '--source', str(input_path),
            '--device', str(device),
            '--project', self.predict_path,
            '--name', self.run_name+'_eval',
            "--save-txt"
        ]
        with open(self.predict_log, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
        shutil.move(self.predict_log, self.predict_path+'/'+self.run_name+'_eval/'+'predict_log_'+self.run_name +'.txt')




        
#test yolo\
#run_name='1_2_yolo_female_fixed'
#yolo_model=YOLOModel(100, run_name)
#yolo_model.predict_class('/home/bshi/Documents/Annotation1_1_fixed/val/Female', '/home/bshi/test/train_results/yolo/yolo_class_2_run/weights/best.pt')
#run_name='1_2_yolo_male_fixed'
#yolo_model=YOLOModel(100, run_name)
#yolo_model.predict_class('/home/bshi/Documents/Annotation1_1_fixed/val/Male', '/home/bshi/test/train_results/yolo/yolo_class_2_run/weights/best.pt')
#run_name='1_1_yolo'
#yolo_model=YOLOModel(50, run_name)
#print('YOLO train '+run_name)
#print(yolo_model)
#ytrain_path=yolo_model.train_class('/home/bshi/Documents/Annotation1_1_fixed')
#print('YOLO detect '+run_name)
run_name='1_1_yolo_female'
yolo_model=YOLOModel(100, run_name)
yolo_model.predict_class('/home/bshi/Documents/Annotation1_1_fixed/val/Female','/home/bshi/test/train_results/yolo/1_1_yolo2/weights/best.pt')
run_name='1_1_yolo_male'
yolo_model=YOLOModel(100, run_name)
yolo_model.predict_class('/home/bshi/Documents/Annotation1_1_fixed/val/Male','/home/bshi/test/train_results/yolo/1_1_yolo2/weights/best.pt')

#print('YOLO DONE')

#test resnet

#resnet_model=ResnetModel(100, run_name)
#print('RESNET train '+run_name)
#rtrain_path=resnet_model.train_model('/home/bshi/Documents/Annotation2_0_yolo/final_resnet/dataset.yaml)
#print('RESNET predict '+run_name)
#resnet_model.predict('/home/bshi/Documents/Annotation3_0/final_resnet/validation', '/home/bshi/test/train_results/resnet/automated_run/automated_runweights.h5')

#rerun for formating changes 
#print('run 1')
#run_name='run_500'
#resnet_model=ResnetModel(500, run_name)
#rtrain_path=resnet_model.train_model('/home/bshi/Documents/Annotation2_0/final_resnet/')
#print('run 2')
#resnet_model.predict('/home/bshi/Documents/Annotation1_1/train/', '/home/bshi/test/train_results/resnet/run_100/run_100weights.h5')
#run_name='run_100_2_0_on_1_1_val'
#resnet_model=ResnetModel(100, run_name)
#resnet_model.predict('/home/bshi/Documents/Annotation1_1/validation/', '/home/bshi/test/train_results/resnet/run_100/run_100weights.h5')
#print('run 3')
#run_name='run_100_2_0_on_2_0_val'
#resnet_model=ResnetModel(100, run_name)
#resnet_model.predict('/home/bshi/Documents/Annotation2_0/final_resnet/validation/', '/home/bshi/test/train_results/resnet/run_100/run_100weights.h5')
