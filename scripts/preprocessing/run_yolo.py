"""
YOLO

The following script manages and runs ml workflows using YOLO.
"""

# Encoding, etc.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
import shutil
import typing
import os
import subprocess
import pdb



# Include File Paths of Images in Dataset with Image Data and Labels
class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index: int) -> tuple:
        """
        Override the __getitem__ method. This is the method that dataloader calls

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            tuple: A tuple with:
                - image (Tensor): Transformed image tensor.
                - label (int): Class index of the image.
                - path (str): File path of the image.
        """

        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
# Train and Predict with Yolo
class YOLOModel:
    def __init__(self, epoch: int, run_name: str) -> None:
        """
        Initializes the YOLOModel Class with Unique Run Name.

        Parameters:
            epoch (int): Number of epochs for training.
            run_name (str): UNique identifier for the training run.
        """

        print(1)
        self.train_script = '/home/path_to_file/yolov5/train.py'
        self.detect_path = '/home/path_to_file/yolov5/detect.py'
        self.predict_run = '/home/path_to_file/yolov5/classify/predict.py'
        self.train_c = '/home/path_to_file/yolov5/classify/train.py'
        self.run_name=run_name
        self.train_path='/home/path_to_file/test/train_results/yolo'
        self.predict_path='/home/path_to_file/test/predict_results/yolo'
        self.train_log='/home/path_to_file/test/train_results/yolo/train_log_'+run_name +'.txt'
        self.predict_log='/home/path_to_file/test/train_results/yolo/predict_log_'+run_name+'.txt'
        self.epoch=epoch
    
    def train(self, data_yaml_path: str, batch_size: int = -1, imgsz: int = 640, device: int = 0, weights: str = "yolov5s.pt") -> str:
        """
        Trains YOLOv5 Model with Given Params

        Parameters:
            data_yaml_path (str): Path to the dataset's YAML configuration file.
            batch_size (int, optional): Batch size for training.
            imgsz (int, optional): Image size for training.
            device (int, optional): Device ID for training.
            weights (str), optional: Path to pre-trained weights.

        Returns:
            str: Path to the model after training
        """

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

    def predict(self, input_path: str, model_path: str, device: int = 0) -> None:
        """
        Runs Object Detection with Trained Model

        Parameters:
            input_path (str): Path to the input file.
            model_path (str): Path to the trained model.
            device (int, optional): Device ID for prediction.
        """

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
    
    def train_class(self, data_yaml_path: str, batch_size: int = 800, imgsz: int = 75, device: int = 0, model: str = "yolov5s-cls.pt") -> str:
        """
        Trains Model for Image Classification

        Parameters:
            data_yaml_path (str): Path to the dataset's YAML configuration file.
            batch_size (int, optional): Batch size for training.
            imgsz (int, optional): Image size for training.
            device (int, optional): Device ID for training.
            model (str), optional: Path to pre-trained weights for classification.

        Returns:
            str: Path to the model after training
        """    
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
    
    def predict_class(self, input_path: str, model_path: str, device: int = 0) -> None:
        """
        Runs Image Classification Using Trained Model

        Parameters:
            input_path (str): Path to the input file for classification.
            model_path (str): Path to the trained classification model.
            device (int): Device ID for prediction.
        """
        
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

def main():
    print()
    # Test YOLOv5 model
    #run_name='1_2_yolo_female_fixed'
    #yolo_model=YOLOModel(100, run_name)
    #yolo_model.predict_class('/home/path_to_file/Documents/Annotation1_1_fixed/val/Female', '/home/path_to_file/test/train_results/yolo/yolo_class_2_run/weights/best.pt')
    #run_name='1_2_yolo_male_fixed'
    #yolo_model=YOLOModel(100, run_name)
    #yolo_model.predict_class('/home/path_to_file/Documents/Annotation1_1_fixed/val/Male', '/home/path_to_file/test/train_results/yolo/yolo_class_2_run/weights/best.pt')
    #run_name='1_1_yolo'
    #yolo_model=YOLOModel(50, run_name)
    #print('YOLO train '+run_name)
    #print(yolo_model)
    #ytrain_path=yolo_model.train_class('/home/path_to_file/Documents/Annotation1_1_fixed')
    #print('YOLO detect '+run_name)

    # Run Classification Prediction on Male and Female Datasets
    run_name='1_1_yolo_female'
    yolo_model=YOLOModel(100, run_name)
    yolo_model.predict_class('/home/path_to_file/Documents/Annotation1_1_fixed/val/Female','/home/path_to_file/test/train_results/yolo/1_1_yolo2/weights/best.pt')
    run_name='1_1_yolo_male'
    yolo_model=YOLOModel(100, run_name)
    yolo_model.predict_class('/home/path_to_file/Documents/Annotation1_1_fixed/val/Male','/home/path_to_file/test/train_results/yolo/1_1_yolo2/weights/best.pt')

    # Test Resnet
    #resnet_model=ResnetModel(100, run_name)
    #print('RESNET train '+run_name)
    #rtrain_path=resnet_model.train_model('/home/path_to_file/Documents/Annotation2_0_yolo/final_resnet/dataset.yaml)
    #print('RESNET predict '+run_name)
    #resnet_model.predict('/home/path_to_file/Documents/Annotation3_0/final_resnet/validation', '/home/path_to_file/test/train_results/resnet/automated_run/automated_runweights.h5')
    #rerun for formating changes 
    #print('run 1')
    #run_name='run_500'
    #resnet_model=ResnetModel(500, run_name)
    #rtrain_path=resnet_model.train_model('/home/path_to_file/Documents/Annotation2_0/final_resnet/')
    #print('run 2')
    #resnet_model.predict('/home/path_to_file/Documents/Annotation1_1/train/', '/home/path_to_file/test/train_results/resnet/run_100/run_100weights.h5')
    #run_name='run_100_2_0_on_1_1_val'
    #resnet_model=ResnetModel(100, run_name)
    #resnet_model.predict('/home/path_to_file/Documents/Annotation1_1/validation/', '/home/path_to_file/test/train_results/resnet/run_100/run_100weights.h5')
    #print('run 3')
    #run_name='run_100_2_0_on_2_0_val'
    #resnet_model=ResnetModel(100, run_name)
    #resnet_model.predict('/home/path_to_file/Documents/Annotation2_0/final_resnet/validation/', '/home/path_to_file/test/train_results/resnet/run_100/run_100weights.h5') 


if __name__ == "__main__":
    main()