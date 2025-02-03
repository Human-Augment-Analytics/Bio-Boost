# Bio-Boost

## Abstract
Object detection has numerous applications, from animal tracking to autonomous vehicles. However, certain object classes exhibit substantial similarity, posing challenges for accurate classification from single images. Standard object detection models may struggle to achieve high precision in such tasks. We propose a novel annotation methodology that combines detection with tracking to generate robust annotation sets from videos for training object detection and classification models with minimal additional annotation effort. By leveraging tracking to identify multiple instances of individual objects, our approach enriches the annotation data efficiently. The BioBoost model trained on our generated annotations achieved an accuracy of 99.6% demonstrating superior performance compared to those relying on traditional annotation methods, without demanding significant supplementary annotation time. This technique is particularly promising for datasets where object tracking can be leveraged to augment annotations in a highly efficient manner, minimizing additional annotations. 

## Experimental Setup
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/setup.PNG)

## Usage
### Object Detection and Preprocessing
**The following scripts are responsible for preprocessing the video data and running object detection on the footage.**  
- **detect_all.sh**: Iterates over `.mp4` videos in the input directory, runs YOLO object detection using specified weights through the `detect.py` script from YOLOv5 GitHub, and saves the detection outputs in organized subdirectories.
  - **Invoke**: `bash detect_all.sh`
- **extract_videos.sh**: Iterates over each `.bag` file in the input directory, extracts images from each file, and compiles frames into MP4 videos 
  - **Invoke**: `bash extract_videos.sh <input_directory>`
- **run_yolo.py**: Runs the machine learning workflows using YOLO. It includes two custom classes, `ImageFolderWithPaths` and `YOLOModel`, which are responsible for handling input data, training and prediction.
    - **Invoke**: `python3 run_yolo.py`
- **cocco2yolo.ipynb**: Converts a COCO-formatted dataset into YOLO format.
    - **Invoke**: 
        - Open in Jupyter Notebook
        - Set the parameters: SRC, IMGS, DEST, and mode
        - Run the notebook cells sequentially 

### Tracking 
**The following scripts are responsible for adding tracks to objects in the annotated footage.**  
- **run_sort_fish_detections.py**: Implements modified SORT that retains the object classification information from YOLO detections with the SORT run.
    - **Invoke**: `python3 run_sort_fish_detections.py <InfileDir> <DetectionsFile> <TracksFile> <BaseName>`
- **run_sort_fish_tracking_preparer.py**: Contains `FishTrackingPreparer` class with functionality that supports: detecting fish objects and classifies them with YOLOv5, automatically identifies bower locations and pertinent information about the bowers.
    - **Invoke**: Import and instantiate the class in separate script.
- **run_sort_cluster_track_association_preparer.py**: Contains `ClusterTrackAssociationPreparer` class that takes in directory information and supports: identifying trays using manual input, interpolating and smoothing depth data, automatically identifying bower locations and analyzing pertinent information about the bowers. 
    - **Invoke**: Import and instantiate the class in separate script.

### Temporal Model Integration
**The following script is responsible for integrating data from the temporal model.**
- **run_temporal_model.py**: Integrates data from the temporal model to refine classification by identifying uncertain predictions using entropy, applying decision rules with temporal features, and weighting classifier confidence with True Positive Rates.
    - **Invoke**: `python3 run_temporal_model.py`

### Evaluation
**The following scripts are responsible for evaluating the results from the previous sections.**
- **manual_annotation_variance_analysis.py**: Evaluates and visualizes the consistency of annotations made by two different annotators. 
    - **Invoke**: `python3 manual_annotation_variance_analysis.py`
- **results_class_metrics.py**: Contains class that can be used to calculate various metrics for binary classification (ex: precision, recall, Fm, etc).
    - **Invoke**: Import and instantiate the class in separate script.
- **results_entropy.py**: Calculates entropy for groups, and calculates accuracy for each percentile. 
    - **Invoke**: `python3 results_entropy.py`

## Variance Challenges
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/variance.PNG)

## Models Used
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/models.PNG)

## Temporal Model Decision Tree
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/tree_viz_1.PNG)
