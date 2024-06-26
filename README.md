# Bio-Boost

## Abstract
Object detection has numerous applications, from animal tracking to autonomous vehicles. However, certain object classes exhibit substantial similarity, posing challenges for accurate classification from single images. Standard object detection models may struggle to achieve high precision in such tasks. We propose a novel annotation methodology that combines detection with tracking to generate robust annotation sets from videos for training object detection and classification models with minimal additional annotation effort. By leveraging tracking to identify multiple instances of individual objects, our approach enriches the annotation data efficiently. The BioBoost model trained on our generated annotations achieved an accuracy of 99.6% demonstrating superior performance compared to those relying on traditional annotation methods, without demanding significant supplementary annotation time. This technique is particularly promising for datasets where object tracking can be leveraged to augment annotations in a highly efficient manner, minimizing additional annotations. 

## Experimental Setup
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/setup.PNG)

## Variance Challenges
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/variance.PNG)

## Models Used
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/models.PNG)

## Temporal Model Decision Tree
![](https://github.com/Human-Augment-Analytics/Bio-Boost/blob/main/imgs/tree_viz_1.PNG)
