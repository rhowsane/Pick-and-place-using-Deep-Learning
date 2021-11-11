This directory has the following tasks for each file/subdirectory:

## 1. run_detection_robust.py: 
It uses openCV to detect the objects in photos or videos. You should 

## 2. classes.txt:
It contains the names used in blender to identify the classes, preferably written in numbers 
from 0 to n. 

## 3. yolov3_custom.cfg
It is the configuration file necessary to the the yolov3 training. Details on how to change them, you can check on:

LEVER, T. Training YOLOv3 Convolutional Neural Networks Using darknet. 2019.
https://medium.com/@thomas.lever.business/training-yolov3-convolutional-neural-networks-using-darknet-2f429858583c

## 4. Create 3 directories:
## 4.1. Data test
Containing 2 directories "Images" and "Videos", so that you can insert the files you want 
them to be detected.

## 4.2. Output
Containing 2 directories "Images_resul" and "Videos_resul", so that you can get the output of the detection.

## 4.3. Weights
Insert the weights obtained from Google Collab.
