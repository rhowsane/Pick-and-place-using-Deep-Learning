This directory has the following tasks for each file/subdirectory:

## 1. obj.data: 
It contains the paths that will be used during the training to look for the files 
on Google Drive. If you use a quantity different from 4 objects, you should change to
the quantity of classes you're really using.

## 2. obj.names:
It contains the names used in blender to identify the classes, preferably written in numbers 
from 1 to n. 

## 3. yolov3_custom.cfg
It is the configuration file necessary to the the yolov3 training.


## 4. For the Training:
You can use the collab below as model to train your data and obtain a Loss vs iteration graph:

https://colab.research.google.com/drive/16OHPS1bnKd9SMhM7vZVmsDcyIA0A4tr9?usp=sharing
