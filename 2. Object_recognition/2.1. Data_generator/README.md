This directory contains the following tasks for each file/subdirectory:

## 1. Data: 
It contains 2 subdirectories (imgs_rot and labels_rot) which will receive 
the files generated in Blender. And the display_annot.py, where you can check if the 
bounding boxes are involving your objects correctly.

## 2. createtrain_test_path.py: 
It will create the paths to the files generated by Blender
on your computer.

## 3. yolo_data_generator.blend: 
It contains the objects in the environment on Blender, 
including the script for the data generator. The file ".blend1" was generated automatically
by Blender.