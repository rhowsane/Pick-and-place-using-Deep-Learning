# Pick-and-place-using-Deep-Learning
Repo used to make a simulation using ROS of a pick-and-place operation using the Panda robot in a autonomous way through deep learning. In resume, there are two independent approaches.

First, a simple pick-and-place simulation, with no automation involved, was tested on the Gazebo simulation environment with the Panda robot. Codes of this first part are in the forked project "Panda_Arm" with proper modifications. But, it was obtained that it is an inherent difficulty for the Gazebo (in verses greaterthan the 2nd) to manage contact interactions like this. But it can be solved using  the  Gazebo-GraspFix.h  plugin.   

In  a  second  step,  it was  started  the  process  for  object recognition, ranging from the generation of synthetic data with the Blender software for training a pre-trained network with Yolov3 architecture to the analysis of the results obtained.  With the adoption of simplistic hypotheses such as camera at a fixed height, manipulation of the same set of objects, similar light intensity range both in the simulation environment and in the data generation environment. It is possible to detect the objects with a dectetion score ranging from 0.4 to 0.98, which can be sufficient under the conditions of the experiment.

REFERENCES

REDMON, J.; FARHADI, A. Yolov3: An incremental improvement. arXiv, 2018.
