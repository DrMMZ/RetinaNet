# RetinaNet for Object Detection

[RetinaNet](https://arxiv.org/abs/1708.02002) is an efficient one-stage object detector trained with the focal loss. This repository is a TensorFlow2 implementation of RetinaNet and its applications, aiming for creating a tool in object detection task that can be easily extended to other datasets or used in building projects. It includes

1. source code of RetinaNet and its configuration;
2. source code of data (RetinaNet's inputs) generator using multiple CPU cores; 
3. source code of utilities such as image/mask preprocessing, augmetation, average precision (AP) metric, visualization and so on;
4. jupyter notebook demonstration using RetinaNet in training and real-time detection on some datasets. Below are example detections on the [nuclei](https://www.kaggle.com/c/data-science-bowl-2018) dataset randomly selected from un-trained images.

<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/nuclei_movie.gif" width='360' height='360'/>
</p> 

### Requirements
`python 3.7.9`, `tensorflow 2.3.1`, `matplotlib 3.3.4`, `numpy 1.19.2`, `opencv 4.5.1`, `scipy 1.6.0`, `scikit-image 0.17.2` and `tensorflow-addons 0.13.0`

### References
1. Lin et al., *Focal Loss for Dense Object Detection*, https://arxiv.org/abs/1708.02002, 2018
2. *Mask R-CNN for Object Detection and Segmentation*, https://github.com/matterport/Mask_RCNN, 2018
