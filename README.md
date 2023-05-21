# Multi-Object object detection

This repository is an exercise for the multi-object object detection project on the PASCAL-VOC 2012 Dataset.
The PASCAL-VOC 2012 contains 20 classes of natural objects. The train/val data has 11,530 images containing 27,450 ROI 
annotated objects and 6,929 segmentations.

Original dataset was downloaded from [http://host.robots.ox.ac.uk/pascal/VOC/index.html](http://host.robots.ox.ac.uk/pascal/VOC/index.html).

# Overview

In this exercise, in order to solve the challenge of multi-object object detection in PASCAL VOC 2012 dataset, two 
different networks have been used. 
### 1- An Innovative Model.

This model includes 10,276,493 parameters, 2,640,933 of which are trainable. The input size of this model is 
(256, 256, 3) and it is trained to 220 epochs.

The structure of the model is shown in the image below:

<img src="./images/innovative model graph.png">

### 2- A model similar to SSD in which there are only convolutional layers of SSD.

This model includes 25,478,936 parameters, 10,764,248 of which are trainable. The input size of this model is 
(300, 300, 3) and it is trained to 300 epochs.

The structure of the model is shown in the image below:

<img src="./images/ssd-like model graph.png">

***Both of these models uses anchor boxes for object detection problem***

These two networks have been trained in Google colab, and for their training, 80% of PASCAL data has been considered as train 
data and 20% of them as validation data.

loss curves for both models are shown in figure below:

### loss curves of innovative model

<img src="./images/class_loss innovative model.png">
<img src="./images/offset_loss innovative model.png">

### loss curve of SSD-Like model

<img src="./images/loss SSD-like model.png">

## Usage

### Requirements

- python 3.9
- Tensorflow == 2.11.0
- pandas == 1.5.3
- numpy == 1.24.2
- matplotlib == 3.6.3
- keras~=2.11.0
- scikit-learn~=1.1.1
- lxml == 4.9.2

### Train

In order to train the innovative network, run file ***train_innovative_model.py*** and in order to train the SSD-Like
network, run file ***train_ssd_like_model.py***.

### Predict

In order to predict using the innovative network, run file ***prediction_innovative_model.py*** and in order to predict 
using the SSD-Like network, run file ***prediction_ssd_like_model.py***.

## Examples

### Innovative Model Examples

<img src="./images/prediction innovative model/fig 1.png">
<img src="./images/prediction innovative model/fig 2.png">
<img src="./images/prediction innovative model/fig 3.png">

### SSD-Like Model Examples

<img src="./images/prediction ssd_like model/fig 1.png">
<img src="./images/prediction ssd_like model/fig 2.png">
<img src="./images/prediction ssd_like model/fig 3.png">

## TODO

Both of these models have a little problem of over-fitting, and the reason for that is not augmenting the data, which 
will be solved in the next versions.

## License
This repository is released under [Apache License V2](http://www.apache.org/licenses/LICENSE-2.0). To develop,
publication and use it, please follow the terms of this license.
