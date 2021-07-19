# Realtime-Face-Mask-detection
In recent trend in world wide Lockdowns due to COVID19 outbreak, as Face Mask is became mandatory for everyone while roaming outside, approach of Deep Learning for Detecting Faces With and Without mask were a good trendy practice.
## Objective

The project was created to classify people with mask and without mask in real time.
Create a Deep Learning model to classify people with masks and without mask and use the model for a rea;l time feed from a camera.

### Methods Used

* Machine Learning
* Deep Learning
* Transfer Learning

### Technologies Used

* Python
* Numpy/ jupyter
* OpenCV
* Tensorflow

## Project Descrition

The [dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) for the project was taken from kaggle.
Data set consists of 7553 RGB images in 2 folders as withmask and withoutmask. Images with masks are 3725 and images of faces without mask are 3828.
Here we have used Resnet 50 as a basemodel and imagenet weights are used. The image arre augumented before training the model. During video feed 
DNN is used to capture the face in the video feed and that face is then sent to the model for predicting if the person in the video is wearing a mask or not.

## Installation
All the installations required are given in the requirement.txt

## Usage
Step 1: Use the ipynb file to train and save the model.

Step 2: Use the py file to get and model that was created and capture the face using the caffe model.

Step 3: The py file will run the camera in the desktop(if its run on a computer) and will show the result which can then be saved


## Project status
The project is COMPLETED as all the requirements for this project is completed. For people who wants to add any new feature or attribute for this project, you may submit a request
to this project maintainers.
