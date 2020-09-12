# Facemask-detection using keras and Opencv 

A Deep Learning Project to detct facemask in Live.This project is implemented in Python using Keras, Tensorflow as Backend and OpenCV.

## Dataset

The original dataset is prepared by [Prajna Bhandary](https://www.linkedin.com/in/prajna-bhandary-0b03a416a/) and available at [Github](https://github.com/prajnasb/observations/tree/master/experiements/data)

## Step 1 : Pre-processing the data
The data set consists of images with mask and without mask. Take two lists named withmask and withoutmask and append all the images of withmask and wihout mask in to the lists respectively
## Step 2 : Using Convolutional Neural Network(CNN) to build model
The Layers should be added are  <br/>
1.2 Conv2d Layers each layer with a Relu layer followed by pooling layer <br/>
2.Flatten layer  <br/>
3.Dropout layer <br/>
4.Dense layer  <br/>
5.softmax layer  <br/>

## Step 3 : Using OpenCV to Detect the Face Mask
First Load the saved model and import the required libraries such as Opencv and do not forget to download ‘haarcascade_frontalface_default.xml’ classifier you can download from the below link.

## Outputs

<img src="https://github.com/Shoyabsyed4m3/FaceMask-Detection/blob/master/Outputs/withoutmask1.PNG" >
<img src="https://github.com/Shoyabsyed4m3/FaceMask-Detection/blob/master/Outputs/withmask1.PNG" >
<img src="https://github.com/Shoyabsyed4m3/FaceMask-Detection/blob/master/Outputs/mask.PNG" >

