#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# with_mask list of name of files or images in data set folder 
with_mask = []

image_path = r'/content/drive/My Drive/Colab Notebooks/with_mask'
for image in os.walk(image_path):
    with_mask.append(image[2]) 
    


# without mask list of name of files or images in data set folder 
without_mask = []

image_path = r'/content/drive/My Drive/Colab Notebooks/without_mask'
for image in os.walk(image_path):
    without_mask.append(image[2]) 
    
    
#Data preprocessing

import numpy as np

with_mask = np.array(with_mask)
without_mask = np.array(without_mask)
without_mask = np.reshape(without_mask,-1)
with_mask = np.reshape(with_mask,-1)
withmask=[]

for i in with_mask:
    try: 
        arr= cv2.imread(image_path + '/' + i)
        arr = cv2.resize(arr,(100,100))                 #resize images to 100*100 pixels
        gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)     #convvert into gray image
        withmask.append(gray)
    except:
        pass
    
withoutmask=[]

for i in without_mask:
    try:
      arr= cv2.imread(image_path + '/' + i)
      arr = cv2.resize(arr,(100,100))
      gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
      withoutmask.append(gray)
    except:
      pass

xtrain = []
ytrain = []

for i in range(len(withmask)):       #I have taken dataset of withmask and without mask of same number of images
  xtrain.append(withmask[i])        # If there are different append the remaining to xtrain list 
  xtrain.append(withoutmask[i])
  ytrain.append(1)
  ytrain.append(0)

xtrain = np.array(xtrain)
xtrain = np.reshape(xtrain,(xtrain.shape[0],100,100,1))

ytrain = np.array(ytrain)


xtrain = xtrain/255   #Normalise the array

from keras.utils import np_utils
ytrain=np_utils.to_categorical(ytrain)  #convert output to categorical [1,0] or [0,1]


#CNN Model

input_shape=(100, 100, 1)

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 50 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(xtrain,ytrain,epochs=20,callbacks=[checkpoint],validation_split=0.2)
#This will upto 30mins based on your dataset

model.save('mymodel')   #to save the model



#Using OpenCv to detect facemask in live

from keras.models import load_model
import cv2
import numpy as np

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)               #To predict

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):     #press Esc to exit
        break
        
cv2.destroyAllWindows()
source.release()

