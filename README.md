# Sign-Language-Speaker
It is a simple CNN model to recognise sign language(Gestures).This software acts as a translator between deaf/mute people and a normal people, basically it can detect a gesture based on trained dataset and speak the output.This is a pretrained model with 6 gestures, however you can add more.OpenCV is used for live detection of the hand gestures performed by the user.We are using indian sign langauge to train the model.

# Requirements
1.python 3.x  
2.OpenCV 3.4.2   
3.Keras  
4.Pyttsx3   

# Project Files Usage
Download and extract files, install all the dependencies, run below files as required.
The project is divided into 3 main files :-
## 1.Recognise.py
Run this file and set the HSV values on the Trackbar window, typically hsv values are L-(0,51,61) and H-(0,0,0), set the values according to your background.Make sure that on hsv window your background should be black on only foreground should be white.
Make sure that background is plane, and less noisy.

# Adding your own gesture
## 2.Capture.py
If you want to add your own gesture to the system, first generate the dataset first.To do so, follow:-    
1. Run capture.py  
2.Enter your gesture name  
3.Set the HSV values according to your Background  
4.Press 'C' to capture the frame, which will save into dataset folder.  
5.Click minimum 250 images(Training set: 1-200, Test set: 201-250)  

## 3.cnn_model.py
After generation of dataset of your gesture, simply run this file to train the model in order to validate the gesture in action.This file will fetch the dataset from 'mydata' folder and training will start, after completion the .h5 file will be generated as 'Trained_model.h5'.   

Note: Every time after new dataset generations, running this file is mandatory.Otherwise it will predict based on old gestures.
