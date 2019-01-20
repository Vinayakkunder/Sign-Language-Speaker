''' This program is to detect and predict the gesture '''

import cv2
import numpy as np

import pyttsx3
import os
import time
import threading


def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')


def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
           time.sleep(1)
           #engine.say('Hello')
           return 'Hello'
       elif result[0][1] == 1:
           time.sleep(1)
           #engine.say('This')
           return 'This'
       elif result[0][2] == 1:
           time.sleep(1)
           #engine.say('none')
           return ''
       elif result[0][3] == 1:
           time.sleep(1)
           #engine.say('is')
           return 'Is'
       elif result[0][4] == 1:
           time.sleep(1)
           #engine.say('Python')
           return 'Python'
  

       

cam = cv2.VideoCapture(1)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")


    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
    
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    #assigning predictor to a variable and defining speak function
    img_text = predictor()
    def speaker(img_text):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate-50)
        engine.setProperty('voice', voices[1].id)
        engine.say(img_text)
        time.sleep(1)
    #speaker(img_text)

    #using multithreading between predictor function and speaker function
    
    t = time.time()
    t1 = threading.Thread(target=predictor, args=() )
    t2 = threading.Thread(target=speaker, args=(img_text))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()

    
    
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
