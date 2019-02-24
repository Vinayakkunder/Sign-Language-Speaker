import cv2
import numpy as np
import pyttsx3
import time
<<<<<<< HEAD
import threading

=======
import os
from threading import Thread
>>>>>>> f4e6d5685a06785f980be80b61d816a82c0c2d38
def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')
tts = pyttsx3.init()
img_text = ''
def speaker():
    voices = tts.getProperty('voices')
    rate = tts.getProperty('rate')
    tts.setProperty('rate', rate+20)
    tts.setProperty('voice', voices[1].id)
    global img_text
    tts.say(img_text)
    tts.runAndWait()
    
curr_T = None


#
def say_text(text):
	engine = pyttsx3.init()
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
<<<<<<< HEAD
       
       global img_text
       
       if result[0][0] == 1:
              return 'None'
            
       elif result[0][1] == 1:
           #img_text='I am V.k.'
           return 'I am V.k.'
       elif result[0][2] == 1:
           #img_text='Whats up'
           return 'Whats up'
       elif result[0][3] == 1:
           #img_text='I am Hungry'
           return 'I am Hungry'
       '''elif result[0][4] == 1:
           return 'ML' '''
       
        
       global curr_T
       if curr_T == None:
           curr_T = threading.Thread(target=speaker, args=())
           curr_T.start()
       elif curr_T.isAlive() == False:
           curr_T = threading.Thread(target=speaker, args=())
           curr_T.start()

=======
       engine = pyttsx3.init()
       voices = engine.getProperty('voices')
       rate = engine.getProperty('rate')
       engine.setProperty('rate', rate-50)
       engine.setProperty('voice', voices[1].id)
       if result[0][0] == 1:
           return 'Hello'
              
       elif result[0][1] == 1:
           
           info = "Hello"
#          Thread(target=say_text, args=(info,)).start()      Problem here.  Please try with and without this line
           return 'This'
       elif result[0][2] == 1:
           return ''
       elif result[0][3] == 1:   
           info = "is"
#           Thread(target=say_text, args=(info,)).start()   
           return 'Is'
       elif result[0][4] == 1:
           info = "python"
#           Thread(target=say_text, args=(info,)).start()   
           return 'Python'
  
       elif result[0][5] == 1:
           return 'F'
       elif result[0][6] == 1:
           return 'G'
       elif result[0][7] == 1:
           return 'H'
       elif result[0][8] == 1:
           return 'I'
       elif result[0][9] == 1:
           return 'J'
       elif result[0][10] == 1:
           return 'K'
       elif result[0][11] == 1:
           return 'L'
       elif result[0][12] == 1:
           return 'M'
       elif result[0][13] == 1:
           return 'N'
       elif result[0][14] == 1:
           return 'O'
       elif result[0][15] == 1:
           return 'P'
       elif result[0][16] == 1:
           return 'Q'
       elif result[0][17] == 1:
           return 'R'
       elif result[0][18] == 1:
           return 'S'
       elif result[0][19] == 1:
           return 'T'
       elif result[0][20] == 1:
           return 'U'
       elif result[0][21] == 1:
           return 'V'
       elif result[0][22] == 1:
           return 'W'
       elif result[0][23] == 1:
           return 'X'
       elif result[0][24] == 1:
           return 'Y'
       elif result[0][25] == 1:
           return 'Z'
      
>>>>>>> f4e6d5685a06785f980be80b61d816a82c0c2d38
       

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

<<<<<<< HEAD

=======
img_text = ''
import time 
>>>>>>> f4e6d5685a06785f980be80b61d816a82c0c2d38
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")


    #img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
<<<<<<< HEAD
    #imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
=======
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
>>>>>>> f4e6d5685a06785f980be80b61d816a82c0c2d38
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    #save_img = cv2.resize(mask, (image_x, image_y))
    save_img = cv2.resize(mask,None,fx=0.1,fy=0.1, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
<<<<<<< HEAD
=======

>>>>>>> f4e6d5685a06785f980be80b61d816a82c0c2d38
    
    
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()