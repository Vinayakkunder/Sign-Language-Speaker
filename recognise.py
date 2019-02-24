import cv2
import numpy as np
import pyttsx3
import time
import threading

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


def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
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
    #imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    
    
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()