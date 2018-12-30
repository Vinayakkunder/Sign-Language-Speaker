# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:00:58 2018

@author: DarkArmy
"""

import cv2
import numpy as np
import pyttsx3
from keras.models import load_model
classifer = load_model('Trained_model.h5')

image_x, image_y = 64,64

def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)