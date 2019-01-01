import cv2
import time
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.mkdir('./mydata/training_set/' + folder_name)
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.mkdir('./mydata/test_set/' + folder_name)
