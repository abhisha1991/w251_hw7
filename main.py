print("Starting facial detection task...")
import numpy as np
import cv2
import time
import requests
import random
import os
import paho.mqtt.client as paho
import uuid

# create temp directory for images
path = os.getcwd() + "/img"
try:
    os.makedirs(path, exist_ok = True)
except e:
    print("Could not create image directory")
    raise e

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_name_gray = "{0}/image-gray-{1}.jpg".format(path, str(uuid.uuid4()))
    img_name_color = "{0}/image-color-{1}.jpg".format(path, str(uuid.uuid4()))
    print("captured an image...")
    #cv2.imwrite(img_name_gray, gray)
    #cv2.imwrite(img_name_color, frame)
    time.sleep(5)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
