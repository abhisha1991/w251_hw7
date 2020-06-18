print("Starting facial detection task...")
import numpy as np
import cv2
import time
import requests
import random
import os
import paho.mqtt.client as paho
import uuid

# init mqtt
broker = "mosquitto1"
port = 1883
def on_publish(client,userdata,result):
    print("data published")

client1 = paho.Client("P1tx2")
client1.on_publish = on_publish
client1.connect(broker, port)
print("Connected to broker")

# create temp directory for images
path = os.getcwd() + "/img"
try:
    os.makedirs(path, exist_ok = True)
except e:
    print("Could not create image directory")
    raise e

# download the xml file for the model
xml_file_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
xml_file = "haarcascade_frontalface_default.xml"
r = requests.get(xml_file_url)
with open(xml_file, 'wb') as f:
    f.write(r.content)

# set up classifier
face_cascade = cv2.CascadeClassifier(xml_file)

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame not captured, sleep 5 seconds
    if ret is False:
        time.sleep(5)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # bad design - writing and then re-reading image, needs to be revisited
        img_name = "{0}/image-{1}.jpg".format(path, str(uuid.uuid4()))
        cv2.imwrite(img_name, roi_gray)
        image = cv2.imread(img_name)
        # finally send the image via mqtt
        ret = client1.publish("fdimagestx2/test", bytes(image))
        print("sent image data!")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
