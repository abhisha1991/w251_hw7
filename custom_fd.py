from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import time
import wget
from tf_trt_models.detection import download_detection_model, build_detection_graph

def download(url, file):
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        wget.download(url, file)
        print('Download Finished') 

FROZEN_GRAPH_NAME = 'frozen_inference_graph_face.pb'
download("https://github.com/yeephycho/tensorflow-face-detection/blob/master/model/frozen_inference_graph_face.pb?raw=true",FROZEN_GRAPH_NAME)

while True:
    time.sleep(10)
