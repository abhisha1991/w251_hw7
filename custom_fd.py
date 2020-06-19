from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import time
import wget
from tf_trt_models.detection import download_detection_model, build_detection_graph

IMAGE_PATH = ''
def detect_image(image_file_path):
	IMAGE_PATH = image_file_path
	ensure_img_dir()
	FROZEN_GRAPH_NAME = 'frozen_inference_graph_face.pb'
	download("https://github.com/yeephycho/tensorflow-face-detection/blob/master/model/frozen_inference_graph_face.pb?raw=true", FROZEN_GRAPH_NAME)

def ensure_img_dir():
	# create temp directory for images
	path = os.getcwd() + "/img"
	try:
		os.makedirs(path, exist_ok = True)
	except e:
		print("Could not create image directory")
		raise e

def download(url, file):
	if not os.path.isfile(file):
		print('Downloading ' + file + '...')
		wget.download(url, file)
		print('Download Finished')

def do_detect():
	output_dir=''
	frozen_graph = tf.GraphDef()
	with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
    		frozen_graph.ParseFromString(f.read())

	# https://github.com/NVIDIA-AI-IOT/tf_trt_models/blob/master/tf_trt_models/detection.py
	INPUT_NAME='image_tensor'
	BOXES_NAME='detection_boxes'
	CLASSES_NAME='detection_classes'
	SCORES_NAME='detection_scores'
	MASKS_NAME='detection_masks'
	NUM_DETECTIONS_NAME='num_detections'

	input_names = [INPUT_NAME]
	output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

	trt_graph = trt.create_inference_graph(
	    input_graph_def=frozen_graph,
	    outputs=output_names,
	    max_batch_size=1,
	    max_workspace_size_bytes=1 << 25,
	    precision_mode='FP16',
	    minimum_segment_size=50
	)

	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True

	tf_sess = tf.Session(config=tf_config)

	# use this if you want to try on the optimized TensorRT graph
	# Note that this will take a while
	# tf.import_graph_def(trt_graph, name='')

	# use this if you want to try directly on the frozen TF graph
	# this is much faster
	tf.import_graph_def(frozen_graph, name='')

	tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
	tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
	tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
	tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
	tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

	image = Image.open(IMAGE_PATH)

	#plt.imshow(image)

	image_resized = np.array(image.resize((300, 300)))
	image = np.array(image)

	scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
	    tf_input: image_resized[None, ...]
	})

	boxes = boxes[0] # index by 0 to remove batch dimension
	scores = scores[0]
	classes = classes[0]
	num_detections = num_detections[0]

	# suppress boxes that are below the threshold..
	DETECTION_THRESHOLD = 0.5

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	img = ax.imshow(image)

	# plot boxes exceeding score threshold
	for i in range(int(num_detections)):
		if scores[i] < DETECTION_THRESHOLD:
	        	continue
		# scale box to image coordinates
		box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

		# display rectangle
		patch = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], color='g', alpha=0.3)
		ax.add_patch(patch)

		# display class index and score
		plt.text(x=box[1] + 10, y=box[2] - 10, s='%d (%0.2f) ' % (classes[i], scores[i]), color='w')


	plt.savefig("{0}/result.jpg".format(path))
	#plt.show()

	tf_sess.close()
