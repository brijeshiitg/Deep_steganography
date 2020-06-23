import cv2
import numpy as np
from arguments import *

def read_data(input_path, start, end):

	train_x = np.zeros((end - start, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
	message = np.zeros((FLAGS.input_size, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
 
	for i in range(start, end):
		img = cv2.imread(input_path + str(i+1) + '.JPEG')
		if img.shape[0] >256 or img.shape[1]>256:
			img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
		elif img.shape[0]<256 or img.shape[1]<256:
			img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
		train_x[i-start,:,:,:] = np.array(img).reshape(256,256,3)    # image as input
		train_x[i-start] /= 255.0
		msg = cv2.imread(FLAGS.message_path + str(i+1) + '.JPEG')
		if msg.shape[0] >256 or msg.shape[1]>256:
			msg = cv2.resize(msg, (256,256), interpolation=cv2.INTER_AREA)
		elif msg.shape[0]<256 or msg.shape[1]<256:
			msg = cv2.resize(msg, (256,256), interpolation=cv2.INTER_CUBIC)
		message[i-start,:,:,:] = np.array(msg).reshape(256,256,3)
		message[i-start] /= 255.0

	return train_x, message