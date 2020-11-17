#!/usr/bin/env python
import cv2
import csv
import numpy as np
import os
from os.path import isfile, join
import random
import string
import sys
sys.path.append('../scripts')
import one_hot

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rospy
import roslib
roslib.load_manifest('enph353_gazebo')

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

o_h = one_hot.OneHot()

ps_path = os.path.dirname(os.path.realpath(__file__)) + "/" + '../scripts/ps_model'
plate_path = os.path.dirname(os.path.realpath(__file__)) + "/" + '../scripts/pl_model3'

class PlateReader():
	def __init__(self):
		# Loading Neural Network
		self.sess = tf.Session()
		self.graph = tf.compat.v1.get_default_graph()
		set_session(self.sess)
		self.ps_model = tf.compat.v1.keras.models.load_model(ps_path)
		self.ps_model._make_predict_function()

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("plate", Image, self.callback)
		self.id_pub = rospy.Publisher("license_plate", String, queue_size=10)

		rospy.init_node('license_plate', anonymous=True)

		self.ps_num = [] # 10 most recent predicted parking spots
		self.plates = []
		self.all_ps = ['1', '2', '3', '4', '5','6', '7', '8'] # All possible parking spots

	# Takes an image of the license plate and returns slices of the parking number and 4 plate characters
	def get_slices(self,img):
		parking_num, plate_1, plate_2, plate_3, plate_4 = o_h.crop(img)
		return parking_num, plate_1, plate_2, plate_3, plate_4

	# Predicts parking spot
	def predict_ps(self, parking_num):
		img_aug = np.expand_dims(parking_num, axis = 0)
		with self.graph.as_default():
			set_session(self.sess)
			Y_predict = self.ps_model.predict(img_aug)[0]
			Y_predict_r = np.zeros(8)
			Y_predict_r[np.argmax(Y_predict)] = 1
			parking_spot = o_h.ps_decoder[tuple(Y_predict_r)]
			return parking_spot

	"""# Predicts single character of license plate		
	def predict_plate(self, c):
		img_aug = np.expand_dims(c, axis = 0)
		with self.graph.as_default():
			set_session(self.sess)
			Y_predict = self.pl_model.predict(img_aug)[0]
			Y_predict_r = np.zeros(36)
			Y_predict_r[np.argmax(Y_predict)] = 1
			plate_char = o_h.pplate_decoder[tuple(Y_predict_r)]
			return plate_char """

	# Adds predicted parking spot to a list which holds the 10 most recent predictions
	def add_ps(self, img):
		if len(self.ps_num) == 10:
			del self.ps_num[0]
		ps = self.predict_ps(img)
		self.ps_num.append(ps)

	"""def add_plate(self, c1, c2, c3, c4):
		if len(self.plates) == 4:
			del self.plates[0]
		plate = self.predict_plate(c1) + self.predict_plate(c2) + self.predict_plate(c3) + self.predict_plate(c4)
		print(plate)
		self.plates.append(plate) """

	# Checks if 10 most recent predictions are the same
	def read_ps(self):
		if len(self.ps_num) == 10:
			if (all(el == self.ps_num[0] for el in self.ps_num)):
				spot = self.ps_num[0] 
				if spot in self.all_ps:
					self.ps_num = []
					self.all_ps.remove(spot)
					return spot
		return None

	"""def read_plate(self):
		if len(self.plates) == 4:
			if (all(el == self.plates[0] for el in self.plates)):
				plate = self.plates[0] 
				return plate
		return None """

	# Publishes plate to score tracker node
	def publish_plate(self, parking_spot, plate):
		msg = ["Team8", "code", str(parking_spot), str(plate)]
		str_msg = ','.join(msg)
		message = str(str_msg)
		self.id_pub.publish(message)

	def callback(self,img_msg):
		try:
			plate = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
			ps, c1, c2, c3, c4 = self.get_slices(plate)
			self.add_ps(ps)
			spot = self.read_ps()
			if spot is not None:
				self.publish_plate(spot, 'KG29')
		except CvBridgeError as e:
			print(e)

def main(args):
	print("launched properly")
	reader = PlateReader()
	
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

