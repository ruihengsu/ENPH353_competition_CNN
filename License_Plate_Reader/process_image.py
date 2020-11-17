#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
import cv2
import rospy
import sys

from PlateDetector import PlateDetector

import roslib
roslib.load_manifest('enph353_gazebo')


class image_processor:

	def __init__(self):
		self.image_pub = rospy.Publisher(
			"plate", Image, queue_size=1)  # publish to node below

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber(
			"/R1/pi_camera/image_raw", Image, self.callback)

		self.detector = PlateDetector()

	def callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(
				data, "bgr8") # desired_encoding='passthrough'
		except CvBridgeError as e:
			print(e)

		#frame = self.detector.draw_contour(cv_image)
		license_plate = self.detector.draw_contour(cv_image)
		if license_plate is not None:
			try:
				img_msg = self.bridge.cv2_to_imgmsg(license_plate, "bgr8")
				self.image_pub.publish(img_msg) 
			except CvBridgeError as e:
				print(e)
		
def main(args):
	ic = image_processor()
	rospy.init_node('testImage', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main(sys.argv)
