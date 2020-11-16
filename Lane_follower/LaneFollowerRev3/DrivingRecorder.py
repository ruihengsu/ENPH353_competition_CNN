#!/usr/bin/env python
from __future__ import print_function
import message_filters
from PlateDetector import PlateDetector

from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

import os
import cv2
import rospy
import sys

import roslib
roslib.load_manifest('enph353_gazebo')

import datetime

class DrivingRecorder:

    def __init__(self):

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.img_callback)

        self.velocity_sub = rospy.Subscriber(
            "/R1/cmd_vel", Twist, self.vel_callback)

        # self.velocity_pub = rospy.Publisher(
        #     "/R1/cmd_vel", Twist)
        
        self.last_cmd = 0
        self.path = "/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_gazebo/nodes/DrivingData"
        import glob
        files = glob.glob(self.path + "/*")
        for f in files:
            os.remove(f)

        self.last_label = ""
        self.num = 0
        
    def vel_callback(self, data):

        self.last_cmd = data

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(e)

        if self.last_cmd != 0:
            label = ""
            label = label + "{}_".format(self.last_cmd.linear.x)
            label = label + "{}_".format(self.last_cmd.angular.z)

            if self.last_label != label:
                self.last_label = label
            resized_img = cv2.resize(cv_image, (0, 0), fx=0.45, fy=0.45)
            cv2.imwrite(
                self.path + "/" + "[{}]_".format(self.num) + self.last_label + ".jpg", resized_img)
            self.num += 1
        else:
            pass

def main(args):
    dc = DrivingRecorder()
    rospy.init_node('DrivingRecorder', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
