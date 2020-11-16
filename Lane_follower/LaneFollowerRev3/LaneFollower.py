#!/usr/bin/env python
from __future__ import print_function
import logging
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

import message_filters

from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

import os
import cv2
import sys
import numpy as np

import rospy
import roslib
roslib.load_manifest('enph353_gazebo')

logger = logging.getLogger("Mr.LaneFollower")

# LINEAR_VEL = 0.55
# ANGULAR_VEL = 5.05447028499

# LINEAR_VEL = 0.23914845
# ANGULAR_VEL = 1.0

# PLEASE TRAIN AT THIS SPEED
LINEAR_VEL = 0.295245
ANGULAR_VEL = 1.4641

class LaneFollower:

    def __init__(self, CNN_path):
        
        # getting the neural net to load
        # these need to come before everything else
        self.sess = tf.Session()
        self.graph = tf.compat.v1.get_default_graph()
        set_session(self.sess)
        self.CNN = tf.compat.v1.keras.models.load_model(CNN_path)
        self.CNN._make_predict_function()

        
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.img_callback)

        # Maybe we publish to an intermediate topic first?
        # self.velocity_pub = rospy.Publisher(
        #     "/R1/intermediate_cmd_vel", Twist)

        self.velocity_pub = rospy.Publisher(
            "/R1/cmd_vel", Twist,  queue_size=1)

    # def vel_callback(self, data):
    #     self.last_cmd = data

    def parse_one_hot(self, one_hot_array):
        """
        Given a rounded one-hot array (only a single element is 1, all other
        elements are zero), return the linear angular velcity command as a 
        Twist object.
        """
        move = Twist()
        index = np.argmax(one_hot_array)

        if index == 0:
            move.linear.x = LINEAR_VEL
            move.angular.z = ANGULAR_VEL

        elif index == 1:
            move.linear.x = LINEAR_VEL
            move.angular.z = -1.0*ANGULAR_VEL

        elif index == 2:
            move.linear.x = LINEAR_VEL
            move.angular.z = 0

        elif index == 3:
            move.linear.x = -1.0*LINEAR_VEL
            move.angular.z = ANGULAR_VEL

        elif index == 4:
            move.linear.x = -1.0*LINEAR_VEL
            move.angular.z = -1.0*ANGULAR_VEL

        else:
            # logger.critical("Invalid one-hot array")
            move.linear.x = 0
            move.angular.z = 0

        return move

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='bgr8')

            resized_img = cv2.resize(cv_image, (0, 0), fx=0.40, fy=0.40)
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)/255.
            img_aug = np.expand_dims(resized_img, axis=0)

            with self.graph.as_default():
                set_session(self.sess)
                pred = np.round(self.CNN.predict(img_aug)[0])
                vel_cmd = self.parse_one_hot(pred)
                print(vel_cmd)
                self.velocity_pub.publish(vel_cmd)

        except Exception as e:
            print(e)

    def make_prediction(self, img_aug):
        print(np.round(self.CNN.predict(img_aug)[0]))


def main(args):

    rospy.init_node('LaneFollower', anonymous=True)

    LF = LaneFollower(
        "/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_gazebo/nodes/LaneFollowerV2.h5")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main(sys.argv)

    # CNN = tf.compat.v1.keras.models.load_model(
    #     "/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_gazebo/nodes/Lane_follower_light.h5")
    # cv_image = cv2.imread("/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_gazebo/nodes/[97]_0.55_-5.05447028499_.jpg")
    # resized_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)/255.
    # img_aug = np.expand_dims(resized_img, axis=0)
    # for _ in range(1000):
    #     print(np.round(CNN.predict(img_aug)[0]))
