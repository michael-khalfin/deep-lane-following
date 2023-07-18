#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, TwistStamped
import os
from artist_class import Artist
import numpy as np
from dl_path_class import DL_Path

start_drive = Bool()

empty = Empty()
bridge = CvBridge()
it = 0
in_process = False
twist_msg = Twist()
twist_msg.linear.x = 0
ARTIST = False

def red_callback(msg):
    if len(msg.data) > 0:
        print(msg.data)
    with open("data.txt", "a") as f:
        f.write(msg.data + "\n")
    return

def camera_callback(ros_image):
    global bridge, twist_msg
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "passthrough")
    except CvBridgeError as e:
        print(e)

    if ARTIST:
        prediction = artist.predict(cv_image)
        twist_msg, path = artist.get_twist(prediction)
        cv2.imshow("Path", path)
        cv2.waitKey(1)
        twist_msg.linear.x = 1.5
    else:
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        twist_msg = dlam.predict(cv_image)

def drive(args):
    if dlam.train: return
    enable_car.publish(empty)
    velocity_pub.publish(twist_msg)
    
    start_drive.data = True
    start_red_detect.publish(start_drive)
    print(f"DRIVE STARTED: {start_drive.data}")

if __name__ == '__main__':
    rospy.loginfo("Follow lane initialized")
    rospy.init_node('jacks_cv', anonymous=True)
    os.chdir("../../../home/reu-actor/actor_ws/src/jacks_pkg/scripts/DL")
    
    artist = Artist()
    dlam = DL_Path()

    start_red_detect = rospy.Publisher('/drive_enabled', Bool, queue_size=1)
    rospy.Subscriber('/red_detect_topic', String, red_callback)

    rospy.Subscriber('/camera/image_raw', Image, camera_callback)
    rospy.Subscriber('/vehicle/twist', TwistStamped, drive)
    enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
    enable_car.publish(Empty())
    velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    