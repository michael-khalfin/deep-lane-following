# Authors: Jack Volgren
# Jun 13
# The purpose of this class is to recieve images and return a angular z value for other classes to publish to /cmd_vel

import rospy
from geometry_msgs.msg import Twist, TwistStamped
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import rospy

class DL_Path:
    
    def __init__(self):
        rospy.loginfo("MODEL INITIALIZED...")
        self.model = tf.keras.models.load_model('jul_16')   #   Works outside, too bright can fail, changes lanes inside
        # self.model = tf.keras.models.load_model('jul_16_03')   #  Shits the bed in light 
        # self.model = tf.keras.models.load_model('jul_16_01')   #   Dog shit
        # self.model = tf.keras.models.load_model('jul_16_04')   #   I think this ones good bad on inside
        # self.model = tf.keras.models.load_model('jul_17_01')   #   Doesnt turn enough
        self.current_yaw = 0
        self.outside = False
        self.train = False
        self.thresh = 0.88

    def set_yaw(self, st):
        self.current_yaw = st.twist.angular.z

    def predict(self, img):
        img1, img2, img3 = self.prep_images(img)
        # img1 = self.prep_images(img)

        p1 = self.model.predict(img1)[0]
        p2 = self.model.predict(img2)[0]
        p3 = self.model.predict(img3)[0]

        twist_msg = self.get_twist([p1[0], p2[0], p3[0]])

        return twist_msg

    def get_twist(self, predictions):
        twist_msg = Twist()

        predictions = self.remove_outliers(predictions)
        pred_avg = sum(predictions) / len(predictions)
        
        twist_msg.angular.z = pred_avg
        self.current_yaw = twist_msg.angular.z
        print(predictions)

        # Set linear
        if abs(twist_msg.angular.z) < 0.25:
            twist_msg.linear.x = 1.
        else:
            twist_msg.linear.x = 1.

        return twist_msg
    
    def remove_outliers(self, predictions):
        p1, p2, p3 = predictions
        # if p1 > 0 and p2 > 0 and p3 < 0:
        #     predictions = p1, p2
        # elif p1 < 0 and p2 > 0 and p3 > 0:
        #     predictions = p3, p2
        # elif p1 > 0 and p2 < 0 and p3 > 0:
        #     predictions = p3, p1

        # elif p1 < 0 and p2 < 0 and p3 > 0:
        #     predictions = p1, p2
        # elif p1 > 0 and p2 < 0 and p3 < 0:
        #     predictions = p3, p2
        # elif p1 < 0 and p2 > 0 and p3 < 0:
        #     predictions = p1, p3


        max_d = 0.2
        rtr = [self.current_yaw]
        d1 = abs(p1 - self.current_yaw)
        d2 = abs(p2 - self.current_yaw)
        d3 = abs(p3 - self.current_yaw)

        if d1 < max_d: rtr.append(p1)
        if d2 < max_d: rtr.append(p2)
        if d3 < max_d: rtr.append(p3)

        # if len(rtr) == 0: rtr.append(self.current_yaw)
        
        # return predictions
        return rtr
        

    def prep_images(self, img):
        image1 = img[900:, :,]
        image1 = cv.resize(image1, (50,50)) / 255.
        ref = np.copy(image1)
        image2 = cv.dilate(ref * 0.75, (5,5))
        _, image3 = cv.threshold(ref, self.thresh, 1, cv.THRESH_BINARY)
        pct_white = np.count_nonzero(image3) / 2500
        if pct_white > 0.1:
            self.thresh += 0.05 if self.thresh < 0.95 else 0.01
        elif pct_white < 0.05:
            self.thresh -= 0.05 if self.thresh > 0.05 else 0
        # image4 = cv.dilate(ref * 0.5, (5,5))


        disp = np.zeros((50, 150), dtype=np.float32)
        disp[:, :50] = image1
        disp[:, 50:100] = image2
        disp[:, 100:150] = image3
        cv.imshow("Prediction Images", disp)
        cv.waitKey(1)

        image1 = np.expand_dims(image1, axis=0)
        image2 = np.expand_dims(image2, axis=0)
        image3 = np.expand_dims(image3, axis=0)
        # image4 = np.expand_dims(image4, axis=0)
        

        return image1, image2, image3

if __name__ == "__main__":
    # Another test
    img = cv.imread("../bags/img/training/ds02/-0.089884452521801_656.png")
    direction_model = DL_Direction("direction_finder_model")
    print(direction_model.predict(img))