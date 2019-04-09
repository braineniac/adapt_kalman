#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np

class SimpleKalmanNode:
    def __init__(self):

        #####################
        ### Kalman filter ###
        #####################

        self.L_k = np.zeros((3,3))          # Kalman gain
        self.P_k_pre = np.zeros((3,3))      # A priori covariance
        self.P_k_post = np.zeros((3,3))     # A posteriori covariance
        self.C_k = np.zeros(3)
        self.x_k_post = np.zeros((3,1))
        self.x_k_pre = np.zeros((3,1))
        self.x_k_extr = np.zeros((3,1))     # extrapolated state
        self.P_k_extr = np.zeros((3,1))     # extrapolated covariance
        self.phi_k = np.zeros((3,3))
        self.D_k = np.zeros((3,3))
        self.gamma_k = np.zeros((3,3))      # control matrix
        self.Q_k = np.zeros((3,3))
        self.R_k = np.zeros((3,3))
        self.G_k = np.zeros((3,3))
        self.H_k = np.zeros((3,3))
        self.y_k = np.zeros((3,1))          # output
        self.u_k = np.zeros((3,1))          # cocntrol vector

        #################
        ### ROS stuff ###
        #################

        self.imu_sub = rospy.Subscriber("/imu/data_raw", Imu, self.imu_cb)
        self.fake_enc_sub = rospy.Subscriber("/fake_wheel/twist_var", TwistWithCovarianceStamped, self.fake_enc_cb)
        self.odom_pub = rospy.Publisher("/simple_kalman/odom", Odometry, queue_size=1)

        self.begin_time = rospy.Time(0.0)
        self.last_time = rospy.Time(0.0)
        self.delta_t = 0.0

        self.initialize()

    def initialize(self):
        self.begin_time = rospy.get_rostime()

        # set initial covariance
        self.R_k[0][0] = np.exp(-10)
        self.R_k[1][1] = (400/1000000) * 9.80655 / 100.0  # fake encoder covariance, Xdot
        self.R_k[2][2] = (400/1000000) * 9.80655          # imu covariance Xdotdot
        self.R_k[0][0] = 1
        self.R_k[1][1] = 10
        self.R_k[2][2] = 1000
        self.Q_k = np.random.normal(np.exp(-10),1.0,(3,3))
        self.D_k = np.identity(3)
        self.P_k_pre = np.random.normal(np.exp(-10),1.0,(3,3))
        self.phi_k = np.array([[1,self.delta_t,1/2*self.delta_t*self.delta_t],[0,1,self.delta_t],[0,0,1.0]])
        self.gamma_k = self.phi_k
        self.C_k = np.identity(3)

    def set_gain(self):
        E = self.C_k.dot(self.P_k_pre).dot(self.C_k.T) + self.R_k
        self.L_k = self.P_k_pre.dot(self.C_k.T).dot(np.linalg.inv(E))

    def update(self):
        F = self.y_k - self.C_k.dot(self.x_k_pre) - self.D_k.dot(self.u_k)
        self.x_k_post = self.x_k_pre + self.L_k.dot(F)
        self.P_k_post = (np.identity(3) - self.L_k.dot(self.C_k)).dot(self.P_k_pre)
        #self.y_k = self.C_k.dot(self.x_k_post)

    def extrapolate(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T)

        # update for next iteration
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr
        self.publish()

    def debug_print(self):
        np.set_printoptions(precision=15)
        rospy.loginfo("State: {}".format(self.x_k_pre))
        rospy.loginfo("Covariance: {}".format(self.P_k_pre))
        rospy.loginfo("Kalman gain: {}".format(self.L_k))
        rospy.loginfo("Posteriori state: {}".format(self.x_k_post))
        rospy.loginfo("Posteriori covariance: {}".format(self.P_k_post))
        rospy.loginfo("Extrapolated state: {}".format(self.x_k_extr))
        rospy.loginfo("Extrapolated covariance: {}".format(self.P_k_extr))
        rospy.loginfo("Output: {}".format(self.y_k))

    def imu_cb(self, imu_msg):
        # update time
        now = rospy.get_rostime()
        self.delta_t = (self.last_time - now).to_sec()
        self.last_time = now

        self.u_k[2] = imu_msg.linear_acceleration.x

    def fake_enc_cb(self, enc_msg):
        # update time
        now = rospy.get_rostime()
        self.delta_t = (self.last_time - now).to_sec()
        self.last_time = now

        self.u_k[1] = enc_msg.twist.twist.linear.x

    def publish(self):
        odom_msg = Odometry()

        odom_msg.header.stamp = rospy.get_rostime()
        odom_msg.header.frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.x_k_extr[0]
        odom_msg.pose.covariance[0] = self.P_k_extr[0][0]

        self.odom_pub.publish(odom_msg)

if __name__ == '__main__':
    rospy.loginfo("Initialising simple_kalman_node")
    rospy.init_node("simple_kalman_node")
    simple_kalman_node = SimpleKalmanNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        simple_kalman_node.set_gain()
        simple_kalman_node.update()
        simple_kalman_node.extrapolate()
        rate.sleep()
