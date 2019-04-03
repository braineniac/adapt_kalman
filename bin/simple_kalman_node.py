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

        # state vector [X Xdot Xdotdot]
        self.state = np.zeros(3)
        self.newstate = np.zeros(3)

        # error covariance
        self.P = np.zeros((3,3))
        self.newp = np.zeros((3,3))
        # contol vector u
        self.u = np.zeros(3)

        # state transition model
        self.F = np.zeros((3,3))

        # control-input model
        self.B = np.zeros((3,3))


        # process noise covariances
        self.Q = np.zeros((3,3))
        # observation noise covariance
        self.R = np.zeros((3,3))

        self.H = np.zeros((1,3))
        self.z = np.zeros(3)
        self.S = np.zeros((3,3))
        self.K = np.zeros(3)
        self.y = np.zeros(3)
        self.y_k = np.zeros(3)
        self.newP = np.zeros((3,3))
        self.tosec = rospy.Time.from_sec(time.time()).to_sec()

        # process noise
        self.w = np.zeros((3,3))

        self.mean = np.zeros(3)

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
        self.R[1][1] = (400/1000000) * 9.80655 / 100.0  # fake encoder covariance, Xdot
        self.R[2][2] = (400/1000000) * 9.80655          # imu covariance Xdotdot
        self.R[1][1] = 100
        self.R[2][2] = 1
        #self.w = np.random.mulivariate_normal(mean,q,1000)
        self.F = np.array([[1,self.delta_t,1/2*self.delta_t*self.delta_t],[0,1,self.delta_t],[0,0,1.0]])
        self.B = np.array([[1,self.delta_t,1/2*self.delta_t*self.delta_t],[0,1,self.delta_t],[0,0,1.0]])
        self.H = np.array([1.0, 0, 0])

    def predict(self):
        """Kalman filter predict step
        """
        self.state = self.F.dot(self.state) + self.B.dot(self.u)
        self.P = self.F.dot(self.P).dot(np.transpose(self.F))

    def debug_print(self):
        #np.set_printoptions(precision=15)
        rospy.loginfo("State: {}".format(self.state))
        rospy.loginfo("Covariance: {}".format(self.P))
        rospy.loginfo("Observer cov: {}".format(self.R))
        rospy.loginfo("Residual: {}".format(self.y))
        rospy.loginfo("Residual covariance:".format(self.S))
        rospy.loginfo("Kalman gain: {}".format(self.K))
        rospy.loginfo("New state: {}".format(self.newstate))
        rospy.loginfo("New covariance: {}".format(self.newP))
        rospy.loginfo("New residual: {}".format(self.y_k))

    def update(self):
        """Kalman filter update step
        """
        self.y = self.state - self.H.dot(self.newstate)
        self.S = np.zeros((3,3))
        self.debug_print()
        self.S = self.R + self.H.dot(self.P).dot(np.transpose(self.H))
        self.debug_print()
        self.K = self.P.dot(np.transpose(self.H)).dot(np.linalg.inv(self.S))
        self.newstate = self.state + self.K.dot(self.y)
        self.newP = (np.identity(3) - self.K.dot(self.H)).dot(self.P).dot(np.transpose(np.identity(3) - self.K.dot(self.H))) + self.K.dot(self.R).dot(np.transpose(self.K))
        self.y_k = self.state - self.H.dot(self.newstate)

        # update for the next cycle
        self.publish()
        self.P = self.newP
        self.state = self.newstate

    def imu_cb(self, imu_msg):
        # update time
        now = rospy.get_rostime()
        self.delta_t = self.tosec(self.last_time - now)
        self.last_time = now

        self.u[2] = imu_msg.linear_acceleration.x

    def fake_enc_cb(self):
        # update time
        now = rospy.get_rostime()
        self.delta_t = self.tosec(self.last_time - now)
        self.last_time = now

        self.u[1] = self.twist.twist.linear.x

    def publish(self):
        odom_msg = Odometry()
        self.odom_pub.publish(odom_msg)

if __name__ == '__main__':
    rospy.loginfo("Initialising simple_kalman_node")
    rospy.init_node("simple_kalman_node")
    simple_kalman_node = SimpleKalmanNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        simple_kalman_node.predict()
        simple_kalman_node.update()
        rate.sleep()
