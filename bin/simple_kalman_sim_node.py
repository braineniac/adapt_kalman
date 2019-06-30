#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

import numpy as np

from simple_kalman_sim import SimpleKalmanSim

class SimpleKalmanSimNode:
    def __init__(self, imu_out="/imu/data",
                       twist_out="/fake_wheel/twist",
                       N=300,
                       ratio=1/3,
                       sim_time=10.0,
                       peak_vel=0.14):
        self.imu_pub = rospy.Publisher(imu_out, Imu, queue_size=1)
        self.twist_pub = rospy.Publisher(twist_out, TwistWithCovarianceStamped, queue_size=1)

        self.simple_kalman_sim = SimpleKalmanSim(N=N, peak_vel=peak_vel, sim_time=sim_time)

        rospy.loginfo("{}".format((N/sim_time)))

        self.twist_freq = 10.
        self.imu_freq = 50.

        self.sim_time = sim_time
        self.N = N

        self.begin_time = rospy.get_rostime().to_sec()

    def set_pub_timers(self):
        rospy.Timer(rospy.Duration(1/self.imu_freq), self.publish_imu)
        rospy.Timer(rospy.Duration(1/self.twist_freq), self.publish_twist)

    def check_shutdown(self):
        if rospy.get_rostime().to_sec() > (self.begin_time + self.sim_time):
            rospy.signal_shutdown("Simulation finished")

    def publish_imu(self, timer):
        imu_msg = Imu()

        ros_time = rospy.get_rostime()
        imu_msg.header.stamp = ros_time
        imu_msg.frame_id = "base_link"

        idx = self.find_nearest_time(ros_time.to_sec() - self.begin_time)

        imu_msg.linear_acceleration.x = self.simple_kalman_sim.accel[idx]
        imu_msg.linear_acceleration_covariance[0] = self.simple_kalman_sim.kalman.R_k[1][1]

        self.imu_pub.publish(imu_msg)

    def publish_twist(self,timer):
        twist_msg = TwistWithCovarianceStamped()

        ros_time = rospy.get_rostime()
        twist_msg.header.stamp = ros_time
        twist_msg.frame_id="base_link"

        idx = self.find_nearest_time(ros_time.to_sec() - self.begin_time)

        twist_msg.twist.twist.linear.x = self.simple_kalman_sim.vel[idx]
        twist_msg.twist.covariance[0] = self.simple_kalman_sim.kalman.Q_k[1][1]

        self.twist_pub.publish(twist_msg)

    def find_nearest_time(self, value):
        idx = np.abs(self.simple_kalman_sim.t - value).argmin()
        return idx

if __name__ == '__main__':
    rospy.loginfo("Initialising simple_kalman_sim_node")
    rospy.init_node("simple_kalman_sim_node")
    simple_kalman_sim_node = SimpleKalmanSimNode()
    simple_kalman_sim_node.set_pub_timers()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        simple_kalman_sim_node.check_shutdown()
        rate.sleep()
