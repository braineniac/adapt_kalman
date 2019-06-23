#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

from simple_kalman_sim import SimpleKalmanSim

class SimpleKalmanSimNode:
    def __init__(self, imu_out="/imu/data",
                       twist_out="/fake_wheel/twist",
                       N=250,
                       sim_time=5.0,
                       peak_vel=0.14):
        self.imu_pub = rospy.Publisher(imu_out, Imu, queue_size=1)
        self.twist_pub = rospy.Publisher(twist_out, TwistWithCovarianceStamped, queue_size=1)

        self.simple_kalman_sim = SimpleKalmanSim(N=N, peak_vel=peak_vel, sim_time=sim_time)

        self.twist_rate = (sim_time/N) / 10.
        self.imu_rate = (sim_time/N)

        self.sim_time = sim_time
        self.N = N

    def set_pub_timers(self):
        rospy.Timer(rospy.Duration(self.imu_rate), self.publish_imu)
        rospy.Timer(rospy.Duration(self.twist_rate), self.publish_twist)

    def publish_imu(self):
        imu_msg = Imu()

        ros_time = rospy.get_rostime()
        imu_msg.header.stamp = ros_time
        ## TODO:

        self.imu_pub.pub(imu_msg)

    def pub_twist(self):
        twist_msg = TwistWithCovarianceStamped()
        ## TODO:

        self.twist_pub.pub(twist_msg)

if __name__ == '__main__':
    rospy.loginfo("Initialising simple_kalman_sim_node")
    rospy.init_node("simple_kalman_sim_node")
    simple_kalman_sim_node = SimpleKalmanSimNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        simple_kalman_sim_node.set_pub_timers()
        rate.sleep()
