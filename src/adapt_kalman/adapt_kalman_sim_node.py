#!/usr/bin/env python
# Copyright (c) 2019 Daniel Hammer. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

from adapt_kalman_sim import AdaptKalmanSim

class AdaptKalmanSimNode(AdaptKalmanSim):

    twist_freq = 10.
    imu_freq = 50.

    def __init__(self, imu_out="/imu/data",
                       twist_out="/fake_wheel/twist",
                       N=300,
                       ratio=1/3,
                       sim_time=10.0,
                       peak_vel=0.14):
        AdaptKalmanSim.__init__(self, N=N, peak_vel=peak_vel, sim_time=sim_time)
        self.N = N
        self.sim_time = sim_time
        self.begin_time = rospy.get_rostime().to_sec()

        self.imu_pub = rospy.Publisher(imu_out, Imu, queue_size=1)
        self.twist_pub = rospy.Publisher(twist_out, TwistWithCovarianceStamped, queue_size=1)

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
        imu_msg.header.frame_id = "base_link"

        idx = self.find_nearest_time(ros_time.to_sec() - self.begin_time)

        imu_msg.linear_acceleration.x = self.accel[idx]
        imu_msg.linear_acceleration_covariance[0] = self.R_k[1][1]

        self.imu_pub.publish(imu_msg)

    def publish_twist(self,timer):
        twist_msg = TwistWithCovarianceStamped()

        ros_time = rospy.get_rostime()
        twist_msg.header.stamp = ros_time
        twist_msg.header.frame_id="base_link"

        idx = self.find_nearest_time(ros_time.to_sec() - self.begin_time)

        twist_msg.twist.twist.linear.x = self.vel[idx]
        twist_msg.twist.covariance[0] = self.Q_k[1][1]

        self.twist_pub.publish(twist_msg)

    def find_nearest_time(self, value):
        idx = np.abs(self.t - value).argmin()
        return idx

if __name__ == '__main__':
    rospy.loginfo("Initialising adapt_kalman_sim_node")
    rospy.init_node("adapt_kalman_sim_node")
    adapt_kalman_sim_node = AdaptKalmanSimNode()
    adapt_kalman_sim_node.set_pub_timers()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        adapt_kalman_sim_node.check_shutdown()
        rate.sleep()
