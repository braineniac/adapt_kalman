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

import tf
import rosbag
import numpy as np


class BagReader(object):
    def __init__(self, bag_path=""):
        if bag_path is None:
            raise AttributeError
        else:
            self.bag = rosbag.Bag(bag_path)

    def read_odom(self, topic=None):
        if topic is None:
            raise ValueError
        else:
            odom_msgs = self.bag.read_messages(topics=topic)
            odom = []
            for odom_msg in odom_msgs:
                t = odom_msg.message.header.stamp.to_sec()

                pos_x = odom_msg.message.pose.pose.position.x
                pos_y = odom_msg.message.pose.pose.position.y
                pos_z = odom_msg.message.pose.pose.position.z

                orient_x = odom_msg.message.pose.pose.orientation.x
                orient_y = odom_msg.message.pose.pose.orientation.y
                orient_z = odom_msg.message.pose.pose.orientation.z
                orient_w = odom_msg.message.pose.pose.orientation.w
                q = [orient_x, orient_y, orient_z, orient_w]
                roll, pitch, yaw = tf.transformations.euler_from_quaternion(q)

                lin_x = odom_msg.message.twist.twist.linear.x
                lin_y = odom_msg.message.twist.twist.linear.y
                lin_z = odom_msg.message.twist.twist.linear.z

                ang_x = odom_msg.message.twist.twist.angular.x
                ang_y = odom_msg.message.twist.twist.angular.y
                ang_z = odom_msg.message.twist.twist.angular.z

                odom_data = (pos_x, pos_y, pos_z, roll, pitch, yaw, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
                odom.append((t, odom_data))
            return odom

    def read_imu(self, topic=None):
        if topic is None:
            raise AttributeError
        else:
            imu = []
            imu_msgs = self.bag.read_messages(topics=topic)
            for imu_msg in imu_msgs:
                t = imu_msg.message.header.stamp.to_sec()
                accel_x = imu_msg.message.linear_acceleration.x
                accel_y = imu_msg.message.linear_acceleration.y
                accel_z = imu_msg.message.linear_acceleration.z

                gyro_x = imu_msg.message.angular_velocity.x
                gyro_y = imu_msg.message.angular_velocity.y
                gyro_z = imu_msg.message.angular_velocity.z

                imu_data = (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
                imu.append((t, imu_data))
            return imu

    def read_twist(self, topic=None):
        if topic is None:
            raise ValueError
        else:
            twist = []
            twist_msgs = self.bag.read_messages(topics=topic)
            for twist_msg in twist_msgs:
                t = twist_msg.message.header.stamp.to_sec()

                lin_x = twist_msg.message.twist.twist.linear.x
                lin_y = twist_msg.message.twist.twist.linear.y
                lin_z = twist_msg.message.twist.twist.linear.z

                ang_x = twist_msg.message.twist.twist.angular.x
                ang_y = twist_msg.message.twist.twist.angular.y
                ang_z = twist_msg.message.twist.twist.angular.z

                twist_data = (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
                twist.append((t, twist_data))
            return twist
