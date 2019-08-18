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

import rosbag

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
                pos = (pos_x,pos_y,pos_z)

                orient_x = odom_msg.message.pose.pose.orientation.x
                orient_y = odom_msg.message.pose.pose.orientation.y
                orient_z = odom_msg.message.pose.pose.orientation.z
                orient_w = odom_msg.message.pose.pose.orientation.w
                q = [x,y,z,w]
                roll,pitch,yaw = tf.transformations.euler_from_quaternion(q)
                orient = (orient_x,orient_y,orient_z,roll,pitch,yaw)

                vel_x = odom_msg.message.twist.twist.linear.x
                vel_y = odom_msg.message.twist.twist.linear.y
                vel_z = odom_msg.message.twist.twist.linear.z
                vel = (vel_x,vel_y,vel_z)

                odom.append((t,pos,orient,vel))
            return odom

    def read_imu(self, topic=None):
        if topic is None:
            raise AttributeError
        else:
            imu = []
            imu_msgs = self.bag.read_messages(topics=topic)
            for imu_msg in msgs:
                t = imu_msg.message.header.stamp.to_sec()
                accel_x = imu_msg.message.linear_acceleration.x
                accel_y = imu_msg.message.linear_acceleration.y
                accel_z = imu_msg.message.linear_acceleration.z
                accel = (accel_x,accel_y,accel_z)

                gyro_x = imu_msg.message.angular_velocity.x
                gyro_y = imu_msg.message.angular_velocity.y
                gyro_z = imu_msg.message.angular_velocity.z
                gyro=(gyro_x,gyro_y,gyro_z)

                imu.append((t,accel,gyro))
            return imu

    def read_twist(self,topic=None):
        if topic is not None:
            raise ValueError
        else:
            twist = []
            twist_msgs = self.bag.read_messages(topics=topic)
            for twist_msg in twist_msgs:
                t = twist_msg.message.header.stamp.to_sec()

                lin_x = twist_msg.message.twist.twist.linear.x
                lin_y = twist_msg.message.twist.twist.linear.y
                lin_z = twist_msg.message.twist.twist.linear.z
                lin = (lin_x,lin_y,lin_z)

                ang_x = twist_msg.message.twist.twist.angular.x
                ang_y = twist_msg.message.twist.twist.angular.y
                ang_z = twist_msg.message.twist.twist.angular.z
                ang = (ang_x,ang_y,ang_z)
                twist.append((t,lin,ang))
            return twist
