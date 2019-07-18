#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from matplotlib import pyplot as plt
import numpy as np
import tf

import rosbag
from geometry_msgs.msg import Quaternion

class EKFExporter:
    def __init__(self, bag_path="", odom_topic="/odometry/filtered"):
        self.bag_path = bag_path
        self.odom_topic = odom_topic

        self.vel = []
        self.pos_x = []
        self.yaw = []
        self.pos_y = []
        self.t = []

    def read_bag(self):
        rosbag_f = rosbag.Bag(self.bag_path)
        odom_msgs = rosbag_f.read_messages(topics=self.odom_topic)
        for odom_msg in odom_msgs:
            pos_x = odom_msg.message.pose.pose.position.x
            pos_y = odom_msg.message.pose.pose.position.y
            vel_x = odom_msg.message.twist.twist.linear.x
            x = odom_msg.message.pose.pose.orientation.x
            y = odom_msg.message.pose.pose.orientation.y
            z = odom_msg.message.pose.pose.orientation.z
            w = odom_msg.message.pose.pose.orientation.w
            q = [x,y,z,w]
            roll,pitch,yaw = tf.transformations.euler_from_quaternion(q)
            t = odom_msg.message.header.stamp.to_sec()
            self.pos_x.append(pos_x)
            self.pos_y.append(pos_y)
            self.yaw.append(yaw)
            self.vel.append(vel_x)
            self.t.append(t)

        self.plot_t = np.abs(np.array(self.t) - self.t[0])

    def plot_all(self):
        plt.figure(1)


        plt.subplot(411)
        plt.title("Velocity in x")
        plt.plot(self.plot_t, self.vel)

        plt.subplot(412)
        plt.plot(self.plot_t,self.pos_x)

        plt.subplot(413)
        plt.plot(self.plot_t,self.pos_y)

        plt.subplot(414)
        plt.plot(self.plot_t,self.yaw)

        plt.figure(2)
        plt.plot(self.pos_x,self.pos_y)

        plt.show()

    def export_loops(self):

        np.savetxt("plots/loop_xy.csv", np.transpose(
            [self.pos_x, self.pos_y]), header='x y', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_yaw.csv", np.transpose(
            [self.plot_t, self.yaw]), header='t yaw', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_x.csv", np.transpose(
            [self.plot_t, self.pos_x]), header='t yaw', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_y.csv", np.transpose(
            [self.plot_t, self.pos_y]), header='t yaw', comments='# ', delimiter=' ', newline='\n')


    def export_test4_all(self):
        b = 100
        e = 75
        new_t_array = []
        for elem in self.plot_t:
            new_t_array.append(elem - self.plot_t[b])

        np.savetxt("plots/ekf_pos.csv", np.transpose(
            [new_t_array[b:-e], self.pos_x[b:-e]]), header='t x', comments='# ', delimiter=' ', newline='\n')
        np.savetxt("plots/ekf_vel.csv", np.transpose(
            [new_t_array[b:-e], self.vel[b:-e]]), header='t v', comments='# ', delimiter=' ', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag of EKF")
    parser.add_argument("-b", "--bag", help="Rosbag path")
    parser.add_argument("-t", "--topic", help="Topic name")
    args = parser.parse_args()
    if args.topic:
        ekf_exporter = EKFExporter(args.bag, args.topic)
    else:
        ekf_exporter = EKFExporter(args.bag)
    ekf_exporter.read_bag()
    #ekf_exporter.plot_all()
    #ekf_exporter.export_loops()
    ekf_exporter.export_test4_all()
