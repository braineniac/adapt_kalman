#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib2tikz
import argparse
from matplotlib import pyplot as plt
import numpy as np

import rosbag


class EKFExporter:
    def __init__(self, bag_path="", odom_topic="/odometry/filtered"):
        self.bag_path = bag_path
        self.odom_topic = odom_topic

        self.vel = []
        self.pos = []
        self.t = []

    def read_bag(self):
        rosbag_f = rosbag.Bag(self.bag_path)
        odom_msgs = rosbag_f.read_messages(topics=self.odom_topic)
        for odom_msg in odom_msgs:
            pos_x = odom_msg.message.pose.pose.position.x
            vel_x = odom_msg.message.twist.twist.linear.x
            t = odom_msg.message.header.stamp.to_sec()
            self.pos.append(pos_x)
            self.vel.append(vel_x)
            self.t.append(t)

        self.plot_t = np.abs(np.array(self.t) - self.t[0])

    def plot_all(self):
        plt.figure(1)

        plt.subplot(211)
        plt.title("EKF distance")
        plt.plot(self.plot_t,self.pos)

        plt.subplot(212)
        plt.title("EKF velocity")
        plt.plot(self.plot_t,self.vel)

        plt.show()

    def export_all(self):
        plt.figure(1)
        plt.title("EKF distance")
        plt.plot(self.plot_t,self.pos)
        matplotlib2tikz.save("plots/ekf_pos.tex",figureheight='4cm', figurewidth='6cm')

        plt.figure(2)
        plt.title("EKF velocity")
        plt.plot(self.plot_t,self.vel)
        matplotlib2tikz.save("plots/ekf_vel.tex",figureheight='4cm', figurewidth='6cm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag of EKF")
    parser.add_argument("-b","--bag",help="Rosbag path")
    parser.add_argument("-t", "--topic", help="Topic name")
    args = parser.parse_args()
    if args.topic:
        ekf_exporter = EKFExporter(args.bag,args.topic)
    else:
        ekf_exporter = EKFExporter(args.bag)
    ekf_exporter.read_bag()
    ekf_exporter.plot_all()
    ekf_exporter.export_all()
