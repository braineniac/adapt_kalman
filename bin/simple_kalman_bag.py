#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import rosbag
import argparse

class SimpleKalmanBag:
    def __init__(self,bagpath=None):
        self.bag_path = bagpath

        self.u = [[],[]]
        self.t = []
        self.input = []

    def read_bag(self):
        bag = rosbag.Bag(self.bag_path)
        imu_msgs = bag.read_messages(topics=['/imu/data_raw'])
        twist_msgs = bag.read_messages(topics=['/fake_wheel/twist'])
        last_vel = 0.0
        last_accel = 0.0
        for imu_msg,twist_msg in zip(imu_msgs,twist_msgs):
            imu_t = imu_msg.message.header.stamp.to_sec()
            twist_t = twist_msg.message.header.stamp.to_sec()
            imu_a_x = imu_msg.message.linear_acceleration.x
            twist_l_x = twist_msg.message.twist.twist.linear.x
            if last_vel != twist_l_x:
                self.input.append((twist_t,twist_l_x,last_accel))
                last_vel = twist_l_x
            if last_accel != imu_a_x:
                self.input.append((imu_t,last_vel,imu_a_x))
                last_accel = imu_a_x
        self.sort2np()

    def sort2np(self):
        sorted_input = sorted(self.input,key=lambda elem : elem[0])
        for t,u0,u1 in sorted_input:
            self.u[0].append(u0)
            self.u[1].append(u1)
            self.t.append(t)

    def plot_all(self):
        plt.figure(1)
        print(self.u)
        print(self.t)
        print(self.input)

        plt.subplot(121)
        plt.title("Imu data")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t,self.u[1])

        plt.subplot(122)
        plt.title("Fake wheel encoder")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity")
        plt.plot(self.t,self.u[0])

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag through a kalman filter")
    parser.add_argument("bagpath",help="Rosbag path")
    args = parser.parse_args()
    simple_kalman_bag = SimpleKalmanBag(args.bagpath)
    simple_kalman_bag.read_bag()
    simple_kalman_bag.plot_all()
