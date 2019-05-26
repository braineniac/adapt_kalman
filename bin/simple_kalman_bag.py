#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import rosbag
import argparse
from simple_kalman import SimpleKalman

class SimpleKalmanBag:
    def __init__(self,bag_path=None):
        self.bag_path = bag_path
        self.bag = rosbag.Bag(bag_path)
        self.kalman = SimpleKalman(1000,5)

        self.u = [[],[]]
        self.t = []
        self.imu = []
        self.t_imu = []
        self.twist = []
        self.t_twist = []

    def read_imu(self, topic):
        msgs = self.bag.read_messages(topics=topic)
        for imu_msg in msgs:
            imu_t = imu_msg.message.header.stamp.to_sec()
            imu_a_x = imu_msg.message.linear_acceleration.x
            self.imu.append((imu_t,imu_a_x))

    def read_twist(self,topic):
        msgs = self.bag.read_messages(topics=topic)
        for twist_msg in msgs:
            twist_t = twist_msg.message.header.stamp.to_sec()
            twist_l_x = twist_msg.message.twist.twist.linear.x
            self.twist.append((twist_t,twist_l_x))

    def upscale_twist(self):
        last_j = (0,0)
        for _j in self.twist:
            t_twist,x_twist = _j
            last_twist_t, last_x_twist = last_j
            for _k in self.imu:
                t_imu,x_imu = _k
                if last_twist_t <= t_imu and t_twist > t_imu:
                    self.u[0].append(last_x_twist)
                    self.u[1].append(x_imu)
                    self.t.append(t_imu)
                elif t_twist < t_imu:
                    break;
            last_j = _j

    def run_filter(self):
        for u,t in zip(zip(self.u[0],self.u[1]), np.diff(self.t)):
            self.kalman.filter(u,t)

    def plot_all(self):
        plt.figure(2)

        plt.subplot(211)
        plt.title("Imu data")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t, self.u[1])

        plt.subplot(212)
        plt.title("Fake wheel encoder")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity")
        plt.plot(self.t, self.u[0])

        self.kalman.plot_all()

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag through a kalman filter")
    parser.add_argument("bagpath",help="Rosbag path")
    args = parser.parse_args()
    simple_kalman_bag = SimpleKalmanBag(args.bagpath)
    simple_kalman_bag.read_imu("/imu")
    simple_kalman_bag.read_twist("/fake_wheel/twist")
    simple_kalman_bag.upscale_twist()
    simple_kalman_bag.run_filter()
    simple_kalman_bag.plot_all()
