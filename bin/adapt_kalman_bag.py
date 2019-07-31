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

import argparse
import numpy as np
from matplotlib import pyplot as plt

import rosbag

from adapt_kalman import AdaptKalman

class AdaptKalmanBag(AdaptKalman):

    u = [[],[]]
    t = []
    imu = []
    twist = []

    def __init__(self, bagpath=None, ratio=1/3.,window="sig",window_size=5,adapt=True):
        AdaptKalman.__init__(self,ratio,window, window_size, adapt)
        self.bag = rosbag.Bag(bagpath)

    def run_filter(self):
        for u,t in zip(zip(self.u[0],self.u[1]), np.diff(self.t)):
            self.filter_step(u,t)

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
        """
        This upscales the fake wheel encoder input to match the imu rate.

        It just replicates the last avalable input until a new one comes.
        """
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

    def export_all(self, begin=0, end=1):

        begin,end = self.slicer()

        new_t_array = []
        for elem in self.plot_t:
            new_t_array.append(elem - self.plot_t[begin])

        self.plot_t = new_t_array

        np.savetxt("plots/real_input_vel.csv", np.transpose([self.plot_t[begin:-end], self.plot_u[begin:-end]]) ,header='t u0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/real_input_accel.csv", np.transpose([self.plot_t[begin:-end], self.plot_a[begin:-end]]) ,header='t a', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/real_robot_dist_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],self.plot_y[begin:-end]]) ,header='t y', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/real_robot_vel_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],self.plot_v[begin:-end]]) ,header='t v', comments='# ',delimiter=' ', newline='\n')

        fill = len(self.plot_t) - len(self.plot_r)
        full_ratio_array = np.insert(self.plot_r, 0, np.full((fill),self.r_k))

        np.savetxt("plots/real_robot_ratio_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],full_ratio_array[begin:-end]]) ,header='t r', comments='# ',delimiter=' ', newline='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag through a kalman filter")
    parser.add_argument("-b", "--bagpath",help="Rosbag path")
    parser.add_argument("-r", "--ratio", type=float, default=1/3., help="Covariance ratio")
    parser.add_argument("-w", "--window", type=str, default="", help="Window type: sig or exp")
    parser.add_argument("-ws", "--window_size", type=int, default=5, help="Window size")
    parser.add_argument("--imu", type=str, default="/imu", help="IMU topic")
    parser.add_argument("--twist", type=str, default="/fake_wheel/twist", help="Twist topic")

    args = parser.parse_args()

    adapt = False
    if args.window != "":
        adapt=True

    adapt_kalman_bag = AdaptKalmanBag(
        bagpath = args.bagpath,
        ratio=args.ratio,
        window=args.window,
        window_size=args.window_size,
        adapt=adapt)
    adapt_kalman_bag.read_imu(args.imu)
    adapt_kalman_bag.read_twist(args.twist)
    adapt_kalman_bag.upscale_twist()
    adapt_kalman_bag.run_filter()
    adapt_kalman_bag.plot_all()
    #adapt_kalman_bag.export_all(200,300)
