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
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

import tf
import rosbag
from geometry_msgs.msg import Quaternion

class EKFReader:

    x_ekf = [[],[],[],[]]
    t = []
    plot_x = [[],[],[],[]]
    plot_xy = []

    def __init__(self, bag_path=""):
        self.bag_path = bag_path

    def read_odom(self, odom_topic):
        rosbag_f = rosbag.Bag(self.bag_path)
        odom_msgs = rosbag_f.read_messages(topics=odom_topic)
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
            self.x_ekf[0].append(pos_x)
            self.x_ekf[1].append(pos_y)
            self.x_ekf[2].append(vel_x)
            self.x_ekf[3].append(yaw)
            self.t.append(t)

        self.plot_t = np.abs(np.array(self.t) - self.t[0])

    def filter_butter(self, array, order=5, fc=1/50.):
        fs = 50
        w = fc / (fs / 2.) # Normalize the frequency
        b, a = signal.butter(order, w, 'low', analog=False)
        output = signal.filtfilt(b, a, array)
        return output

    def find_slice(self, start=0.0, finish=np.Inf):
        begin = 0
        end = -1
        for elem in self.t:
            if elem <= finish:
                end = end + 1
            if elem <= start:
                begin = begin + 1
        end = len(self.t) - end
        return begin,end

    def set_zero_time(self, begin):
        new_t_array = []
        print(self.t[begin])
        for elem in self.t:
            new_t_array.append(elem - self.t[begin])
        self.t = new_t_array

    def plot_all(self, start_t=0, end_t=np.inf):
        begin,end = self.find_slice(start_t,end_t)
        self.set_zero_time(begin)
        plt.figure(1)

        plt.subplot(411)
        plt.title("x pos")
        self.plot_x[0] = plt.plot(self.t[begin:-end],self.x_ekf[0][begin:-end])
        #plt.plot(self.t[begin:-end], self.filter_butter(self.x_ekf[0][begin:-end],2,1/50.), "r", label="1/50")
        #plt.legend()

        plt.subplot(412)
        plt.title("y pos")
        self.plot_x[1] = plt.plot(self.t[begin:-end],self.x_ekf[1][begin:-end])
        #plt.plot(self.t[begin:-end], self.filter_butter(self.x_ekf[1][begin:-end],2,1/50.), "r", label="1/50")
        #plt.legend()

        plt.subplot(413)
        plt.title("Velocity in x")
        self.plot_x[2] = plt.plot(self.t[begin:-end], self.x_ekf[2][begin:-end])

        plt.subplot(414)
        plt.title("Phi")
        self.plot_x[3] = plt.plot(self.t[begin:-end],self.x_ekf[3][begin:-end])

        plt.figure(2)
        self.plot_xy = plt.plot(self.x_ekf[0][begin:-end],self.x_ekf[1][begin:-end])

        #plt.show()

    def export_loops(self, post=""):

        begin,end = self.slicer()

        np.savetxt("plots/loop_xy_{}.csv".format(post), np.transpose(
            [self.pos_x[begin:-end], self.pos_y[begin:-end]]), header='x y', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_yaw_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.yaw[begin:-end]]), header='t yaw', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_x_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.pos_x[begin:-end]]), header='t x', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_x_filter_50_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],5,1/50.)]), header='t x', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_x_filter_80_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],5,1/80.)]), header='t x', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_y_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.pos_y[begin:-end]]), header='t y', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_y_filter_50_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],5,1/50.)]), header='t y', comments='# ', delimiter=' ', newline='\n')

        np.savetxt("plots/loop_y_filter_80_{}.csv".format(post), np.transpose(
            [self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],5,1/80.)]), header='t y', comments='# ', delimiter=' ', newline='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag of EKF")
    parser.add_argument("-b", "--bag", help="Rosbag path")
    parser.add_argument("-t", "--topic", default="/odometry/filtered", help="Topic name")
    parser.add_argument("-t0", "--begin", type=float, default=0, help="Beginning of the slice")
    parser.add_argument("-t1", "--end", type=float, default=np.inf, help="End of slice")
    parser.add_argument("-p" ,"--post", type=str, default="", help="Post export text")

    args = parser.parse_args()

    ekf_reader = EKFReader(args.bag)
    ekf_reader.read_odom(args.topic)
    ekf_reader.plot_all(args.begin,args.end)
