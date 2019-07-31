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

class EKFExporter:

    vel = []
    pos_x = []
    yaw = []
    pos_y = []
    t = []

    def __init__(self, bag_path="", odom_topic="/odometry/filtered"):
        self.bag_path = bag_path
        self.odom_topic = odom_topic

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

    def filter_pos(self, array, order=5, fc=1/50.):
        fs = 50
        w = fc / (fs / 2.) # Normalize the frequency
        b, a = signal.butter(order, w, 'low', analog=False)
        output = signal.filtfilt(b, a, array)
        return output

    def plot_all(self, begin=0, end=1):

        begin,end = self.slicer()
        plt.figure(1)

        plt.subplot(411)
        plt.title("Velocity in x")
        plt.plot(self.plot_t[begin:-end], self.vel[begin:-end])

        plt.subplot(412)
        plt.plot(self.plot_t[begin:-end],self.pos_x[begin:-end])

        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],1,1/75.), "g", label="1/75")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],2,1/50.), "m", label="1/50")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],3,1/85.), "k", label="1/85")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],4,1/80.), "r", label="1/80")
        #plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_x[begin:-end],5,1/100.), "c", label="1/100")
        plt.legend()

        plt.subplot(413)
        plt.plot(self.plot_t[begin:-end],self.pos_y[begin:-end])
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],1,1/75.), "g", label="1/75")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],2,1/50.), "m", label="1/50")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],3,1/85.), "k", label="1/85")
        plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],4,1/80.), "r", label="1/80")
        #plt.plot(self.plot_t[begin:-end], self.filter_pos(self.pos_y[begin:-end],5,1/100.), "c", label="1/100")
        plt.legend()

        plt.subplot(414)
        plt.plot(self.plot_t[begin:-end],self.yaw[begin:-end])

        plt.figure(2)
        plt.plot(self.pos_x[begin:-end],self.pos_y[begin:-end])

        plt.show()

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

    def slicer(self, start=5.0, finish=16.0):
        begin = 0
        end = 0
        for elem in self.plot_t:
            if elem <= finish:
                end = end + 1
            if elem <= start:
                begin = begin + 1
        end = len(self.plot_t) - end
        return begin,end

    def plot_export_line(self, start=5.0, finish=16.0, post=""):

        begin,end = self.slicer(start,finish)

        new_t_array = []
        for elem in self.plot_t:
            new_t_array.append(elem - self.plot_t[begin])

        plt.figure(1)
        plt.title("Velocity in x")
        plt.plot(new_t_array[begin:-end], self.vel[begin:-end])

        plt.figure(2)
        plt.plot(new_t_array[begin:-end], self.pos_x[begin:-end])

        plt.show()

        np.savetxt("plots/ekf_pos_{}.csv".format(post), np.transpose(
            [new_t_array[begin:-end], self.pos_x[begin:-end]]), header='t x', comments='# ', delimiter=' ', newline='\n')
        np.savetxt("plots/ekf_vel_{}.csv".format(post), np.transpose(
            [new_t_array[begin:-end], self.vel[begin:-end]]), header='t v', comments='# ', delimiter=' ', newline='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag of EKF")
    parser.add_argument("-b", "--bag", help="Rosbag path")
    parser.add_argument("-t", "--topic", help="Topic name")
    parser.add_argument("-e", "--exp", default="loops", help="Type of experiment ran")
    parser.add_argument("--post", default="", help="Name postfix")

    args = parser.parse_args()
    if args.topic:
        ekf_exporter = EKFExporter(args.bag, args.topic)
    else:
        ekf_exporter = EKFExporter(args.bag)
    ekf_exporter.read_bag()


    if args.exp == "loops":
        ekf_exporter.export_loops(args.post)
        ekf_exporter.plot_all()
    elif args.exp == "line":
        ekf_exporter.plot_export_line(args.post)
