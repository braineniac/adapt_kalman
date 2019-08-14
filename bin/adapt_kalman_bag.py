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

    def __init__(self, alpha=1.0,beta=1.0,bagpath=None, r1=1/3., r2 = 1.,window="",ws=5, o1=5, o2=5,turn=0.0):
        super(AdaptKalmanBag,self).__init__(alpha=alpha, beta=beta, r1=r1, r2=r2, window=window, ws=ws, o1=o1, o2=o2, x0=[0,0,0,turn])
        self.t_bag = []
        self.u_bag = [[],[]]
        self.y_bag = [[],[]]
        self.imu = []
        self.twist = []
        self.bag = rosbag.Bag(bagpath)

    def run_filter(self):
        for u,y,t in zip(zip(self.u_bag[0],self.u_bag[1]), zip(self.y_bag[0], self.y_bag[1]), np.diff(self.t_bag)):
            self.filter_step(u,y,t)

    def read_imu(self, topic):
        msgs = self.bag.read_messages(topics=topic)
        for imu_msg in msgs:
            imu_t = imu_msg.message.header.stamp.to_sec()
            imu_a_x = imu_msg.message.linear_acceleration.x
            imu_g_z = imu_msg.message.angular_velocity.z
            self.imu.append((imu_t,imu_a_x, imu_g_z))

    def read_twist(self,topic):
        msgs = self.bag.read_messages(topics=topic)
        for twist_msg in msgs:
            twist_t = twist_msg.message.header.stamp.to_sec()
            twist_l_x = twist_msg.message.twist.twist.linear.x
            twist_a_z = twist_msg.message.twist.twist.angular.z
            self.twist.append((twist_t,twist_l_x,-twist_a_z))

    def upscale_twist(self):
        """
        This upscales the fake wheel encoder input to match the imu rate.

        It just replicates the last avalable input until a new one comes.
        """
        last_j = (0,0,0)
        for _j in self.twist:
            t_twist,x_twist,a_z_twist = _j
            last_twist_t, last_x_twist, last_a_z_twist = last_j
            for _k in self.imu:
                t_imu,x_imu,g_z_imu = _k
                if last_twist_t <= t_imu and t_twist > t_imu:
                    self.u_bag[0].append(last_x_twist)
                    self.u_bag[1].append(last_a_z_twist)
                    self.y_bag[0].append(x_imu)
                    self.y_bag[1].append(g_z_imu)
                    self.t_bag.append(t_imu)
                elif t_twist < t_imu:
                    break;
            last_j = _j

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag through a kalman filter")
    parser.add_argument("-b", "--bagpath",help="Rosbag path")
    parser.add_argument("--turn", type=float, default=0.0,help="Robot turn in degrees")
    parser.add_argument("--imu", type=str, default="/imu", help="IMU topic")
    parser.add_argument("--twist", type=str, default="/fake_encoder/twist", help="Twist topic")
    parser.add_argument("--alpha", type=float, default=1.0,help="Alpha")
    parser.add_argument("--beta", type=float,default=1.0,help="Beta")
    parser.add_argument("-r1", "--ratio1", type=float, default=1/3., help="Covariance ratio1")
    parser.add_argument("-r2", "--ratio2", type=float, default=1., help="Covariance ratio2")
    parser.add_argument("-w", "--window", type=str, default="", help="Window type: sig or exp")
    parser.add_argument("-ws", "--window_size", type=int, default=5, help="Window size")
    parser.add_argument("-o1", "--order1", type=int, default=3, help="Adaptive order1")
    parser.add_argument("-o2", "--order2", type=int, default=3, help="Adaptive order2")
    parser.add_argument("-t0", "--begin", type=float, default=0, help="Beginning of the slice")
    parser.add_argument("-t1", "--end", type=float, default=np.inf, help="End of slice")
    parser.add_argument("-p" ,"--post", type=str, default="", help="Post export text")

    args = parser.parse_args()

    adapt_kalman_bag = AdaptKalmanBag(
        bagpath = args.bagpath,
        alpha=args.alpha,
        beta=args.beta,
        r1=args.ratio1,
        r2=args.ratio2,
        window=args.window,
        ws1=args.window_size,
        o1=args.order1,
        o2=args.order2,
        turn=args.turn
        )
    adapt_kalman_bag.read_imu(args.imu)
    adapt_kalman_bag.read_twist(args.twist)
    adapt_kalman_bag.upscale_twist()
    adapt_kalman_bag.run_filter()
    adapt_kalman_bag.plot_all(args.begin,args.end)
    #adapt_kalman_bag.export_all(args.begin,args.end, "real" ,args.post)
