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

import numpy as np
from matplotlib import pyplot as plt
import argparse

from adapt_kalman import AdaptKalman

class AdaptKalmanSimCircle(AdaptKalman):
    u_sim = [[],[]]
    y_sim = [[],[]]
    t_sim = []

    def __init__(self, alpha=1.0, beta=1.0, N=200,sim_time=5.0,r1=1/3.,r2=1., window="sig", ws1=5, ws2=5, o1=3,o2=1):
        AdaptKalman.__init__(self, alpha=alpha,beta=beta,r1=r1,r2=r2,window_type=window, ws1=ws1, ws2=ws2, o1=o1, o2=o2)
        self.t_sim = np.linspace(0,sim_time,N)
        self.u_sim[0] = self.set_vel(sim_time,0.5,N)
        self.u_sim[1] = self.set_a_z(sim_time)
        self.y_sim[0] = self.set_accel(sim_time,N)
        self.y_sim[1] = self.set_dphi(sim_time)

    def run_filter(self):
        for u,y,t in zip(zip(self.u_sim[0],self.u_sim[1]),zip(self.y_sim[0],self.y_sim[1]), np.diff(self.t_sim)):
            self.filter_step(u,y,t)

    def set_dphi(self,sim_time):
        turn_rate = 2*np.pi / sim_time
        k = 0.01 # drift over time
        dphi = []
        for t in self.t_sim:
            dphi.append(t*k*turn_rate + turn_rate)
        return np.array(dphi)

    def set_a_z(self,sim_time):
        turn_rate = 2*np.pi / sim_time
        t = self.t_sim
        return np.piecewise(t, t>0, [turn_rate])

    def set_vel(self,sim_time,peak_vel,N):
        array = np.linspace(0,sim_time,N)
        N_8 = N / 8
        for i in range(8):
            self.line_vel(array,i*N_8, (i+1) * N_8,peak_vel)
        return array

    def line_vel(self,array,begin,end,peak_vel):
        t = self.t_sim[begin:end]
        t_start = self.t_sim[int(begin+0.1*len(t))]
        t_stop = self.t_sim[int(begin+0.9*len(t))]
        array[begin:end] = np.piecewise(t, [t<t_start,t>t_start,t>t_stop], [0,peak_vel,0])

    def set_accel(self,sim_time,N):
        accel = 0
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.u_sim[0],gauss/gauss.sum(), mode="same")
        grad = 25*np.gradient(conv)
        noise_still = np.random.normal(0,0.08,N)
        noise_moving = self.get_noise_moving(3)
        offset = 0.3

        accel += grad
        accel += noise_still
        accel += noise_moving
        #accel += offset

        return accel

    def get_noise_moving(self, peak_coeff):
        noise_moving = []
        for x in self.u_sim[0]:
            # fill staying still with zeros
            if abs(x) < 0.01:
                noise_moving.append(0.0)
            else:
                noise_moving.append(np.random.normal(0,x*peak_coeff))

        return noise_moving

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Garry rosbag circle simulation")
    parser.add_argument("-N", type=int, default=500, help="Number of points")
    parser.add_argument("-t", "--sim_time", type=float, default=5.0, help="Simulation time span")
    parser.add_argument("--alpha", default=1.0,help="Alpha")
    parser.add_argument("--beta", default=1.0,help="Beta")
    parser.add_argument("-r1", "--ratio1", type=float, default=1/3., help="Covariance ratio1")
    parser.add_argument("-r2", "--ratio2", type=float, default=1., help="Covariance ratio2")
    parser.add_argument("-w", "--window", type=str, default="", help="Window type: sig or exp")
    parser.add_argument("-ws1", "--window_size1", type=int, default=5, help="Window size1")
    parser.add_argument("-ws2", "--window_size2", type=int, default=5, help="Window size2")
    parser.add_argument("-o1", "--order1", type=int, default=3, help="Adaptive order1")
    parser.add_argument("-o2", "--order2", type=int, default=1, help="Adaptive order2")
    parser.add_argument("-t0", "--begin", type=float, default=0, help="Beginning of the slice")
    parser.add_argument("-t1", "--end", type=float, default=np.inf, help="End of slice")
    parser.add_argument("-p" ,"--post", type=str, default="", help="Post export text")
    args = parser.parse_args()

    adapt_kalman_sim_circle = AdaptKalmanSimCircle(
        N= args.N,
        sim_time = args.sim_time,
        alpha=args.alpha,
        beta=args.beta,
        r1=args.ratio1,
        r2=args.ratio2,
        window=args.window,
        ws1=args.window_size1,
        ws2=args.window_size2,
        o1=args.order1,
        o2=args.order2
        )
    adapt_kalman_sim_circle.run_filter()
    adapt_kalman_sim_circle.plot_all(args.begin,args.end)
    adapt_kalman_sim_circle.export_all(args.begin,args.end, "sim_circle", args.post)
