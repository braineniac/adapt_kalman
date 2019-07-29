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

class AdaptKalmanSim(AdaptKalman):

    def __init__(self, N=200,sim_time=5.0,peak_vel=0.14,ratio=1/3, window="sig", window_size=4, adapt=False):
        AdaptKalman.__init__(self, ratio,window, window_size, adapt)
        self.t = np.linspace(0,sim_time,N)
        self.vel = self.set_vel(sim_time,peak_vel,N)
        self.accel = self.set_accel(sim_time,N)

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.filter_step(u,t)

    def set_vel(self,sim_time,peak_vel, N):
        t = self.t
        box_function = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        return box_function

    def set_accel(self,sim_time,N):
        accel = 0
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.vel,gauss/gauss.sum(), mode="same")
        grad = 50*np.gradient(conv)
        noise_still = np.random.normal(0,0.05,N)
        noise_moving = self.get_noise_moving(3)
        offset = 0.3

        accel += grad
        accel += noise_still
        accel += noise_moving
        accel += offset

        return accel

    def get_noise_moving(self, peak_coeff):
        noise_moving = []
        for x in self.vel:
            # fill staying still with zeros
            if abs(x) < 0.01:
                noise_moving.append(0.0)
            else:
                noise_moving.append(np.random.normal(0,x*peak_coeff))

        return noise_moving

    def export_all(self, begin=0, end=1):

        new_t_array = []
        for elem in self.plot_t:
            new_t_array.append(elem - self.plot_t[begin])

        self.plot_t = new_t_array

        np.savetxt("plots/sim_input_vel.csv", np.transpose([self.plot_t[begin:-end], self.plot_u[begin:-end]]) ,header='t u0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/sim_input_accel.csv", np.transpose([self.plot_t[begin:-end], self.plot_a[begin:-end]]) ,header='t a', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/sim_robot_dist_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],self.plot_y[begin:-end]]) ,header='t y', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/sim_robot_vel_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],self.plot_v[begin:-end]]) ,header='t v', comments='# ',delimiter=' ', newline='\n')

        fill = len(self.plot_t) - len(self.plot_r)
        full_ratio_array = np.insert(self.plot_r, 0, np.full((fill),self.r_k))

        np.savetxt("plots/sim_robot_ratio_{}.csv".format(self.window), np.transpose([self.plot_t[begin:-end],full_ratio_array[begin:-end]]) ,header='t r', comments='# ',delimiter=' ', newline='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Garry rosbag simulation")
    parser.add_argument("-N", type=int, default=200, help="Number of points")
    parser.add_argument("-t", "--sim_time", type=float, default=5.0, help="Simulation time span")
    parser.add_argument("-p", "--peak_vel", type=float, default=0.14, help="Peak velocity")
    parser.add_argument("-r", "--ratio", type=float, default=1/3., help="Covariance ratio")
    parser.add_argument("-w", "--window", type=str, default="", help="Window type: sig or exp")
    parser.add_argument("-ws", "--window_size", type=int, default=5, help="Window size")
    args = parser.parse_args()

    adapt = False
    if args.window != "":
        adapt=True

    adapt_kalman_sim = AdaptKalmanSim(
        N=args.N,
        sim_time=args.sim_time,
        peak_vel=args.peak_vel,
        ratio=args.ratio,
        window=args.window,
        window_size=args.window_size,
        adapt=adapt
    )
    adapt_kalman_sim.run_filter()
    adapt_kalman_sim.plot_all(10)
    adapt_kalman_sim.export_all(10)
