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
plt.subplots_adjust(hspace=1.5)
from simple_kalman import SimpleKalman
import argparse

class SimpleKalmanSim:

    def __init__(self, N=200,
                       sim_time=5.0,
                       peak_vel=0.14,
                       ratio=1/3,
                       window="sig",
                       window_size=4,
                       adapt=False):
        self.t = np.linspace(0,sim_time,N)
        self.vel = self.set_vel(sim_time,peak_vel,N)
        self.accel = self.set_accel(sim_time,N)
        self.kalman = SimpleKalman(ratio,window,window_size,adapt)

    def set_vel(self,sim_time,peak_vel, N):
        t = self.t
        box_function = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        return box_function

    def plot_all(self):
        plt.figure(1)

        xticks = len(self.kalman.plot_t)

        plt.subplot(611)
        plt.title("Simulated robot velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.t,self.vel)

        plt.subplot(612)
        plt.title("Robot distance")
        plt.xlabel("Time in s")
        plt.ylabel("Distance in m")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t,self.kalman.plot_y)

        plt.subplot(613)
        plt.title("Robot velocity post")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t,self.kalman.plot_v)

        plt.subplot(614)
        plt.title("Robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t, self.kalman.plot_a)

        # plt.subplot(615)
        # plt.title("test")
        # fill = len(self.kalman.plot_t) - len(self.kalman.test_a)
        # full_test_array = np.insert(self.kalman.test_a,0, np.zeros(fill))
        # plt.plot(self.kalman.plot_t,full_test_array)

        plt.subplot(615)
        plt.title("Coeff")
        #plt.xticks(np.arange(0, len(self.kalman.plot_t[30:]), step=0.2))
        fill = len(self.kalman.plot_t) - len(self.kalman.coeff_a)
        full_coeff_array = np.insert(self.kalman.coeff_a,0, np.ones(fill))
        plt.plot(self.kalman.plot_t,full_coeff_array)

        plt.subplot(616)
        plt.title("Ratio")
        #plt.xticks(np.arange(0, len(self.kalman.plot_t[30:]), step=0.2))
        fill = len(self.kalman.plot_t) - len(self.kalman.ratio_a)
        full_ratio_array = np.insert(self.kalman.ratio_a, 0, np.full((fill),self.kalman.ratio))
        plt.plot(self.kalman.plot_t,full_ratio_array)

        plt.show()

    def get_noise_moving(self, peak_coeff):
        noise_moving = []
        for x in self.vel:
            # fill staying still with zeros
            if abs(x) < 0.01:
                noise_moving.append(0.0)
            else:
                noise_moving.append(np.random.normal(0,x*peak_coeff))

        return noise_moving

    def set_accel(self,sim_time,N):
        accel = 0
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.vel,gauss/gauss.sum(), mode="same")
        grad = 50*np.gradient(conv)
        grad_shift = 0.7 * np.roll(grad,10)
        noise_still = np.random.normal(0,0.05,N)
        noise_moving = self.get_noise_moving(3)
        offset = 0.3

        accel += grad
        #accel += grad_shift
        accel += noise_still
        accel += noise_moving
        accel += offset

        return accel

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.kalman.filter(u,t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Garry rosbag simulation")
    parser.add_argument("-N", type=int, default=200, help="Number of points")
    parser.add_argument("-t", "--sim_time", type=float, default=5.0, help="Simulation time span")
    parser.add_argument("-p", "--peak_vel", type=float, default=0.14, help="Peak velocity")
    parser.add_argument("-r", "--ratio", type=float, default=1/3., help="Covariance ratio")
    parser.add_argument("-w", "--window", type=str, default="sig", help="Window type: sig or exp")
    parser.add_argument("-ws", "--window_size", type=int, default=5, help="Window size")
    parser.add_argument("-a", "--adapt", action="store_true", help="Use adaptive covariances")
    args = parser.parse_args()
    simple_kalman_sim = SimpleKalmanSim(
        N=args.N,
        sim_time=args.sim_time,
        peak_vel=args.peak_vel,
        ratio=args.ratio,
        window=args.window,
        window_size=args.window_size,
        adapt=args.adapt
    )
    simple_kalman_sim.run_filter()
    simple_kalman_sim.plot_all()
