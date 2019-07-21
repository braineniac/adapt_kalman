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
import matplotlib
from matplotlib import pyplot as plt

import pandas as pd
from fractions import Fraction

class SimpleKalman:
    def __init__(self, ratio=1/10, window="sig", window_size=10, adapt=True):
        small_val = np.exp(-20)
        # set initial covariance
        #imu_stdev = (400/1000000) * 9.80655
        #fake_enc_stdev = (400/1000000) * 9.80655 / 100.

        self.use_adapt = adapt
        self.window_list = ["sig", "exp"]
        self.window = window
        self.window_size = window_size

        self.ratio = ratio
        fake_enc_stdev, imu_stdev = self.decomp_fraction(ratio)

        self.u = None
        self.t = 0

        self.delta_t = 0.0
        self.last_array = []
        self.peak = 0

        self.L_k = np.zeros((2,2))                                  # Kalman gain
        #self.P_k_pre = np.random.normal(small_val,1.0,(2,2))        # A priori covariance
        self.P_k_pre = np.zeros((2,2))
        self.P_k_post = np.zeros((2,2))                             # A posteriori covariance
        self.C_k = np.zeros(2)
        self.x_k_post = np.zeros((2,1))
        self.x_k_pre = np.zeros((2,1))
        self.x_k_extr = np.zeros((2,1))                             # extrapolated state
        self.P_k_extr = np.zeros((2,1))                             # extrapolated covariance
        self.phi_k = np.zeros((2,2))
        self.D_k = np.zeros((2,2))
        self.gamma_k = np.zeros((2,2))                              # control matrix
        self.R_k = np.zeros((2,2))                                  # measurement covariance
        self.Q_k = np.zeros((2,2))                                  # state covariance
        self.G_k = np.zeros((2,2))
        self.H_k = np.zeros((2,2))
        self.y_k = np.zeros((2,1))                                  # output
        self.u_k = np.zeros((2,1))                                  # control vector

        self.R_k[0][0] = small_val
        self.R_k[1][1] = imu_stdev*imu_stdev
        self.Q_k[0][0] = small_val
        self.Q_k[1][1] = fake_enc_stdev*fake_enc_stdev

        self.H_k[1][1] = 1
        self.G_k[1][1] = 1

        #  arrays for plotting
        self.sum_t = 0.0
        self.plot_t = []
        self.plot_y = []
        self.plot_v = []
        self.plot_u0 = []
        self.plot_a = []
        self.ratio_a = []
        self.coeff_a = []

    def time_update(self):
        self.D_k = np.array([[0,0],[0,self.delta_t]])
        self.phi_k = np.array([[1,self.delta_t],[0,0]])
        self.gamma_k = np.array([[0,0],[0,1]])
        self.C_k = np.array([[0,0],[1/2*self.delta_t*self.delta_t,self.delta_t]])

    def set_gain(self):
        E = self.C_k.dot(self.P_k_pre).dot(self.C_k.T) + self.H_k.dot(self.Q_k).dot(self.H_k.T) + self.R_k
        self.L_k = self.P_k_pre.dot(self.C_k.T).dot(np.linalg.inv(E))

    def update(self):
        F = self.y_k - self.C_k.dot(self.x_k_pre) - self.D_k.dot(self.u_k)
        self.x_k_post = self.x_k_pre + self.L_k.dot(F)
        self.P_k_post = (np.identity(2) - self.L_k.dot(self.C_k)).dot(self.P_k_pre)

    def extrapolate(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T) + self.G_k.dot(self.Q_k).dot(self.G_k.T)

        # update for next iteration
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr

    def filter(self, u,t):

        self.delta_t = t
        self.u_k[0] = 0
        self.u_k[1] = u[0]
        self.y_k[0] = 0
        self.y_k[1] = u[1]

        if self.use_adapt:
            self.adapt_covar()

        self.time_update()
        self.set_gain()
        self.update()
        self.extrapolate()

        # append to arrays for plotting
        self.sum_t = self.sum_t + t
        self.plot_y.append(self.x_k_post[0])
        self.plot_v.append(self.x_k_post[1])
        self.plot_u0 = np.append(self.plot_u0, [u[0]])
        self.plot_a = np.append(self.plot_a,[u[1]])
        self.plot_t.append(self.sum_t)

        #self.print_debug()

    def get_current_N(self, N):
        array = []
        for i in range(0,N):
            array.append(self.plot_u0[-i-1])
            #print(self.plot_u0[-i-1])
        return np.array(array)

    def sigmoid_mirror_avg(self,array):
        n = len(array)
        newarray = np.zeros(n)
        alpha = 5
        x = np.linspace(-1,1,n)
        w = np.zeros(n)

        # initilise weights
        for i in range(n):
            w[i] = 1 / (1 + np.exp(alpha * x[i]))

        for j in range(n):
            newarray[j] = array[j]*w[j] / np.sum(w)

        return np.sum(newarray)

    def exponential_avg(self,array):
        series = pd.Series(array)
        avg = np.average(series.ewm(span=len(array)).mean().values)
        return avg

    def adapt_covar(self):
        N = self.window_size
        if len(self.plot_u0) == N:
            self.last_array = self.get_current_N(N)

        elif len(self.plot_u0) >= 2*N:
            current_array = self.get_current_N(N)
            last_array = self.last_array

            if self.window in self.window_list:
                if self.window == "exp":
                    current_avg = self.exponential_avg(current_array)
                    last_avg = self.exponential_avg(last_array)

                elif self.window == "sig":
                    current_avg = self.sigmoid_mirror_avg(current_array)
                    last_avg = self.sigmoid_mirror_avg(last_array)

            avg_diff = abs(current_avg - last_avg)

            # set the biggest peak detected
            peak = avg_diff
            if peak > self.peak:
                self.peak = peak
            if self.peak != 0:
                k = (np.reciprocal(self.ratio) - 1) / self.peak
            else:
                k = 0

            # offset by 1 and scale with max peak
            coeff = avg_diff * k + 1

            # update covariances
            u0_stdev,u1_stdev = self.decomp_fraction(self.ratio * coeff)
            self.Q_k[1][1] = u0_stdev*u0_stdev
            self.R_k[1][1] = u1_stdev*u1_stdev

            # update for next iteration
            self.last_array = current_array

            # append for plotting
            self.coeff_a.append(coeff)
            self.ratio_a.append(self.ratio * coeff)

            if self.plot_v[-4] > 1:
                exit(1)

    def decomp_fraction(self, frac):
        ratio = Fraction(frac).limit_denominator(200)

        num_len = len(str(ratio.numerator))
        denum_len = len(str(ratio.denominator))

        if num_len > denum_len:
            dec_shift = num_len
        else:
            dec_shift = denum_len

        num = ratio.numerator / (10.**dec_shift)
        denum = ratio.denominator / (10.**dec_shift)

        return [num,denum]

    def print_debug(self):
        print("Vel:{}".format(self.plot_v[-1]))
        print("Accel:{}".format(self.plot_a[-1]))
        print("u0:{}".format(self.plot_u0[-1]))
        #print("Coeff:{}".format(self.coeff_a[-1]))
        #print("Ratio:{}".format(self.ratio_a[-1]))
        print("Q_k:{}".format(self.Q_k[1][1]))
        print("R_k:{}".format(self.R_k[1][1]))
        print("P_k_pre:{}".format(self.P_k_pre))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def plot_all(self):
        plt.figure(1)

        plt.subplot(311)
        plt.title("Robot distance")
        plt.xlabel("Time in s")
        plt.ylabel("Distance in m")
        plt.plot(self.plot_t,self.plot_y)

        plt.subplot(312)
        plt.title("Robot velocity post")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.plot_t,self.plot_v_post)

        plt.subplot(313)
        plt.title("Robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.plot_t, self.plot_a)

        plt.show()
