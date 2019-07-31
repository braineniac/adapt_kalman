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
from fractions import Fraction

class Kalman:

    # main vectors and matrices
    L_k = np.zeros((3,1))                                  # Kalman gain
    x_k_pre = np.zeros((3,1))                              # A priori state
    P_k_pre = np.zeros((3,3))                              # A priori covariance
    x_k_post = np.zeros((3,1))                             # A posteriori state
    P_k_post = np.zeros((3,3))                             # A posteriori covariance
    x_k_extr = np.zeros((3,1))                             # extrapolated state
    P_k_extr = np.zeros((3,3))                             # extrapolated covariance
    C_k = np.zeros(3)
    phi_k = np.zeros((3,3))
    D_k = np.zeros((3,3))
    gamma_k = np.zeros((3,3))                              # control matrix
    R_k = np.zeros((3,3))                                  # observation noise covaraiance
    Q_k = np.zeros((3,3))                                  # process noise covariance
    G_k = np.zeros((3,1))
    H_k = np.zeros((3,1))
    y_k = np.zeros((3,1))                                  # output
    u_k = np.zeros((3,1))                                  # control vector
    H_k = np.array([1,1,1])
    G_k = np.array([1,1,1])
    small_val = np.exp(-99)
    R_k.fill(small_val)
    Q_k.fill(small_val)
    P_k_pre.fill(small_val)

    # plotting
    sum_t = 0.0
    plot_t = []
    plot_y = []
    plot_v = []
    plot_u = []
    plot_a = []

    def __init__(self, ratio=1/3.):
        self.r_k = ratio
        fake_enc_stdev, imu_stdev = self.decomp_fraction(ratio)
        self.R_k = imu_stdev*imu_stdev
        self.Q_k[1][1] = fake_enc_stdev*fake_enc_stdev

    def set_time_delta(self):
        t = self.delta_t
        self.D_k = np.array([0,0,0])
        self.phi_k = np.array([[1,t,0],[0,0,0],[0,0,0]])
        self.gamma_k = np.array([[0,0,0],[0,1,0],[0,0,0]])
        self.C_k = np.array([0,0,1])

    def set_gain(self):
        E = self.C_k.dot(self.P_k_pre).dot(self.C_k.T) + self.H_k.dot(self.Q_k).dot(self.H_k.T) + self.R_k
        self.L_k = self.P_k_pre.dot(self.C_k.T).dot(1/E)

    def update_state(self):
        F = self.y_k - self.C_k.dot(self.x_k_pre) - self.D_k.dot(self.u_k)
        self.x_k_post = self.x_k_pre + self.L_k.dot(F)

    def update_error_covar(self):
        self.P_k_post = (np.identity(3) - self.L_k.dot(self.C_k)).dot(self.P_k_pre)

    def extr_state(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)

    def extr_error_covar(self):
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T) + self.G_k.dot(self.Q_k).dot(self.G_k.T)

    def set_next_iter(self):
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr

    def filter_iter(self, u=None, t=None):
        if u and t:
            self.delta_t = t
            self.u_k[0] = 0
            self.u_k[1] = u[0]
            self.y_k[0] = 0
            self.y_k[2] = u[1]

            self.set_time_delta()
            self.set_gain()
            self.update_state()
            self.update_error_covar()
            self.extr_state()
            self.extr_error_covar()
            self.set_next_iter()

            # append to arrays for plotting
            self.sum_t = self.sum_t + t
            self.plot_y.append(self.x_k_post[0])
            self.plot_v.append(self.x_k_post[1])
            self.plot_u = np.append(self.plot_u, [u[0]])
            self.plot_a = np.append(self.plot_a,[u[1]])
            self.plot_t.append(self.sum_t)

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
        plt.plot(self.plot_t,self.plot_v)

        plt.subplot(313)
        plt.title("Robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.plot_t, self.plot_a)

        plt.show()
