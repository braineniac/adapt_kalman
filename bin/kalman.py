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

class Kalman:

    L_k = np.zeros((4,1))                # Kalman gain matrix
    x_k_pre = np.zeros((4,1))            # A priori state vector
    P_k_pre = np.zeros((4,4))            # A priori covariance matrix
    x_k_post = np.zeros((4,1))           # A posteriori state vector
    P_k_post = np.zeros((4,4))           # A posteriori covariance matrix
    x_k_extr = np.zeros((4,1))           # extrapolated state vector
    P_k_extr = np.zeros((4,4))           # extrapolated covariance matrix
    C_k = np.zeros((2,4))                # measurement matrix
    phi_k = np.zeros((4,4))              # dynamics matrix
    D_k = np.zeros((2,2))                # measurement input matrix
    gamma_k = np.zeros((4,2))            # control matrix
    R_k = np.zeros((2,2))                # observation noise covaraiance matrix
    Q_k = np.zeros((2,2))                # process noise covariance matrix
    G_k = np.zeros((4,2))                #
    H_k = np.zeros((2,2))                #
    y_k = np.zeros((2,1))                # observation vector
    u_k = np.zeros((2,1))                # control input vector
    r_k = np.zeros((2,2))                # ratio matrix

    G_k[2][0] = 1
    G_k[3][1] = 1
    H_k[0][0] = 1
    H_k[1][1] = 1

    t = 0.0
    dt = 0.0
    t_a = []
    x_a = [[],[],[],[]]
    u_a = [[],[]]
    y_a = [[],[]]

    def __init__(self, ratio1=1/3., ratio2=1.):
        self.r_k[0][0] = ratio1
        self.r_k[1][1] = ratio2
        imu_stdev = 0.04
        gyro_stdev = 0.02
        fake_enc_stdev = ratio1 * imu_stdev
        ang_z_stdev = ratio2 * gyro_stdev
        self.R_k[0][0] = imu_stdev*imu_stdev
        self.R_k[1][1] = gyro_stdev*gyro_stdev
        self.Q_k[0][0] = fake_enc_stdev*fake_enc_stdev
        self.Q_k[1][1] = ang_z_stdev*ang_z_stdev

    def set_time_delta(self):
        self.phi_k = np.array([
        [1,0,self.dt*np.cos(self.x_k_post[3]),0],
        [0,1,self.dt*np.sin(self.x_k_post[3]),0],
        [0,0,0,0],
        [0,0,0,1]
        ])
        self.gamma_k = np.array([
        [0,0],
        [0,0],
        [1,0],
        [0,self.dt],
        ])
        self.C_k = np.array([
        [0,0,1/self.dt,0],
        [0,0,0,0]
        ])
        self.D_k = np.array([
        [-1/self.dt,0],
        [0,1]
        ])

    def set_gain(self):
        E = self.C_k.dot(self.P_k_pre).dot(self.C_k.T) + self.H_k.dot(self.Q_k).dot(self.H_k.T) + self.R_k
        self.L_k = self.P_k_pre.dot(self.C_k.T).dot(np.linalg.inv(E))

    def update_state(self):
        F = self.y_k - self.C_k.dot(self.x_k_pre) - self.D_k.dot(self.u_k)
        self.x_k_post = self.x_k_pre + self.L_k.dot(F)

    def update_error_covar(self):
        self.P_k_post = (np.identity(4) - self.L_k.dot(self.C_k)).dot(self.P_k_pre)

    def extr_state(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)

    def extr_error_covar(self):
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T) + self.G_k.dot(self.Q_k).dot(self.G_k.T)

    def set_next_iter(self):
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr

    def filter_iter(self, u=None, y=None, dt=None):
        if u and y and dt:
            self.dt = dt
            self.u_k[0] = u[0]
            self.u_k[1] = u[1]
            self.y_k[0] = y[0]
            self.y_k[1] = y[1]

            # execute iteration steps
            self.set_time_delta()
            self.set_gain()
            self.update_state()
            self.update_error_covar()
            self.extr_state()
            self.extr_error_covar()
            self.set_next_iter()

            # append to arrays
            self.t = self.t + dt
            self.t_a.append(self.t)
            self.x_a[0].append(self.x_k_post[0])
            self.x_a[1].append(self.x_k_post[1])
            self.x_a[2].append(self.x_k_post[2])
            self.x_a[3].append(self.x_k_post[3])
            self.u_a[0].append(u[0])
            self.u_a[1].append(u[1])
            self.y_a[0].append(y[0])
            self.y_a[1].append(y[1])
