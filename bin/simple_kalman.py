#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg',warn=False, force=True)
from matplotlib import pyplot as plt

import pandas as pd
from fractions import Fraction

class SimpleKalman:
    def __init__(self, u0_stdev=0.01,u1_stdev=0.01):
        small_val = np.exp(-20)
        # set initial covariance
        #imu_stdev = (400/1000000) * 9.80655
        #fake_enc_stdev = (400/1000000) * 9.80655 / 100.0
        #imu_stdev = 0.5
        #fake_enc_stdev = 0.5
        imu_stdev = u1_stdev
        fake_enc_stdev = u0_stdev

        self.u = None
        self.t = 0

        self.delta_t = 0.0
        self.sum_t = 0.0
        self.plot_t = []
        self.plot_y = []
        self.plot_v_post = []
        self.plot_v_pre = []
        self.plot_u0 = []
        self.plot_a = []

        self.ratio = u0_stdev/u1_stdev
        self.coeff_array = []
        self.last_array = []
        self.peak = 0

        self.L_k = np.zeros((2,2))                                  # Kalman gain
        self.P_k_pre = np.random.normal(small_val,1.0,(2,2))        # A priori covariance
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
        self.plot_v_pre.append(self.x_k_pre[1])
        self.delta_t = t
        self.u_k[0] = 0
        self.u_k[1] = u[0]
        self.y_k[0] = 0
        self.y_k[1] = u[1]

        self.adapt_covar(20)
        self.time_update()
        self.set_gain()
        self.update()
        self.extrapolate()

        self.sum_t = self.sum_t + t
        self.plot_y.append(self.x_k_post[0])
        self.plot_v_post.append(self.x_k_post[1])
        self.plot_u0 = np.append(self.plot_u0, [u[0]])
        self.plot_a = np.append(self.plot_a,[u[1]])
        self.plot_t.append(self.sum_t)

    def get_abs_array(self, array = []):
        if array == []:
            return None

        array_pos = []
        for x in array:
            if x < 0:
                x = -x
            array_pos.append(x)
        return array_pos

    def get_diff_sum(self, array = [], N=0):
        if array == []:
            return None

        np_array = np.array(array).reshape((1,N))[0]
        print(np_array)
        #array_diff = np.diff(np_array)
        #print(array_diff)
        #array_pos = self.get_abs_array(array_diff)
        #print(array_pos)
        array_sum = np.sum(np_array)
        if array_sum < 1.0e-10:
            array_sum = 1.0e-10

        return array_sum

    def get_current_N(self, N):
        array = []
        for i in range(0,N):
            array.append(self.plot_u0[-i-1])
            print(self.plot_u0[-i-1])
        return array

    def adapt_covar(self, N=0):
        if len(self.plot_u0) == N:
            self.last_array = self.get_current_N(N)

        elif len(self.plot_u0) >= 2*N:
            current_array = self.get_current_N(N)
            last_array = self.last_array
            last_diff_sum = self.get_diff_sum(last_array, N)
            current_diff_sum = self.get_diff_sum(current_array,N)

            last_series = pd.Series(last_array)
            current_series = pd.Series(current_array)
            peak = np.amax(current_array)
            if peak > self.peak:
                self.peak = peak

            current_avg = np.average(current_series.ewm(span=N).mean().values)
            last_avg = np.average(last_series.ewm(span=N).mean().values)

            coeff = np.log(current_avg/last_avg)

            self.coeff_array.append(coeff)

            new_ratio = Fraction(self.ratio * coeff).limit_denominator(200)

            u0_stdev = new_ratio.numerator
            u1_stdev = new_ratio.denominator
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("coeff: {}".format(coeff))
            print("u0 stdev: {}".format(u0_stdev))
            print("u1 stdev: {}".format(u1_stdev))
            print("last diff sum: {}".format(last_diff_sum))
            print("current diff sum: {}".format(current_diff_sum))
            self.Q_k[1][1] = u0_stdev*u0_stdev
            self.R_k[1][1] = u1_stdev*u1_stdev

            self.last_array = current_array

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
