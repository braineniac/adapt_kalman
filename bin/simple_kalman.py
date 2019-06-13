#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg',warn=False, force=True)
from matplotlib import pyplot as plt

import pandas as pd
from fractions import Fraction

class SimpleKalman:
    def __init__(self, ratio=1/10):
        small_val = np.exp(-20)
        # set initial covariance
        #imu_stdev = (400/1000000) * 9.80655
        #fake_enc_stdev = (400/1000000) * 9.80655 / 100.0

        self.ratio = Fraction(ratio).limit_denominator(200)
        fake_enc_stdev = self.ratio.numerator
        imu_stdev = self.ratio.denominator

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
        self.test_a = []


        self.ratio_a = []
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

    def filter(self, u,t,N=10):
        self.plot_v_pre.append(self.x_k_pre[1])
        self.delta_t = t
        self.u_k[0] = 0
        self.u_k[1] = u[0]
        self.y_k[0] = 0
        self.y_k[1] = u[1]

        self.adapt_covar(N)
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

    # def get_diff_sum(self, array = [], N=0):
    #     if array == []:
    #         return None
    #
    #     np_array = np.array(array).reshape((1,N))[0]
    #     #print(np_array)
    #     #array_diff = np.diff(np_array)
    #     #print(array_diff)
    #     #array_pos = self.get_abs_array(array_diff)
    #     #print(array_pos)
    #     array_sum = np.sum(np_array)
    #     if array_sum < 1.0e-10:
    #         array_sum = 1.0e-10
    #
    #     return array_sum

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

        #plt.figure(3)
        #plt.plot(x,w)
        #plt.show()

        return np.sum(newarray)


    def adapt_covar(self, N=0):
        if len(self.plot_u0) == N:
            self.last_array = self.get_current_N(N)

        elif len(self.plot_u0) >= 2*N:
            current_array = self.get_current_N(N)
            last_array = self.last_array
            # last_diff_sum = self.get_diff_sum(last_array, N)
            # current_diff_sum = self.get_diff_sum(current_array,N)

            last_series = pd.Series(last_array)
            current_series = pd.Series(current_array)

            current_avg = np.average(current_series.ewm(span=N).mean().values)
            last_avg = np.average(last_series.ewm(span=N).mean().values)
#############################################################################
            test_current_series = pd.Series(current_array)
            test_last_series = pd.Series(last_array)
            test_current_series_flip = pd.Series(np.flip(current_array))
            test_last_series_flip = pd.Series(np.flip(last_array))

            test_current_avg = np.average(test_current_series.ewm( alpha=0.001).mean().values)
            test_last_avg = np.average(test_last_series.ewm(span=N).mean().values)
            test_current_avg_flip = np.average(test_current_series_flip.ewm(span=N).mean().values)
            test_last_avg_flip = np.average(test_last_series_flip.ewm(span=N).mean().values)
            test_diff = abs(test_current_avg - test_last_avg)
            test_diff_flip = abs(test_current_avg_flip - test_last_avg_flip)

            current_sig = self.sigmoid_mirror_avg(current_array)
            last_sig = self.sigmoid_mirror_avg(last_array)
            #print("$$$$$$$$$$$$$$$$$$$$$$$$")
            #print(current_sig)
            #print(last_sig)
            #print(abs(current_sig - last_sig))
            #self.test_a.append(current_sig - last_sig)
##############################################################################
            avg_diff = abs(current_sig - last_sig)
            peak = avg_diff
            if peak > self.peak:
                self.peak = peak

            if self.peak != 0:
                k = (np.reciprocal(self.ratio) - 1) / self.peak
            else:
                k = 0
            coeff = avg_diff * k + 1

            self.coeff_array.append(coeff)
            self.ratio_a.append(self.ratio * coeff)

            new_ratio = Fraction(self.ratio * coeff).limit_denominator(200)

            u0_stdev = new_ratio.numerator
            u1_stdev = new_ratio.denominator
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # print("coeff: {}".format(coeff))
            # print("u0 stdev: {}".format(u0_stdev))
            # print("u1 stdev: {}".format(u1_stdev))
            # print("last diff sum: {}".format(last_diff_sum))
            # print("current diff sum: {}".format(current_diff_sum))
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
