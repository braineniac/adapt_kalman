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
import pandas as pd
from matplotlib import pyplot as plt

from kalman import Kalman

class AdaptKalman(Kalman):
    delta_w_hat = np.zeros((2,1))  # delta_w peak vector
    u = [[],[]]                    # all control input vectors
    y = [[],[]]                    # all measuremnt vectors
    r_a = [[[],[]],[[],[]]]
    order = np.zeros((2,2))
    N = np.zeros((2,1))
    w_k_pre = np.zeros((2,1))
    w_k = np.zeros((2,1))
    wsize = np.zeros((2,1))
    # s = np.zeros((2,2))
    #R_k_pre = np.zeros((2,2))
    #R_k_set = False
    window_list = ["sig", "exp"]

    def __init__(self,alpha=1.0,beta=1.0,r1=1/3.,r2=1.0,window_type="sig",ws1=5, ws2=5, o1=3, o2=1):
        Kalman.__init__(self, r1, r2, alpha,beta)
        self.window_type = window_type
        self.N[0] = ws1
        self.N[1] = ws2
        self.order[0][0] = o1
        self.order[1][1] = o2
        self.wsize[0] = ws1
        self.wsize[1] = ws2
        #self.R_k_pre = self.R_k
        # self.s[0][0] = 1
        # self.s[1][1] = 1

    def filter_step(self, u=None, y=None,t=None):
        if u and t:
            self.adapt_covar()
            self.filter_iter(u,y,t)

    def adapt_covar(self):
        c_k = np.identity(2)

        if len(self.u_a[0]) >= 2*self.N[0] and len(self.u_a[1]) >= 2*self.N[1]:
            w_k = self.set_windows()
            w_k_pre = self.w_k_pre

            delta_w_k = abs(w_k - w_k_pre)

            self.set_peak(delta_w_k)

            c_k = (self.order/10).dot(self.delta_w_hat).dot(delta_w_k.T) + np.identity(2)

            # update for next iteration
            self.w_k_pre = w_k

        elif len(self.u_a[0]) >= self.N[0] and len(self.u_a[1]) >= self.N[1]:
                self.w_k_pre = self.set_windows()

        # if np.array_equal(np.identity(2), c_k):
        #     self.R_k_set = False
        #     self.R_k = self.R_k_pre
        # elif not self.R_k_set:
        #     self.R_k = self.R_k.dot(self.s)
        #     self.R_k_set = True

        r_k_post = c_k.dot(self.r_k)
        self.Q_k = self.R_k.dot(r_k_post)
        self.r_a[0][0].append(r_k_post[0][0])
        self.r_a[0][1].append(r_k_post[0][1])
        self.r_a[1][0].append(r_k_post[1][0])
        self.r_a[1][1].append(r_k_post[1][1])

    def set_windows(self):
        w_avg = np.zeros((2,1))
        w_avg[0] = self.get_window_avg(self.u_a[0], self.N[0])
        w_avg[1] = self.get_window_avg(self.u_a[1], self.N[1])
        return w_avg

    def get_window_avg(self, array, N):
        window = []
        avg = 0
        for i in range(0,N):
            window.append(array[-i-1])
        if self.window_type == "exp":
            avg = self.exp_avg(window)
        elif self.window_type == "sig":
            avg = self.sig_avg(window)
        return avg

    def set_peak(self, delta_w):
        for row in delta_w:
            for cell in delta_w:
                row = int(row)
                cell = int(cell)
                if self.delta_w_hat[row][cell] < delta_w[row][cell]:
                    self.delta_w_hat[row][cell] = 1/delta_w[row][cell]

    def sig_avg(self,array):
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

    def exp_avg(self,array):
        series = pd.Series(array)
        avg = np.average(series.ewm(span=len(array)).mean().values)
        return avg


    def find_slice(self, start=0.0, finish=np.Inf):
        begin = 0
        end = -1
        for elem in self.t_a:
            if elem <= finish:
                end = end + 1
            if elem <= start:
                begin = begin + 1
        end = len(self.t_a) - end
        return begin,end

    def set_zero_time(self, begin):
        new_t_array = []
        for elem in self.t_a:
            new_t_array.append(elem - self.t_a[begin])
        self.t_a = new_t_array

    def psi_fix(self):
        psi_a = []
        for psi in self.x_a[3]:
            k = abs(int(psi / (2*np.pi)))
            if psi > 2*np.pi:
                psi -= 2*np.pi*k
            elif psi < -2*np.pi*k:
                psi += 2*np.pi*(k+1)
            psi_a.append(psi*180/np.pi)
        self.x_a[3] = psi_a

    def plot_all(self, start=0.0,finish=np.Inf):
        begin,end = self.find_slice(start,finish)
        self.set_zero_time(begin)
        self.psi_fix()

        plt.figure(1)
        plt.subplot(411)
        plt.title("Robot distance in x")
        plt.ylabel("Distance in m")
        plt.xlabel("Time in s")
        plt.plot(self.t_a[begin:-end],self.x_a[0][begin:-end])

        plt.subplot(412)
        plt.title("Robot distance in y")
        plt.ylabel("Distance in m")
        plt.xlabel("Time in s")
        plt.plot(self.t_a[begin:-end], self.x_a[1][begin:-end])

        plt.subplot(413)
        plt.title("Robot velocity")
        plt.ylabel("Velocity in m/s")
        plt.xlabel("Time in s")
        plt.plot(self.t_a[begin:-end], self.x_a[2][begin:-end])

        plt.subplot(414)
        plt.title("phi")
        plt.ylabel("Phi in degrees")
        plt.xlabel("Time in s")
        plt.plot(self.t_a[begin:-end],self.x_a[3][begin:-end])

        plt.figure(2)
        plt.title("xy")
        plt.xlabel("x distance")
        plt.ylabel("y distance")
        plt.plot(self.x_a[0][begin:-end],self.x_a[1][begin:-end])

        plt.figure(3)
        plt.subplot(411)
        plt.title("fake wheel encoder input")
        plt.plot(self.t_a[begin:-end],[self.alpha*x for x in self.u_a[0][begin:-end]])

        plt.subplot(412)
        plt.title("joystick turn input")
        plt.plot(self.t_a[begin:-end],[self.beta*x for x in self.u_a[1][begin:-end]])

        plt.subplot(413)
        plt.title("IMU input")

        plt.plot(self.t_a[begin:-end],self.y_a[0][begin:-end])

        plt.subplot(414)
        plt.title("gyro input")
        plt.plot(self.t_a[begin:-end],self.y_a[1][begin:-end])

        plt.figure(4)

        plt.subplot(211)
        plt.title("Ratio 00")
        plt.plot(self.t_a[begin:-end],self.r_a[0][0][begin:-end])

        plt.subplot(212)
        plt.title("Ratio 11")
        plt.plot(self.t_a[begin:-end],self.r_a[1][1][begin:-end])

        plt.show()

    def export_all(self, start=0, finish=np.inf,pre="",post=""):
        begin,end = self.find_slice(start,finish)
        self.set_zero_time(begin)
        self.psi_fix()

        np.savetxt("plots/{}_u0_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], [self.alpha*x for x in self.u_a[0][begin:-end]]]) ,header='t u0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_u1_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], [self.beta*x for x in self.u_a[1][begin:-end]]]) ,header='t u1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_y0_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], self.y_a[0][begin:-end]]) ,header='t y0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_y1_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], self.y_a[1][begin:-end]]) ,header='t y1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x0_{}_{}.csv".format(pre,self.window,post), np.transpose([self.t_a[begin:-end],self.x_a[0][begin:-end]]) ,header='t x0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x1_{}_{}.csv".format(pre,self.window,post), np.transpose([self.t_a[begin:-end],self.x_a[1][begin:-end]]) ,header='t x1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x2_{}_{}.csv".format(pre,self.window,post), np.transpose([self.t_a[begin:-end],self.x_a[2][begin:-end]]) ,header='t x2', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x3_{}_{}.csv".format(pre,self.window,post), np.transpose([self.t_a[begin:-end],self.x_a[3][begin:-end]]) ,header='t x3', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_r0_{}_{}.csv".format(pre,self.window,post), np.transpose([self.t_a[begin:-end],self.r_a[0][begin:-end]]) ,header='t r0', comments='# ',delimiter=' ', newline='\n')
