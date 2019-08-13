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

    def __init__(self,alpha=1.0,beta=1.0,r1=1/3.,r2=1.0,window="sig",ws=5, o1=3, o2=1,x0=[0,0,0,0]):
        super(AdaptKalman,self).__init__(r1=r1, r2=r2, alpha=alpha,beta=beta,x0=x0)
        self.u = [[],[]]                    # all control input vectors
        self.y = [[],[]]                    # all measuremnt vectors
        self.N = ws
        self.M_k = np.zeros((2,2))
        self.Ro_k = np.zeros((2,2))
        self.Lambda_k = np.zeros((2,2))
        self.window = window
        self.M_k[0][0] = o1
        self.M_k[1][1] = o2
        self.Ro_k[0][0] = r1
        self.Ro_k[1][1] = r2
        self.window_list = ["sig", "exp"]

        self.plot_u = [[],[]]
        self.plot_y = [[],[]]
        self.plot_x = [[],[],[],[]]
        self.plot_xy = []
        self.plot_r = [[],[]]
        self.r_a = [[],[]]

    def filter_step(self, u=None, y=None,t=None):
        if u and t:
            self.adapt_covar()
            self.filter_iter(u,y,t)

    def adapt_covar(self):
        if len(self.u_a[0]) >= 2*self.N and len(self.u_a[1]) >= 2*self.N:
            w_k = self.set_windows()
            self.Lambda_k = np.identity(2)
            if w_k[0] != 0.0:
                self.Lambda_k[0][0] = abs(self.u_a[0][-1]) / abs(w_k[0])
            if w_k[1] != 0.0:
                self.Lambda_k[1][1] = abs(self.u_a[1][-1]) / abs(w_k[1])
        print(self.M_k)
        print(self.Lambda_k)
        print(self.Ro_k)
        self.Q_k = self.M_k.dot(self.Lambda_k).dot(self.Ro_k).dot(self.R_k)
        self.r_a[0].append(self.M_k.dot(self.Lambda_k).dot(self.Ro_k)[0][0])
        self.r_a[1].append(self.M_k.dot(self.Lambda_k).dot(self.Ro_k)[1][1])

    def set_windows(self):
        w_avg = np.zeros((2,1))
        w_avg[0] = self.get_window_avg(self.u_a[0], self.N)
        w_avg[1] = self.get_window_avg(self.u_a[1], self.N)
        return w_avg

    def get_window_avg(self, array, N):
        window = []
        avg = 0
        for i in range(0,N):
            window.append(array[-i-1])
        if self.window == "exp":
            avg = self.exp_avg(window)
        elif self.window == "sig":
            avg = self.sig_avg(window)
        return avg

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

    def add_plots(self, start=0, finish=np.inf):
        begin,end = self.find_slice(start,finish)
        self.set_zero_time(begin)
        self.psi_fix()

        self.plot_x[0] = (self.t_a[begin:-end],self.x_a[0][begin:-end])
        self.plot_x[1] = (self.t_a[begin:-end], self.x_a[1][begin:-end])
        self.plot_x[2] = (self.t_a[begin:-end], self.x_a[2][begin:-end])
        self.plot_x[3] = (self.t_a[begin:-end],self.x_a[3][begin:-end])
        self.plot_xy = (self.x_a[0][begin:-end],self.x_a[1][begin:-end])
        self.plot_u[0] = (self.t_a[begin:-end],[self.alpha*x for x in self.u_a[0][begin:-end]])
        self.plot_u[1] = (self.t_a[begin:-end],[self.beta*x for x in self.u_a[1][begin:-end]])
        self.plot_y[0] = (self.t_a[begin:-end],self.y_a[0][begin:-end])
        self.plot_y[1] = (self.t_a[begin:-end],self.y_a[1][begin:-end])
        self.plot_r[0] = (self.t_a[begin:-end],self.r_a[0][0][begin:-end])
        self.plot_r[1] = (self.t_a[begin:-end],self.r_a[1][1][begin:-end])

    def plot_all(self, start=0.0,finish=np.Inf):
        begin,end = self.find_slice(start,finish)
        self.set_zero_time(begin)
        self.psi_fix()

        plt.figure(1)
        plt.subplot(411)
        plt.title("Robot distance in x")
        plt.ylabel("Distance in m")
        plt.xlabel("Time in s")
        self.plot_x[0] = plt.plot(self.t_a[begin:-end],self.x_a[0][begin:-end])

        plt.subplot(412)
        plt.title("Robot distance in y")
        plt.ylabel("Distance in m")
        plt.xlabel("Time in s")
        self.plot_x[1] = plt.plot(self.t_a[begin:-end], self.x_a[1][begin:-end])

        plt.subplot(413)
        plt.title("Robot velocity")
        plt.ylabel("Velocity in m/s")
        plt.xlabel("Time in s")
        self.plot_x[2] = plt.plot(self.t_a[begin:-end], self.x_a[2][begin:-end])

        plt.subplot(414)
        plt.title("phi")
        plt.ylabel("Phi in degrees")
        plt.xlabel("Time in s")
        self.plot_x[3] = plt.plot(self.t_a[begin:-end],self.x_a[3][begin:-end])

        plt.figure(2)
        plt.title("xy")
        plt.xlabel("x distance")
        plt.ylabel("y distance")
        self.plot_xy = plt.plot(self.x_a[0][begin:-end],self.x_a[1][begin:-end])

        plt.figure(3)
        plt.subplot(411)
        plt.title("fake wheel encoder input")
        self.plot_u[0] = plt.plot(self.t_a[begin:-end],[self.alpha*x for x in self.u_a[0][begin:-end]])

        plt.subplot(412)
        plt.title("joystick turn input")
        self.plot_u[1] = plt.plot(self.t_a[begin:-end],[self.beta*x for x in self.u_a[1][begin:-end]])

        plt.subplot(413)
        plt.title("IMU input")

        self.plot_y[0] = plt.plot(self.t_a[begin:-end],self.y_a[0][begin:-end])

        plt.subplot(414)
        plt.title("gyro input")
        self.plot_y[1] = plt.plot(self.t_a[begin:-end],self.y_a[1][begin:-end])

        plt.figure(4)

        plt.subplot(211)
        plt.title("Ratio 00")
        self.plot_r[0] = plt.plot(self.t_a[begin:-end],self.r_a[0][begin:-end])

        plt.subplot(212)
        plt.title("Ratio 11")
        self.plot_r[1] = plt.plot(self.t_a[begin:-end],self.r_a[1][begin:-end])

        plt.show()

    def export_all(self, start=0, finish=np.inf,pre="",post=""):
        begin,end = self.find_slice(start,finish)
        self.set_zero_time(begin)
        self.psi_fix()

        np.savetxt("plots/{}_u0_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], [self.alpha*x for x in self.u_a[0][begin:-end]]]) ,header='t u0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_u1_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], [self.beta*x for x in self.u_a[1][begin:-end]]]) ,header='t u1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_y0_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], self.y_a[0][begin:-end]]) ,header='t y0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_y1_{}.csv".format(pre,post), np.transpose([self.t_a[begin:-end], self.y_a[1][begin:-end]]) ,header='t y1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x0_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.x_a[0][begin:-end]]) ,header='t x0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x1_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.x_a[1][begin:-end]]) ,header='t x1', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x2_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.x_a[2][begin:-end]]) ,header='t x2', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_x3_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.x_a[3][begin:-end]]) ,header='t x3', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_r0_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.r_a[0][begin:-end]]) ,header='t r0', comments='# ',delimiter=' ', newline='\n')

        np.savetxt("plots/{}_r1_{}_{}.csv".format(pre,self.window_type,post), np.transpose([self.t_a[begin:-end],self.r_a[1][begin:-end]]) ,header='t r1', comments='# ',delimiter=' ', newline='\n')
