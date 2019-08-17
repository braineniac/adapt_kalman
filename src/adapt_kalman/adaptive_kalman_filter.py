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

from kalman_filter import KalmanFilter

class AdaptiveKalmanFilter(KalmanFilter):

    def __init__(self,alpha=1.0,beta=1.0,Q_k=None,R_k=None,window="sig",ws=5, o1=3, o2=1,x0=[0,0,0,0,0]):
        super(AdaptiveKalmanFilter,self).__init__(alpha=alpha,beta=beta,Q_k=Q_k,R_k=R_k,x0=x0)

        self._N = ws
        self._u_window = np.zeros((2,1,ws))

        self.M_k = np.zeros((2,2))
        self.Ro_k = np.zeros((2,2))
        self.Lambda_k = np.zeros((2,2))
        self.window = window
        self.M_k[0][0] = o1
        self.M_k[1][1] = o2
        self.Ro_k[0][0] = r1*r1
        self.Ro_k[1][1] = r2*r2
        self.window_list = ["sig", "exp"]

        self.plot_u = [[],[]]
        self.plot_y = [[],[]]
        self.plot_x = [[],[],[],[],[]]
        self.plot_xy = []
        self.plot_r = [[],[]]
        self.r_a = [[],[]]

    def filter_step(self, u=None, y=None,t=None):
        if u and y and t:
            self.adapt_covar()
            self.filter_iter(u,y,t)

    def adapt_covar(self):
        if len(self.u_a[0]) >= 2*self.N and len(self.u_a[1]) >= 2*self.N:
            w_k = self.set_windows()
            self.Lambda_k = np.identity(2)
            if w_k[0] != 0.0:
                self.Lambda_k[0][0] = self.M_k[0][0] * abs(self.u_a[0][-1]) / abs(w_k[0])
            if w_k[1] != 0.0:
                self.Lambda_k[1][1] = self.M_k[1][1] * abs(self.u_a[1][-1]) / abs(w_k[1])

        self.Q_k = self.Lambda_k.dot(self.Ro_k).dot(self.R_k)
        self.r_a[0].append(self.Lambda_k.dot(self.Ro_k)[0][0])
        self.r_a[1].append(self.Lambda_k.dot(self.Ro_k)[1][1])

    def set_windows(self):
        w_avg = np.zeros((2,1))
        w_avg[0] = self.get_window_avg(self.u_a[0], self.N)
        w_avg[1] = self.get_window_avg(self.u_a[1], self.N)
        return w_avg


    # def find_slice(self, start=0.0, finish=np.Inf):
    #     begin = 0
    #     end = -1
    #     for elem in self.t_a:
    #         if elem <= finish:
    #             end = end + 1
    #         if elem <= start:
    #             begin = begin + 1
    #     end = len(self.t_a) - end
    #     return begin,end

    # def set_zero_time(self, begin):
    #     new_t_array = []
    #     for elem in self.t_a:
    #         new_t_array.append(elem - self.t_a[begin])
    #     self.t_a = new_t_array
    #
    # def psi_fix(self):
    #     psi_a = []
    #     for psi in self.x_a[3]:
    #         k = abs(int(psi / (2*np.pi)))
    #         if psi > 2*np.pi:
    #             psi -= 2*np.pi*k
    #         elif psi < -2*np.pi*k:
    #             psi += 2*np.pi*(k+1)
    #         psi_a.append(psi*180/np.pi)
    #     self.x_a[3] = psi_a

    # def add_plots(self, start=0, finish=np.inf):
    #     begin,end = self.find_slice(start,finish)
    #     #self.set_zero_time(begin)
    #     #print(begin,end)
    #     self.psi_fix()
    #
    #     self.plot_x[0] = (self.t_a[begin:-end],self.x_a[0][begin:-end])
    #     self.plot_x[1] = (self.t_a[begin:-end], self.x_a[1][begin:-end])
    #     self.plot_x[2] = (self.t_a[begin:-end], self.x_a[2][begin:-end])
    #     self.plot_x[3] = (self.t_a[begin:-end], self.x_a[3][begin:-end])
    #     self.plot_x[4] = (self.t_a[begin:-end],self.x_a[4][begin:-end])
    #     self.plot_xy = (self.x_a[0][begin:-end],self.x_a[1][begin:-end])
    #     self.plot_u[0] = (self.t_a[begin:-end],[self.alpha*x for x in self.u_a[0][begin:-end]])
    #     self.plot_u[1] = (self.t_a[begin:-end],[self.beta*x for x in self.u_a[1][begin:-end]])
    #     self.plot_y[0] = (self.t_a[begin:-end],self.y_a[0][begin:-end])
    #     self.plot_y[1] = (self.t_a[begin:-end],self.y_a[1][begin:-end])
    #     self.plot_r[0] = (self.t_a[begin:-end],self.r_a[0][begin:-end])
    #     self.plot_r[1] = (self.t_a[begin:-end],self.r_a[1][begin:-end])
