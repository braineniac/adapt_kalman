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
from adaptive_kalman_filter import AdaptiveKalmanFilter
from bag_reader import BagReader


class StateEstimator(object):

    def __init__(self):
        self.states = []
        self.input = []
        self.output = []
        self.time = []

    def get_states(self):
        raise NotImplementedError


class KalmanStateEstimator(StateEstimator):
    def __init__(self, bag_path=None, kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter) or bag_path is None:
            raise AttributeError
        else:
            super(KalmanStateEstimator, self).__init__()
            self.bag_path = bag_path
            self.kalman_filter = kalman_filter

    def get_states(self):
        if len(self.time) != len(self.input) or len(self.input) != len(self.output):
            raise ValueError
        elif len(self.states) == len(self.time):
            return self.states
        else:
            for t, u, y in zip(self.time, self.input, self.output):
                self.kalman_filter.filter_iter((t, u, y))
                self.states.append((t, self.kalman_filter.get_post_states()))
            return self.states

    def set_input_bag_twist(self, topic=None):
        if topic is None or mask is None:
            raise ValueError
        else:
            bagreader = BagReader(self.bag_path)
            t, bagreader.read_twist(topic)

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
