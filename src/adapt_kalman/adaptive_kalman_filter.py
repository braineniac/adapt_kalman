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
from collections import deque

from kalman_filter import KalmanFilter
from moving_weighted_window import MovingWeightedWindow

class AdaptiveKalmanFilter(KalmanFilter):

    def __init__(self,Q_k=None,R_k=None,alpha=1.0,beta=1.0, window=None, M_k=None,x0=[0,0,0,0,0]):
        if not isinstance(window,MovingWeightedWindow) or np.count_nonzero(M_k)<2:
            raise AttributeError
        else:
            super(AdaptiveKalmanFilter,self).__init__(alpha=alpha,beta=beta,Q_k=Q_k,R_k=R_k,x0=x0)
            self._window = window
            self._M_k = M_k
            self._Lambda_k = np.identity(2)
            self._Ro_k = self._Q_k.dot(np.linalg.inv(self._R_k))
            self._u_buffer = deque([],self._window.get_size())

    def filter_iter(self, tuy=(None,None,None)):
        t,u,y = tuy
        self._u_buffer.append(u)
        self._adapt_covar()
        super(AdaptiveKalmanFilter,self).filter_iter(tuy)

    def _adapt_covar(self):
        if len(self._u_buffer) >= self._window.get_size():
            w_k = self._get_windowed_input()
            self._Lambda_k = np.identity(2)
            if w_k[0] != 0.0:
                self._Lambda_k[0][0] = self._M_k[0][0] * abs(self._u_buffer[0][-1]) / abs(w_k[0])
            if w_k[1] != 0.0:
                self._Lambda_k[1][1] = self._M_k[1][1] * abs(self._u_buffer[1][-1]) / abs(w_k[1])
        self._Q_k = self._Lambda_k.dot(self._Ro_k).dot(self._R_k)

    def _get_windowed_input(self):
        u_win = self._sort_input()
        w_k = []
        for j in range(len(u_win)):
            w_k.append([])
            self._window.set_window(u_win[j])
            w_k[j] = self._window.get_avg()
        return w_k

    def _sort_input(self):
        u_win = []
        for u in self._u_buffer:
            for i in range(len(u)):
                try:
                    u_win[i].append(u[i])
                except IndexError:
                    u_win.append([])
                    u_win[i].append(u[i])
        return u_win
