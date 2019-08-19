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

from moving_weighted_window import MovingWeightedWindow

class KalmanFilter(object):

    def __init__(self, Q_k=None, R_k=None, alpha=1.,beta=1.,x0 = [0,0,0,0,0]):
        if np.count_nonzero(Q_k) <= 1 or np.count_nonzero(R_k) <= 1:
            raise AttributeError
        else:
            self._alpha = alpha
            self._beta = beta

            self._R_k = R_k                            # observation noise covaraiance matrix
            self._Q_k = Q_k                            # process noise covariance matrix

            self._u_k = np.zeros((2,1))                # control input vector
            self._y_k = np.zeros((2,1))                # observation vector
            self._L_k = np.zeros((5,2))                # Kalman gain matrix

            self._x_k_pre = np.zeros((5,1))            # A priori state vector
            self._x_k_post = np.zeros((5,1))           # A posteriori state vector
            self._x_k_extr = np.zeros((5,1))           # extrapolated state vector

            x0[3] = np.radians(x0[3])
            self._x_k_pre = np.array(x0).reshape((5,1))

            self._P_k_pre = np.zeros((5,5))            # A priori covariance matrix
            self._P_k_post = np.zeros((5,5))           # A posteriori covariance matrix
            self._P_k_extr = np.zeros((5,5))           # extrapolated covariance matrix

            self._phi_k = np.zeros((5,5))              # dynamics matrix
            self._gamma_k = np.zeros((5,2))            # control matrix
            self._G_k = np.zeros((5,2))                #
            self._G_k[2][0] = self._alpha
            self._G_k[4][1] = self._beta

            self._C_k = np.zeros((2,5))                # measurement matrix
            self._D_k = np.zeros((2,2))                # measurement input matrix
            self._H_k = np.zeros((2,2))                #

            self._dt = 0
            self._t = 0

    def filter_iter(self, tuy = (None,None,None)):
        u,y,t = tuy
        if t is None and u is None and y is None:
            raise ValueError
        else:
            self._dt = t - self._t
            self._t = t

            self._u_k[0] = u[0]
            self._u_k[1] = u[1]
            self._y_k[0] = y[0]
            self._y_k[1] = y[1]

            # execute iteration steps
            self._update_matrices()
            self._set_gain()
            self._update_states()
            self._update_error_covars()
            self._extr_states()
            self._extr_error_covars()
            self._setup_next_iter()

    def get_post_states(self):
        return tuple(self._x_k_post)

    def _update_matrices(self):
        if self._dt:
            self._phi_k = np.array([
                [1,0,self._dt*np.cos(self._x_k_post[3]),0,0],
                [0,1,self._dt*np.sin(self._x_k_post[3]),0,0],
                [0,0,0,0,0],
                [0,0,0,1,self._dt],
                [0,0,0,0,0]
            ])
            self._gamma_k = np.array([
                [0,0],
                [0,0],
                [self._alpha,0],
                [0,0],
                [0,self._beta]
            ])
            self._C_k = np.array([
                [0,0,-1/self._dt,0,0],
                [0,0,0,0,1]
            ])
            self._D_k = np.array([
                [self._alpha/self._dt,0],
                [0,0]
            ])

    def _set_gain(self):
        E = self._C_k.dot(self._P_k_pre).dot(self._C_k.T) + self._H_k.dot(self._Q_k).dot(self._H_k.T) + self._R_k
        self._L_k = self._P_k_pre.dot(self._C_k.T).dot(np.linalg.inv(E))

    def _update_states(self):
        F = self._y_k - self._C_k.dot(self._x_k_pre) - self._D_k.dot(self._u_k)
        self._x_k_post = self._x_k_pre + self._L_k.dot(F)

    def _update_error_covars(self):
        self._P_k_post = (np.identity(5) - self._L_k.dot(self._C_k)).dot(self._P_k_pre)

    def _extr_states(self):
        self._x_k_extr = self._phi_k.dot(self._x_k_post) + self._gamma_k.dot(self._u_k)

    def _extr_error_covars(self):
        self._P_k_extr = self._phi_k.dot(self._P_k_post).dot(self._phi_k.T) + self._G_k.dot(self._Q_k).dot(self._G_k.T)

    def _setup_next_iter(self):
        self._x_k_pre = self._x_k_extr
        self._P_k_pre = self._P_k_extr

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
