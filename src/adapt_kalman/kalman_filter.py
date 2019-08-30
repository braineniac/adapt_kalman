#!/usr/bin/env python

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

    def __init__(self, Q_k=None, R_k=None, alpha=1., beta=1., x0=(0, 0, 0, 0, 0)):
        if np.count_nonzero(Q_k) <= 1 or np.count_nonzero(R_k) <= 1:
            raise ValueError
        else:
            self._alpha = alpha
            self._beta = beta

            self._R_k = R_k  # Observation Covariance Matrix
            self._Q_k = Q_k  # Process Covariance Matrix

            self._u_k = np.zeros((2, 1))  # Input Vector
            self._y_k = np.zeros((2, 1))  # Measurement Vector
            self._L_k = np.zeros((5, 2))  # Kalman Gain Matrix

            x0_rad = list(x0)
            x0_rad[3] = np.radians(x0[3])
            x0 = np.array(x0_rad).reshape((5, 1))  # Initial State Vector
            self._X_k_pre = x0  # A Priori state vector
            self._X_k_post = np.zeros((5, 1))  # A Posteriori state vector
            self._X_k_extr = np.zeros((5, 1))  # Extrapolated state vector

            self._P_k_pre = np.zeros((5, 5))  # A Priori Parameter Covariance Matrix
            self._P_k_post = np.zeros((5, 5))  # A Posteriori Parameter Covariance Matrix
            self._P_k_extr = np.zeros((5, 5))  # Extrapolated Parameter Covariance Matrix

            self._Phi_k = np.zeros((5, 5))  # Dynamic Coefficient Matrix
            self._Gamma_k = np.zeros((5, 2))  # Input Coupling Matrix
            self._G_k = np.zeros((5, 2))  # Process Noise Input Coupling Matrix
            self._G_k[2][0] = self._alpha
            self._G_k[4][1] = self._beta

            self._C_k = np.zeros((2, 5))  # Measurement Sensitivity Matrix
            self._D_k = np.zeros((2, 2))  # Output Coupling Matrix
            self._H_k = np.zeros((2, 2))  # Process Noise Output Coupling Matrix

            self._dt = 0
            self._t = 0

    def filter_iter(self, tuy=(None, None, None)):
        t, u, y = tuy
        if not t and not u and not y:
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
        return self._X_k_post

    def get_Q(self):
        return tuple(self._Q_k)

    def _update_matrices(self):
        if self._dt:
            self._Phi_k = np.array([
                [1, 0, self._dt * np.cos(self._X_k_post[3]), 0, 0],
                [0, 1, self._dt * np.sin(self._X_k_post[3]), 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, self._dt],
                [0, 0, 0, 0, 0]
            ])
            self._Gamma_k = np.array([
                [0, 0],
                [0, 0],
                [self._alpha, 0],
                [0, 0],
                [0, self._beta]
            ])
            self._C_k = np.array([
                [0, 0, -1 / self._dt, 0, 0],
                [0, 0, 0, 0, 1]
            ])
            self._D_k = np.array([
                [self._alpha / self._dt, 0],
                [0, 0]
            ])

    def _set_gain(self):
        E = self._C_k.dot(self._P_k_pre).dot(self._C_k.T) + self._H_k.dot(self._Q_k).dot(self._H_k.T) + self._R_k
        self._L_k = self._P_k_pre.dot(self._C_k.T).dot(np.linalg.inv(E))

    def _update_states(self):
        F = self._y_k - self._C_k.dot(self._X_k_pre) - self._D_k.dot(self._u_k)
        self._X_k_post = self._X_k_pre + self._L_k.dot(F)

    def _update_error_covars(self):
        self._P_k_post = (np.identity(5) - self._L_k.dot(self._C_k)).dot(self._P_k_pre)

    def _extr_states(self):
        self._X_k_extr = self._Phi_k.dot(self._X_k_post) + self._Gamma_k.dot(self._u_k)

    def _extr_error_covars(self):
        self._P_k_extr = self._Phi_k.dot(self._P_k_post).dot(self._Phi_k.T) + self._G_k.dot(self._Q_k).dot(self._G_k.T)

    def _setup_next_iter(self):
        self._X_k_pre = self._X_k_extr
        self._P_k_pre = self._P_k_extr


class AdaptiveKalmanFilter(KalmanFilter):

    def __init__(self, Q_k=None, R_k=None, alpha=1.0, beta=1.0, window=None, M_k=None, x0=(0, 0, 0, 0, 0)):
        if not isinstance(window, MovingWeightedWindow) or np.count_nonzero(M_k) < 2:
            raise ValueError
        else:
            super(AdaptiveKalmanFilter, self).__init__(alpha=alpha, beta=beta, Q_k=Q_k, R_k=R_k, x0=x0)
            self._window = window
            self._M_k = M_k
            self._Lambda_k = np.identity(2)
            self._Ro_k = self._Q_k.dot(np.linalg.inv(self._R_k))
            self._du_buffer = [deque([], self._window.get_size()),deque([], self._window.get_size())]
            self._last_u = (0, 0)

    def filter_iter(self, tuy=(None, None, None)):
        t, u, y = tuy
        self._du_buffer[0].append(abs(u[0] - self._last_u[0]))
        self._du_buffer[1].append(abs(u[1] - self._last_u[1]))
        self._last_u = u
        self._adapt_covariance()
        super(AdaptiveKalmanFilter, self).filter_iter(tuy)

    @staticmethod
    def sum_du_w(du_buffer=None, w_k=None):
        sum = 0
        du_buffer = np.array(du_buffer)[::-1]
        for du, Q in zip(du_buffer, w_k):
            sum += du * Q
        return sum

    def _adapt_covariance(self):
        if len(self._du_buffer[0]) >= self._window.get_size():
            ones = np.ones(self._window.get_size())
            self._window.set_window(ones)
            w_k = self._window.get_window()
            self._Lambda_k = np.identity(2)
            if np.max(self._du_buffer[0]) != 0.0:
                self._Lambda_k[0][0] = 1 + self._M_k[0][0] / np.max(self._du_buffer[0]) * \
                                       self.sum_du_w(self._du_buffer[0], w_k)
            if np.max(self._du_buffer[1]) != 0.0:
                self._Lambda_k[1][1] = 1 + self._M_k[1][1] / np.max(self._du_buffer[1]) * \
                                       self.sum_du_w(self._du_buffer[1], w_k)
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
