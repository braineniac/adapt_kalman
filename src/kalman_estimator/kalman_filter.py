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

    def __init__(self,
                 Q_k=np.zeros((2, 2)), R_k=np.zeros((2, 2)),
                 alpha=1, beta=1,
                 mass=1,
                 micro_theta=1, micro_eta=1,
                 x0=(0, 0, 0, 0, 0, 0, 0)):
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise ValueError("Alpha is a number!")
        if not isinstance(beta, float) and not isinstance(beta, int):
            raise ValueError("Beta is a number!")
        if not isinstance(mass, float) and not isinstance(mass, int):
            raise ValueError("Mass is a number!")
        if not isinstance(micro_theta, float) and not isinstance(micro_theta, int):
            raise ValueError("micro_theta is a number!")
        if not isinstance(micro_eta, float) and not isinstance(micro_eta, int):
            raise ValueError("micro_eta is a number!")
        if not isinstance(Q_k, np.ndarray) or not isinstance(R_k, np.ndarray):
            raise ValueError("Q_k or R_k is not a numpy array!")
        if np.count_nonzero(Q_k) < 2 or np.count_nonzero(R_k) < 2:
            raise ValueError("Q_k or R_k covariance underdefined!")
        if np.array(x0).shape != (7, ):
            raise ValueError("Incorrect shape for x0!")
        self._alpha = alpha
        self._beta = beta
        self._mass = mass
        self._micro_theta = micro_theta
        self._micro_eta = micro_theta
        self._R_k = R_k  # Observation Covariance Matrix
        self._Q_k = Q_k  # Process Covariance Matrix
        self._x0 = x0  # Initial State Vector

        self._u_k = np.zeros((2, 1))  # Input Vector
        self._y_k = np.zeros((2, 1))  # Measurement Vector
        self._L_k = np.zeros((7, 2))  # Kalman Gain Matrix

        self._x_k_pre = x0  # A Priori state vector
        self._x_k_post = np.zeros((7, 1))  # A Posteriori state vector
        self._x_k_extr = np.zeros((7, 1))  # Extrapolated state vector

        # A Priori Parameter Covariance Matrix
        self._P_k_pre = np.zeros((7, 7))
        # A Posteriori Parameter Covariance Matrix
        self._P_k_post = np.zeros((7, 7))
        # Extrapolated Parameter Covariance Matrix
        self._P_k_extr = np.zeros((7, 7))

        self._Phi_k = np.zeros((7, 7))  # Dynamic Coefficient Matrix
        self._Gamma_k = np.zeros((7, 2))  # Input Coupling Matrix
        self._Gamma_k[3][0] = self._alpha / self._mass
        self._Gamma_k[6][1] = self._beta / self._mass
        self._G_k = np.zeros((7, 2))  # Process Noise Input Coupling Matrix
        self._G_k[3][0] = self._alpha / self._mass
        self._G_k[6][1] = self._beta / self._mass

        self._C_k = np.zeros((2, 7))  # Measurement Sensitivity Matrix
        self._C_k[0][3] = 1
        self._C_k[1][5] = 1
        self._D_k = np.zeros((2, 7))  # Output Coupling Matrix
        self._H_k = np.zeros((2, 7))  # Process Noise Output Coupling Matrix

        self._dt = 0
        self._t = 0

    def filter_iter(self, tuy=(None, None, None)):
        if not isinstance(tuy, tuple) and not isinstance(tuy, list):
            raise ValueError("Iteration input is not a list or a tuple!")
        if not any(tuy):
            raise ValueError("Iteration input contains an empty element!")
        t, u, y = tuy
        self._dt = t - self._t
        self._t = t

        self._u_k[0] = u[0]
        self._u_k[1] = u[1]
        self._y_k[0] = y[0]
        self._y_k[1] = y[1]

        # execute iteration steps
        self._update_Phi_k()
        self._set_gain()
        self._update_states()
        self._update_error_covars()
        self._extr_states()
        self._extr_error_covars()
        self._setup_next_iter()

    def get_post_states(self):
        return self._x_k_post

    def get_Q(self):
        return tuple(self._Q_k)

    def _update_Phi_k(self):
        self._Phi_k = np.array([
            [1, 0,
             self._dt * float(np.cos(self._x_k_post[3])),
             0.5 * self._dt * self._dt * float(np.cos(self._x_k_post[3])),
             0, 0, 0],
            [1, 0,
             self._dt * float(np.sin(self._x_k_post[3])),
             0.5 * self._dt * self._dt * float(np.sin(self._x_k_post[3])),
             0, 0, 0],
            [0, 0, 1, self._dt, 0, 0, 0],
            [0, 0, - self._micro_theta / self._mass, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, self._dt, 0.5 * self._dt * self._dt],
            [0, 0, 0, 0, 0, 1, self._dt],
            [0, 0, 0, 0, 0, - self._micro_eta / self._mass, 0]
        ])

    def _set_gain(self):
        self._L_k = self._P_k_pre.dot(self._C_k.T).dot(np.linalg.inv(
            self._C_k.dot(self._P_k_pre).dot(self._C_k.T)
            + self._H_k.dot(self._Q_k).dot(self._H_k.T)
            + self._R_k
        ))

    def _update_states(self):
        self._x_k_post = self._x_k_pre + self._L_k.dot(
            self._y_k - self._C_k.dot(self._x_k_pre) - self._D_k.dot(self._u_k)
        )

    def _update_error_covars(self):
        self._P_k_post = (np.identity(7) - self._L_k.dot(self._C_k)).dot(self._P_k_pre)

    def _extr_states(self):
        self._X_k_extr = self._Phi_k.dot(self._x_k_post) + self._Gamma_k.dot(self._u_k)

    def _extr_error_covars(self):
        self._P_k_extr = self._Phi_k.dot(self._P_k_post).dot(self._Phi_k.T) \
            + self._G_k.dot(self._Q_k).dot(self._G_k.T)

    def _setup_next_iter(self):
        self._x_k_pre = self._x_k_extr
        self._P_k_pre = self._P_k_extr


class AdaptiveKalmanFilter(KalmanFilter):

    def __init__(self,
                 Q_k=np.zeros((2, 2)), R_k=np.zeros((2, 2)),
                 alpha=1, beta=1,
                 mass=1,
                 micro_theta=1, micro_eta=1,
                 window=None, M_k=np.zeros((2, 2)),
                 x0=(0, 0, 0, 0, 0, 0, 0)):
        if not isinstance(window, MovingWeightedWindow):
            raise ValueError("Window is not a MovingWeightedWindow object!")
        if np.count_nonzero(M_k) < 2:
            raise ValueError("M_k underdefined!")
        super(AdaptiveKalmanFilter, self).__init__(
            Q_k=Q_k, R_k=R_k,
            alpha=alpha, beta=beta,
            mass=mass,
            micro_theta=micro_theta, micro_eta=micro_eta,
            x0=x0)
        self._window = window
        self._M_k = M_k
        self._Lambda_k = np.identity(2)
        self._Ro_k = self._Q_k.dot(np.linalg.inv(self._R_k))
        self._du_buffer = [
            deque([], self._window.get_size()),
            deque([], self._window.get_size())]
        self._last_u = (0, 0)

    def filter_iter(self, tuy=(None, None, None)):
        if not isinstance(tuy, tuple) and not isinstance(tuy, list):
            raise ValueError("Iteration input is not a list or a tuple!")
        if not any(tuy):
            raise ValueError("Iteration input contains an empty element!")
        t, u, y = tuy
        self._du_buffer[0].append(abs(u[0] - self._last_u[0]))
        self._du_buffer[1].append(abs(u[1] - self._last_u[1]))
        self._last_u = u
        self._adapt_covariance()
        super(AdaptiveKalmanFilter, self).filter_iter(tuy)

    def _adapt_covariance(self):
        if len(self._du_buffer[0]) >= self._window.get_size():
            if np.max(self._du_buffer[0]) != 0.0:
                self._Lambda_k[0][0] = 1 \
                    + self._M_k[0][0] / np.max(self._du_buffer[0]) \
                    * self._window.get_weighted_sum(self._du_buffer[0])
            if np.max(self._du_buffer[1]) != 0.0:
                self._Lambda_k[1][1] = 1 \
                    + self._M_k[1][1] / np.max(self._du_buffer[1]) \
                    * self._window.get_weighted_sum(self._du_buffer[0])
        self._Q_k = self._Lambda_k.dot(self._Ro_k).dot(self._R_k)
