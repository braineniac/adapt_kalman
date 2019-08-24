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

from kalman_filter import KalmanFilter, AdaptiveKalmanFilter
from moving_weighted_window import MovingWeightedExpWindow, MovingWeightedSigWindow
from state_estimator import StateEstimator, KalmanStateEstimator, StatePlotHandler
from bag_system_io import BagSystemIO
from bag_reader import BagReader
from system_io_simulator import LineSimulator, OctagonSimulator
from bag_generator import EKFGenerator, IMUTransformGenerator
from shell import *

import numpy as np


class ThesisExporter(object):
    def __init__(self):
        self.R_k = np.zeros((2, 2))
        self.R_k[0][0] = 0.04
        self.R_k[1][1] = 0.02

        self.Q_k = np.zeros((2, 2))
        self.Q_k[0][0] = self.R_k[0][0] * 0.001
        self.Q_k[1][1] = self.R_k[1][1] * 1

        self.alpha = 1
        self.beta = 1

        self.slice_times = (0, np.inf)
        self.bag = None


class AlphaTuner(ThesisExporter):
    def __init__(self):
        super(AlphaTuner, self).__init__()
        self.alpha = [0.8, 0.9, 1, 1.1, 1.2]
        self.bag = "/home/dan/ws/rosbag/garry3/5m_slow.bag"

    def set_kalman(self):
        reader = BagReader(self.bag)
        bag_system_io = BagSystemIO()
        imu_raw = reader.read_imu("/imu")
        twist_raw = reader.read_twist("/fake_encoder/twist")
        sys_output = bag_system_io.get_output(imu_raw)
        sys_input = bag_system_io.get_input(twist_raw)
        plot_list = []
        for alpha in self.alpha:
            kalman_filter = KalmanFilter(
                alpha=alpha,
                beta=self.beta,
                Q_k=self.Q_k,
                R_k=self.R_k
            )
            state_estimator = KalmanStateEstimator(kalman_filter)
            state_estimator.set_stamped_output(sys_output)
            state_estimator.set_stamped_input(sys_input)

            plot_handler = StatePlotHandler(state_estimator)
            plot_list.append(plot_handler)
        for plot_handler in plot_list:
            plot_handler.get_input()
            plot_handler.get_output()
            plot_handler.get_states()


class ThesisDataExporter(object):
    _R_k = np.zeros((2, 2))
    _R_k[0][0] = 0.04
    _R_k[1][1] = 0.02

    def __init__(self):
        self.exp_window = MovingWeightedExpWindow(5)
        self.sig_window = MovingWeightedSigWindow(5)

    def export_alphas(self):
        Q_k = np.zeros((2, 2))
        Q_k[0][0] = self._R_k[0][0] * 0.01
        Q_k[1][1] = self._R_k[1][1] * 1
        kalman_filter_list = []
        kalman_state_estimator_list = []
        plot_handler_list = []
        beta = 1
        alphas = [1, 1.1, 1.2, 0.9, 0.8]
        for alpha in alphas:
            kalman_filter_list.append(KalmanFilter(Q_k, self._R_k, alpha, beta))
        for kalman_filter in kalman_filter_list:
            kalman_state_estimator = KalmanStateEstimator(kalman_filter)

            kalman_state_estimator_list.append()
        for kalman_state_estimator in kalman_state_estimator_list:
            plot_handler_list.append(StatePlotHandler(kalman_state_estimator))
        for plot_handler in plot_handler_list:
            plot_handler.get_states()
            plot_handler.get_input()
            plot_handler.get_output()

    def export_betas(self):
        pass

    def export_line_sim(self):
        pass

    def export_octagon_sim(self):
        pass

    def export_sig(self):
        pass

    def export_exp(self):
        pass


if __name__ == '__main__':
    export_alphas = AlphaTuner()
    export_alphas.set_kalman()
