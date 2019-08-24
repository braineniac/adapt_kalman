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
from matplotlib import pyplot as plt


class Thesis(object):
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

        self.figure = 1

    def add_figure(self):
        plt.figure(self.figure)
        self.figure += 1


class AlphaTuner(Thesis):
    def __init__(self):
        super(AlphaTuner, self).__init__()
        self.alpha = [0.8, 0.9, 1, 1.1, 1.2]
        self.bag = "/home/dan/ws/rosbag/garry3/5m_slow.bag"
        self.imu_topic = "/imu"
        self.twist_topic = "/fake_encoder/twist"
        self.slice = [0, 35]

        self.bag_reader = BagReader(self.bag)
        self.bag_system_io = BagSystemIO()
        self.state_plots = []

    def get_sys_input(self):
        return self.bag_system_io.get_input(self.bag_reader.read_twist(self.twist_topic))

    def get_sys_output(self):
        return self.bag_system_io.get_output(self.bag_reader.read_imu(self.imu_topic))

    def set_state_plots(self):
        for alpha in self.alpha:
            kalman_filter = KalmanFilter(
                alpha=alpha,
                beta=self.beta,
                Q_k=self.Q_k,
                R_k=self.R_k
            )
            state_estimator = KalmanStateEstimator(kalman_filter)
            state_estimator.set_stamped_input(self.get_sys_input())
            state_estimator.set_stamped_output(self.get_sys_output())
            plot_handler = StatePlotHandler(state_estimator)
            plot_handler.set_slice_times(self.slice[0], self.slice[1])
            self.state_plots.append(plot_handler)

    def plot(self):
        options = ["b", "k", "r", "m", "g"]
        legends = [str(x) for x in self.alpha]
        self.plot_input_figure(options, legends)
        self.plot_output_figure()
        self.plot_states_figure(options, legends)
        plt.show()

    @staticmethod
    def add_plot(stamped_plot=None, dimension=None, option=None, legend=None):
        if stamped_plot is None or dimension is None:
            raise ValueError
        else:
            t, plot = stamped_plot
            if option is None or legend is None:
                plt.plot(t, plot[dimension])
            else:
                plt.plot(t, plot[dimension], option, label=legend)
                plt.legend()

    def plot_input_figure(self, options=None, legends=None):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_input_titles())
        for i in range(num_titles):
            plt.subplot(num_titles*100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_input_titles()[i])
            plt.xlabel("Time [s]")
            for state_plot, option, legend in zip(self.state_plots, options, legends):
                self.add_plot(state_plot.get_input_plot(), i, option, legend)

    def plot_output_figure(self):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_output_titles())
        for i in range(num_titles):
            plt.subplot(num_titles*100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_output_titles()[i])
            plt.xlabel("Time [s]")
            self.add_plot(self.state_plots[0].get_output_plot(), i)

    def plot_states_figure(self, options=None, legends=None):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_states_titles())
        for i in range(num_titles):
            plt.subplot(num_titles*100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_states_titles()[i])
            plt.xlabel("Time [s]")
            for state_plot, option, legend in zip(self.state_plots, options, legends):
                self.add_plot(state_plot.get_states_plot(), i, option, legend)

    def export(self, pre="alphas_"):
        for state_plot in self.state_plots:
            state_plot.export_input(pre)
            state_plot.export_output(pre)
            state_plot.export_states(pre)

if __name__ == '__main__':
    alphas = AlphaTuner()
    alphas.set_state_plots()
    alphas.plot()
    alphas.export()
