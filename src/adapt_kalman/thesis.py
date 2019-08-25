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
    figure = 1

    @staticmethod
    def add_figure():
        plt.figure(Thesis.figure)
        Thesis.figure += 1

    @staticmethod
    def get_sys_input(bag=None, topic=None):
        bag_system_io = BagSystemIO()
        bag_reader = BagReader(bag)
        sys_input = bag_system_io.get_input(bag_reader.read_twist(topic))
        return sys_input

    @staticmethod
    def get_sys_output(bag=None, topic=None):
        bag_system_io = BagSystemIO()
        bag_reader = BagReader(bag)
        sys_output = bag_system_io.get_output(bag_reader.read_imu(topic))
        return sys_output

    @staticmethod
    def get_sys_states(bag=None, topic=None):
        bag_system_io = BagSystemIO()
        bag_reader = BagReader(bag)
        sys_states = bag_system_io.get_states(bag_reader.read_odom(topic))
        return sys_states

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

    def __init__(self):
        self.R_k = np.zeros((2, 2))
        self.R_k[0][0] = 0.04
        self.R_k[1][1] = 0.02

        self.Q_k = np.zeros((2, 2))
        self.Q_k[0][0] = self.R_k[0][0] * 0.001
        self.Q_k[1][1] = self.R_k[1][1] * 1

        self.alpha = 1
        self.beta = 1

        self.slice = (0, np.inf)
        self.bag = None


class Tuner(Thesis):
    def __init__(self):
        super(Tuner, self).__init__()
        self.bag = []
        self.state_plots = []
        self.imu_topic = None
        self.twist_topic = None
        self.slice = None
        self.legends = []

    def get_kalman_filters(self):
        raise NotImplementedError

    def run(self):
        kalman_filters = self.get_kalman_filters()
        input = self.get_sys_input(self.bag, self.twist_topic)
        output = self.get_sys_output(self.bag, self.imu_topic)
        self.state_plots = self.get_kalman_state_plots(kalman_filters, input, output)

    def get_kalman_state_plots(self, kalman_filters=None, input=None, output=None):
        state_plots = []
        for kalman_filter in kalman_filters:
            state_estimator = KalmanStateEstimator(kalman_filter)
            state_estimator.set_stamped_input(input)
            state_estimator.set_stamped_output(output)
            plot_handler = StatePlotHandler(state_estimator)
            plot_handler.set_slice_times(self.slice[0], self.slice[1])
            state_plots.append(plot_handler)
        return state_plots

    def plot(self):
        options = ["b", "k", "r", "m", "g"]
        self.plot_input_figure(options, self.legends)
        self.plot_output_figure()
        self.plot_states_figure(options, self.legends)

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


class CompareTwo(Tuner):
    def __init__(self, bag_base=None, bag_compare=None, twist_topic=None, imu_topic=None, slice=None):
        super(Tuner, self).__init__()
        self.bag_base = bag_base
        self.bag_compare = bag_compare
        self.imu_topic = imu_topic
        self.twist_topic = twist_topic
        self.slice = slice
        self.legends = ["simple", "multiple"]

    def get_kalman_filters(self):
        kalman_filters = []
        kalman_filters.append(KalmanFilter(alpha=self.alpha, beta=self.beta, Q_k=self.Q_k, R_k=self.R_k))
        return kalman_filters

    def run(self):
        kalman_filter_base = self.get_kalman_filters()
        kalman_filter_compare = self.get_kalman_filters()
        input_base = self.get_sys_input(self.bag_base, self.twist_topic)
        input_compare = self.get_sys_input(self.bag_compare, self.twist_topic)
        output_base = self.get_sys_output(self.bag_base, self.imu_topic)
        output_compare = self.get_sys_output(self.bag_compare, self.imu_topic)
        state_plots_base = self.get_kalman_state_plots(kalman_filter_base, input_base, output_base)
        state_plots_compare = self.get_kalman_state_plots(kalman_filter_compare, input_compare, output_compare)
        self.state_plots = state_plots_base + state_plots_compare


class AlphaTuner(Tuner):
    def __init__(self, alphas=None, bag=None, twist_topic=None,imu_topic=None, slice=None):
        super(AlphaTuner, self).__init__()
        self.alpha = alphas
        self.bag = bag
        self.imu_topic = imu_topic
        self.twist_topic = twist_topic
        self.slice = slice
        self.legends = [str(x) for x in self.alpha]

    def get_kalman_filters(self):
        kalman_filters = []
        for alpha in self.alpha:
            kalman_filter = KalmanFilter(
                alpha=alpha,
                beta=self.beta,
                Q_k=self.Q_k,
                R_k=self.R_k
            )
            kalman_filters.append(kalman_filter)
        return kalman_filters


class BetaTuner(Tuner):
    def __init__(self, betas=None, bag=None, twist_topic=None,imu_topic=None ,slice=None):
        super(BetaTuner, self).__init__()
        self.beta = betas
        self.bag = bag
        self.imu_topic = imu_topic
        self.twist_topic = twist_topic
        self.legends = [str(x) for x in self.beta]
        self.slice = slice

    def get_kalman_filters(self):
        kalman_filters = []
        for beta in self.beta:
            kalman_filter = KalmanFilter(
                alpha=self.alpha,
                beta=beta,
                Q_k=self.Q_k,
                R_k=self.R_k
            )
            kalman_filters.append(kalman_filter)
        return kalman_filters


if __name__ == '__main__':
    imu_topic = "/imu"
    twist_topic = "/fake_encoder/twist"

    alphas_bag = "/home/dan/ws/rosbag/garry3/5m_slow.bag"
    alphas = [0.8, 0.9, 1, 1.1, 1.2]
    alpha_slice = [0, 35]
    betas = [0.8, 0.9, 1, 1.1, 1.2]
    betas_bag = "/home/dan/ws/rosbag/garry3/5turns.bag"
    betas_slice = [0, 30]

    alphas = AlphaTuner(alphas, alphas_bag, twist_topic, imu_topic, alpha_slice)
    alphas.run()
    alphas.plot()
    betas = BetaTuner(betas, betas_bag,twist_topic, imu_topic, betas_slice)
    betas.run()
    betas.plot()
    #alphas.export()
    compare_bag = "/home/dan/ws/rosbag/garry3/5m_m.bag"
    compare = CompareTwo(alphas_bag, compare_bag,  twist_topic,imu_topic, alpha_slice)
    compare.run()
    compare.plot()
    plt.show()
