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

import os.path

import numpy as np
from matplotlib import pyplot as plt

from kalman_estimator.kalman_filter import KalmanFilter, AdaptiveKalmanFilter
from kalman_estimator.moving_weighted_window import MovingWeightedSigWindow
from kalman_estimator.kalman_estimator import BagSysIO, StateEstimator, KalmanEstimator, EstimationPlots
from bag_reader import BagReader
from simulator import LineSimulator, OctagonSimulator
from bag_generator import EKFGenerator, IMUTransformGenerator


def check_file(bag=None):
    if os.path.isfile(bag):
        if not os.access(bag, os.R_OK):
            raise ValueError
        return True
    else:
        return False


def check_directory(dir=None):
    if not dir:
        raise ValueError
    elif not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory " + dir)
    return True


class Experiment(object):

    def __init__(self,
                 bag_sys_io=None,
                 kalman_filter=None,
                 slice=(0, np.inf), legend=[]):
        if not isinstance(bag_sys_io, BagSysIO):
            raise ValueError("Passed bag_sys_io not a BagSysIO!")
        if not isinstance(kalman_filter, KalmanFilter):
            raise ValueError("Passed kalman_filter not a KalmanFilter!")
        self._bag_sys_io = bag_sys_io
        self._kalman_filter = kalman_filter
        self._slice = slice
        self._legend = legend

    def _get_estimation(self):
        state_estimator = KalmanEstimator(self._kalman_filter)
        state_estimator.set_stamped_input(self._bag_sys_io.get_input())
        state_estimator.set_stamped_output(self._bag_sys_io.get_output())
        return state_estimator

    def get_estimation_plots(self):
        estimation = self._get_estimation()
        estimation_plotter = EstimationPlotter(estimation,
                                               self._slice, self._legend)
        return estimation_plotter


class ExperimentSuite(object):

    def __init__(self, name="", experiments=None):
        if not isinstance(experiments, list):
            raise ValueError("Pass a list of Experiment!")
        if not all(isinstance(exp, Experiment) for exp in experiments):
            raise ValueError("Pass a list only containing Experiment!")
        self._name = name
        self._experiments = experiments

    def plot_suite(self):
        experiment_plotter = ExperimentPlotter(self._experiments)
        experiment_plotter.plot()

    def export_suite(self):
        for i in range(len(self._experiments)):
            estimation_plots = self._experiments[i].get_estimation_plots()
            estimation_plots.export_input(self._name + str(i) + "_")
            estimation_plots.export_output(self._name + str(i) + "_")
            estimation_plots.export_states(self._name + str(i) + "_")
            estimation_plots.export_x0x1(self._name + str(i) + "_")
            estimation_plots.export_Q(self._name + str(i) + "_")


class ExperimentPlotter(object):
    figure = 1

    @staticmethod
    def add_figure():
        plt.figure(ExperimentPlotter.figure)
        ExperimentPlotter.figure += 1

    @staticmethod
    def _add_plot(stamped_plot=None, dimension=None, option=None, legend=None):
        if not stamped_plot:
            raise ValueError
        else:
            t, plot = stamped_plot
            if dimension is not None:
                plot = plot[dimension]
            if not option or not legend:
                plt.plot(t, plot)
            else:
                plt.plot(t, plot, option, label=legend)
                plt.legend()

    def __init__(self, experiments=None):
        if not isinstance(experiments, list):
            raise ValueError("Pass a list of Experiment!")
        if not all(isinstance(exp, Experiment) for exp in experiments):
            raise ValueError("Pass a list only containing Experiment!")
        self._experiments = experiments
        self._all_estimation_plots = []
        self._options = ["b", "r", "k", "m", "g"]

    def plot(self):
        for experiment in self._experiments:
            estimation_plots = experiment.get_estimation_plots()
            self._all_estimation_plots.append(estimation_plots)
        self._plot_input_figure()
        self._plot_output_figure()
        self._plot_states_figure()
        self._plot_xy_state_figure()
        self._plot_Q_figure()

    def _plot_input_figure(self):
        self.add_figure()
        input_titles = self._all_estimation_plots[0].get_input_titles()
        for i in range(len(input_titles)):
            plt.subplot(len(input_titles) * 100 + 10 + 1 + i)
            plt.ylabel(input_titles[i])
            plt.xlabel("Time [s]")
            for estimation_plots, option in
            zip(self._all_estimation_plots, self._options):
                legend = estimation_plots.get_legend()
                self._add_plot(estimation_plots.get_input_plot(),
                              i,
                              option,
                              legend)

    def _plot_output_figure(self):
        self.add_figure()
        output_titles = self._all_estimation_plots[0].get_output_titles()
        for i in range(len(output_titles)):
            plt.subplot(len(output_titles) * 100 + 10 + 1 + i)
            plt.ylabel(output_titles[i])
            plt.xlabel("Time [s]")
            for estimation_plots, option in
            zip(self._all_estimation_plots, self._options):
                legend = estimation_plots.get_legend()
                self._add_plot(estimation_plots.get_output_plot(),
                              i,
                              option,
                              legend)

    def _plot_states_figure(self):
        self.add_figure()
        states_titles = self._all_estimation_plots.get_states_titles()
        for i in range(len(states_titles)):
            plt.subplot(len(states_titles) * 100 + 10 + 1 + i)
            plt.ylabel(states_titles[i])
            plt.xlabel("Time [s]")
            for estimation_plots, option in
            zip(self._all_estimation_plots, self._options):
                legend = estimation_plots.get_legend()
                self._add_plot(estimation_plots.get_states_plot(),
                              i,
                              option,
                              legend)

    def _plot_xy_state_figure(self):
        self.add_figure()
        plt.xlabel("x")
        plt.ylabel("y")
        for estimation_plots, option in
            zip(self._all_estimation_plots, self._options):
                legend = estimation_plots.get_legend()
                self.add_plot(estimation_plots.get_x0x1_plot(),
                              None,
                              option,
                              legend)

    def _plot_Q_figure(self):
        self.add_figure()
        Q_titles = self._all_estimation_plots[0].get_Q_titles()
        for i in range(len(Q_titles)):
            plt.subplot(len(Q_titles) * 100 + 10 + 1 + i)
            plt.xlabel("Time [s]")
            plt.ylabel(Q_titles[i])
            for estimation_plots, option in
            zip(self._all_estimation_plots, self._options):
                legend = estimation_plots.get_legend()
                self._add_plot(estimation_plots.get_Q_plot(),
                              i,
                              option,
                              legend)
