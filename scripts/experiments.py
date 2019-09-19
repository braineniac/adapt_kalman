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




class AlphaExperiment(KalmanExperiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=[]):
        super(AlphaExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)

    def set_kalman_state_plots(self, kalman_filters=None, input=None, output=None):
        if not isinstance(kalman_filters, list) or not input or not output:
            raise ValueError
        else:
            for kalman_filter in kalman_filters:
                state_estimator = KalmanStateEstimator(kalman_filter)
                state_estimator.set_stamped_input(input)
                state_estimator.set_stamped_output(output)
                state_estimator.set_u1y1_zero()
                plot_handler = StatePlotter(state_estimator)
                plot_handler.set_slice_times(self.slice[0], self.slice[1])
                self.state_plots.append(plot_handler)


class AlphasExperiment(AlphaExperiment):
    def __init__(self, bag_sys_io=None, slice=(0, np.inf)):
        self.legend = [str(x) for x in alphas]
        super(AlphasExperiment, self).__init__(bag_sys_io, slice, legends)

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None):
        if not isinstance(alpha, list):
            raise ValueError
        else:
            kalman_filters = []
            for a in alpha:
                kalman_filters.append(self.get_kalman_filter(alpha=a, beta=beta, Q_k=Q_k, R_k=R_k)[0])
            return kalman_filters


class BetasExperiment(KalmanExperiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=[]):
        super(BetasExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None):
        if not isinstance(beta, list):
            raise ValueError
        else:
            kalman_filters = []
            for b in beta:
                kalman_filters.append(self.get_kalman_filter(alpha=alpha, beta=b, Q_k=Q_k, R_k=R_k)[0])
            return kalman_filters

class BetasRatiosExperiment(KalmanExperiment):

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None):
        if not isinstance(Q_k, list):
            raise ValueError
        else:
            kalman_filters = []
            for Q in Q_k:
                kalman_filters.append(self.get_kalman_filter(alpha=alpha, beta=beta, Q_k=Q, R_k=R_k)[0])
            return kalman_filters


class AdaptiveKalmanExperiment(KalmanExperiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=None):
        super(AdaptiveKalmanExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        return self.get_adaptive_kalman_filter(alpha, beta, Q_k, R_k, window, M_k)

    def run(self, alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        kalman_filters = self.get_adaptive_kalman_filter(alpha, beta, Q_k, R_k, window, M_k)
        input = self.get_sys_input(self.bag, self.twist_topic)
        output = self.get_sys_output(self.bag, self.imu_topic)
        self.set_kalman_state_plots(kalman_filters, input, output)


class AdaptiveKalmanAlphaExperiment(AlphaExperiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=None):
        super(AdaptiveKalmanAlphaExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        return self.get_adaptive_kalman_filter(alpha, beta, Q_k, R_k, window, M_k)

    def run(self, alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        kalman_filters = self.get_adaptive_kalman_filter(alpha, beta, Q_k, R_k, window, M_k)
        input = self.get_sys_input(self.bag, self.twist_topic)
        output = self.get_sys_output(self.bag, self.imu_topic)
        self.set_kalman_state_plots(kalman_filters, input, output)


class EKFExperiment(KalmanExperiment):
    def __init__(self, bag=None, odom_topic=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=[]):
        super(EKFExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)
        if not odom_topic:
            raise ValueError
        else:
            self.odom_topic = odom_topic

    def run(self):
        input = self.get_sys_input(self.bag, self.twist_topic)
        output = self.get_sys_output(self.bag, self.imu_topic)
        states = self.get_sys_states(self.bag, self.odom_topic)
        self.set_state_plots(states, input, output)


class KalmanSimulator(Experiment):
    def __init__(self, input=None, output=None, slice=(0, np.inf), legend=[]):
        if not input or not output:
            raise ValueError
        else:
            super(KalmanSimulator, self).__init__(slice, legend)
            self.input = input
            self.output = output

    def run(self, kalman_filter=None):
        self.set_kalman_state_plots([kalman_filter], self.input, self.output)


class AdaptiveKalmanSimulator(KalmanSimulator):
    def __init__(self, input=None, output=None, slice=(0, np.inf), legend=[]):
        super(AdaptiveKalmanSimulator, self).__init__(input, output, slice, legend)

    def run(self, kalman_filter):
        self.set_kalman_state_plots([kalman_filter], self.input, self.output)


class Compare(Experiment):
    def __init__(self, experiments=None, slice=(0, np.inf), legends=[]):
        if not isinstance(experiments, list):
            raise ValueError
        else:
            super(Compare, self).__init__(slice, legends)
            self.experiments = experiments




class ThesisDataExporter(object):

    def __init__(self):
        self.alpha = 10.905
        self.beta = 1.5267
        self.r1 = 0.1
        self.r2 = 10
        self.micro_v = 50
        self.micro_dpsi = 1
        self.mass = 1.02
        self.length = 0.25
        self.width = 0.14

        self.R_k = np.zeros((2, 2))
        self.R_k[0][0] = 0.04 * 0.04
        self.R_k[1][1] = 0.02 * 0.02
        self.Q_k = np.zeros((2, 2))
        self.Q_k[0][0] = self.R_k[0][0] * self.r1 * self.r1
        self.Q_k[1][1] = self.R_k[1][1] * self.r2 * self.r2

        self.window = MovingWeightedSigWindow(50)
        self.M_k = np.zeros((2, 2))
        self.M_k[0][0] = 100
        self.M_k[1][1] = 1

        self.twist_topic = "/fake_encoder/twist"
        self.imu_topic = "/imu_out/data"
        self.twist_ekf_topic = "/fake_encoder/twist_var"
        self.imu_ekf_topic = "/imu/var"
        self.odom_topic = "/odometry/filtered"

        self.output = "/home/dan/ws/rosbag/garry3/"
        self.out_trans = self.output + "trans/"
        self.out_ekf = self.output + "ekf/"

        self.alphas_single_bag = "5m_medium.bag"
        self.betas_single_bag = "5turns.bag"
        self.alphas_bag = "5m_m.bag"
        self.betas_bag = "5turns_m.bag"
        self.alphas_multi_bag = "10m.bag"
        self.betas_multi_bag = "10turns.bag"
        self.octagon_bag = "loops_7-8.bag"
        self.floor_bag = "floor.bag"

        self.alphas = [1.6, 1.8, 2, 2.2, 2.4]
        self.betas = [6.6, 7.0, 7.4, 7.8, 8.2]
        self.beta_ratios = [0.01, 0.1, 1, 10, 100]

        self.alphas_slice = [9, 32]
        self.alpha_multi_slice = []
        self.betas_slice = [0, 50]
        self.line_slice = (0.5, np.inf)
        self.octagon_slice = (0, np.inf)

        self.legend_multi = ["single", "multi"]
        self.legend_adapt = ["reference","single", "multi"]
        self.legend_ekf = ["EKF", "multi adapt"]
        self.legend_sim = ["no adapt", "adapt"]

        self.line_sim_time = 3
        self.octagon_sim_time = 50
        self.peak_vel = 1
        self.peak_turn = 1.7

    def run_alphas(self, alphas_bag=None, alphas=None, slice=(0, np.inf)):
        if not check_file(alphas_bag) or not alphas:
            raise ValueError
        else:
            legend = [str(x) for x in alphas]
            alphas_exp = AlphasExperiment(alphas_bag, self.twist_topic, self.imu_topic, slice, legend)
            alphas_exp.run(alphas, self.beta, self.Q_k, self.R_k)
            alphas_exp.plot()
            alphas_exp.export("alphas_")
            return alphas_exp

    def run_betas(self, betas_bag=None, betas=None, slice=(0, np.inf)):
        if not check_file(betas_bag) or not betas:
            raise ValueError
        else:
            legend = [str(x) for x in betas]
            betas_exp = BetasExperiment(betas_bag, self.twist_topic, self.imu_topic, slice, legend)
            betas_exp.run(self.alpha, betas, self.Q_k, self.R_k)
            betas_exp.plot()
            betas_exp.export("betas_")
            return betas_exp

    def run_beta_ratios(self, betas_bag=None, ratios=None, slice=(0, np.inf)):
        if not check_file(betas_bag) or not ratios:
            raise ValueError
        else:
            legend = [str(x) for x in ratios]
            Q_ks = []
            for ratio in ratios:
                Q_k = np.zeros((2, 2))
                Q_k[0][0] = self.Q_k[0][0]
                Q_k[1][1] = self.R_k[1][1] * ratio * ratio
                Q_ks.append(Q_k)
            beta_ratios_exp = BetasRatiosExperiment(betas_bag, self.twist_topic, self.imu_topic, slice, legend)
            beta_ratios_exp.run(self.alpha, self.beta, Q_ks, self.R_k)
            beta_ratios_exp.plot()
            beta_ratios_exp.export("r2_ratios_")

    def get_kalman_experiment(self, bag=None, slice=(0, np.inf)):
        if not check_file(bag):
            raise ValueError
        else:
            kalman_experiment = KalmanExperiment(bag, self.twist_topic, self.imu_topic, slice)
            kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_experiment

    def get_kalman_alpha_experiment(self, bag=None, slice=(0, np.inf)):
        if not check_file(bag):
            raise ValueError
        else:
            kalman_experiment = AlphaExperiment(bag, self.twist_topic, self.imu_topic, slice)
            kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_experiment

    def get_adaptive_kalman_alpha_experiment(self, bag=None, slice=(0, np.inf)):
        if not check_file(bag):
            raise ValueError
        else:
            adaptive_kalman_experiment = AdaptiveKalmanAlphaExperiment(bag, self.twist_topic, self.imu_topic, slice)
            adaptive_kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return adaptive_kalman_experiment

    def get_adaptive_kalman_experiment(self, bag=None, slice=(0, np.inf)):
        if not check_file(bag):
            raise ValueError
        else:
            adaptive_kalman_experiment = AdaptiveKalmanExperiment(bag, self.twist_topic, self.imu_topic, slice)
            adaptive_kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return adaptive_kalman_experiment

    def get_ekf_experiment(self, bag=None, slice=(0, np.inf)):
        if not check_file(bag):
            raise ValueError
        else:
            ekf_experiment = EKFExperiment(bag, self.odom_topic, self.twist_ekf_topic, self.imu_ekf_topic, slice)
            ekf_experiment.run()
            return ekf_experiment

    @staticmethod
    def run_compare(experiments=[], slice=(0, np.inf), legend=[]):
        if not experiments:
            raise ValueError
        else:
            compare = Compare(experiments, slice, legend)
            compare.run()
            compare.plot()

    def get_line_kalman_simulation(self, time=None, peak_vel=None, slice=(0, np.inf), legend=[]):
        if not time or not peak_vel:
            raise ValueError
        else:
            line_sim = LineSimulator(time, peak_vel)
            line_sim.run()
            line_sim_input = line_sim.get_stamped_input()
            line_sim_output = line_sim.get_stamped_output()
            kalman_sim = KalmanSimulator(
                line_sim_input, line_sim_output, slice, legend)
            kalman_filter = KalmanFilter(self.Q_k, self.R_k,
                                         self.alpha, self.beta,
                                         self.mass,
                                         self.micro_theta, self.micro_eta)
            kalman_sim.run(kalman_filter)
            return kalman_sim

    def get_line_adaptive_kalman_simulation(self,time=None, peak_vel=None, slice=(0, np.inf), legend=[]):
        if not time or not peak_vel:
            raise ValueError
        else:
            line_sim = LineSimulator(time, peak_vel)
            line_sim.run()
            line_sim_input = line_sim.get_stamped_input()
            line_sim_output = line_sim.get_stamped_output()
            window = MovingWeightedSigWindow(10)
            adaptive_kalman_sim = AdaptiveKalmanSimulator(line_sim_input, line_sim_output, slice, legend)
            adaptive_kalman_filter = AdaptiveKalmanFilter(self.Q_k, self.R_k,
                                                          self.alpha, self.beta,
                                                          self.mass,
                                                          self.micro_theta, self.micro_eta,
                                                          window,
                                                          self.M_k)
            adaptive_kalman_sim.run(adaptive_kalman_filter)
            return adaptive_kalman_sim

    def get_octagon_kalman_simulation(self, time=None, peak_vel=None, peak_turn=None, slice=(0, np.inf), legend=[]):
        if not time or not peak_vel or not peak_turn:
            raise ValueError
        else:
            octagon_sim = OctagonSimulator(time, peak_vel, peak_turn)
            octagon_sim.run()
            octagon_sim_input = octagon_sim.get_stamped_input()
            octagon_sim_output = octagon_sim.get_stamped_output()
            kalman_sim = KalmanSimulator(octagon_sim_input, octagon_sim_output, slice, legend)
            kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_sim

    def get_octagon_adaptive_kalman_simulation(self, time=None, peak_vel=None, peak_turn=None, slice=(0, np.inf), legend=[]):
        if not time or not peak_vel or not peak_turn:
            raise ValueError
        else:
            octagon_sim = OctagonSimulator(time, peak_vel, peak_turn)
            octagon_sim.run()
            octagon_sim_input = octagon_sim.get_stamped_input()
            octagon_sim_output = octagon_sim.get_stamped_output()
            kalman_sim = AdaptiveKalmanSimulator(octagon_sim_input, octagon_sim_output, slice, legend)
            kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return kalman_sim

    # def transform_all_ekf(self):
    #     bags = [
    #         self.out_trans + "trans_" + self.alphas_bag,
    #         self.out_trans + "trans_" + self.betas_bag,
    #         self.out_trans + "trans_" + self.octagon_bag,
    #         self.out_trans + "trans_" + self.floor_bag,
    #         self.out_trans + "trans_" + self.alphas_single_bag,
    #         self.out_trans + "trans_" + self.betas_single_bag,
    #         self.out_trans + "trans_" + self.alphas_multi_bag,
    #         self.out_trans + "trans_" + self.betas_multi_bag,
    #     ]
    #     self.run_ekf_transforms(bags, self.out_ekf)

    def transform_all_IMU(self):
        bags = [
            self.output + self.alphas_single_bag,
            self.output + self.betas_single_bag,
            self.output + self.alphas_bag,
            self.output + self.betas_bag,
            self.output + self.alphas_multi_bag,
            self.output + self.betas_multi_bag,
            self.output + self.octagon_bag,
            self.output + self.floor_bag
        ]
        self.run_IMU_transforms(bags, self.out_trans)

    def export_all(self):
        # self.export_alphas()
        # self.export_betas()
        # self.export_betas_ratios()
        # self.export_alpha_single()
        # self.export_alpha_multi()
        # self.export_beta_single()
        # self.export_beta_multi()
        # self.export_alpha_adapt()
        # self.export_beta_adapt()
        # self.export_alpha_ekf()
        # self.export_beta_ekf()
        self.export_line_sim()
        # self.export_octagon_sim()
        # self.export_octagon()
        # self.export_floor()

    def export_alphas(self):
        self.run_alphas(self.out_trans + "trans_" + self.alphas_bag, self.alphas, self.alphas_slice)

    def export_betas(self):
        self.run_betas(self.out_trans + "trans_" + self.betas_bag, self.betas, self.betas_slice)

    def export_betas_ratios(self):
        self.run_beta_ratios(self.out_trans + "trans_" + self.betas_bag, self.beta_ratios, self.betas_slice)

    def export_alpha_single(self):
        alphas_single_exp = self.get_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_single_bag, self.alphas_slice)
        alphas_single_exp.export("alpha_single_")
        alphas_exp = self.get_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_bag, self.alphas_slice)
        alphas_exp.export("alpha_")
        self.run_compare([alphas_single_exp, alphas_exp], self.alphas_slice, self.legend_multi)

    def export_alpha_multi(self):
        alphas_exp = self.get_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_bag, self.alphas_slice)
        alphas_exp.export("alpha_")
        alphas_multi_exp = self.get_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_multi_bag, self.alphas_slice)
        alphas_multi_exp.export("alpha_multi_")
        self.run_compare([alphas_exp, alphas_multi_exp], self.alphas_slice, self.legend_multi)

    def export_beta_single(self):
        betas_single_exp = self.get_kalman_experiment(self.out_trans + "trans_" + self.betas_single_bag, self.betas_slice)
        betas_single_exp.export("beta_single_")
        betas_exp = self.get_kalman_experiment(self.out_trans + "trans_" + self.betas_bag, self.betas_slice)
        betas_exp.export("beta_")
        self.run_compare([betas_single_exp, betas_exp], self.betas_slice, self.legend_multi)

    def export_beta_multi(self):
        betas_exp = self.get_kalman_experiment(self.out_trans + "trans_" + self.betas_bag, self.betas_slice)
        betas_exp.export("beta_")
        betas_multi_exp = self.get_kalman_experiment(self.out_trans + "trans_" + self.betas_multi_bag, self.betas_slice)
        betas_multi_exp.export("beta_multi_")
        self.run_compare([betas_exp, betas_multi_exp], self.betas_slice, self.legend_multi)

    def export_alpha_adapt(self):
        alpha_adapt_exp = self.get_adaptive_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_bag, self.alphas_slice)
        alpha_adapt_exp.export("alpha_adapt_")
        alphas_single_exp = self.get_adaptive_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_single_bag, self.alphas_slice)
        alphas_single_exp.export("alpha_single_adapt_")
        alphas_multi_exp = self.get_adaptive_kalman_alpha_experiment(self.out_trans + "trans_" + self.alphas_multi_bag, self.alphas_slice)
        alphas_multi_exp.export("alpha_multi_adapt_")
        self.run_compare([alpha_adapt_exp, alphas_single_exp, alphas_multi_exp], self.alphas_slice, self.legend_adapt)

    def export_beta_adapt(self):
        beta_adapt_exp = self.get_adaptive_kalman_experiment(self.out_trans + "trans_" + self.betas_bag, self.betas_slice)
        beta_adapt_exp.export("beta_adapt_")
        betas_single_exp = self.get_adaptive_kalman_experiment(self.out_trans + "trans_" + self.betas_single_bag, self.betas_slice)
        betas_single_exp.export("beta_single_adapt_")
        betas_multi_exp = self.get_adaptive_kalman_experiment(self.out_trans + "trans_" + self.betas_multi_bag, self.betas_slice)
        betas_multi_exp.export("beta_multi_adapt_")
        self.run_compare([beta_adapt_exp, betas_single_exp, betas_multi_exp], self.betas_slice, self.legend_adapt)

    def export_octagon(self):
        octagon_adapt_exp = self.get_adaptive_kalman_experiment(self.out_trans + "trans_" + self.octagon_bag)
        octagon_adapt_exp.export("octagon_adapt_")
        octagon_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.octagon_bag)
        octagon_ekf_exp.export("octagon_ekf_")
        self.run_compare([octagon_ekf_exp, octagon_adapt_exp], (0, np.inf), self.legend_ekf)

    def export_floor(self):
        floor_adapt_exp = self.get_adaptive_kalman_experiment(self.out_trans + "trans_" + self.floor_bag)
        floor_adapt_exp.export("floor_adapt_")
        floor_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.floor_bag)
        floor_ekf_exp.export("floor_ekf_")
        self.run_compare([floor_ekf_exp, floor_adapt_exp], (0, np.inf), self.legend_ekf)

    def export_alpha_ekf(self):
        alpha_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.alphas_bag, self.alphas_slice)
        alpha_ekf_exp.export("alpha_ekf_")
        alphas_single_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.alphas_single_bag, self.alphas_slice)
        alphas_single_ekf_exp.export("alpha_single_ekf_")
        alphas_multi_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.alphas_multi_bag, self.alphas_slice)
        alphas_multi_ekf_exp.export("alpha_multi_ekf_")
        self.run_compare([alpha_ekf_exp, alphas_single_ekf_exp, alphas_multi_ekf_exp], self.alphas_slice, self.legend_ekf)

    def export_beta_ekf(self):
        beta_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.betas_bag, self.betas_slice)
        beta_ekf_exp.export("beta_ekf_")
        betas_single_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.betas_single_bag, self.betas_slice)
        betas_single_ekf_exp.export("beta_single_ekf_")
        betas_multi_ekf_exp = self.get_ekf_experiment(self.out_ekf + "ekf_trans_" + self.betas_multi_bag, self.betas_slice)
        betas_multi_ekf_exp.export("beta_multi_ekf_")
        self.run_compare([beta_ekf_exp, betas_single_ekf_exp, betas_multi_ekf_exp], self.betas_slice, self.legend_ekf)

    def export_line_sim(self):
        line_kalman_sim = self.get_line_kalman_simulation(self.line_sim_time, self.peak_vel, self.line_slice)
        # line_kalman_sim.export("sim_line_")
        line_adaptive_kalman_sim = self.get_line_adaptive_kalman_simulation(
             self.line_sim_time, self.peak_vel, self.line_slice)
        # line_adaptive_kalman_sim.export("sim_line_adaptive_")
        self.run_compare([line_kalman_sim, line_adaptive_kalman_sim], self.line_slice, self.legend_sim)

    def export_octagon_sim(self):
        octagon_kalman_sim = self.get_octagon_kalman_simulation(
            self.octagon_sim_time, self.peak_vel, self.peak_turn, self.octagon_slice)
        octagon_kalman_sim.export("sim_octagon_")
        octagon_adaptive_kalman_sim = self.get_octagon_adaptive_kalman_simulation(
            self.octagon_sim_time, self.peak_vel, self.peak_turn, self.octagon_slice)
        octagon_adaptive_kalman_sim.export("sim_octagon_adaptive_")
        self.run_compare([octagon_kalman_sim, octagon_adaptive_kalman_sim], self.octagon_slice, self.legend_sim)


    def run_ekf_transforms(self, input_bags=[], output_folder=None):
        if not input_bags or not check_directory(output_folder):
            raise ValueError
        else:
            for bag in input_bags:
                ekf_generator = EKFGenerator(bag, output_folder)
                ekf_generator.generate(self.r1, self.r2, self.alpha, self.beta)

    @staticmethod
    def run_IMU_transforms(input_bags=[], output_folder=None):
        if not input_bags or not check_directory(output_folder):
            raise ValueError
        else:
            for bag in input_bags:
                imu_trans_generator = IMUTransformGenerator(bag, output_folder)
                imu_trans_generator.generate()


if __name__ == '__main__':
    thesis = ThesisDataExporter()
    thesis.transform_all_IMU()
    thesis.transform_all_ekf()
    thesis.export_all()
    plt.show()
