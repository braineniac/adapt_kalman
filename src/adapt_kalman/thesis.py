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

from kalman_filter import KalmanFilter, AdaptiveKalmanFilter
from moving_weighted_window import MovingWeightedWindow, MovingWeightedExpWindow, MovingWeightedSigWindow
from state_estimator import BagSystemIO, StateEstimator, KalmanStateEstimator, StatePlotter
from bag_reader import BagReader
from simulator import LineSimulator, OctagonSimulator
from bag_generator import EKFGenerator, IMUTransformGenerator


def check_file(bag=None):
    if not os.path.isfile(bag) and not os.access(bag, os.R_OK):
        raise ValueError
    else:
        return True


def check_directory(dir=None):
    if not dir:
        raise ValueError
    elif not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory " + dir)
    else:
        return True


class Experiment(object):
    figure = 1

    def __init__(self, slice=(0, np.inf), legends=[]):
        self.slice = slice
        self.legends = legends
        self.state_plots = []

    @staticmethod
    def add_figure():
        plt.figure(Experiment.figure)
        Experiment.figure += 1

    @staticmethod
    def get_sys_input(bag=None, topic=None):
        if not bag or not topic:
            raise ValueError
        else:
            bag_system_io = BagSystemIO()
            bag_reader = BagReader(bag)
            sys_input = bag_system_io.get_input(bag_reader.read_twist(topic))
            return sys_input

    @staticmethod
    def get_sys_output(bag=None, topic=None):
        if not bag or not topic:
            raise ValueError
        else:
            bag_system_io = BagSystemIO()
            bag_reader = BagReader(bag)
            sys_output = bag_system_io.get_output(bag_reader.read_imu(topic))
            return sys_output

    @staticmethod
    def get_sys_states(bag=None, topic=None):
        if not bag or not topic:
            raise ValueError
        else:
            bag_system_io = BagSystemIO()
            bag_reader = BagReader(bag)
            sys_states = bag_system_io.get_states(bag_reader.read_odom(topic))
            return sys_states

    @staticmethod
    def add_plot(stamped_plot=None, dimension=None, option=None, legend=None):
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

    @staticmethod
    def get_kalman_filter(alpha=None, beta=None, Q_k=None, R_k=None):
        if not alpha or not beta or np.count_nonzero(Q_k) <= 1 or np.count_nonzero(R_k) <= 1:
            raise ValueError
        else:
            return [KalmanFilter(alpha=alpha, beta=beta, Q_k=Q_k, R_k=R_k)]

    @staticmethod
    def get_adaptive_kalman_filter(alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        if not alpha or not beta or np.count_nonzero(Q_k) <= 1 or np.count_nonzero(R_k) <= 1 or \
                not isinstance(window, MovingWeightedWindow) or np.count_nonzero(M_k) <= 1:
            raise ValueError
        else:
            return [AdaptiveKalmanFilter(alpha=alpha, beta=beta, Q_k=Q_k, R_k=R_k, window=window, M_k=M_k)]

    def run(self):
        raise NotImplementedError

    def set_kalman_state_plots(self, kalman_filters=None, input=None, output=None):
        if not isinstance(kalman_filters, list) or not input or not output:
            raise ValueError
        else:
            for kalman_filter in kalman_filters:
                state_estimator = KalmanStateEstimator(kalman_filter)
                state_estimator.set_stamped_input(input)
                state_estimator.set_stamped_output(output)
                plot_handler = StatePlotter(state_estimator)
                plot_handler.set_slice_times(self.slice[0], self.slice[1])
                self.state_plots.append(plot_handler)

    def set_state_plots(self, states=None, input=None, output=None):
        if not isinstance(states, list) or not input or not output:
            raise ValueError
        else:
            state_estimator = StateEstimator()
            state_estimator.set_stamped_input(input)
            state_estimator.set_stamped_output(output)
            state_estimator.set_stamped_states(states)
            plot_handler = StatePlotter(state_estimator)
            plot_handler.set_slice_times(self.slice[0], self.slice[1])
            self.state_plots.append(plot_handler)

    def get_state_plots(self):
        return self.state_plots

    def plot(self):
        options = ["b", "r", "k", "m", "g"]
        self.plot_input_figure(options, self.legends)
        self.plot_output_figure()
        self.plot_states_figure(options, self.legends)
        self.plot_xy_state_figure(options, self.legends)
        self.plot_Q_figure()

    def plot_input_figure(self, options=None, legends=None):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_input_titles())
        for i in range(num_titles):
            plt.subplot(num_titles * 100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_input_titles()[i])
            plt.xlabel("Time [s]")
            for state_plot, option, legend in zip(self.state_plots, options, legends):
                self.add_plot(state_plot.get_input_plot(), i, option, legend)

    def plot_output_figure(self):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_output_titles())
        for i in range(num_titles):
            plt.subplot(num_titles * 100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_output_titles()[i])
            plt.xlabel("Time [s]")
            self.add_plot(self.state_plots[0].get_output_plot(), i)

    def plot_states_figure(self, options=None, legends=None):
        self.add_figure()
        num_titles = len(self.state_plots[0].get_states_titles())
        for i in range(num_titles):
            plt.subplot(num_titles * 100 + 10 + 1 + i)
            plt.ylabel(self.state_plots[0].get_states_titles()[i])
            plt.xlabel("Time [s]")
            for state_plot, option, legend in zip(self.state_plots, options, legends):
                self.add_plot(state_plot.get_states_plot(), i, option, legend)

    def plot_xy_state_figure(self,options=None, legends=None):
        self.add_figure()
        plt.xlabel("x")
        plt.ylabel("y")
        for state_plot, option, legend in zip(self.state_plots, options, legends):
            self.add_plot(state_plot.get_x0x1_plot(), None, option, legend)

    def plot_Q_figure(self,options=None, legends=None):
        self.add_figure()
        for i in range(0, 2):
            plt.subplot(2 * 100 + 10 + 1 + i)
            plt.xlabel("Time [s]")
            plt.ylabel("Q" + str(i))
            for state_plot in self.state_plots:
                self.add_plot(state_plot.get_Q_plot(), i, options, legends)

    def export(self, pre=""):
        for state_plot in self.state_plots:
            state_plot.export_input(pre)
            state_plot.export_output(pre)
            state_plot.export_states(pre)
            state_plot.export_x0x1(pre)
            state_plot.export_Q(pre)


class KalmanExperiment(Experiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=[]):
        if not bag or not twist_topic or not imu_topic:
            raise ValueError
        else:
            super(KalmanExperiment, self).__init__(slice, legends)
            self.bag = bag
            self.twist_topic = twist_topic
            self.imu_topic = imu_topic

    def get_kalman_filters(self, alpha=None, beta=None, Q_k=None, R_k=None):
        kalman_filter = self.get_kalman_filter(alpha=alpha, beta=beta, Q_k=Q_k, R_k=R_k)
        return kalman_filter

    def run(self, alpha=None, beta=None, Q_k=None, R_k=None):
        kalman_filters = self.get_kalman_filters(alpha, beta, Q_k, R_k)
        input = self.get_sys_input(self.bag, self.twist_topic)
        output = self.get_sys_output(self.bag, self.imu_topic)
        self.set_kalman_state_plots(kalman_filters, input, output)


class AlphasExperiment(KalmanExperiment):
    def __init__(self, bag=None, twist_topic=None, imu_topic=None, slice=(0, np.inf), legends=[]):
        super(AlphasExperiment, self).__init__(bag, twist_topic, imu_topic, slice, legends)

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

    def run(self, alpha=None, beta=None, Q_k=None, R_k=None):
        kalman_filter = self.get_kalman_filter(alpha, beta, Q_k, R_k)
        self.set_kalman_state_plots(kalman_filter, self.input, self.output)


class AdaptiveKalmanSimulator(KalmanSimulator):
    def __init__(self, input=None, output=None, slice=(0, np.inf), legend=[]):
        super(AdaptiveKalmanSimulator, self).__init__(input, output, slice, legend)

    def run(self, alpha=None, beta=None, Q_k=None, R_k=None, window=None, M_k=None):
        kalman_filter = self.get_adaptive_kalman_filter(alpha, beta, Q_k, R_k, window, M_k)
        self.set_kalman_state_plots(kalman_filter, self.input, self.output)


class Compare(Experiment):
    def __init__(self, experiments=None, slice=(0, np.inf), legends=[]):
        if not isinstance(experiments, list):
            raise ValueError
        else:
            super(Compare, self).__init__(slice, legends)
            self.experiments = experiments

    def run(self):
        for experiment in self.experiments:
            if not experiment.get_state_plots():
                raise ValueError
            else:
                self.state_plots.append(experiment.get_state_plots()[0])


class ThesisDataExporter(object):
    def __init__(self, alpha=None, beta=None, r1=None, r2=None):
        self.alpha = alpha
        self.beta = beta
        self.r1 = r1
        self.r2 = r2

        self.R_k = np.zeros((2, 2))
        self.R_k[0][0] = 0.04
        self.R_k[1][1] = 0.02
        self.Q_k = np.zeros((2, 2))
        self.Q_k[0][0] = self.R_k[0][0] * self.r1
        self.Q_k[1][1] = self.R_k[1][1] * self.r2

        self.window = MovingWeightedExpWindow(10)
        self.M_k = np.zeros((2, 2))
        self.M_k[0][0] = 5
        self.M_k[1][1] = 1

        self.twist_topic = "/fake_encoder/twist"
        self.imu_topic = "/imu"
        self.odom_topic = "/odometry/filtered"

    def run_alphas(self, bag=None, alphas=[], slice=(0, np.inf)):
        if not check_file(bag) or not alphas:
            raise ValueError
        else:
            legend = [str(x) for x in alphas]
            alphas_exp = AlphasExperiment(bag, self.twist_topic, self.imu_topic, slice, legend)
            alphas_exp.run(alphas, self.beta, self.Q_k, self.R_k)
            alphas_exp.plot()
            return alphas_exp

    def run_betas(self, bag=None, betas=[], slice=(0, np.inf)):
        if not check_file(bag) or not betas:
            raise ValueError
        else:
            legend = [str(x) for x in betas]
            betas_exp = BetasExperiment(bag, self.twist_topic, self.imu_topic, slice, legend)
            betas_exp.run(self.alpha, betas, self.Q_k, self.R_k)
            betas_exp.plot()
            return betas_exp

    def get_kalman_experiment(self, bag=None):
        if not check_file(bag):
            raise ValueError
        else:
            kalman_experiment = KalmanExperiment(bag, self.twist_topic, self.imu_topic)
            kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_experiment

    def get_adaptive_kalman_experiment(self, bag=None):
        if not check_file(bag):
            raise ValueError
        else:
            adaptive_kalman_experiment = AdaptiveKalmanExperiment(bag, self.twist_topic, self.imu_topic)
            adaptive_kalman_experiment.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return adaptive_kalman_experiment

    def get_ekf_experiment(self, bag=None):
        if not check_file(bag):
            raise ValueError
        else:
            ekf_experiment = EKFExperiment(bag, self.odom_topic, self.twist_topic, self.imu_topic)
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

    def get_line_kalman_simulation(self, time=None, peak_vel=None):
        if not time or not peak_vel:
            raise ValueError
        else:
            line_sim = LineSimulator(time, peak_vel)
            line_sim.run()
            line_sim_input = line_sim.get_stamped_input()
            line_sim_output = line_sim.get_stamped_output()
            kalman_sim = KalmanSimulator(line_sim_input, line_sim_output)
            kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_sim

    def get_line_adaptive_kalman_simulation(self, time=None, peak_vel=None):
        if not time or not peak_vel:
            raise ValueError
        else:
            line_sim = LineSimulator(time, peak_vel)
            line_sim.run()
            line_sim_input = line_sim.get_stamped_input()
            line_sim_output = line_sim.get_stamped_output()
            adaptive_kalman_sim = AdaptiveKalmanSimulator(line_sim_input, line_sim_output)
            adaptive_kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return adaptive_kalman_sim

    def get_octagon_kalman_simulation(self, time=None, peak_vel=None, peak_turn=None):
        if not time or not peak_vel or not peak_turn:
            raise ValueError
        else:
            octagon_sim = OctagonSimulator(time, peak_vel, peak_turn)
            octagon_sim.run()
            octagon_sim_input = octagon_sim.get_stamped_input()
            octagon_sim_output = octagon_sim.get_stamped_output()
            kalman_sim = KalmanSimulator(octagon_sim_input, octagon_sim_output)
            kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k)
            return kalman_sim

    def get_octagon_adaptive_kalman_simulation(self, time=None, peak_vel=None, peak_turn=None):
        if not time or not peak_vel or not peak_turn:
            raise ValueError
        else:
            octagon_sim = OctagonSimulator(time, peak_vel, peak_turn)
            octagon_sim.run()
            octagon_sim_input = octagon_sim.get_stamped_input()
            octagon_sim_output = octagon_sim.get_stamped_output()
            kalman_sim = AdaptiveKalmanSimulator(octagon_sim_input, octagon_sim_output)
            kalman_sim.run(self.alpha, self.beta, self.Q_k, self.R_k, self.window, self.M_k)
            return kalman_sim

    def run_ekf_transforms(self, input_bags=[], output_folder=None):
        if not input_bags or not check_directory(output_folder):
            raise ValueError
        else:
            for bag in input_bags:
                if not check_file(bag):
                    raise ValueError
                else:
                    ekf_generator = EKFGenerator(bag, output_folder)
                    ekf_generator.generate(self.r1, self.r2, self.alpha, self.beta)

    @staticmethod
    def run_IMU_transforms(input_bags=[], output_folder=None):
        if not input_bags or check_directory(output_folder):
            raise ValueError
        else:
            for bag in input_bags:
                if not check_file(bag):
                    raise ValueError
                else:
                    imu_trans_generator = IMUTransformGenerator(bag, output_folder)
                    imu_trans_generator.generate()


if __name__ == '__main__':
    thesis = ThesisDataExporter(1, 1, r1=0.0001, r2=1)

    alphas_bag = "/home/dan/ws/rosbag/garry3/5m_slow.bag"
    alphas = [0.8, 0.9, 1, 1.1, 1.2]
    alphas_slice = [0, 35]
    #thesis.run_alphas(alphas_bag, alphas, alphas_slice)

    betas = [0.8, 0.9, 1, 1.1, 1.2]
    betas_bag = "/home/dan/ws/rosbag/garry3/5turns.bag"
    betas_slice = [0, 30]
    #thesis.run_betas(betas_bag, betas, betas_slice)

    alphas_multi_bag = "/home/dan/ws/rosbag/garry3/5m_m.bag"
    alphas_single_exp = thesis.get_kalman_experiment(alphas_bag)
    alphas_single_exp.export("alphas_")
    alphas_multi_exp = thesis.get_kalman_experiment(alphas_multi_bag)
    alphas_multi_exp.export("alphas_multi_")
    legend = ["single", "multi"]
    #thesis.run_compare([alphas_single_exp, alphas_multi_exp], alphas_slice, legend)

    betas_multi_bag = "/home/dan/ws/rosbag/garry3/5turns_m.bag"
    betas_single_exp = thesis.get_kalman_experiment(betas_bag)
    betas_single_exp.export("betas_")
    betas_multi_exp = thesis.get_kalman_experiment(betas_multi_bag)
    betas_multi_exp.export("betas_multi_")
    legend = ["single", "multi"]
    #thesis.run_compare([betas_single_exp, betas_multi_exp], betas_slice, legend)

    alpha_multi_adapt_exp = thesis.get_adaptive_kalman_experiment(alphas_multi_bag)
    alpha_multi_adapt_exp.export("alpha_multi_adapt_")
    legend = ["single", "multi adapt"]
    #thesis.run_compare([alphas_single_exp, alpha_multi_adapt_exp], alphas_slice, legend)

    beta_multi_adapt_exp = thesis.get_adaptive_kalman_experiment(betas_bag)
    beta_multi_adapt_exp.export("beta_multi_adapt_")
    legend = ["single", "multi adapt"]
    #thesis.run_compare([betas_single_exp, beta_multi_adapt_exp], betas_slice, legend)

    alpha_multi_ekf_bag = "/home/dan/ws/rosbag/garry3/ekf_5m_m.bag"
    alpha_multi_ekf_exp = thesis.get_ekf_experiment(alpha_multi_ekf_bag)
    alpha_multi_ekf_exp.export("alpha_ekf_multi_")
    legend = ["multi adapt", "EKF"]
    #thesis.run_compare([alpha_multi_adapt_exp, alpha_multi_ekf_exp], alphas_slice, legend)

    beta_multi_ekf_bag = "/home/dan/ws/rosbag/garry3/ekf_5turns_m.bag"
    beta_multi_ekf_exp = thesis.get_ekf_experiment(beta_multi_ekf_bag)
    beta_multi_ekf_exp.export("beta_multi_ekf_")
    legend = ["multi adapt", "EKF"]
    #thesis.run_compare([beta_multi_adapt_exp, beta_multi_ekf_exp], betas_slice, legend)

    sim_time = 10
    peak_vel = 0.14
    peak_turn = np.pi / 2
    line_kalman_sim = thesis.get_line_kalman_simulation(sim_time, peak_vel)
    line_kalman_sim.export("sim_line_")
    line_adaptive_kalman_sim = thesis.get_line_adaptive_kalman_simulation(sim_time, peak_vel)
    line_adaptive_kalman_sim.export("sim_line_adaptive_")
    legend = ["no adapt", "adapt"]
    line_slice = (0, np.inf)
    thesis.run_compare([line_kalman_sim, line_adaptive_kalman_sim], line_slice, legend)

    octagon_kalman_sim = thesis.get_octagon_kalman_simulation(sim_time, peak_vel, peak_turn)
    octagon_kalman_sim.export("sim_octagon_")
    octagon_adaptive_kalman_sim = thesis.get_octagon_adaptive_kalman_simulation(sim_time, peak_vel, peak_turn)
    octagon_adaptive_kalman_sim.export("sim_octagon_adaptive_")
    legend = ["no adapt", "adapt"]
    octagon_slice = (0, np.inf)
    #thesis.run_compare([octagon_kalman_sim, octagon_adaptive_kalman_sim], octagon_slice, legend)

    plt.show()
