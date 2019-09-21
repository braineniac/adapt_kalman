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

from kalman_estimator import KalmanFilter, AdaptiveKalmanFilter
from kalman_estimator import MovingWeightedSigWindow
from kalman_estimator import SysIO, SimSysIO, BagSysIO
from kalman_estimator import KalmanEstimator, EstimationPlots
from kalman_estimator import BagReader

from experiments import Experiment, NoRotationExperiment
from experiments import ExperimentPlotter, ExperimentSuite
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


class ThesisConfig(object):
    alpha = 10.905
    beta = 1.5267
    r1 = 0.1
    r2 = 10
    micro_v = 6
    micro_dpsi = 0.147
    mass = 1.02
    length = 0.25
    width = 0.14

    R_k = np.zeros((2, 2))
    R_k[0][0] = 0.04 * 0.04
    R_k[1][1] = 0.02 * 0.02
    Q_k = np.zeros((2, 2))
    Q_k[0][0] = R_k[0][0] * r1 * r1
    Q_k[1][1] = R_k[1][1] * r2 * r2

    window = MovingWeightedSigWindow(50)
    M_k = np.zeros((2, 2))
    M_k[0][0] = 100
    M_k[1][1] = 100

    twist_topic = "/fake_encoder/twist"
    imu_topic = "/imu_out/data"
    twist_ekf_topic = "/fake_encoder/twist_var"
    imu_ekf_topic = "/imu/var"
    odom_topic = "/odometry/filtered"

    output = "/home/dan/ws/rosbag/garry3/"
    out_trans = output + "trans/"
    out_ekf = output + "ekf/"

    straight_nojerk_bag_name = "5m_medium.bag"
    straight_nojerk_bag = out_trans + "trans_" + straight_nojerk_bag_name

    turn_nojerk_bag_name = "5turns.bag"
    turn_nojerk_bag = out_trans + "trans_" + turn_nojerk_bag_name

    alphas_single_bag = "5m_medium.bag"
    betas_single_bag = "5turns.bag"
    alphas_bag = "5m_m.bag"
    betas_bag = "5turns_m.bag"
    alphas_multi_bag = "10m.bag"
    betas_multi_bag = "10turns.bag"
    octagon_bag = "loops_7-8.bag"
    floor_bag = "floor.bag"

    micro_v_list = [5, 6, 7]
    micro_v_slice = (0, np.inf)
    micro_v_legend = [str(x) for x in micro_v_list]
    micro_dpsi_list = [0.143, 0.145, 0.147, 0.149, 0.151]
    micro_dpsi_slice = (0, np.inf)
    micro_dpsi_legend = [str(x) for x in micro_dpsi_list]

    line_sim_time = 5
    line_sim_slice = (0, np.inf)
    line_sim_legend = ["KF", "aKF"]
    line_sim_window = MovingWeightedSigWindow(300)

    legend_multi = ["single", "multi"]
    legend_adapt = ["reference", "single", "multi"]
    legend_ekf = ["EKF", "multi adapt"]
    legend_sim = ["no adapt", "adapt"]

    line_sim_time = 10
    octagon_sim_time = 50
    peak_vel = 1
    peak_turn = 1.7

    @staticmethod
    def get_Q_k(r1=None, r2=None):
        if r1 and r2:
            Q_k = np.zeros((2, 2))
            Q_k[0][0] = ThesisConfig.R_k[0][0] * r1 * r1
            Q_k[1][1] = ThesisConfig.R_k[1][1] * r2 * r2
            return Q_k
        return ThesisConfig.Q_k


class ThesisExperimentSuite(ExperimentSuite):

    def __init__(self, name=""):
        super(ThesisExperimentSuite, self).__init__(name)
        self._sys_IOs = []
        self._kalman_filters = []

        self._set_io()
        self._set_kalman_filters()
        self._set_experiments()

    def _get_bag_ios(self, bags=[]):
        bags_sys_IO = []
        for bag in bags:
            bag_reader = BagReader(bag)
            bag_sys_IO = BagSysIO(bag_reader,
                                  ThesisConfig.twist_topic,
                                  ThesisConfig.imu_topic)
            bags_sys_IO.append(bag_sys_IO)
        return bags_sys_IO

    def _set_io(self):
        raise NotImplementedError

    def _set_kalman_filters(self):
        raise NotImplementedError

    def _set_experiments(self):
        raise NotImplementedError


class MicroVTune(ThesisExperimentSuite):

    def __init__(self):
        super(MicroVTune, self).__init__("micro_v_tune")

    def _set_io(self):
        self._sys_IOs = self._get_bag_ios([ThesisConfig.straight_nojerk_bag])

    def _set_kalman_filters(self):
        for micro_v in ThesisConfig.micro_v_list:
            kalman_filter = KalmanFilter(
                ThesisConfig.get_Q_k(0.00001, 0.00001), ThesisConfig.R_k,
                ThesisConfig.alpha, ThesisConfig.beta,
                ThesisConfig.mass,
                ThesisConfig.length, ThesisConfig.width,
                micro_v, ThesisConfig.micro_dpsi
            )
            self._kalman_filters.append(kalman_filter)

    def _set_experiments(self):
        for kalman_filter, legend in \
                zip(self._kalman_filters, ThesisConfig.micro_v_legend):
            experiment = NoRotationExperiment(
                self._sys_IOs[0],
                kalman_filter,
                ThesisConfig.micro_v_slice,
                legend)
            self._experiments.append(experiment)


class MicroDPsiTune(ThesisExperimentSuite):

    def __init__(self):
        super(MicroDPsiTune, self).__init__("micro_dpsi_tune")

    def _set_io(self):
        self._sys_IOs = self._get_bag_ios([ThesisConfig.turn_nojerk_bag])

    def _set_kalman_filters(self):
        for micro_dpsi in ThesisConfig.micro_dpsi_list:
            kalman_filter = KalmanFilter(
                ThesisConfig.get_Q_k(0.00001, 0.00001), ThesisConfig.R_k,
                ThesisConfig.alpha, ThesisConfig.beta,
                ThesisConfig.mass,
                ThesisConfig.length, ThesisConfig.width,
                ThesisConfig.micro_v, micro_dpsi
            )
            self._kalman_filters.append(kalman_filter)

    def _set_experiments(self):
        for kalman_filter, legend in \
            zip(self._kalman_filters, ThesisConfig.micro_dpsi_legend):
                experiment = Experiment(
                    self._sys_IOs[0],
                    kalman_filter,
                    ThesisConfig.micro_dpsi_slice,
                    legend)
                self._experiments.append(experiment)


class LineSimulation(ThesisExperimentSuite):

    def __init__(self):
        super(LineSimulation, self).__init__("line_sim")

    def _set_io(self):
        line_sim = LineSimulator(
            ThesisConfig.line_sim_time, ThesisConfig.peak_vel)
        line_sim.run()
        sim_io = SimSysIO(line_sim.get_input(), line_sim.get_output())
        self._sys_IOs = sim_io

    def _set_kalman_filters(self):
        kalman_filter = KalmanFilter(
            ThesisConfig.Q_k, ThesisConfig.R_k,
            ThesisConfig.alpha, ThesisConfig.beta,
            ThesisConfig.mass,
            ThesisConfig.length, ThesisConfig.width,
            ThesisConfig.micro_v, ThesisConfig.micro_dpsi
        )
        self._kalman_filters.append(kalman_filter)
        adaptive_kalman_filter = AdaptiveKalmanFilter(
            ThesisConfig.Q_k, ThesisConfig.R_k,
            ThesisConfig.alpha, ThesisConfig.beta,
            ThesisConfig.mass,
            ThesisConfig.length, ThesisConfig.width,
            ThesisConfig.micro_v, ThesisConfig.micro_dpsi,
            ThesisConfig.line_sim_window, ThesisConfig.M_k
        )
        self._kalman_filters.append(adaptive_kalman_filter)

    def _set_experiments(self):
        for kalman_filter, legend in \
            zip(self._kalman_filters, ThesisConfig.line_sim_legend):
                experiment = Experiment(
                    self._sys_IOs,
                    kalman_filter,
                    ThesisConfig.line_sim_slice,
                    legend)
                self._experiments.append(experiment)

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
    #
    # def transform_all_IMU(self):
    #     bags = [
    #         self.output + self.alphas_single_bag,
    #         self.output + self.betas_single_bag,
    #         self.output + self.alphas_bag,
    #         self.output + self.betas_bag,
    #         self.output + self.alphas_multi_bag,
    #         self.output + self.betas_multi_bag,
    #         self.output + self.octagon_bag,
    #         self.output + self.floor_bag
    #     ]
    #     self.run_IMU_transforms(bags, self.out_trans)
    #
    #
    # def run_ekf_transforms(self, input_bags=[], output_folder=None):
    #     if not input_bags or not check_directory(output_folder):
    #         raise ValueError
    #     else:
    #         for bag in input_bags:
    #             ekf_generator = EKFGenerator(bag, output_folder)
    #             ekf_generator.generate(self.r1, self.r2, self.alpha, self.beta)
    #
    # @staticmethod
    # def run_IMU_transforms(input_bags=[], output_folder=None):
    #     if not input_bags or not check_directory(output_folder):
    #         raise ValueError
    #     else:
    #         for bag in input_bags:
    #             imu_trans_generator = IMUTransformGenerator(bag, output_folder)
    #             imu_trans_generator.generate()


if __name__ == '__main__':
    # micro_v_tune = MicroVTune()
    # micro_v_tune.plot()
    # micro_dpsi_tune = MicroDPsiTune()
    # micro_dpsi_tune.plot()
    line_sim = LineSimulation()
    line_sim.plot()
