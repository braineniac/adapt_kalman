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

from kalman_filter import KalmanFilter,AdaptiveKalmanFilter
from moving_weighted_window import MovingWeightedExpWindow,MovingWeightedSigWindow
from state_estimator import StateEstimator,KalmanStateEstimator,StatePlotHandler
from bag_system_io import BagSystemIO
from system_io_simulator import LineSimulator,OctagonSimulator
from bag_generator import EKFGenrator,IMUTransformGenerator
from shell import *

import numpy as np

if __name__ == '__main__':
    alpha = 1
    beta = 1
    Q_k = np.zeros((2,2))
    R_k = np.zeros((2,2))
    Q_k[0][0] = 0.004
    Q_k[1][1] = 0.002
    R_k[0][0] = 0.04
    R_k[1][1] = 0.02
    M_k = np.zeros((2,2))
    M_k[0][0] = 2
    M_k[1][1] = 1
    x0 = [0,0,0,0,0]

    # uydt_array = (((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    # ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    # ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    # ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    # ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    # ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),)
    #
    # window_exp = MovingWeightedExpWindow(5)
    # window_sig = MovingWeightedSigWindow(5)
    #
    # adaptive_kalman_filter = AdaptiveKalmanFilter(Q_k,R_k,alpha,beta,window_sig,M_k,x0)
    # kalman_filter = KalmanFilter(Q_k,R_k,alpha,beta,x0)
    #
    # for uydt in uydt_array:
    #     kalman_filter.filter_iter(uydt)
    #     #print(kalman_filter.get_post_states())
    # for uydt in uydt_array:
    #     adaptive_kalman_filter.filter_iter(uydt)
    #     #print(adaptive_kalman_filter.get_post_states())

    rosbag = "/home/dan/ws/rosbag/garry3/5m_slow.bag"
    bag_system_filter = BagSystemFilter(rosbag)
    mask = [1,0,0,0,0,1]
    mask1 = [0,0,0,0,0,0,1,0,0,0,0,1]
    t_in,sys_in = bag_system_filter.get_input("/fake_encoder/twist",mask)
    t_out,sys_out = bag_system_filter.get_output("/imu",mask1)
    print(len(t_in),len(t_out))

    # np.savetxt("{}/ekf_{}_x0.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[0][0],ekf_reader.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')
