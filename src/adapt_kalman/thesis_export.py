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

from kalman_filter import KalmanFilter
from adaptive_kalman_filter import AdaptiveKalmanFilter
from moving_weighted_window import MovingWeightedExpWindow,MovingWeightedSigWindow

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

    uydt_array = (((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),
    ((0,0),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,1),(0,1),0.1),((0,0),(0,1),0.1),)

    window_exp = MovingWeightedExpWindow(5)
    window_sig = MovingWeightedSigWindow(5)

    adaptive_kalman_filter = AdaptiveKalmanFilter(Q_k,R_k,alpha,beta,window_sig,M_k,x0)
    kalman_filter = KalmanFilter(Q_k,R_k,alpha,beta,x0)

    for uydt in uydt_array:
        kalman_filter.filter_iter(uydt)
        #print(kalman_filter.get_post_states())
    for uydt in uydt_array:
        adaptive_kalman_filter.filter_iter(uydt)
        #print(adaptive_kalman_filter.get_post_states())
