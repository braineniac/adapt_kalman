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
    x0 = [0,0,0,0,0]
    kalman_fiter = KalmanFilter(Q_k,R_k,alpha,beta,x0)

    array = [1,2,3,4,5,6,7,7,0,9,9]
    same_array = [5,5,5,5,5,5]
    window_exp = MovingWeightedExpWindow(5)
    window_sig = MovingWeightedSigWindow(5)
    window_exp.set_window(array)
    print(window_exp.get_window())
    avg = window_exp.get_avg()
    print(avg)
    window_sig.set_window(array)
    print(window_sig.get_window())
    print(window_sig.get_avg())
