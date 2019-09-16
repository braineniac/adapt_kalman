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

import numpy as np


class MovingWeightedWindow(object):

    def __init__(self, size=0):
        if not size or not isinstance(size, int):
            raise ValueError("Invalid window size!")
        self._size = size
        self._weights = self._set_weights()

    # def get_avg(self):
    #     raise NotImplementedError

    def get_size(self):
        return self._size

    def get_weighted_sum(self, array=[]):
        if not isinstance(array, list) or not isinstance(array, tuple):
            raise ValueError("Input array is neither an array nor a tuple!")
        if not array or not all(array):
            raise ValueError("Input array is empty!")
        sum = 0
        for elem, weight in zip(array, self._weights):
            sum += elem * weight
        return sum

    def _set_weights(self):
        raise NotImplementedError


class MovingWeightedSigWindow(MovingWeightedWindow):

    def __init__(self, size, alpha=10):
        super(MovingWeightedSigWindow, self).__init__(size)
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise ValueError("Alpha must be float or int!")
        if not alpha:
            raise ValueError("Alpha can't be zero!")
        self._alpha = alpha

    # def get_avg(self):
    #     y = np.zeros(self._size)
    #     for i in range(self._size):
    #         upper_sum = 0
    #         for k in range(i + 1):
    #             upper_sum += self._window[k] * self._weights[k]
    #         y[i] = upper_sum / np.sum(self._weights[:i + 1])
    #     return y[-1]

    def _set_weights(self):
        x = np.linspace(1, 0, self._size)
        w = np.zeros(self._size)
        for i in range(self._size):
            w[i] = 1 / (1 + np.exp(self._alpha * (x[i] - 0.5)))
        return w
