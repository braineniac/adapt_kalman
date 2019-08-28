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
import pandas as pd


class MovingWeightedWindow(object):

    def __init__(self, size=0):
        if not size:
            raise ValueError
        else:
            self._size = size
            self._weights = []
            self._window = []

    def get_avg(self):
        raise NotImplementedError

    def get_size(self):
        return self._size

    def set_window(self, array=None):
        if not array:
            raise ValueError
        else:
            window = []
            for i in range(self._size):
                window.append(array[-i - 1])
            self._window = window

    def get_window(self):
        return self._window


class MovingWeightedExpWindow(MovingWeightedWindow):

    def __init__(self, size):
        super(MovingWeightedExpWindow, self).__init__(size)

    def get_avg(self):
        series = pd.Series(np.flip(self._window))
        ewm = series.ewm(com=self._size / 2).mean().values
        return ewm[-1]


class MovingWeightedSigWindow(MovingWeightedWindow):

    def __init__(self, size, alpha=10):
        super(MovingWeightedSigWindow, self).__init__(size)
        if not alpha:
            raise ValueError
        else:
            self._alpha = alpha
            self._set_weights()

    def get_avg(self):
        y = np.zeros(self._size)
        for i in range(self._size):
            upper_sum = 0
            for k in range(i + 1):
                upper_sum += self._window[k] * self._weights[k]
            y[i] = upper_sum / np.sum(self._weights[:i + 1])
        return y[-1]

    def _set_weights(self):
        x = np.linspace(0, 1, self._size)
        w = np.zeros(self._size)
        for i in range(self._size):
            w[i] = 1 / (1 + np.exp(self._alpha * (x[i] - 0.5)))
        self._weights = w
