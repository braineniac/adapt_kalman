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

from simulator_tools import *


class SystemIOSimulator(object):

    def __init__(self, time=None):
        if time is None:
            raise ValueError
        else:
            self._input = None
            self._output = None
            self._time = tuple(np.linspace(0, time, time * 50))  # system output is 50Hz

    def run(self):
        self._set_input()
        self._set_output()

    def get_stamped_input(self):
        return tuple(self._time, self._input)

    def get_stamped_output(self):
        return tuple(self._time, self._output)

    def _set_input(self):
        raise NotImplementedError

    def _set_output(self):
        raise NotImplementedError


class LineSimulator(SystemIOSimulator):

    def __init__(self, time=None, peak_vel=None):
        if peak_vel is None:
            raise ValueError
        else:
            super(LineSimulator, self).__init__(time)
            self._peak_vel = peak_vel

    def _set_input(self):
        box_function = get_boxcar(self._time, 0.8, self._peak_vel)
        zeros = np.zeros(self._time)
        self._input = (box_function, zeros)

    def _set_output(self):
        _t, u = self._input
        u0, _u1 = u

        gauss = get_gauss(0.01, self._time)
        conv = np.convolve(u0, gauss, mode="same")
        grad = 50 * np.gradient(conv)
        moving_noise = get_moving_noise(u0, 1, 0.1)

        accel = 0
        accel += grad
        accel += noise_still
        accel += noise_moving

        zeros = np.zeros(self._time)
        self.output = (accel, zeros)


class OctagonSimulator(SystemIOSimulator):

    def __init__(self, peak_vel=None, peak_turn=None, time=None):
        if peak_vel is None or peak_turn is None:
            raise ValueError
        else:
            super(OctagonSimulator, self).__init__(time)
        self._peak_vel = peak_vel
        self._peak_turn = peak_turn

    def _set_input(self):
        u0 = self._get_u0()
        u1 = self._get_u1(u0)
        self._input = (u0, u1)

    def _get_u0(self):
        u0 = np.linspace(0, self._time, self._N)
        u0_sections = divide_into_sections(u0, 8)
        for u0_section in u0_sections:
            u0_section = get_boxcar(u0_section, 0.8, self._peak_vel)
        u0 = merge_sublist(u0_sections)
        return u0

    def _get_u1(self, u0=None):
        if u0 is None:
            return ValueError
        else:
            section_indexes = self.get_zero_section_indexes(u0)
            sections = self.get_sections_by_indexes(u0, section_indexes)
            for section in sections:
                section = self.get_boxcar(section, 0.6, self._peak_turn)
            u1 = np.linspace(0, self._time, self._N)
            u1 = self.set_sections_by_indexes(u1, sections, section_indexes)
            return u1

    def _set_output(self):
        y0 = self._get_y0()
        y1 = self._get_y1()
        self._output = (y0, y1)

    def _get_y0(self):
        accel = np.linspace(0, self._time, self._N)
        u0 = self._input[0]
        gauss = get_gauss(0.1)
        conv = np.convolve(u0, gauss, mode="same")
        grad = 20 * np.gradient(conv)
        noise_still = np.random.normal(0, 0.08, self._N)
        noise_moving = get_moving_noise(accel, 3, 0.1)
        return grad + noise_still + noise_moving

    def _get_y1(self):
        dpsi = np.linspace(0, self._time, self._N)
        gauss = get_gauss(0.1, 3 / 7.)
        u1 = self._input[1]
        conv = np.convolve(u1, gauss, mode="same")
        noise = get_moving_noise(conv, 1, 0.1)
        return conv + noise
