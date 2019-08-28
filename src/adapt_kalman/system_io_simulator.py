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
import sys
np.set_printoptions(threshold=sys.maxsize)
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
        input_list = []
        for t, u0, u1 in zip(self._time, *self._input):
            input_list.append((t, (u0, u1)))
        return input_list

    def get_stamped_output(self):
        output_list = []
        for t, y0, y1 in zip(self._time, *self._output):
            output_list.append((t, (y0, y1)))
        return output_list

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
        u0 = np.zeros(len(self._time)).tolist()
        box_function = get_boxcar(u0, 0.8, self._peak_vel)
        zeros = np.zeros(len(self._time))
        self._input = (box_function, zeros)

    def _set_output(self):
        u0, _u1 = self._input

        gauss = get_gauss(0.01, self._time)
        conv = np.convolve(u0, gauss, mode="same")
        grad = 50 * np.gradient(conv)
        moving_noise = get_moving_noise(u0, 1, 0.1)

        accel = 0
        accel += grad
        accel += moving_noise

        zeros = np.zeros(len(self._time))
        self._output = (accel, zeros)


class OctagonSimulator(SystemIOSimulator):

    def __init__(self, time=None, peak_vel=None, peak_turn=None):
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
        u0 = np.zeros(len(self._time)).tolist()
        u0_sections = divide_into_sections(u0, 8)
        new_u0_sections = []
        for u0_section in u0_sections:
            new_u0_sections.append(get_boxcar(u0_section, 0.8, self._peak_vel))
        u0 = np.concatenate(new_u0_sections).tolist()
        return u0

    def _get_u1(self, u0=None):
        if u0 is None:
            return ValueError
        else:
            u0 = self._get_u0()
            section_indexes = get_zero_section_indexes(u0)
            sections = get_sections_by_indexes(u0, section_indexes)
            new_sections = []
            for section in sections:
                new_sections.append(get_boxcar(section, 0.6, self._peak_turn))
            u1 = np.zeros(len(self._time))
            u1 = set_sections_by_indexes(u1, new_sections, section_indexes)
            return u1

    def _set_output(self):
        y0 = self._get_y0()
        y1 = self._get_y1()
        self._output = (y0, y1)

    def _get_y0(self):
        u0 = self._get_u0()
        gauss = get_gauss(0.1)
        conv = np.convolve(u0, gauss, mode="same")
        grad = 20 * np.gradient(conv)
        noise_still = np.random.normal(0, 0.08, len(u0))
        noise_moving = get_moving_noise(conv, 3, 0.1)
        return grad + noise_still + noise_moving

    def _get_y1(self):
        gauss = get_gauss(0.1, (3/7., 3/7.))
        u1 = self._get_u1(self._get_u0())
        conv = np.convolve(u1, gauss, mode="same")
        noise = get_moving_noise(conv, 1, 0.1)
        return conv + noise
