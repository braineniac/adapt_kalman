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


def get_gauss(sigma=None, slice_tuple=None):
    if not sigma:
        raise ValueError
    else:
        N_gauss = 100
        x = np.linspace(-1, 1, N_gauss)
        gauss = np.exp(-(x / sigma) ** 2 / 2)
        gauss = gauss / gauss.sum()  # normalize
        if slice_tuple is not None:
            for i in range(int(N_gauss * slice_tuple[0])-1):
                gauss[i] = 0
            for j in range(int(N_gauss * slice_tuple[1])):
                gauss[-j] = 0
        return gauss


def get_noise(array=None, peak_still=None, peak_moving=None, moving_threshold=None):
    if array is None or not peak_still or not moving_threshold:
        raise ValueError
    else:
        noise = []
        for x in array:
            if abs(x) > moving_threshold:
                noise.append(np.random.normal(0, abs(x * peak_moving)))
            else:
                noise.append(np.random.normal(0, abs(peak_still)))
        return noise


def get_zero_section_indexes(array=None):
    if not array:
        raise ValueError
    else:
        last_x = 1
        pair = []
        pairs=[]
        for i in range(len(array)-1):
            if array[i] == 0 and last_x != 0:
                pair.append(i)
            elif array[i] != 0 and last_x == 0:
                pair.append(i)
            if len(pair) == 2:
                tuple_pair = (pair[0], pair[1])
                pairs.append(tuple_pair)
                pair.remove(pair[0])
                pair.remove(pair[0])
            last_x = array[i]
        return pairs


def get_sections_by_indexes(array=None, indexes=None):
    if not array or not indexes:
        raise ValueError
    else:
        sections = []
        for index in indexes:
            start, end = index
            section = array[start:end]
            sections.append(section)
        return sections


def set_sections_by_indexes(array=None, sections=None, indexes=None):
    if array is None or sections is None or indexes is None:
        raise ValueError
    else:
        for index, section in zip(indexes, sections):
            start, end = index
            for i in range(start, end-1):
                array[i] = section[i-start]
        return array


def get_boxcar(array=None, high_percent=None, peak=None):
    if not isinstance(array, list) or not high_percent or not peak:
        raise ValueError
    else:
        start_index = int(len(array) * (1 - high_percent) / 2)
        stop_index = int(start_index + len(array) * high_percent)
        for i in range(len(array)-1):
            if i < start_index or i > stop_index:
                array[i] = 0
            else:
                array[i] = peak
        return array


def divide_into_sections(array=None, num_of_sections=None):
    if not array or not num_of_sections:
        raise ValueError
    else:
        N_section = len(array) / num_of_sections
        sections = []
        for i in range(num_of_sections):
            sections.append(array[i * N_section:(i + 1) * N_section])
        return sections


class SystemIOSimulator(object):

    def __init__(self, time=None):
        if not time:
            raise ValueError
        else:
            self._input = None
            self._output = None
            self._time = np.linspace(0, time, time * 1000)  # system output is 50Hz

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
        if not time or not peak_vel:
            raise ValueError
        else:
            super(LineSimulator, self).__init__(time)
            self._peak_vel = peak_vel

    def _set_input(self):
        u0 = np.zeros(len(self._time)).tolist()
        box_function = get_boxcar(u0, 0.4, self._peak_vel)
        zeros = np.zeros(len(self._time))
        self._input = (box_function, zeros)
        #self._input = (zeros, zeros)

    def _set_output(self):
        u0 = np.zeros(len(self._time)).tolist()
        u0 = get_boxcar(u0, 0.4, self._peak_vel)
        #gauss = get_gauss(0.03, (0,0.5))
        gauss = get_gauss(0.3)
        conv = np.convolve(u0, gauss, mode="same")
        grad = 200 * np.gradient(conv)
        moving_noise = get_noise(grad, 3, 2, 0.1)
        accel = 0
        accel += grad
        accel += moving_noise
        accel = np.roll(accel, 30)
        # accel = np.negative(accel)
        zeros = np.zeros(len(self._time))
        self._output = (accel, zeros)


class OctagonSimulator(SystemIOSimulator):

    def __init__(self, time=None, peak_vel=None, peak_turn=None):
        if not peak_vel or not peak_turn:
            raise ValueError
        else:
            super(OctagonSimulator, self).__init__(time)
        self._peak_vel = peak_vel
        self._peak_turn = peak_turn

    def _set_input(self):
        u0 = self._get_u0()
        u1 = self._get_u1()
        self._input = (u0, u1)

    def _get_u0(self):
        u0 = np.zeros(len(self._time)).tolist()
        u0_sections = divide_into_sections(u0, 8)
        new_u0_sections = []
        for u0_section in u0_sections:
            new_u0_sections.append(get_boxcar(u0_section, 0.8, self._peak_vel))
        u0 = np.concatenate(new_u0_sections).tolist()
        return u0

    def _get_u1(self):
        u0 = self._get_u0()
        section_indexes = get_zero_section_indexes(u0)
        sections = get_sections_by_indexes(u0, section_indexes)
        new_sections = []
        for section in sections:
            new_sections.append(get_boxcar(section, 0.4, self._peak_turn))
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
        noise_moving = get_moving_noise(conv, 4, 0.1)
        return grad + noise_still + noise_moving

    def _get_y1(self):
        gauss = get_gauss(0.1, (3/7., 3/7.))
        u1 = self._get_u1()
        conv = np.convolve(u1, gauss, mode="same")
        noise = get_moving_noise(conv, 0.3, 0.1)
        return conv + noise
