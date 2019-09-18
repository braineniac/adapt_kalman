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
from itertools import compress, chain

import numpy as np
from scipy import signal

from kalman_filter import KalmanFilter
from bag_reader import BagReader


def check_directory(dir=None):
    if not dir:
        raise ValueError
    elif not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory " + dir)
    return True


class BagSystemIO(object):

    def __init__(self,
                 bag_reader=None,
                 input_twist=None,
                 output_imu=None,
                 state_odom=None):
        if not isinstance(bag_reader, BagReader):
            raise ValueError("Passed bag_reader not BagReader!")
        self._bag_reader = bag_reader
        self._input_twist = input_twist
        self._output_imu = output_imu
        self._state_odom = state_odom

        self._input_mask = [1, 0, 0, 0, 0, 1]
        self._output_mask = [1, 0, 0, 0, 0, 1]
        self._state_mask = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]

    def get_input(self):
        if not self._input_twist:
            raise ValueError("Input topic not defined!")
        stamped_input = self._bag_reader.read_twist(self._input_twist)
        return self._filter(stamped_input, self._input_mask)

    def get_output(self, stamped_output=None):
        if not self._output_imu:
            raise ValueError("Output topic not defined!")
        stamped_output = self._bag_reader.read_imu(self._output_imu)
        return self._filter(stamped_output, self._output_mask)

    def get_states(self, stamped_states=None):
        if not self._state_odom:
            raise ValueError("State topic not defined!")
        stamped_states = self._bag_reader.read_odom(self._state_odom)
        return self._filter(stamped_states, self._state_mask)

    @staticmethod
    def _filter(stamped_points=None, mask=None):
        if not stamped_points or not mask:
            raise ValueError
        else:
            time, points = zip(*stamped_points)
            mask = np.array(mask, dtype=bool)
            filtered_stamped_points = []
            for t, point in zip(time, points):
                filtered_stamped_points.append((t, tuple(compress(point, mask))))
            return filtered_stamped_points


class StateEstimator(object):

    def __init__(self):
        self._stamped_states = []
        self._stamped_input = []
        self._stamped_output = []
        self._stamped_Q = []
        self._time = []

    def get_stamped_states(self):
        return self._stamped_states

    def get_stamped_input(self):
        return self._stamped_input

    def get_stamped_output(self):
        return self._stamped_output

    def get_stamped_Q(self):
        return self._stamped_Q

    def set_stamped_states(self, stamped_states=None):
        if not stamped_states:
            raise ValueError
        else:
            new_stamped_states = []
            for stamp, states in stamped_states:
                states = self._v_state_form(states)
                states = self._psi_state_limit(states)
                new_stamped_states.append((stamp, states))
            self._stamped_states = new_stamped_states

    def set_stamped_input(self, stamped_input=None):
        if not stamped_input:
            raise ValueError
        else:
            self._stamped_input = stamped_input
            self._add_time_from_input()

    def set_u1y1_zero(self):
        new_stamped_input = []
        new_stamped_output = []
        if self._stamped_input and self._stamped_output:
            for t, u in self._stamped_input:
                u0, u1 = u
                new_stamped_input.append((t, (u0, 0)))
            for t, y in self._stamped_output:
                y0, y1 = y
                new_stamped_output.append((t, (y0, 0)))
            self._stamped_input = new_stamped_input
            self._stamped_output = new_stamped_output

    def set_stamped_output(self, stamped_output=None):
        if not stamped_output:
            raise ValueError
        else:
            self._stamped_output = stamped_output
            self._add_time_from_output()

    # @staticmethod
    # def _v_state_form(state=None):
    #     if not state:
    #         raise ValueError
    #     else:
    #         x, y, psi, xdot, ydot, psidot = state
    #         v = np.sqrt(xdot * xdot + ydot * ydot)
    #         return x, y, v, psi, psidot
    #
    @staticmethod
    def _psi_state_limit(state=None):
        if state is None:
            raise ValueError
        else:
            x, y, v, a, psi, dpsi, ddpsi = state
            k = abs(int(psi / (2 * np.pi)))
            if psi > 2 * np.pi:
                psi -= 2 * np.pi * k
            elif psi < -2 * np.pi * k:
                psi += 2 * np.pi * (k + 1)
            return x, y, v, a, psi, dpsi, ddpsi

    # @staticmethod
    # def _order_state(state=None):
    #     if not state:
    #         raise ValueError
    #     else:
    #         x, y, psi, v, dpsi = state
    #         return x, y, v, psi, dpsi

    def _add_time_from_output(self):
        for t_y, _y in self._stamped_output:
            self._time.append(t_y)

    def _add_time_from_input(self):
        for t_u, _u in self._stamped_input:
            self._time.append(t_u)


class KalmanEstimator(StateEstimator):

    def __init__(self, kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter):
            raise ValueError
        else:
            super(KalmanEstimator, self).__init__()
            self._kalman_filter = kalman_filter

    def get_stamped_states(self):
        if len(self._stamped_states) == len(self._time):
            return self._stamped_states
        else:
            self._run_kalman()
            return self._stamped_states

    def _run_kalman(self):
        if len(self._stamped_input) <= 1 or len(self._stamped_output) <= 1:
            raise ValueError
        else:
            u_index = 0
            y_index = 0
            u = (0, 0)
            y = (0, 0)
            stamped_states = []
            stamped_Q = []
            for t in np.sort(self._time):
                last_u_t, last_u = self._stamped_input[u_index]
                last_y_t, last_y = self._stamped_output[y_index]
                if t == last_u_t:
                    u = last_u
                    if u_index != len(self._stamped_input)-1:
                        u_index += 1
                elif t == last_y_t:
                    y = last_y
                    if y_index != len(self._stamped_output)-1:
                        y_index += 1
                self._kalman_filter.filter_iter((t, u, y))
                states = self._kalman_filter.get_post_states()
                states = list(chain(*states))
                states = self._psi_state_limit(states)
                stamped_states.append((t, states))
                Q = self._kalman_filter.get_Q()
                stamped_Q.append((t, (Q[0][0], Q[1][1])))
            self._stamped_states = stamped_states
            self._stamped_Q = stamped_Q


class StatePlotter(object):
    def __init__(self, state_estimator=None):
        if not isinstance(state_estimator, StateEstimator):
            raise ValueError
        else:
            self._state_estimator = state_estimator
            self._start_slice = 0
            self._end_slice = np.inf
            self._input_titles = ["Input u0", "Input u1"]
            self._output_titles = ["Output y0", "Output y1"]
            self._states_titles = [
                "x state", "y state",
                "v state", "a state",
                "phi state", "dphi state", "ddphi_state"
            ]

    def get_input_titles(self):
        return self._input_titles

    def get_output_titles(self):
        return self._output_titles

    def get_states_titles(self):
        return self._states_titles

    def get_input_plot(self):
        plot_u = [[], []]
        plot_u_t = []
        stamped_input = self._state_estimator.get_stamped_input()
        sliced_stamped_input = self._slice_data(stamped_input)
        for u_stamped in sliced_stamped_input:
            t, u = u_stamped
            u0, u1 = u
            plot_u[0].append(u0)
            plot_u[1].append(u1)
            plot_u_t.append(t)
        return plot_u_t, plot_u

    def get_output_plot(self):
        plot_y = [[], []]
        plot_y_t = []
        stamped_output = self._state_estimator.get_stamped_output()
        sliced_stamped_output = self._slice_data(stamped_output)
        for y_stamped in sliced_stamped_output:
            t, y = y_stamped
            y0, y1 = y
            plot_y[0].append(y0)
            plot_y[1].append(y1)
            plot_y_t.append(t)
        return plot_y_t, plot_y

    def get_states_plot(self):
        plot_states = [[], [], [], [], [], [], []]
        plot_states_t = []
        stamped_states = self._state_estimator.get_stamped_states()
        sliced_stamped_states = self._slice_data(stamped_states)
        for states_stamped in sliced_stamped_states:
            t, states = states_stamped
            for i in range(len(states)):
                plot_states[i].append(states[i])
            plot_states_t.append(t)
        return plot_states_t, plot_states

    def get_x0x1_plot(self):
        plot_x0_state = []
        plot_x1_state = []
        stamped_states = self._state_estimator.get_stamped_states()
        slices_stamped_states = self._slice_data(stamped_states)
        for states_stamped in slices_stamped_states:
            t, states = states_stamped
            plot_x0_state.append(states[0])
            plot_x1_state.append(states[1])
        return plot_x0_state, plot_x1_state

    def get_Q_plot(self):
        plot_Q0 = []
        plot_Q1 = []
        plot_t = []
        if self._state_estimator.get_stamped_Q():
            sliced_Q = self._slice_data(self._state_estimator.get_stamped_Q())
            for stamped_Q in sliced_Q:
                t, Q = stamped_Q
                Q0, Q1 = Q
                plot_Q0.append(Q0)
                plot_Q1.append(Q1)
                plot_t.append(t)
            return plot_t, (plot_Q0, plot_Q1)

    def set_slice_times(self, start=0, end=np.inf):
        if start > end:
            raise ValueError
        else:
            self._start_slice = start
            self._end_slice = end

    def export_output(self, pre="", post=""):
        t, y = self.get_output_plot()
        y0, y1 = y
        dir = "../../data/{}".format(pre)
        np.savetxt("{}/{}y0{}.csv".format(dir, pre, post),
                   np.transpose([t, y0]), header='t y0',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}y1{}.csv".format(dir, pre, post),
                   np.transpose([t, y1]), header='t y1',
                   comments='# ', delimiter=' ', newline='\n')

    def export_input(self, pre="", post=""):
        t, u = self.get_input_plot()
        u0, u1 = u
        dir = "../../data/{}".format(pre)
        check_directory(dir)
        np.savetxt("{}/{}u0{}.csv".format(dir, pre, post),
                   np.transpose([t, u0]), header='t u0',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}u1{}.csv".format(dir, pre, post),
                   np.transpose([t, u1]), header='t u1',
                   comments='# ', delimiter=' ', newline='\n')

    def export_states(self, pre="", post=""):
        t, x = self.get_states_plot()
        x0, x1, x2, x3, x4, x5, x6 = x
        dir = "../../data/{}".format(pre)
        check_directory(dir)
        np.savetxt("{}/{}x0{}.csv".format(dir, pre, post),
                   np.transpose([t, x0]), header='t x0',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x1{}.csv".format(dir, pre, post),
                   np.transpose([t, x1]), header='t x1',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x2{}.csv".format(dir, pre, post),
                   np.transpose([t, x2]), header='t x2',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x3{}.csv".format(dir, pre, post),
                   np.transpose([t, x3]), header='t x3',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x4{}.csv".format(dir, pre, post),
                   np.transpose([t, x4]), header='t x4',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x5{}.csv".format(dir, pre, post),
                   np.transpose([t, x5]), header='t x5',
                   comments='# ', delimiter=' ', newline='\n')
        np.savetxt("{}/{}x6{}.csv".format(dir, pre, post),
                   np.transpose([t, x6]), header='t x6',
                   comments='# ', delimiter=' ', newline='\n')

    def export_x0x1(self, pre="", post=""):
        x0, x1 = self.get_x0x1_plot()
        dir = "../../data/{}".format(pre)
        check_directory(dir)
        np.savetxt("{}/{}x0x1{}.csv".format(dir, pre, post),
                   np.transpose([x0, x1]), header='x0 x1',
                   comments='# ', delimiter=' ', newline='\n')

    def export_Q(self, pre="", post=""):
        if self.get_Q_plot():
            t, Q = self.get_Q_plot()
            Q0, Q1 = Q
            dir = "../../data/{}".format(pre)
            check_directory(dir)
            np.savetxt("{}/{}Q0{}.csv".format(dir, pre, post),
                       np.transpose([t, Q0]), header='t Q0',
                       comments='# ', delimiter=' ', newline='\n')
            np.savetxt("{}/{}Q1{}.csv".format(dir, pre, post),
                       np.transpose([t, Q1]), header='t Q1',
                       comments='# ', delimiter=' ', newline='\n')

    @staticmethod
    def filter_butter(array, order=5, fc=1 / 50.):
        fs = 50
        w = fc / (fs / 2.)  # Normalize the frequency
        b, a = signal.butter(order, w, 'low', analog=False)
        output = signal.filtfilt(b, a, array)
        return output

    def _slice_data(self, stamped_data=None):
        if not stamped_data:
            raise ValueError
        else:
            t_list = []
            data_list = []
            for t, data in stamped_data:
                t_list.append(t)
                data_list.append(data)
            t_list = self._set_zero_time(0, t_list)
            t_list, data_list = self._extend_slice(t_list, data_list)
            start, end = self._find_slice(t_list)
            t_list = self._set_zero_time(start, t_list)
            new_points = []
            for t, data in zip(t_list[start:end], data_list[start:end]):
                new_points.append((t, data))
            return new_points

    def _extend_slice(self, t_list=None, data_list=None):
        if not isinstance(t_list, list) or data_list is None:
            raise ValueError
        else:
            if t_list[-1] < self._end_slice != np.inf:
                t_list.append(self._end_slice)
                data_list.append(data_list[-1])
            return t_list, data_list

    def _find_slice(self, t_array=None):
        if not t_array:
            raise ValueError
        else:
            start_index = 0
            end_index = 0
            for t in t_array:
                if t <= self._end_slice:
                    end_index += 1
                if t < self._start_slice:
                    start_index += 1
            return start_index, end_index

    @staticmethod
    def _set_zero_time(start_index=0, t_array=None):
        if not start_index and start_index != 0 or not t_array:
            raise ValueError
        else:
            new_t_array = []
            for t in t_array:
                new_t_array.append(t - t_array[start_index])
            return new_t_array
