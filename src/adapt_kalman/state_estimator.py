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
from scipy import signal

from kalman_filter import KalmanFilter, AdaptiveKalmanFilter


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

    def set_states(self, stamped_states=None):
        if stamped_states is None:
            raise ValueError
        else:
            new_stamped_states = []
            for stamp, states in stamped_states:
                states = self._v_state_form(states)
                states = self._order_states(states)
                states = self._psi_state_limit(states)
                new_stamped_states.append((stamp, states))
            self._stamped_states = new_stamped_states

    def set_stamped_input(self, stamped_input=None):
        if stamped_input is None:
            raise ValueError
        else:
            self._stamped_input = stamped_input
            self._add_time_from_input()

    def set_stamped_output(self, stamped_output=None):
        if stamped_output is None:
            raise ValueError
        else:
            self._stamped_output = stamped_output
            self._add_time_from_output()

    @staticmethod
    def _v_state_form(states=None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x, y, psi, xdot, ydot, psidot in zip(states):
                v = np.sqrt(xdot * xdot + ydot * ydot)
                new_states.append((x, y, v, psi, psidot))
            return new_states[0]

    @staticmethod
    def _psi_state_limit(states=None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x, y, v, psi, dpsi in zip(*states):
                k = abs(int(psi / (2 * np.pi)))
                if psi > 2 * np.pi:
                    psi -= 2 * np.pi * k
                elif psi < -2 * np.pi * k:
                    psi += 2 * np.pi * (k + 1)
                new_states.append((x, y, v, psi, dpsi))
            return new_states[0]

    @staticmethod
    def _order_states(states=None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x, y, psi, v, dpsi in zip(states):
                new_states.append((x, y, v, psi, dpsi))
            return new_states[0]

    def _add_time_from_output(self):
        for t_y, _y in self._stamped_output:
            self._time.append(t_y)

    def _add_time_from_input(self):
        for t_u, _u in self._stamped_input:
            self._time.append(t_u)


class KalmanStateEstimator(StateEstimator):

    def __init__(self, kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter):
            raise AttributeError
        else:
            super(KalmanStateEstimator, self).__init__()
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
                states = self._psi_state_limit(states)
                stamped_states.append((t, states))
                stamped_Q.append((t, self._kalman_filter.get_Q()))
            self._stamped_states = stamped_states
            self._stamped_Q = stamped_Q


class StatePlotHandler(object):
    def __init__(self, state_estimator=None):
        if not isinstance(state_estimator, StateEstimator):
            raise ValueError
        else:
            self._state_estimator = state_estimator
            self._start_slice = 0
            self._end_slice = np.inf
            self._input_titles = ["Input u0", "Input u1"]
            self._output_titles = ["Output y0", "Output y1"]
            self._states_titles = ["x state", "y state", "v state", "phi state", "dphi state"]

    def get_input_titles(self):
        return self._input_titles

    def get_output_titles(self):
        return self._output_titles

    def get_states_titles(self):
        return self._states_titles

    def get_input_plot(self):
        plot_u = [[], []]
        plot_u_t = []
        sliced_stamped_input = self._slice_data(self._state_estimator.get_stamped_input())
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
        sliced_stamped_output = self._slice_data(self._state_estimator.get_stamped_output())
        for y_stamped in sliced_stamped_output:
            t, y = y_stamped
            y0, y1 = y
            plot_y[0].append(y0)
            plot_y[1].append(y1)
            plot_y_t.append(t)
        return plot_y_t, plot_y

    def get_states_plot(self):
        plot_states = [[], [], [], [], []]
        plot_states_t = []
        slices_stamped_states = self._slice_data(self._state_estimator.get_stamped_states())
        for states_stamped in slices_stamped_states:
            t, states = states_stamped
            x0, x1, x2, x3, x4 = states
            plot_states[0].append(x0)
            plot_states[1].append(x1)
            plot_states[2].append(x2)
            plot_states[3].append(x3)
            plot_states[4].append(x4)
            plot_states_t.append(t)
        return plot_states_t, plot_states

    def get_Q_plot(self):
        plot_Q = []
        plot_t = []
        for stamped_Q in self._state_estimator.get_stamped_Q():
            t, Q = stamped_Q
            plot_Q.append(Q)
            plot_t.append(t)
        return plot_t, plot_Q

    def set_slice_times(self, start=0, end=np.inf):
        if start > end:
            raise ValueError
        else:
            self._start_slice = start
            self._end_slice = end

    def get_Q(self):
        stamped_Q = self._state_estimator.get_stamped_Q()
        return self._slice_data(stamped_Q)

    def export_output(self, pre="", post=""):
        t, y = self.get_output_plot()
        y0, y1 = y
        np.savetxt("../../data/{}y0{}.csv".format(pre, post), np.transpose([t, y0]), header='t y0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}y1{}.csv".format(pre, post), np.transpose([t, y1]), header='t y1', comments='# ',
                   delimiter=' ', newline='\n')

    def export_input(self, pre="", post=""):
        t, u = self.get_input_plot()
        u0, u1 = u
        np.savetxt("../../data/{}u0{}.csv".format(pre, post), np.transpose([t, u0]), header='t u0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}u1{}.csv".format(pre, post), np.transpose([t, u1]), header='t u1', comments='# ',
                   delimiter=' ', newline='\n')

    def export_states(self, pre="", post=""):
        t, x = self.get_states_plot()
        x0, x1, x2, x3, x4 = x
        np.savetxt("../../data/{}x0{}.csv".format(pre, post), np.transpose([t, x0]), header='t x0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}x1{}.csv".format(pre, post), np.transpose([t, x1]), header='t x1', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}x2{}.csv".format(pre, post), np.transpose([t, x2]), header='t x2', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}x3{}.csv".format(pre, post), np.transpose([t, x3]), header='t x3', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}x4{}.csv".format(pre, post), np.transpose([t, x4]), header='t x4', comments='# ',
                   delimiter=' ', newline='\n')

    def export_Q(self, pre="", post=""):
        t, Q = self.get_Q_plot()
        np.savetxt("../../data/{}Q{}.csv".format(pre, post), np.transpose([t, Q]), header='t Q', comments='# ',
                   delimiter=' ', newline='\n')

    @staticmethod
    def filter_butter(array, order=5, fc=1 / 50.):
        fs = 50
        w = fc / (fs / 2.)  # Normalize the frequency
        b, a = signal.butter(order, w, 'low', analog=False)
        output = signal.filtfilt(b, a, array)
        return output

    def _slice_data(self, stamped_data=None):
        if stamped_data is None:
            raise ValueError
        else:
            t_list = []
            data_list = []
            for t, data in stamped_data:
                t_list.append(t)
                data_list.append(data)
            t_list = self._set_zero_time(0, t_list)
            start, end = self._find_slice(t_list)
            t_list = self._set_zero_time(start, t_list)
            new_points = []
            for t, data in zip(t_list[start:end], data_list[start:end]):
                new_points.append((t, data))
            return new_points

    def _find_slice(self, t_array=None):
        if t_array is None:
            raise ValueError
        else:
            start_index = 0
            end_index = -1
            for t in t_array:
                if t <= self._end_slice:
                    end_index += 1
                if t < self._start_slice:
                    start_index += 1
            end_index = end_index - len(t_array)
            return start_index, end_index

    @staticmethod
    def _set_zero_time(start_index=0, t_array=None):
        if start_index is None or t_array is None:
            raise ValueError
        else:
            new_t_array = []
            for t in t_array:
                new_t_array.append(t - t_array[start_index])
            return new_t_array
