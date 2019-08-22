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
from bag_system_filter import BagSystemFilter


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
            stamps, states = stamped_states
            states = self._v_state_form(states)
            states = self._psi_state_limit(states)
            self._stamped_states = (stamps, states)

    def set_stamped_input(self, stamped_input=None):
        if stamped_input is None:
            raise ValueError
        else:
            self._stamped_input = stamped_input

    def set_stamped_output(self, stamped_output=None):
        if stamped_output is None:
            raise ValueError
        else:
            self._stamped_output = stamped_output

    @staticmethod
    def _v_state_form(states=None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x, y, psi, xdot, ydot, psidot in zip(states):
                v = np.sqrt(xdot * xdot + ydot * ydot)
                new_states.append((x, y, v, psi, psidot))
            return new_states

    @staticmethod
    def _psi_state_limit(states=None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x, y, psi, v, dpsi in zip(states):
                k = abs(int(psi / (2 * np.pi)))
                if psi > 2 * np.pi:
                    psi -= 2 * np.pi * k
                elif psi < -2 * np.pi * k:
                    psi += 2 * np.pi * (k + 1)
                new_states.append((x, y, v, psi, dpsi))
            return new_states


class KalmanStateEstimator(StateEstimator):

    def __init__(self, kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter):
            raise AttributeError
        else:
            super(KalmanStateEstimator, self).__init__()
            self._kalman_filter = kalman_filter

    def get_stamped_states(self):
        if len(self._time) != len(self._stamped_input) or len(self._stamped_input) != len(self._stamped_output):
            raise ValueError
        elif len(self._stamped_states) == len(self._time):
            return self._stamped_states
        else:
            self._add_time_from_input()
            self._add_time_from_output()
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
            stamp = []
            states = []
            Q = []
            for t in np.sort(self._time):
                last_u_t, last_u = self._stamped_input[u_index]
                last_y_t, last_y = self._stamped_output[y_index]
                if t == last_u_t:
                    u = last_u
                    u_index += 1
                elif t == last_y_t:
                    y = last_y
                    y_index += 1
                self._kalman_filter.filter_iter((t, u, y))
                states.append(self._kalman_filter.get_post_states())
                Q.append(self._kalman_filter.get_Q())
                stamp.append(t)
            states = self._psi_state_limit(states)
            self._stamped_states = (stamp, states)
            self._stamped_Q = (stamp, Q)

    def _add_time_from_output(self):
        for t_y, _y in self._stamped_output:
            self._time.append(t_y)

    def _add_time_from_input(self):
        for t_u, _u in self._stamped_input:
            self._time.append(t_u)


class StatePlotHandler(object):
    def __init__(self, state_estimator=None):
        if not isinstance(state_estimator, StateEstimator):
            raise ValueError
        else:
            self._state_estimator = state_estimator
            self._start_slice = 0
            self._end_slice = np.inf

    def set_slice_times(self, start=None, end=None):
        if not isinstance(start, float) or not isinstance(end, float) or start > end:
            raise ValueError
        else:
            self._start_slice = start
            self._end_slice = end

    def get_states(self):
        stamped_states = self._state_estimator.get_stamped_states()
        return self._slice_data(stamped_states)

    def get_input(self):
        stamped_input = self._state_estimator.get_stamped_input()
        return self._slice_data(stamped_input)

    def get_output(self):
        stamped_output = self._state_estimator.get_stamped_output()
        return self._slice_data(stamped_output)

    def get_Q(self):
        stamped_Q = self._state_estimator.get_stamped_Q()
        return self._slice_data(stamped_Q)

    def export_output(self, pre="", post=""):
        t, y = self.get_output()
        y0, y1 = y
        np.savetxt("../../data/{}_y0_{}.csv".format(pre, post), np.transpose([t, y0]), header='t y0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_y1_{}.csv".format(pre, post), np.transpose([t, y1]), header='t y1', comments='# ',
                   delimiter=' ', newline='\n')

    def export_input(self, pre="", post=""):
        t, u = self.get_input()
        u0, u1 = u
        np.savetxt("../../data/{}_u0_{}.csv".format(pre, post), np.transpose([t, u0]), header='t u0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_u1_{}.csv".format(pre, post), np.transpose([t, u1]), header='t u1', comments='# ',
                   delimiter=' ', newline='\n')

    def export_states(self, pre="", post=""):
        t, x = self.get_states()
        x0, x1, x2, x3, x4 = x
        np.savetxt("../../data/{}_x0_{}.csv".format(pre, post), np.transpose([t, x0]), header='t x0', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_x1_{}.csv".format(pre, post), np.transpose([t, x1]), header='t x1', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_x2_{}.csv".format(pre, post), np.transpose([t, x2]), header='t x2', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_x3_{}.csv".format(pre, post), np.transpose([t, x3]), header='t x3', comments='# ',
                   delimiter=' ', newline='\n')
        np.savetxt("../../data/{}_x4_{}.csv".format(pre, post), np.transpose([t, x4]), header='t x4', comments='# ',
                   delimiter=' ', newline='\n')

    def export_Q(self, pre="", post=""):
        t, Q = self.get_stamped_Q()
        np.savetxt("../../data/{}_Q_{}.csv".format(pre, post), np.transpose([t, Q]), header='t Q', comments='# ',
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
            t, data = zip(stamped_data)
            start, end = self._find_slice(t)
            for points in data:
                points = points[start:end]
            t = self._set_zero_time(start, t)
            return tuple(t, data)

    def _find_slice(self, t_array=None):
        if t_array is None:
            raise ValueError
        else:
            start_index = 0
            end_index = -1
            for t in t_array:
                if t <= self._end_slice:
                    end_index += 1
                if t <= self._start_slice:
                    start_index += 1
            end_index = len(t_array) - end_index
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
