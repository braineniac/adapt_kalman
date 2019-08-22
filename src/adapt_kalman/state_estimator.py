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

from kalman_filter import KalmanFilter,AdaptiveKalmanFilter
from bag_system_filter import BagSystemFilter

class StateEstimator(object):

    def __init__(self):
        self._states = []
        self._input = []
        self._output = []
        self._time = []

    def get_stamped_states(self):
        return self._states

    def get_stamped_input(self):
        return self._input

    def get_stamped_output(self):
        return self._output

    def set_states(self, stamped_states=None):
        if stamped_states is None:
            raise ValueError
        else:
            stamps,states = stamped_states
            states = self._v_state_form(states)
            states = self._psi_state_limit(states)
            self._states = states
            self._time = stamps

    def set_stamped_input(self, input=None):
        if input is None:
            raise ValueError
        else:
            self._input = input

    def set_stamped_output(self, output=None):
        if output is None:
            raise ValueError
        else:
            self._output = output

    def _v_state_form(self, states = None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x,y,psi,xdot,ydot,psidot in zip(states):
                v = np.sqrt(xdot*xdot + ydot*ydot)
                new_states.append((x,y,v,psi,psidot))
            return new_states

    def _psi_state_limit(self, states = None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x,y,psi,v,psidot in zip(states):
                k = abs(int(psi / (2*np.pi)))
                if psi > 2*np.pi:
                    psi -= 2*np.pi*k
                elif psi < -2*np.pi*k:
                    psi += 2*np.pi*(k+1)
                new_states.append((x,y,v,psi,psidot))
            return new_states

class KalmanStateEstimator(StateEstimator):

    def __init__(self,kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter):
            raise AttributeError
        else:
            super(KalmanStateEstimator, self).__init__()
            self._kalman_filter = kalman_filter

    def get_stamped_states(self):
        if len(self._time) != len(self._input) or len(self._input) != len(self._output):
            raise ValueError
        elif len(self._states) == len(self._time):
            return self._states
        else:
            self._add_time_from_input()
            self._add_time_from_output()
            self._run_kalman()
            return self._states

    def _run_kalman(self):
        if len(self._input) <= 1 or len(self._output) <= 1:
            raise ValueError
        else:
            u_index = 0
            y_index = 0
            u = (0,0)
            y = (0,0)
            stamped_state = []
            stamp = []
            states = []
            for t in sort(self._time):
                last_u_t,last_u = self._get_u(u_index)
                last_y_t,last_y = self._get_y(y_index)
                if t == last_u_t:
                    u = last_u
                    u_index += 1
                elif t == last_y_t:
                    y = last_y
                    y_index += 1
                self._kalman_filter.filter_iter((t, u, y))
                states.append(self._kalman_filter.get_post_states())
                stamp.append(t)
            states = self._psi_state_limit(states)
            return (stamp,states)

    def _get_y(self, index=0):
        return self._output[index][0],self._output[index][1]

    def _get_u(self,index=0):
        return self._input[index][0],self._input[index][1]

    def _add_time_from_output(self):
        for t_y,_y in self._output:
            self._time.append(t_y)

    def _add_time_from_input(self):
        for t_u,_u in self._input:
            self._time.append(t_u)

class StatePlotHandler(object):
    def __init__(self, state_estimator=None):
        if not isinstance(state_estimator,StateEstimator):
            raise ValueError
        else:
            self._state_estimator = state_estimator
            self._start_slice = 0
            self._end_slice = np.inf

    def set_slice_times(self, start=None, end=None):
        if not isinstance(start,float) or not isinstance(end,float) or start>end:
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

    def filter_butter(self, array, order=5, fc=1/50.):
        fs = 50
        w = fc / (fs / 2.) # Normalize the frequency
        b, a = signal.butter(order, w, 'low', analog=False)
        output = signal.filtfilt(b, a, array)
        return output

    def _slice_data(self, stamped_data=None):
        if stamped_data is None:
            raise ValueError
        else:
            t,data = zip(stamped_data)
            start,end = self._find_slice(t)
            for points in data:
                points = points[start:end]
            t = self._set_zero_time(begin,t)
            return (t,data)

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
            return start_index,end_index

    def _set_zero_time(self, start_index=0, t_array=None):
        if start_index is None or t_array is None:
            raise ValueError
        else:
            new_t_array = []
            for t in t_array:
                new_t_array.append(t - t_array[start_index])
            return new_t_array
