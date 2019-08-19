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

from kalman_filter import KalmanFilter
from adaptive_kalman_filter import AdaptiveKalmanFilter
from bag_system_filter import BagSystemFilter

class StateEstimator(object):

    def __init__(self):
        self._states = []
        self._input = []
        self._output = []
        self._time = []

    def get_states(self):
        return self._states

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    def set_states(self, states=None):
        if states is None:
            raise ValueError
        else:
            t,states = states
            states = self.v_state_correction(states)
            self._states = states
            self._time = t_states

    def set_input(self, input=None):
        if input is None:
            raise ValueError
        else:
            self._input = input

    def set_output(self, output=None):
        if output is None:
            raise ValueError
        else:
            self._output = output

    def _v_state_correction(self, states = None):
        if states is None:
            raise ValueError
        else:
            new_states = []
            for x,y,psi,xdot,ydot,psidot in zip(states):
                v = np.sqrt(xdot*xdot + ydot*ydot)
                new_states.append((x,y,v,psi,psidot))
            return states

class KalmanStateEstimator(StateEstimator):

    def __init__(self,kalman_filter=None):
        if not isinstance(kalman_filter, KalmanFilter):
            raise AttributeError
        else:
            super(KalmanStateEstimator, self).__init__()
            self._kalman_filter = kalman_filter

    def get_states(self):
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
                self._states.append((t, self._kalman_filter.get_post_states()))

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
