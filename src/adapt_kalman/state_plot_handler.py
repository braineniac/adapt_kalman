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

from state_estimator import StateEstimator,KalmanStateEstimator

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
