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
from itertools import compress

from bag_reader import BagReader


class BagSystemIO(object):

    def __init__(self):
        self._input_mask = [1, 0, 0, 0, 0, 1]
        self._output_mask = [1, 0, 0, 0, 0, 1]
        self._state_mask = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]

    def get_input(self, stamped_input=None):
        if stamped_input is None:
            raise ValueError
        else:
            return self._filter(stamped_input, self._input_mask)

    def get_output(self, stamped_output=None):
        if stamped_output is None:
            raise ValueError
        else:
            return self._filter(stamped_output, self._output_mask)

    def get_states(self, stamped_states=None):
        if stamped_states is None:
            raise ValueError
        else:
            return self._filter(stamped_states, self._state_mask)

    @staticmethod
    def _filter(stamped_points=None, mask=None):
        if stamped_points is None or mask is None:
            raise ValueError
        else:
            t, points = stamped_points
            mask = np.array(mask, dtype=bool)
            filtered_points = []
            for point in points:
                filtered_points.append(tuple(compress(point, mask)))
            return tuple(t, filtered_points)
