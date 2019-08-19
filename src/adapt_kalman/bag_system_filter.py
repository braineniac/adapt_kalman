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

class BagSystemFilter(object):

    def __init__(self):
        if bag_path is None:
            raise AttributeError
        else:
            self.bag_reader = BagReader(bag_path)

    def get_input(self, topic=None):
        if topic is None:
            raise ValueError
        else:
            mask = [1,0,0,0,0,1]
            t,twist_data = zip(*self.bag_reader.read_twist(topic))
            return (t,self._filter(twist_data,mask))

    def get_output(self, topic=None):
        if topic is None:
            raise ValueError
        else:
            mask = [1,0,0,0,0,1]
            t,imu_data = zip(*self.bag_reader.read_imu(topic))
            return (t,self._filter(imu_data,mask))

    def get_states(self,topic=None):
        if topic is None or mask is None:
            raise ValueError
        else:
            mask = [1,1,0,0,0,1,1,1,0,0,0,1]
            t,odom_data = zip(*self.bag_reader.read_odom(topic))
            return (t,self._filter(odom_data,mask))

    def _filter(self, points=None, mask=None):
        if points is None or mask is None:
            raise ValueError
        else:
            mask = np.array(mask,dtype=bool)
            fil_points = []
            for point in points:
                fil_points.append(tuple(compress(point,mask)))
            return fil_points
