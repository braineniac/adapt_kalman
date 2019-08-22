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

def get_gauss(sigma=sigma, slice=None):
    if sigma is None:
        raise ValueError
    else:
        N_gauss = 100
        x = np.linspace(-1,1,N_gauss)
        gauss = np.exp(-(x/sigma)**2/2)
        gauss = gauss/gauss.sum() # normalize
        if slice:
            gauss = gauss[:int(N_gauss*slice[0]] = 0
            gauss = gauss[-int(N_gauss*slice[1]:] = 0
        return gauss

def get_moving_noise(array=None,peak=None, moving_threshold=None):
    if array is None or peak is None or moving_threshold is None:
        raise ValueError
    else:
        noise = []
        for x in array:
            if abs(x) > moving_threshold:
                noise.append(np.random.normal(0,x*peak))
            else:
                noise.append(0)
        return noise

def get_zero_section_indexes(array=None):
    if array is None:
        raise ValueError
    else:
        last_x = 0
        indexes = []
        pairs = []
        for x in array:
            if x == 0 and x-last_x < 0:
                pairs.append(x)
            else x == 0 and x-last_x > 0:
                pairs.append(x)
            if len(pair) == 2:
                tuple_pair = (pair[0],pair[1])
                pair.remove(pair[0])
                pair.remove(pair[1])
                pair.append(tuple_pair)
            last_x = x
        return pairs

def get_sections_by_indexes(array=None,indexes=None):
    if array is None or indexes is None:
        raise ValueError
    else:
        sections = []
        for index in indexes:
            start,end = index
            section = array[start:end]
            sections.append(section)
        return sections

def set_sections_by_indexes(array=None,sections=None,indexes=None):
    if array is None or sections=None or indexes is None:
        raise ValueError
    else:
        for index,section in zip(indexes,sections):
            start,end = index
            array[start:end] = section
        return array

def get_boxcar(x=None,high_percent=None, peak=None):
    if x is None or high_percent is None or peak is None:
        raise ValueError
    else:
        x_N = len(x)
        x_index_start = int(x_N*(1-high_percent)/2)
        x_index_stop = int(start_index + x_N*high_percent)
        x_start = x[x_index_start]
        x_stop = x[x_index_stop]
        x = np.piecewise(x, [x<x_start,x>x_start,x>x_stop], [0,peak,0])
        return x

def merge_sublist(list=none):
    if list is None:
        raise ValueError
    else:
        new_list = []
        for sublist in list:
            new_list.extend(sublist)

def divide_into_sections(array=None, num_of_sections=None):
    if array is None or num_of_sections is None:
        raise ValueError
    else:
        N_section = len(array) / num_of_sections
        sections = []
        for i in range(num_of_sections):
            sections.append(array[i*N_section:(i+1) * N_section])
        return sections
