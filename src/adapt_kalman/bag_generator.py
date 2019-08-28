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

import roslaunch


def get_filename_from_path(path=None):
    if not path:
        raise ValueError
    else:
        name = path.split('/')[-1]
        return name


class BagGenerator(object):
    def __init__(self, bag_path=None, out_path=None, prefix=None):
        if not bag_path or not out_path:
            raise ValueError
        else:
            self.bag_path = bag_path
            self.out_path = out_path
            self.prefix = prefix

    @staticmethod
    def _generate(cli_args=None):
        if not cli_args:
            raise ValueError
        else:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
            roslaunch_args = cli_args[2:]
            parent = roslaunch.parent.ROSLaunchParent(uuid, [(roslaunch_file[0], roslaunch_args), ])
            parent.start()
            parent.spin()


class EKFGenerator(BagGenerator):
    def __init__(self, bag_path=None, out_path=None, prefix="ekf_"):
        super(EKFGenerator, self).__init__(bag_path, out_path, prefix)

    def generate(self, r1=None, r2=None, alpha=None, beta=None, postfix=""):
        if not r1 or not r2 or not alpha or not beta:
            raise ValueError
        else:
            cli_args = [
                "adapt_kalman",
                "ekf.launch",
                "bag:={}".format(self.bag_path),
                "output:={}".format(self.out_path + self.prefix + get_filename_from_path(self.bag_path) + postfix),
                "r1:={}".format(r1),
                "r2:={}".format(r2),
                "alpha:={}".format(alpha),
                "beta:={}".format(beta)
            ]
            super(EKFGenerator, self)._generate(cli_args)


class IMUTransformGenerator(BagGenerator):
    def __init__(self, bag_path=None, out_path=None, prefix="trans_"):
        super(IMUTransformGenerator, self).__init__(bag_path, out_path, prefix)

    def generate(self, postfix=""):
        cli_args = [
            "adapt_kalman",
            "transform_data.launch",
            "bag:={}".format(self.bag_path),
            "output:={}".format(self.out_path + self.prefix + get_filename_from_path(self.bag_path) + postfix)
        ]
        super(IMUTransformGenerator, self)._generate(cli_args)
