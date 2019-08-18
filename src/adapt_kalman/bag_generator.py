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

class BagGenerator(object):
    def __init__(self, bag_path=None, out_path=None, prefix=None):
        if bag_path is None or out_path is None:
            raise ValueError
        else:
            self.bag_path = bag_path
            self.out_path = out_path
            self.prefix = prefix

    def _generate(self, cli_args=None):
        if cli_args is None:
            raise AttributeError
        else:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
            roslaunch_args = cli_args[2:]
            parent = roslaunch.parent.ROSLaunchParent(uuid, [(roslaunch_file[0], roslaunch_args),])
            parent.start()
            parent.spin()


class EKFGenerator(BagGenerator):
    def __init__(self, bag_path=None, out_path=None, prefix="ekf_"):
        super(EKFGenerator, self).__init__(bag_path,out_path,prefix)

    def generate(self, postfix=None, r1=None,r2=None,alpha=None,beta=None):
        if r1 is None or r2 is None or alpha is None or beta is None:
            raise ValueError
        else:
            cli_args = [
            "adapt_kalman",
            "ekf.launch",
            "bag:={}".format(self.bag_path),
            "output:={}".format(self.out_path + self.prefix + get_filename_from_path(self.bagpath) + postfix),
            "r1:={}".format(self.r1),
            "r2:={}".format(self.r2),
            "alpha:={}".format(self.alpha),
            "beta:={}".format(self.beta)
            ]
            super(EKFGenerator,self)._generate(cli_args)

class IMUTransformGenerator(BagGenerator):
    def __init__(self, bag_path=None, out_path=None, prefix="trans_"):
        super(EKFGenerator, self).__init__(bag_path,out_path,prefix)

    def generate(self, postfix=None):
        cli_args = [
        "adapt_kalman",
        "transform_data.launch",
        "bag:={}".format(self.bag_path),
        "output:={}".format(self.out_path + self.prefix + get_filename_from_path(self.bagpath) + postfix)
        ]
        super(EKFGenerator,self)._generate(cli_args)

def get_filename_from_path(path=None):
    if path is not None:
        name = path.split('/')[-1]
        print(name)
