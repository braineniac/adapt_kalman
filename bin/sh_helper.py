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

from sh import test, mkdir
from sh import ErrorReturnCode_1

def file_exists(file_name=None):
    try:
        test("-e", file_name)
        return True
    except ErrorReturnCode_1:
        print("File {} doesn't exist!".format(file_name))
        return False

def folder_exists(folder_name=None):
    try:
        test("-d", folder_name)
        return True
    except ErrorReturnCode_1:
        print("Folder {} doesn't exist!".format(folder_name))
        return create_folder(folder_name)

def create_folder(folder=None):
    try:
        mkdir("-p", folder)
        print("Folder {} created!".format(folder))
        return True
    except ErrorReturnCode_1:
        print("Folder creation in {} failed!".format(folder))
        return False
