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
from matplotlib import pyplot as plt

import roslaunch

from sh_helper import file_exists, folder_exists

from adapt_kalman_bag import AdaptKalmanBag
from adapt_kalman_sim_line import AdaptKalmanSimLine
from adapt_kalman_sim_octagon import AdaptKalmanSimOctagon
from ekf_reader import EKFReader

class Exporter:

    def __init__(self):
        self.bag_folder = "/home/dan/ws/rosbag/garry3/"
        folder_exists(self.bag_folder)
        self.img_folder = self.bag_folder + "images/"
        folder_exists(self.img_folder)
        self.plot_folder = self.bag_folder + "data/"
        folder_exists(self.plot_folder)
        self.trans_folder = self.bag_folder + "trans/"
        folder_exists(self.trans_folder)
        self.ekf_folder = self.bag_folder + "ekf/"
        folder_exists(self.ekf_folder)

        self.line_bag = "5m_medium2.bag"
        self.line_multi_bag = "5m_m.bag"
        self.turn_bag = "10turns.bag"
        self.turn_multi_bag = "10turns_m.bag"
        self.octagon_bag = "loops_5-6.bag"

        self.kalman_line_base =None
        self.kalman_turn_base = None

        self.line_start=0
        self.line_end=np.inf

        self.line_m_start = 0
        self.line_m_end= np.inf

        self.turn_start=0
        self.turn_end=np.inf

        self.turn_m_start=0
        self.turn_m_end=np.inf

        self.fig_count = 1

        self.alpha = 1.65
        self.beta= 10.4 #10.4
        self.r1= 0.005
        self.r2=9999

        self.fig_dims = [40,25]

        self.run_transforms()
        #self.run_ekfs()

        #self.export_line()
        self.export_turn()

        #self.export_ekf(self.line_bag, self.line_start, self.line_end)
        #self.export_ekf(self.turn_bag,self.turn_start,self.turn_end)
        #self.export_ekf(self.line_multi_bag, self.line_start, self.line_end)
        #self.export_ekf(self.turn_multi_bag,self.turn_start,self.turn_end)
        #self.no_kalman_line()
        #self.no_kalman_turn()
        #self.compare_multi_line()
        #self.compare_multi_turn()
        plt.show()

    def run_transforms(self):
        self.transform_rosbag(self.line_bag)
        self.transform_rosbag(self.turn_bag)
        self.transform_rosbag(self.turn_multi_bag)
        self.transform_rosbag(self.line_multi_bag)

    def run_ekfs(self):
        self.run_ekf(self.line_bag)
        self.run_ekf(self.turn_bag)
        self.run_ekf(self.line_bag)
        self.run_ekf(self.line_multi_bag)
        self.run_ekf(self.turn_multi_bag)

    def no_kalman_line(self):
        single_bagpath = self.trans_folder + "trans_" + self.line_bag
        multi_bagpath = self.trans_folder + "trans_" + self.line_multi_bag
        if file_exists(single_bagpath) and file_exists(multi_bagpath):
            single_kalman_bag_line = self.kalman_line_base
            r1 = 99999.9
            r2 = 99999.9
            multi_kalman_bag_line =self.run_bag(multi_bagpath, self.alpha, self.beta,r1,r2, self.line_m_start,self.line_m_end)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.subplot(211)
            plt.title("Comparison of single vs multi, no kalman")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance x [m]")
            plt.plot(single_kalman_bag_line.plot_x[0][0], single_kalman_bag_line.plot_x[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_x[0][0],multi_kalman_bag_line.plot_x[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_x0.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_x[0][0],multi_kalman_bag_line.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.title("Comparison of single vs multi")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity v [m/s]")
            plt.plot(single_kalman_bag_line.plot_x[2][0], single_kalman_bag_line.plot_x[2][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_x[2][0],multi_kalman_bag_line.plot_x[2][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_x2.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_x[2][0],multi_kalman_bag_line.plot_x[2][1]]),header='t v', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_nokalman_states.png".format(self.img_folder, self.line_bag), quality=95, dpi=300)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.subplot(211)
            plt.title("Fake wheel encoder input")
            plt.xlabel("Time [s]")
            plt.ylabel("Speed in x [m/s]")
            plt.plot(single_kalman_bag_line.plot_u[0][0], single_kalman_bag_line.plot_u[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_u[0][0], multi_kalman_bag_line.plot_u[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_u0.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_u[0][0],multi_kalman_bag_line.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.title("Acceleration output")
            plt.xlabel("Time [s]")
            plt.ylabel("Accel in x [m/s2]")
            plt.plot(single_kalman_bag_line.plot_y[0][0], single_kalman_bag_line.plot_y[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_y[0][0], multi_kalman_bag_line.plot_y[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_y0.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_y[0][0],multi_kalman_bag_line.plot_y[0][1]]),header='t y0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_nokalman_u0y0.png".format(self.img_folder, self.line_bag), quality=95, dpi=300)

    def no_kalman_turn(self):
        single_bagpath = self.trans_folder + "trans_" + self.turn_bag
        multi_bagpath = self.trans_folder + "trans_" + self.turn_multi_bag
        if file_exists(single_bagpath) and file_exists(multi_bagpath):
            single_kalman_bag_turn = self.kalman_turn_base
            r1 =99999.9
            r2 =99999.9
            multi_kalman_bag_turn =self.run_bag(multi_bagpath, self.alpha, self.beta,r1,r2, self.turn_m_start,self.turn_m_end)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.title("Comparison of single vs multi")
            plt.xlabel("Time [s]")
            plt.ylabel("Phi [degrees]")
            plt.plot(single_kalman_bag_turn.plot_x[3][0], single_kalman_bag_turn.plot_x[3][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_x[3][0],multi_kalman_bag_turn.plot_x[3][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_x3.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_turn.plot_x[3][0],multi_kalman_bag_turn.plot_x[3][1]]),header='t phi', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_nokalman_x3.png".format(self.img_folder, self.turn_bag), quality=95, dpi=300)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.subplot(211)
            plt.xlabel("Time [s]")
            plt.ylabel("Joystick angular z input [rad/s]")
            plt.plot(single_kalman_bag_turn.plot_u[1][0], single_kalman_bag_turn.plot_u[1][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_u[1][0], multi_kalman_bag_turn.plot_u[1][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_u1.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_turn.plot_u[1][0],multi_kalman_bag_turn.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.xlabel("Time [s]")
            plt.ylabel("Gyro output[rad/s]")
            plt.plot(single_kalman_bag_turn.plot_y[1][0], single_kalman_bag_turn.plot_y[1][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_y[1][0], multi_kalman_bag_turn.plot_y[1][1], "r", label="multi")
            np.savetxt("{}/multi_{}_nokalman_y1.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_turn.plot_y[1][0],multi_kalman_bag_turn.plot_y[1][1]]),header='t y1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_nokalman_u1y1.png".format(self.img_folder, self.turn_bag), quality=95, dpi=300)

    def compare_multi_line(self):
        single_bagpath = self.trans_folder + "trans_" + self.line_bag
        multi_bagpath = self.trans_folder + "trans_" + self.line_multi_bag
        if file_exists(single_bagpath) and file_exists(multi_bagpath):
            single_kalman_bag_line = self.kalman_line_base
            multi_kalman_bag_line =self.run_bag(multi_bagpath, self.alpha, self.beta,self.r1,self.r2, self.line_m_start,self.line_m_end)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.suptitle("{}/{}".format(single_bagpath,multi_bagpath))

            plt.subplot(211)
            plt.title("Comparison of single vs multi")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance x [m]")
            plt.plot(single_kalman_bag_line.plot_x[0][0], single_kalman_bag_line.plot_x[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_x[0][0],multi_kalman_bag_line.plot_x[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_x0.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_x[0][0],multi_kalman_bag_line.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.title("Comparison of single vs multi")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity v [m/s]")
            plt.plot(single_kalman_bag_line.plot_x[2][0], single_kalman_bag_line.plot_x[2][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_x[2][0],multi_kalman_bag_line.plot_x[2][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_x2.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_x[2][0],multi_kalman_bag_line.plot_x[2][1]]),header='t v', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_comp_states.png".format(self.img_folder, self.line_bag), quality=95, dpi=300)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.subplot(211)
            plt.title("Fake wheel encoder input")
            plt.xlabel("Time [s]")
            plt.ylabel("Speed in x [m/s]")
            plt.plot(single_kalman_bag_line.plot_u[0][0], single_kalman_bag_line.plot_u[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_u[0][0], multi_kalman_bag_line.plot_u[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_u0.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_line.plot_u[0][0],multi_kalman_bag_line.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.title("Acceleration output")
            plt.xlabel("Time [s]")
            plt.ylabel("Accel in x [m/s2]")
            plt.plot(single_kalman_bag_line.plot_y[0][0], single_kalman_bag_line.plot_y[0][1], "b", label="single")
            plt.plot(multi_kalman_bag_line.plot_y[0][0], multi_kalman_bag_line.plot_y[0][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_y0.csv".format(self.plot_folder,self.line_bag), np.transpose([multi_kalman_bag_line.plot_y[0][0],multi_kalman_bag_line.plot_y[0][1]]),header='t y0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_comp_u0y0.png".format(self.img_folder, self.turn_bag), quality=95, dpi=300)

    def compare_multi_turn(self):
        single_bagpath = self.trans_folder + "trans_" + self.turn_bag
        multi_bagpath = self.trans_folder + "trans_" + self.turn_multi_bag
        if file_exists(single_bagpath) and file_exists(multi_bagpath):
            single_kalman_bag_turn = self.kalman_turn_base
            multi_kalman_bag_turn =self.run_bag(multi_bagpath, self.alpha, self.beta,self.r1,self.r2, self.turn_m_start,self.turn_m_end)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.suptitle("{}/{}".format(single_bagpath,multi_bagpath))

            plt.title("Comparison of single vs multi")
            plt.xlabel("Time [s]")
            plt.ylabel("Phi [degrees]")
            plt.plot(single_kalman_bag_turn.plot_x[3][0], single_kalman_bag_turn.plot_x[3][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_x[3][0],multi_kalman_bag_turn.plot_x[3][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_x3.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_turn.plot_x[3][0],multi_kalman_bag_turn.plot_x[3][1]]),header='t phi', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_comp_x3.png".format(self.img_folder, self.turn_bag), quality=95, dpi=300)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1

            plt.subplot(211)
            plt.xlabel("Time [s]")
            plt.ylabel("Joystick angular z input [rad/s]")
            plt.plot(single_kalman_bag_turn.plot_u[1][0], single_kalman_bag_turn.plot_u[1][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_u[1][0], multi_kalman_bag_turn.plot_u[1][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_u1.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_turn.plot_u[1][0],multi_kalman_bag_turn.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(212)
            plt.xlabel("Time [s]")
            plt.ylabel("Gyro output[rad/s]")
            plt.plot(single_kalman_bag_turn.plot_y[1][0], single_kalman_bag_turn.plot_y[1][1], "b", label="single")
            plt.plot(multi_kalman_bag_turn.plot_y[1][0], multi_kalman_bag_turn.plot_y[1][1], "r", label="multi")
            np.savetxt("{}/multi_{}_comp_y1.csv".format(self.plot_folder,self.turn_bag), np.transpose([multi_kalman_bag_turn.plot_y[1][0],multi_kalman_bag_turn.plot_y[1][1]]),header='t y1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/multi_{}_comp_u1y1.png".format(self.img_folder, self.turn_bag), quality=95, dpi=300)

    def transform_rosbag(self, bag_name=None):
        input_bag_file = self.bag_folder + bag_name
        output_bag_file = self.trans_folder + "trans_" + bag_name
        if file_exists(input_bag_file) and not file_exists(output_bag_file):
            # set roslaunch logging
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)

            # run transform launch file
            cli_args = [
            "adapt_kalman",
            "transform_data.launch",
            "bag:={}".format(input_bag_file),
            "output:={}".format(output_bag_file)
            ]
            roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
            roslaunch_args = cli_args[2:]
            parent = roslaunch.parent.ROSLaunchParent(uuid, [(roslaunch_file[0], roslaunch_args),])
            parent.start()
            parent.spin()

    def run_bag(self,bagpath=None,alpha=1.0,beta=1.0,r1=1.0,r2=1.0, time_start=0.0,time_finish=np.inf,window_type=""):
        if file_exists(bagpath):
            adapt_kalman_bag = AdaptKalmanBag(bagpath=bagpath, alpha=alpha,beta=beta,r1=r1,r2=r2)
            adapt_kalman_bag.read_imu("/imu_out/data")
            adapt_kalman_bag.read_twist("/fake_encoder/twist")
            adapt_kalman_bag.upscale_twist()
            adapt_kalman_bag.run_filter()
            adapt_kalman_bag.add_plots(time_start,time_finish)
            return adapt_kalman_bag

    def export_line(self):
        line_bag = "trans_" + self.line_bag
        line_bag_path = self.trans_folder + line_bag
        alpha = self.alpha
        beta = self.beta
        r1 = self.r1
        r2 = self.r2
        t_s = self.line_start
        t_e = self.line_end
        if file_exists(line_bag_path):

            kalman_line_base = self.run_bag(alpha=alpha,beta=beta,r1=r1,r2=r2,bagpath=line_bag_path,time_start=t_s,time_finish=t_e)
            self.kalman_line_base = kalman_line_base
            kalman_line_plus10 = self.run_bag(alpha=alpha*1.1,beta=beta,r1=r1,r2=r2,bagpath=line_bag_path,time_start=t_s,time_finish=t_e)
            kalman_line_plus20 = self.run_bag(alpha=alpha*1.2,beta=beta,r1=r1,r2=r2,bagpath=line_bag_path,time_start=t_s,time_finish=t_e)
            kalman_line_minus10 = self.run_bag(alpha=alpha*0.9,beta=beta,r1=r1,r2=r2,bagpath=line_bag_path,time_start=t_s,time_finish=t_e)
            kalman_line_minus20 = self.run_bag(alpha=alpha*0.8,beta=beta,r1=r1,r2=r2,bagpath=line_bag_path,time_start=t_s,time_finish=t_e)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.suptitle("{}".format(line_bag_path))
            plt.subplot(411)
            plt.title("Comparison of different alphas in x")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")

            plt.plot(kalman_line_base.plot_x[0][0],kalman_line_base.plot_x[0][1], "b",label=alpha)
            np.savetxt("{}/alphas_{}_x0_base.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_base.plot_x[0][0],kalman_line_base.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus10.plot_x[0][0],kalman_line_plus10.plot_x[0][1], "r",label=alpha*1.1)
            np.savetxt("{}/alphas_{}_x0_plus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus10.plot_x[0][0],kalman_line_plus10.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus20.plot_x[0][0],kalman_line_plus20.plot_x[0][1], "m",label=alpha*1.2)
            np.savetxt("{}/alphas_{}_x0_plus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus20.plot_x[0][0],kalman_line_plus20.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus10.plot_x[0][0],kalman_line_minus10.plot_x[0][1], "g",label=alpha*0.9)
            np.savetxt("{}/alphas_{}_x0_minus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus10.plot_x[0][0],kalman_line_minus10.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus20.plot_x[0][0],kalman_line_minus20.plot_x[0][1], "k",label=alpha*0.8)
            np.savetxt("{}/alphas_{}_x0_minus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus20.plot_x[0][0],kalman_line_minus20.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(412)
            plt.title("Comparison of different alphas in v")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")

            plt.plot(kalman_line_base.plot_x[2][0],kalman_line_base.plot_x[2][1], "b",label=alpha)
            np.savetxt("{}/alphas_{}_x2_base.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_base.plot_x[2][0],kalman_line_base.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus10.plot_x[2][0],kalman_line_plus10.plot_x[2][1], "r",label=alpha*1.1)
            np.savetxt("{}/alphas_{}_x2_plus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus10.plot_x[2][0],kalman_line_plus10.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus20.plot_x[2][0],kalman_line_plus20.plot_x[2][1], "m",label=alpha*1.2)
            np.savetxt("{}/alphas_{}_x2_plus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus20.plot_x[2][0],kalman_line_plus20.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus10.plot_x[2][0],kalman_line_minus10.plot_x[2][1], "g",label=alpha*0.9)
            np.savetxt("{}/alphas_{}_x2_minus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus10.plot_x[2][0],kalman_line_minus10.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus20.plot_x[2][0],kalman_line_minus20.plot_x[2][1], "k",label=alpha*0.8)
            np.savetxt("{}/alphas_{}_x2_minus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus20.plot_x[2][0],kalman_line_minus20.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(413)
            plt.title("Comparison of different alphas, system input")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")

            plt.plot(kalman_line_base.plot_u[0][0],kalman_line_base.plot_u[0][1], "b",label=alpha)
            np.savetxt("{}/alphas_{}_u0_base.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_base.plot_u[0][0],kalman_line_base.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus10.plot_u[0][0],kalman_line_plus10.plot_u[0][1], "r",label=alpha*1.1)
            np.savetxt("{}/alphas_{}_u0_plus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus10.plot_u[0][0],kalman_line_plus10.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_plus20.plot_u[0][0],kalman_line_plus20.plot_u[0][1], "m",label=alpha*1.2)
            np.savetxt("{}/alphas_{}_u0_plus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_plus20.plot_u[0][0],kalman_line_plus20.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus10.plot_u[0][0],kalman_line_minus10.plot_u[0][1], "g",label=alpha*0.9)
            np.savetxt("{}/alphas_{}_u0_minus10.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus10.plot_u[0][0],kalman_line_minus10.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_line_minus20.plot_u[0][0],kalman_line_minus20.plot_u[0][1], "k",label=alpha*0.8)
            np.savetxt("{}/alphas_{}_u0_minus20.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_minus20.plot_u[0][0],kalman_line_minus20.plot_u[0][1]]),header='t u0', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(414)
            plt.title("Comparison of different alphas, system output")
            plt.xlabel("Time [s]")
            plt.ylabel("Acceleration [m/s2]")

            plt.plot(kalman_line_base.plot_y[0][0],kalman_line_base.plot_y[0][1], "b",label=alpha)
            np.savetxt("{}/alphas_{}_y0_base.csv".format(self.plot_folder,line_bag), np.transpose([kalman_line_base.plot_y[0][0],kalman_line_base.plot_y[0][1]]),header='t y0', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/alphas_{}.png".format(self.img_folder, line_bag), quality=95, dpi=300)

    def export_turn(self):
        turn_bag = "trans_" + self.turn_bag
        turn_bag_path = self.trans_folder + turn_bag
        beta = self.beta
        alpha = self.alpha
        r1 = self.r1
        r2 = self.r2
        t_s = self.turn_start
        t_e = self.turn_end
        if file_exists(turn_bag_path):

            kalman_turn_base = self.run_bag(beta=beta,alpha=alpha,r1=r1,r2=r2,bagpath=turn_bag_path,time_start=t_s,time_finish=t_e)
            self.kalman_turn_base = kalman_turn_base
            kalman_turn_plus10 = self.run_bag(beta=beta*1.01,alpha=alpha,r1=r1,r2=r2,bagpath=turn_bag_path,time_start=t_s,time_finish=t_e)
            kalman_turn_plus20 = self.run_bag(beta=beta*1.02,alpha=alpha,r1=r1,r2=r2,bagpath=turn_bag_path,time_start=t_s,time_finish=t_e)
            kalman_turn_minus10 = self.run_bag(beta=beta*0.99,alpha=alpha,r1=r1,r2=r2,bagpath=turn_bag_path,time_start=t_s,time_finish=t_e)
            kalman_turn_minus20 = self.run_bag(beta=beta*0.98,alpha=alpha,r1=r1,r2=r2,bagpath=turn_bag_path,time_start=t_s,time_finish=t_e)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.suptitle("{}".format(turn_bag_path))

            plt.subplot(511)
            plt.title("Comparison of different betas in x")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")

            plt.plot(kalman_turn_base.plot_x[0][0],kalman_turn_base.plot_x[0][1], "b",label=beta)
            np.savetxt("{}/betas_{}_x0_base.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_base.plot_x[0][0],kalman_turn_base.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus10.plot_x[0][0],kalman_turn_plus10.plot_x[0][1], "r",label=beta*1.01)
            np.savetxt("{}/betas_{}_x0_plus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus10.plot_x[0][0],kalman_turn_plus10.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus20.plot_x[0][0],kalman_turn_plus20.plot_x[0][1], "m",label=beta*1.02)
            np.savetxt("{}/betas_{}_x0_plus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus20.plot_x[0][0],kalman_turn_plus20.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus10.plot_x[0][0],kalman_turn_minus10.plot_x[0][1], "g",label=beta*0.99)
            np.savetxt("{}/betas_{}_x0_minus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus10.plot_x[0][0],kalman_turn_minus10.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus20.plot_x[0][0],kalman_turn_minus20.plot_x[0][1], "k",label=beta*0.98)
            np.savetxt("{}/betas_{}_x0_minus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus20.plot_x[0][0],kalman_turn_minus20.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(512)
            plt.title("Comparison of different betas in y")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m/s]")

            plt.plot(kalman_turn_base.plot_x[1][0],kalman_turn_base.plot_x[1][1], "b",label=beta)
            np.savetxt("{}/betas_{}_x2_base.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_base.plot_x[1][0],kalman_turn_base.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus10.plot_x[1][0],kalman_turn_plus10.plot_x[1][1], "r",label=beta*1.01)
            np.savetxt("{}/betas_{}_x2_plus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus10.plot_x[1][0],kalman_turn_plus10.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus20.plot_x[1][0],kalman_turn_plus20.plot_x[1][1], "m",label=beta*1.02)
            np.savetxt("{}/betas_{}_x2_plus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus20.plot_x[1][0],kalman_turn_plus20.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus10.plot_x[1][0],kalman_turn_minus10.plot_x[1][1], "g",label=beta*0.99)
            np.savetxt("{}/betas_{}_x2_minus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus10.plot_x[1][0],kalman_turn_minus10.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus20.plot_x[1][0],kalman_turn_minus20.plot_x[1][1], "k",label=beta*0.98)
            np.savetxt("{}/betas_{}_x2_minus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus20.plot_x[1][0],kalman_turn_minus20.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(513)
            plt.title("Comparison of different betas in phi")
            plt.xlabel("Time [s]")
            plt.ylabel("Phi [degrees]")

            plt.plot(kalman_turn_base.plot_x[3][0],kalman_turn_base.plot_x[3][1], "b",label=beta)
            np.savetxt("{}/betas_{}_x2_base.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_base.plot_x[3][0],kalman_turn_base.plot_x[3][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus10.plot_x[3][0],kalman_turn_plus10.plot_x[3][1], "r",label=beta*1.01)
            np.savetxt("{}/betas_{}_x2_plus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus10.plot_x[3][0],kalman_turn_plus10.plot_x[3][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus20.plot_x[3][0],kalman_turn_plus20.plot_x[3][1], "m",label=beta*1.02)
            np.savetxt("{}/betas_{}_x2_plus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus20.plot_x[3][0],kalman_turn_plus20.plot_x[3][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus10.plot_x[3][0],kalman_turn_minus10.plot_x[3][1], "g",label=beta*0.99)
            np.savetxt("{}/betas_{}_x2_minus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus10.plot_x[3][0],kalman_turn_minus10.plot_x[3][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus20.plot_x[3][0],kalman_turn_minus20.plot_x[3][1], "k",label=beta*0.98)
            np.savetxt("{}/betas_{}_x2_minus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus20.plot_x[3][0],kalman_turn_minus20.plot_x[3][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.subplot(514)
            plt.title("Comparison of different betas, system input")
            plt.xlabel("Time [s]")
            plt.ylabel("Turn velocity [rad/s]")

            plt.plot(kalman_turn_base.plot_u[1][0],kalman_turn_base.plot_u[1][1], "b",label=beta)
            np.savetxt("{}/betas_{}_u0_base.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_base.plot_u[1][0],kalman_turn_base.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus10.plot_u[1][0],kalman_turn_plus10.plot_u[1][1], "r",label=beta*1.01)
            np.savetxt("{}/betas_{}_u0_plus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus10.plot_u[1][0],kalman_turn_plus10.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_plus20.plot_u[1][0],kalman_turn_plus20.plot_u[1][1], "m",label=beta*1.02)
            np.savetxt("{}/betas_{}_u0_plus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_plus20.plot_u[1][0],kalman_turn_plus20.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus10.plot_u[1][0],kalman_turn_minus10.plot_u[1][1], "g",label=beta*0.99)
            np.savetxt("{}/betas_{}_u0_minus10.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus10.plot_u[1][0],kalman_turn_minus10.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')

            plt.plot(kalman_turn_minus20.plot_u[1][0],kalman_turn_minus20.plot_u[1][1], "k",label=beta*0.98)
            np.savetxt("{}/betas_{}_u0_minus20.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_minus20.plot_u[1][0],kalman_turn_minus20.plot_u[1][1]]),header='t u1', comments='# ',delimiter=' ', newline='\n')

            plt.legend()

            plt.subplot(515)
            plt.title("Comparison of different betas, system output")
            plt.xlabel("Time [s]")
            plt.ylabel("Turn velocity [rad/s]")

            plt.plot(kalman_turn_base.plot_y[1][0],kalman_turn_base.plot_y[1][1], "b",label=beta)
            np.savetxt("{}/betas_{}_y1_base.csv".format(self.plot_folder,turn_bag), np.transpose([kalman_turn_base.plot_y[1][0],kalman_turn_base.plot_y[0][1]]),header='t y1', comments='# ',delimiter=' ', newline='\n')
            plt.legend()

            plt.savefig("{}/betas_{}.png".format(self.img_folder, turn_bag), quality=95, dpi=300)

    def run_ekf(self, bag_name=None):
        rosbag = self.bag_folder + bag_name
        output = self.ekf_folder + "ekf_" + bag_name
        if file_exists(rosbag) and not file_exists(output):
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)

            # run transform launch file
            cli_args = [
            "adapt_kalman",
            "ekf.launch",
            "bag:={}".format(rosbag),
            "output:={}".format(output),
            "r1:={}".format(self.r1),
            "r2:={}".format(self.r2),
            "alpha:={}".format(self.alpha),
            "beta:={}".format(self.beta)
            ]
            roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
            roslaunch_args = cli_args[2:]
            parent = roslaunch.parent.ROSLaunchParent(uuid, [(roslaunch_file[0], roslaunch_args),])
            parent.start()
            parent.spin()

    def export_ekf(self, bagname=None, start_t=0,end_t=np.inf):
        bagpath = self.ekf_folder + "ekf_" + bagname
        if file_exists(bagpath):
            ekf_reader = EKFReader(bagpath)
            ekf_reader.read_odom("/odometry/filtered")
            ekf_reader.add_plots(start_t,end_t)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.suptitle("{}".format(bagpath))

            plt.subplot(411)
            plt.title("EKF in x")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.plot(ekf_reader.plot_x[0][0],ekf_reader.plot_x[0][1], "b")
            np.savetxt("{}/ekf_{}_x0.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[0][0],ekf_reader.plot_x[0][1]]),header='t x0', comments='# ',delimiter=' ', newline='\n')

            plt.subplot(412)
            plt.title("EKF in y")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.plot(ekf_reader.plot_x[1][0],ekf_reader.plot_x[1][1], "b")
            np.savetxt("{}/ekf_{}_x1.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[1][0],ekf_reader.plot_x[1][1]]),header='t x1', comments='# ',delimiter=' ', newline='\n')

            plt.subplot(413)
            plt.title("EKF in v")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.plot(ekf_reader.plot_x[2][0],ekf_reader.plot_x[2][1], "b")
            np.savetxt("{}/ekf_{}_x2.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[2][0],ekf_reader.plot_x[2][1]]),header='t x2', comments='# ',delimiter=' ', newline='\n')

            plt.subplot(414)
            plt.title("EKF in Psi")
            plt.xlabel("Time [s]")
            plt.ylabel("Phi [degrees]")
            plt.plot(ekf_reader.plot_x[3][0],ekf_reader.plot_x[3][1], "b")
            np.savetxt("{}/ekf_{}_x3.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[3][0],ekf_reader.plot_x[3][1]]),header='t x3', comments='# ',delimiter=' ', newline='\n')

            plt.savefig(self.img_folder+ "ekf_" + bagname.split("/")[0]+ "_states.png", quality=95, dpi=300)

            plt.figure(self.fig_count, figsize=self.fig_dims)
            self.fig_count += 1
            plt.title("EKF in xy")
            plt.xlabel("Distance [x]")
            plt.ylabel("Distance [y]")
            plt.plot(ekf_reader.plot_x[0][0],ekf_reader.plot_x[1][0], "b")
            np.savetxt("{}/ekf_{}_xy.csv".format(self.plot_folder, bagname), np.transpose([ekf_reader.plot_x[0][0],ekf_reader.plot_x[1][0]]),header='x y', comments='# ',delimiter=' ', newline='\n')

            plt.savefig(self.img_folder+ "ekf_" + bagname.split("/")[0]+ "_xy.png", quality=95, dpi=300)

if __name__ == '__main__':

    exporter = Exporter()
