import numpy as np
from matplotlib import pyplot as plt
import rosbag
import argparse

class SimpleKalman:
    def __init__(self, path_bag):
        self.path_bag = path_bag

        self.small_val = np.exp(-20)
        self.delta_t = 0.0

        self.sum_t = 0.0
        self.plot_t = []
        self.plot_y = []
        self.plot_v_post = []
        self.plot_v_pre = []
        self.plot_a = []
        self.plot_accel = []

        self.u = [[],[]]
        self.t = []
        self.input = []

        self.L_k = np.zeros((2,2))          # Kalman gain
        self.P_k_pre = np.random.normal(self.small_val,1.0,(2,2))      # A priori covariance
        self.P_k_post = np.zeros((2,2))     # A posteriori covariance
        self.C_k = np.zeros(2)
        self.x_k_post = np.zeros((2,1))
        self.x_k_pre = np.zeros((2,1))
        self.x_k_extr = np.zeros((2,1))     # extrapolated state
        self.P_k_extr = np.zeros((2,1))     # extrapolated covariance
        self.phi_k = np.zeros((2,2))
        self.D_k = np.zeros((2,2))
        self.gamma_k = np.zeros((2,2))      # control matrix
        self.R_k = np.zeros((2,2))
        self.Q_k = np.zeros((2,2))
        self.G_k = np.zeros((2,2))
        self.H_k = np.zeros((2,2))
        self.y_k = np.zeros((2,1))          # output
        self.u_k = np.zeros((2,1))          # control vector

        # set initial covariance
        #imu_stdev = (400/1000000) * 9.80655
        #fake_enc_stdev = (400/1000000) * 9.80655 / 100.0
        imu_stdev = 0.5
        fake_enc_stdev = 0.5
        self.R_k[0][0] = self.small_val
        self.R_k[1][1] = imu_stdev*imu_stdev
        self.Q_k[0][0] = self.small_val
        self.Q_k[1][1] = fake_enc_stdev*fake_enc_stdev

        self.H_k[1][1] = 1
        self.G_k[1][1] = 1

    def time_update(self):
        self.D_k = np.array([[0,0],[0,self.delta_t]])
        self.phi_k = np.array([[1,self.delta_t],[0,0]])
        self.gamma_k = np.array([[0,0],[0,1]])
        self.C_k = np.array([[0,0],[1/2*self.delta_t*self.delta_t,self.delta_t]])

    def set_gain(self):
        E = self.C_k.dot(self.P_k_pre).dot(self.C_k.T) + self.H_k.dot(self.Q_k).dot(self.H_k.T) + self.R_k
        self.L_k = self.P_k_pre.dot(self.C_k.T).dot(np.linalg.inv(E))

    def update(self):
        F = self.y_k - self.C_k.dot(self.x_k_pre) - self.D_k.dot(self.u_k)
        self.x_k_post = self.x_k_pre + self.L_k.dot(F)
        self.P_k_post = (np.identity(2) - self.L_k.dot(self.C_k)).dot(self.P_k_pre)

    def extrapolate(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T) + self.G_k.dot(self.Q_k).dot(self.G_k.T)

        # update for next iteration
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr

    def sort2np(self):
        sorted_input = sorted(self.input,key=lambda elem : elem[0])
        for t,u0,u1 in sorted_input:
            self.u[0].append(u0)
            self.u[1].append(u1)
            self.t.append(t)

    def kalman_filter(self):
        for u0,u1,t in zip(self.u[0],self.u[1], np.diff(np.array(self.t))):
            # exclude weird data at the end of the rosbag
            if t>0.5:
                continue
            self.plot_v_pre.append(self.x_k_pre[1])
            self.delta_t = t
            self.u_k[0] = 0
            self.u_k[1] = u0
            self.y_k[0] = 0
            self.y_k[1] = -u1
            self.time_update()
            self.set_gain()
            self.update()
            self.extrapolate()
            self.sum_t = self.sum_t + t
            self.plot_y.append(self.x_k_post[0])
            self.plot_v_post.append(self.x_k_post[1])
            self.plot_a = np.append(self.plot_a,[u1])
            self.plot_t.append(self.sum_t)

        self.plot_accel = self.plot_a[1:-2] + self.plot_a[0:-3] + self.plot_a[2:-1]
        #self.plot_a = np.convolve(self.plot_a,np.random.normal(0.14,0.05,np.ndarray.size(self.plot_a)))
        self.plot_output()

    def plot_output(self):
#        plt.subplot(131)
        plt.figure(1)
        plt.title("Kalman filter x state")
        plt.ylabel("Distance travelled in m")
        plt.xlabel("Time in sec")
        plt.plot(self.plot_t, self.plot_y)

#        plt.subplot(132)
#        plt.figure(2)
#        plt.title("Kalman filter ẋ before state")
#        plt.xlabel("Time in sec")
#        plt.ylabel("Velocity in m/s")
#        plt.plot(self.plot_t,self.plot_v_pre)

#        plt.figure(3)
#        plt.title("Kalman filter ẋ after state")
#        plt.xlabel("Velocity in m/s")
#        plt.xlabel("Time in sec")
#        plt.plot(self.plot_t,self.plot_v_post)

#        plt.subplot(133)
        plt.figure(4)
        plt.title("Imu data")
        plt.xlabel("Time in sec")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.plot_t[1:-2],self.plot_accel)
        plt.show()

    def read_bag(self):
        bag = rosbag.Bag(self.path_bag)
        imu_msgs = bag.read_messages(topics=['/imu/data_raw'])
        twist_msgs = bag.read_messages(topics=['/fake_wheel/twist'])
        last_vel = 0.0
        last_accel = 0.0
        for imu_msg,twist_msg in zip(imu_msgs,twist_msgs):
            imu_t = imu_msg.message.header.stamp.to_sec()
            twist_t = twist_msg.message.header.stamp.to_sec()
            imu_a_x = imu_msg.message.linear_acceleration.x
            twist_l_x = twist_msg.message.twist.twist.linear.x
            if last_vel != twist_l_x:
                self.input.append((twist_t,twist_l_x,last_accel))
                last_vel = twist_l_x
            if last_accel != imu_a_x:
                self.input.append((imu_t,last_vel,imu_a_x))
                last_accel = imu_a_x
        self.sort2np()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process rosbag through a kalman filter")
    parser.add_argument("bag", help="Rosbag path")
    args = parser.parse_args()
    simple_kalman_np = SimpleKalman(args.bag)
    simple_kalman_np.read_bag()
    simple_kalman_np.kalman_filter()
