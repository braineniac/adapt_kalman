import numpy as np
from matplotlib import pyplot as plt

class SimpleKalman:
    def __init__(self):

        self.small_val = np.exp(-20)
        self.delta_t = 0.0

        self.sum_t = 0.0
        self.plot_t = []
        self.plot_y = []
        self.plot_v = []
        self.plot_a = []

        self.u = [[],[]]
        self.t = []

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
        #self.y_k = self.C_k.dot(self.x_k_post)

    def extrapolate(self):
        self.x_k_extr = self.phi_k.dot(self.x_k_post) + self.gamma_k.dot(self.u_k)
        self.P_k_extr = self.phi_k.dot(self.P_k_post).dot(self.phi_k.T) + self.G_k.dot(self.Q_k).dot(self.G_k.T)

        # update for next iteration
        self.x_k_pre = self.x_k_extr
        self.P_k_pre = self.P_k_extr

    def load_array(self):
        data = np.load("/home/dan/ros/src/simple_kalman/numpy/straight_dmp.npy")

        u,t = zip(data[0],data[1])
        for u0,u1,t0,t1 in zip(u[0],u[1],t[0],t[1]):
            print(u)
            self.u[0].append(u0)
            self.t.append(t1)
            self.u[1].append(u1)
            self.t.append(t1)

    def kalman_filter(self):
        for u0,u1,t in zip(np.array(self.u[0]),np.array(self.u[1]),np.diff(np.array(self.t))):
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
            self.plot_y.append(self.x_k_pre[0])
            self.plot_v.append(self.x_k_pre[1])
            self.plot_a.append(-u1)
            self.plot_t.append(self.sum_t)
            #self.print_debug()
        self.plot_output()

    def plot_output(self):
        plt.subplot(131)
        plt.title("Kalman filter x state")
        plt.ylabel("Distance travelled in m")
        plt.xlabel("Time in sec")
        plt.plot(self.plot_t, self.plot_y)

        plt.subplot(132)
        plt.title("Kalman filter ẋ state")
        plt.xlabel("Time in sec")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.plot_t,self.plot_v)

        plt.subplot(133)
        plt.title("Imu data")
        plt.xlabel("Time in sec")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.plot_t,self.plot_a)
        plt.show()

    def print_debug(self):

        #print("y={}".format(self.plot_y))
        #print("t={}".format(self.plot_t))
        np.set_printoptions(precision=32)
        #print("State: {}".format(self.x_k_pre))
        #print("Covariance: {}".format(self.P_k_pre))
        #print("Gamma: {}".format(self.gamma_k))
        #print("C: {}".format(self.C_k))
        #print("Kalman gain: {}".format(self.L_k))
        #print("Posteriori state: {}".format(self.x_k_post))
        #print("Posteriori covariance: {}".format(self.P_k_post))
        #print("Extrapolated state: {}".format(self.x_k_extr))
        #print("Extrapolated covariance: {}".format(self.P_k_extr))
        #print("Output: {}".format(self.y_k))
        #print("u_k: {}".format(self.u_k))

if __name__ == '__main__':
    simple_kalman_np = SimpleKalman()
    simple_kalman_np.load_array()
    simple_kalman_np.kalman_filter()
