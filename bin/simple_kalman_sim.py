#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from simple_kalman import SimpleKalman

class SimpleKalmanSim:

    def __init__(self, N=1000, sim_time=10.0,sigma=0.1, peak_vel=1.14):
        t = np.linspace(0,sim_time,N)
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        self.t = t
        self.x = x
        self.vel = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        self.gauss = np.exp(-(x/sigma)**2/2)
        self.conv = None
        self.accel = None
        self.kalman = SimpleKalman(100,5)

    def plot_all(self):
        plt.figure(1)

        plt.subplot(221)
        plt.title("Simulated robot velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.t,self.vel)

        plt.subplot(222)
        plt.title("Gauss used for the convolution")
        plt.plot(self.x, self.gauss)

        plt.subplot(223)
        plt.title("Result of the convolution")
        plt.plot(self.t,self.conv)

        plt.subplot(224)
        plt.title("Simulated robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t,self.accel)

        plt.show()

    def simulate_input(self):
        self.conv = np.convolve(self.vel,self.gauss/self.gauss.sum(), mode="same")
        self.accel = 100*np.gradient(self.conv)
        #self.plot_all()

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.kalman.filter(u,t)
        self.kalman.plot_all()

if __name__ == '__main__':
    simple_kalman_sim = SimpleKalmanSim()
    simple_kalman_sim.simulate_input()
    simple_kalman_sim.run_filter()
