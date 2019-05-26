#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from simple_kalman import SimpleKalman

class SimpleKalmanSim:

    def __init__(self, N=1000, sim_time=5.0, peak_vel=1.14):
        self.t = np.linspace(0,sim_time,N)
        self.vel = self.set_vel(sim_time,peak_vel,N)
        self.accel = self.set_accel(sim_time,N)
        self.kalman = SimpleKalman(1000,5)

    def set_vel(self,sim_time,peak_vel, N):
        t = self.t
        box_function = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        return box_function

    def plot_all(self):
        plt.figure(2)

        plt.subplot(211)
        plt.title("Simulated robot velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.t,self.vel)

        plt.subplot(212)
        plt.title("Simulated robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t,self.accel)

        plt.show()

    def set_accel(self,sim_time,N):
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.vel,gauss/gauss.sum(), mode="same")
        grad = 100*np.gradient(conv)
        noise = np.random.normal(0,6,N)
        offset = 1.5
        return grad + noise + offset

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.kalman.filter(u,t)
        self.kalman.plot_all()
        self.plot_all()

if __name__ == '__main__':
    simple_kalman_sim = SimpleKalmanSim()
    simple_kalman_sim.run_filter()
