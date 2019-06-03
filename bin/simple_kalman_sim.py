#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
plt.subplots_adjust(hspace=0.5)
import matplotlib2tikz
from simple_kalman import SimpleKalman

class SimpleKalmanSim:

    def __init__(self, N=1000, sim_time=5.0, peak_vel=0.14):
        self.t = np.linspace(0,sim_time,N)
        self.vel = self.set_vel(sim_time,peak_vel,N)
        self.accel = self.set_accel(sim_time,N)
        self.kalman = SimpleKalman(1,3)

    def set_vel(self,sim_time,peak_vel, N):
        t = self.t
        box_function = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        return box_function

    def plot_all(self):
        plt.figure(1)
        #plt.title("Simulated robot velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.t,self.vel)
        matplotlib2tikz.save("plots/input_vel_sim.tex",figureheight='4cm', figurewidth='6cm')

        plt.figure(2)
        #plt.title("Simulated robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t,self.accel)
        matplotlib2tikz.save("plots/input_accel_sim.tex",figureheight='4cm', figurewidth='6cm' )

        plt.figure(3)
        #plt.title("Robot distance")
        plt.xlabel("Time in s")
        plt.ylabel("Distance in m")
        plt.plot(self.kalman.plot_t,self.kalman.plot_y)
        matplotlib2tikz.save("plots/robot_dist_sim.tex",figureheight='3cm', figurewidth='14cm' )

        plt.figure(4)
        #plt.title("Robot velocity post")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.kalman.plot_t,self.kalman.plot_v_post)
        plt.ticklabel_format(axis='both', style='plain')
        matplotlib2tikz.save("plots/robot_vel_sim.tex",figureheight='3cm', figurewidth='14cm' )

        plt.figure(5)
        #plt.title("Robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.kalman.plot_t, self.kalman.plot_a)
        matplotlib2tikz.save("plots/robot_accel_sim.tex",figureheight='3cm', figurewidth='14cm' )

        #plt.show()

    def set_accel(self,sim_time,N):
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.vel,gauss/gauss.sum(), mode="same")
        grad = 50*np.gradient(conv)
        grad_shift = 0.7 * np.roll(grad,10)
        noise = np.random.normal(0,0.2,N)
        offset = 0.3
        return grad + grad_shift + noise + offset

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.kalman.filter(u,t)

if __name__ == '__main__':
    simple_kalman_sim = SimpleKalmanSim()
    simple_kalman_sim.run_filter()
    simple_kalman_sim.plot_all()
