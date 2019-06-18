#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
plt.subplots_adjust(hspace=0.5)
import matplotlib2tikz
from simple_kalman import SimpleKalman

class SimpleKalmanSim:

    def __init__(self, N=200,
                       sim_time=5.0,
                       peak_vel=0.14,
                       ratio=1/3,
                       window="sig",
                       window_size=4,
                       adapt=False):
        self.t = np.linspace(0,sim_time,N)
        self.vel = self.set_vel(sim_time,peak_vel,N)
        self.accel = self.set_accel(sim_time,N)
        self.kalman = SimpleKalman(ratio,window,window_size,adapt)

    def set_vel(self,sim_time,peak_vel, N):
        t = self.t
        box_function = np.piecewise(t, [t<0.1*sim_time,t>1,t>0.9*sim_time], [0,peak_vel,0])
        return box_function

    def plot_all(self):
        plt.figure(1)

        xticks = len(self.kalman.plot_t)

        plt.subplot(611)
        plt.title("Simulated robot velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.t,self.vel)

        plt.subplot(612)
        plt.title("Robot distance")
        plt.xlabel("Time in s")
        plt.ylabel("Distance in m")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t,self.kalman.plot_y)

        plt.subplot(613)
        plt.title("Robot velocity post")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t,self.kalman.plot_v)

        plt.subplot(614)
        plt.title("Robot acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        #plt.xticks(np.arange(0, xticks, step=0.2))
        plt.plot(self.kalman.plot_t, self.kalman.plot_a)

        # plt.subplot(615)
        # plt.title("test")
        # fill = len(self.kalman.plot_t) - len(self.kalman.test_a)
        # full_test_array = np.insert(self.kalman.test_a,0, np.zeros(fill))
        # plt.plot(self.kalman.plot_t,full_test_array)

        plt.subplot(615)
        plt.title("Coeff")
        #plt.xticks(np.arange(0, len(self.kalman.plot_t[30:]), step=0.2))
        fill = len(self.kalman.plot_t) - len(self.kalman.coeff_a)
        full_coeff_array = np.insert(self.kalman.coeff_a,0, np.ones(fill))
        plt.plot(self.kalman.plot_t,full_coeff_array)

        plt.subplot(616)
        plt.title("Ratio")
        #plt.xticks(np.arange(0, len(self.kalman.plot_t[30:]), step=0.2))
        fill = len(self.kalman.plot_t) - len(self.kalman.ratio_a)
        full_ratio_array = np.insert(self.kalman.ratio_a, 0, np.full((fill),self.kalman.ratio))
        plt.plot(self.kalman.plot_t,full_ratio_array)

        plt.show()

    def export_plots(self):
        plt.figure(1)
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.t,self.vel)
        matplotlib2tikz.save("plots/input_vel_sim.tex",figureheight='4cm', figurewidth='6cm')

        plt.figure(2)
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.t,self.accel)
        matplotlib2tikz.save("plots/input_accel_sim.tex",figureheight='4cm', figurewidth='6cm' )

        plt.figure(3)
        plt.xlabel("Time in s")
        plt.ylabel("Distance in m")
        plt.plot(self.kalman.plot_t,self.kalman.plot_y)
        matplotlib2tikz.save("plots/robot_dist_sim.tex",figureheight='3cm', figurewidth='14cm' )

        plt.figure(4)
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in m/s")
        plt.plot(self.kalman.plot_t,self.kalman.plot_v)
        plt.ticklabel_format(axis='both', style='plain')
        matplotlib2tikz.save("plots/robot_vel_sim.tex",figureheight='3cm', figurewidth='14cm' )

        plt.figure(5)
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")
        plt.plot(self.kalman.plot_t, self.kalman.plot_a)
        matplotlib2tikz.save("plots/robot_accel_sim.tex",figureheight='3cm', figurewidth='14cm' )

    def get_noise_moving(self, peak_coeff):
        noise_moving = []
        for x in self.vel:
            # fill staying still with zeros
            if abs(x) < 0.01:
                noise_moving.append(0.0)
            else:
                noise_moving.append(np.random.normal(0,x*peak_coeff))

        return noise_moving

    def set_accel(self,sim_time,N):
        accel = 0
        sigma = 0.01
        x = np.linspace(-sim_time/2.0,sim_time/2.0,N)
        gauss = np.exp(-(x/sigma)**2/2)
        conv = np.convolve(self.vel,gauss/gauss.sum(), mode="same")
        grad = 50*np.gradient(conv)
        grad_shift = 0.7 * np.roll(grad,10)
        noise_still = np.random.normal(0,0.05,N)
        noise_moving = self.get_noise_moving(3)
        offset = 0.3

        accel += grad
        #accel += grad_shift
        accel += noise_still
        accel += noise_moving
        accel += offset

        return accel

    def run_filter(self):
        for u,t in zip(zip(self.vel,self.accel), np.diff(self.t)):
            self.kalman.filter(u,t)

if __name__ == '__main__':
    simple_kalman_sim = SimpleKalmanSim()
    simple_kalman_sim.run_filter()
    simple_kalman_sim.plot_all()
