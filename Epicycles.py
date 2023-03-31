#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:44:39 2023

@author: ezequiel
"""

# ------------------------------------------------------------------------------
# Program:
# -------
#
# This program computes the trajectory of several planets using the approximate
# description given by the epicycles modes of the trajectory.
# 
# The output is an animation that illustrate the motion. 
# 
#
# ------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

def epicycle(t, k, L, omega, theta_0):
    """
    Returns the x and y coordinates of a point on an epicycle with radius k,
    center at (L, 0), and angular velocity omega, starting at angle theta_0.
    """
    theta = omega * t + theta_0
    x = L + k * np.cos(theta)
    y = k * np.sin(theta)
    return x, y

def planetary_orbit(t, k, L, omega, theta_0):
    """
    Returns the x and y coordinates of a planet in orbit based on the epicycle model.
    """
    x, y = 0, 0
    for i in range(len(k)):
        x_, y_ = epicycle(t, k[i], L[i], omega[i], theta_0[i])
        x += x_
        y += y_
    return x, y

# Define the parameters of the epicycles
k = [1, 0.5, 0.2]
L = [0, 1, 2]
omega = [1, 1.5, 2]
theta_0 = [0, 0, 0]

# Define the time range to simulate the orbit
t = np.linspace(0, 10, 1000)

# Calculate the x and y coordinates of the planet's orbit
x, y = [], []
for i in range(len(t)):
    x_, y_ = planetary_orbit(t[i], k, L, omega, theta_0)
    x.append(x_)
    y.append(y_)

# Plot the result
plt.plot(x, y)
plt.title("Simulated Planetary Orbit")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


##############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def epicycle(t, k, L, omega, theta_0):
    """
    Returns the x and y coordinates of a point on an epicycle with radius k,
    center at (L, 0), and angular velocity omega, starting at angle theta_0.
    """
    theta = omega * t + theta_0
    x = L + k * np.cos(theta)
    y = k * np.sin(theta)
    return x, y

def planetary_orbit(t, k, L, omega, theta_0):
    """
    Returns the x and y coordinates of a planet in orbit based on the epicycle model.
    """
    x, y = 0, 0
    for i in range(len(k)):
        x_, y_ = epicycle(t, k[i], L[i], omega[i], theta_0[i])
        x += x_
        y += y_
    return x, y

# Define the parameters of the epicycles
k = [1, 0.5, 0.2]
L = [0, 1, 2]
omega = [1, 1.5, 2]
theta_0 = [0, 0, 0]

# Define the time range to simulate the orbit
t = np.linspace(0, 10, 1000)

# Initialize the figure for the animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], lw=2)

# Define the animation function
def animate(i):
    x_, y_ = planetary_orbit(t[i], k, L, omega, theta_0)
    line.set_data(x_, y_)
    return line,

# Animate the planetary orbit
ani = FuncAnimation(fig, animate, frames=len(t), blit=True)
ani.save("planetary_orbit.mp4")

#ani = FuncAnimation(fig, animate, frames=len(t), blit=True)
#plt.title("Simulated Planetary Orbit")
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show()

#########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the number of points in the orbit
n_points = 1000

# Define the planet's parameters
r = 100
d = 200

# Define the x and y coordinates of the center of the orbit
x = np.zeros(n_points)
y = np.zeros(n_points)

# Define the parameters of the epicycle
a = 50
b = 30
omega = 2 * np.pi / 100

# Simulate the planet's orbit using epicycles
for i in range(n_points):
    x[i] = x[i-1] + a * np.cos(omega * i)
    y[i] = y[i-1] + b * np.sin(omega * i)

# Plot the results
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=10)

def update(frame):
    scat.set_offsets(np.c_[x[:frame], y[:frame]])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=n_points, interval=20, blit=True)
#ani.save("planetary_orbit.mp4")

plt.show()





#########################################################################
#########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Number of planets
N = 3

# Orbit parameters
R = np.array([50, 70, 90])
d = np.array([0, 0, 0])
w = np.array([0.01, 0.03, 0.05])

# Starting angles for each planet
theta = np.zeros(N)

fig, ax = plt.subplots()
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect('equal')

planets = []
for i in range(N):
    planet, = ax.plot([], [], 'ro', markersize=5)
    planets.append(planet)

def update(frame):
    global theta
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = d[i] + R[i]*np.cos(w[i]*frame + theta[i])
        y[i] = R[i]*np.sin(w[i]*frame + theta[i])
        for j in range(i):
            x[i] += R[j]*np.cos(w[j]*frame + theta[j])
            y[i] += R[j]*np.sin(w[j]*frame + theta[j])
    for i in range(N):
        planets[i].set_data(x[i], y[i])
    theta += 0.01
    return planets

ani = animation.FuncAnimation(fig, update, frames=1000, interval=20, blit=True)
plt.show()


f = r"animation.gif"
writergif = animation.PillowWriter(fps=30)
ani.save(f, writer=writergif)
video = ani.to_html5_video()
