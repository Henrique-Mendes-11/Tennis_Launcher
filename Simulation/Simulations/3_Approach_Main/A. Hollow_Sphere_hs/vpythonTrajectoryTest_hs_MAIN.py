# 3D ball trajectroy using vpython 
"""FIRST RUN FILE: LagrangianEq_wXYZ_hs_MAIN.py to generate the data file"""


import vpython as vp
from vpython import *
import numpy as np
import os

# Tennis Ball variables
ball_mass_g = 58.3
ball_mass_kg = ball_mass_g / 1000
ball_diameter_cm = 6.5
ball_radius_cm = ball_diameter_cm / 2
ball_radius_m = ball_radius_cm / 100
ball_racket_contact_time_ms = 30
ball_pressure_kg = 8.165
ball_forward_deformation_cm = 0.6
ball_return_deformation_cm = 0.95
ball_air_avg_velocity_kmh = 125.5
ball_air_avg_velocity_ms = 34.8
ball_sphere_drag_coefficient = 0.47

R = 0.3 #Radius of the ball for the animation only

# Court variables (m)
court_length = 23.77 
doubles_court_width = 10.97 
singles_court_width = 8.23 
net_height_middle = 0.915
net_height_sides = 1.065

# Import data
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '3DBallTest_hs_MAIN.npy')
x, y, z, alpha, beta, gamma, t, v0x, v0y, v0z, w0x, w0y, w0z = np.load(file_path)


# Create the scene with the ball and court
ball = vp.sphere(color = color.yellow, radius = R, make_trail=True, retain=20)
ground = vp.box(pos=vp.vector(11.8,-0.25/2,0), size=vp.vector(court_length,0.25,singles_court_width), color = vp.color.green)


# Velocities 
v0x_value = v0x[0]
v0y_value = v0y[0]
v0z_value = v0z[0]
ball.v = vp.vector(v0x_value, v0z_value, v0y_value)

w0x_value = w0x[0]
w0y_value = w0y[0]
w0z_value = w0z[0]
w = vp.vector(w0x_value, w0z_value, w0y_value)


# Initial position of the ball
ball.pos = vector(x[0], z[0], y[0])
#balldot = sphere(pos = vector(x[0], z[0]+R, y[0]), radius = 0.1, color = color.red)


# Start of the simulation based on the data downloaded
print('Start')
i = 0
while True:
    rate(500)
    i = i + 1
    i = i % len(x)
    if z[i] > 0:
        ball.pos = vector(x[i], z[i], y[i])
        #balldot.pos = vector(x[i], z[i]+R, y[i])