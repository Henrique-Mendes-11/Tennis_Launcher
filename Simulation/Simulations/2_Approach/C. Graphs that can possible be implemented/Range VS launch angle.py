import json
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import vpython as vp


# Performance variables
rhythm = 0
force = 0
initial_angle_ground = 18 #º
initial_angle_x = 0 #º
initial_angle_y = 20 #º
initial_velocity = 17 #m/s
rotations = 0
a_gravity = 9.8 #m/s^2
air_friction = 0
avg_dist_cp = 0.95 #m 

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

# Court variables
court_length = 23.77 #m
doubles_court_width = 10.97 #m  # | --->
singles_court_width = 8.23 #m # | ->
net_height_middle = 0.915 #m
net_height_sides = 1.065 #m


##### Graph - Range as a function of launch angle

g1 = vp.graph(title='Range VS launch angle', xtitle='theta [deg]', ytitle='range [m]', width = 500, height = 250)
f1 = vp.gcurve(color=vp.color.yellow)

def prange(ttheta, v0, r0):
    g = vp.vector(0,-9.8,0)
    m = ball_mass_kg
    p = m*v0*vp.vector(mt.cos(ttheta),mt.sin(ttheta),0)
    t = 0
    dt = 0.001
    r = r0
    while r.y >= 0:
        F = m*g
        p = p + F*dt
        r = r + p*dt/m
        t = t + dt
    return (r.x)

def find_max(range_theta_matrix):
    max_range = 0
    max_range_angle = 0
    for i in range(len(range_theta_matrix)):
        if range_theta_matrix[i][0] > max_range:
            max_range = range_theta_matrix[i][0]
            max_range_angle = range_theta_matrix[i][1]
    return max_range, max_range_angle

#theta = mt.radians(initial_angle_ground) #initial_angle
theta = mt.radians(1) 
dtheta = mt.radians(1)
v0 = initial_velocity    
r = vp.vector(0,avg_dist_cp,0)

range_theta_matrix = []

while theta < mt.radians(90):
    trange = prange(theta, v0, r)
    f1.plot(mt.degrees(theta), trange)
    range_theta_matrix.append([trange, mt.degrees(theta)])
    theta = theta + dtheta

print("Max range: ", find_max(range_theta_matrix)[0], "m")
print("Max range angle: ", find_max(range_theta_matrix)[1], "º")

