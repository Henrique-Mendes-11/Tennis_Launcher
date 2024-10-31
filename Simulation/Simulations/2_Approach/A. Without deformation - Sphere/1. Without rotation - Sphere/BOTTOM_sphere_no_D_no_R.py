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
initial_angle_ground = 71 #º
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


#####

def find_max(list):
    """Find the max value and its time in a list"""
    max = 0
    t = 0
    for i in range(len(list)):
        if list[i][1] > max:
            t = list[i][0]
            max = list[i][1]
    return t, max


# BALL 
# In "middle":
x0 = -round(court_length/2, 3)
pos_init = vp.vector(x0,avg_dist_cp,0)
ball = vp.sphere(pos=pos_init, radius=ball_radius_m, color=vp.color.yellow, make_trail=True)

# COURT
# Singles court
ground = vp.box(pos=vp.vector(0,-0.25/2,0), size=vp.vector(court_length,0.25,singles_court_width), color = vp.color.green)

##### 






# BOTTOM

title = 'Ball <b><i>positions</i></b> and trajectory'
title += '<br>(Choose the graph by clicking 2 times on its label)'
gmix = vp.graph(title=title, xtitle='t[s]', ytitle='distance[m]', fast = False, width = 800)
f1 = vp.gcurve(graph = gmix, color=vp.color.purple, markers = True, label = 'Y', width=2)
f2 = vp.gcurve(graph = gmix, color=vp.color.red, markers = True, label = 'X', width=2)
f3 = vp.gcurve(graph = gmix, color=vp.color.blue, markers = True, label = 'Z', width=2)
g = vp.vector(0,-9.8,0)
v0 = initial_velocity
theta = mt.radians(initial_angle_ground) 

# Ball 
ball.m = ball_mass_kg
ball.v = v0*vp.vector(mt.cos(theta),mt.sin(theta),0) 
ball.p = ball.m*v0*vp.vector(mt.cos(theta),mt.sin(theta),0) 

# Reference frame
arrow_scale_factor = 0.5
axis_x = vp.arrow(pos=ball.pos, axis=arrow_scale_factor*vp.vector(1, 0, 0), color=vp.color.red, shaftwidth=0.025)
axis_y = vp.arrow(pos=ball.pos, axis=arrow_scale_factor*vp.vector(0, 1, 0), color=vp.color.purple, shaftwidth=0.025)
axis_z = vp.arrow(pos=ball.pos, axis=arrow_scale_factor*vp.vector(0, 0, 1), color=vp.color.blue, shaftwidth=0.025)
reference_frame = vp.compound([axis_x, axis_y, axis_z])

vscale1 = 0.03 
varrow = vp.arrow(pos=ball.pos, axis=vscale1*ball.v, color=vp.color.cyan)


t = 0
dt = 0.001
max_z = []

while ball.pos.y >= ground.pos.y+ball.radius+ground.size.y/2:
    vp.rate(1000) 
     
    # Force on the ball
    F = ball.m*g
    
    ball.p = ball.p + F*dt 
    ball.pos = ball.pos + ball.p*dt/ball.m 
    
    a = F/ball.m
    ball.v = ball.v + a*dt
    
    varrow.pos = ball.pos 
    varrow.axis = vscale1*ball.v 

    reference_frame.pos = ball.pos

    max_z.append([t, ball.pos.y])

    t = t + dt
    f1.plot(t, ball.pos.y)
    f2.plot(t, ball.pos.x)
    f3.plot(t, ball.pos.z)
    
    
d_time = t    
print("Time to hit the ground: ", round(d_time, 3), "s")
print("Distance y: ", round(ball.pos.x - pos_init.x, 3), "m")
print("Distance x: ", round(ball.pos.z - pos_init.z, 3), "m")
t_max, max_z = find_max(max_z)
print("Max z: ", round(max_z, 3), "m at t = ", round(t_max, 3), "s")
print("Final velocity: ", round(ball.v.y, 3), "m/s")
    

    
