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
avg_dist_cp = 0.95 #m - average distance/height between the point of contact and the ground

# Tennis Ball variables
ball_mass_g = 58.3
ball_mass_kg = ball_mass_g / 1000
ball_diameter_cm = 6.5
ball_radius_cm = ball_diameter_cm / 2
ball_radius_m = ball_radius_cm / 100
ball_racket_contact_time_ms = 30
ball_racket_contact_time_s = ball_racket_contact_time_ms / 1000
ball_pressure_kg = 8.165
ball_forward_deformation_cm = 0.6
ball_return_deformation_cm = 0.95
ball_air_avg_velocity_kmh = 125.5
ball_air_avg_velocity_ms = 34.8
ball_sphere_drag_coefficient = 0.47

# Tennis variables: Ellipsoid - Oblate spheroid at the beggining and then prolate spheroid the rest of the time beacause of the spin
ellipsoid_mass_g = ball_mass_g
ellipsoid_mass_kg = ball_mass_g / 1000
#Considering one prolate spheroid with radius: a, b, c
ellipsoid_a_cm = 3
ellipsoid_b_cm = 3
ellipsoid_c_cm = 4
ellipsoid_a_m = ellipsoid_a_cm/100
ellipsoid_b_m = ellipsoid_b_cm/100
ellipsoid_c_m = ellipsoid_c_cm/100
ellipsoid_p = 1.6075 
ellipsoid_drag_coefficient = 0.53

# Court variables
court_length = 23.77 #m
doubles_court_width = 10.97 #m  # | --->
singles_court_width = 8.23 #m # | ->
net_height_middle = 0.915 #m
net_height_sides = 1.065 #m


#####
# Ellipsoid functions
def ellipsoid_inertia():
    """Calculates the inertia of the ellipsoid"""
    inertia = 1/5 * ellipsoid_mass_kg * (ellipsoid_a_m**2 + ellipsoid_c_m**2)
    return inertia #kg*m^2

def ellipsoid_initial_angular_velocity():
    """Calculates the initial angular velocity of the ellisoid when it is hit by the raquet"""
    w0 = initial_velocity / ellipsoid_b_m 
    return w0 #rad/s

def ellipsoid_initial_angular_acceleration(w0):
    """Calculates the initial angular acceleration of the ellipsoid when it is hit by the raquet"""
    w_initial = 0
    w_dot0 = (w0-w_initial) / ball_racket_contact_time_s
    return w_dot0 #rad/s^2

def ellipsoid_compute_angular_velocity(v_linear):
    """Calculates the angular velocity of the ellipsoid"""
    w = v_linear / ellipsoid_b_m 
    return w #rad/s
    
def ellipsoid_compute_angular_acceleration(w, w0, t):
    """Calculates the angular acceleration of the ellipsoid"""
    w_dot = (w - w0) / t
    return w_dot #rad/s^2

def ellipsoid_compute_torque_force(force):
    """Calculates the initial torque applied to the ellipsoid when it is hit by the raquet"""
    # a and c are the semi-axes of the ellipsoid used rotation around the x-axis (b)
    l = ellipsoid_a_m * mt.cos(mt.radians(initial_angle_ground))
    torque = l * (force*(mt.sin(mt.radians(initial_angle_ground))*ellipsoid_c_m))  
    torque = round(torque.y, 9) 
    return torque #Nm

def ellipsoid_compute_torque_angular(w, w_dot):
    """Calculates the torque when the ellipsoid is spinning"""
    # In this case, because its a top spin, the rotation is around the x-axis, which is the b semi-axis and 
    #   for that reason we use the b semi-axis as the radius of rotation for torque
    inertia = ellipsoid_inertia()
    torque = inertia * w_dot + w * (inertia * w)
    torque = round(torque, 9) 
    return torque #Nm

def compute_net_torque(torque_angular, torque_force):
    """Calculates the net torque on the ellipsoid"""
    # Sum of the torques due to both angular acceleration and external forces
    net_torque = torque_angular + torque_force
    net_torque = round(net_torque, 9) 
    return net_torque #Nm

def sum_of_torques(torques):
    """Calculates the sum of all the torques applied on the ellipsoid"""
    sum_torques = np.sum(torques)
    return sum_torques #Nm

def find_max(list):
    """Find the max value and its time in a list"""
    max = 0
    t = 0
    for i in range(len(list)):
        if list[i][1] > max:
            t = list[i][0]
            max = list[i][1]
    return t, max

#####

# ELLIPSOID
# In "origin": 
#ball = vp.sphere(pos=vp.vector(-round(court_length/2, 3),avg_dist_cp,-singles_court_width/2), radius=0.08, color=vp.color.yellow, make_trail=True)
# In "middle":
x0 = -round(court_length/2, 3)
pos_init = vp.vector(x0,avg_dist_cp,0)
#ellipsoid = vp.ellipsoid(pos=vp.vector(x0,avg_dist_cp,0), length=ellipsoid_a_m, height=ellipsoid_c_m, width=ellipsoid_b_m, color=vp.color.yellow, make_trail=True)
ellipsoid = vp.ellipsoid(pos=pos_init, length=ellipsoid_a_m, height=ellipsoid_c_m, width=ellipsoid_b_m, texture=vp.textures.earth, make_trail=False)
# In "right":
#ball = vp.sphere(pos=vp.vector(-round(court_length/2, 3),avg_dist_cp,singles_court_width/2), radius=0.08, color=vp.color.yellow, make_trail=True)

# COURT
# Singles court
ground = vp.box(pos=vp.vector(0,-0.25/2,0), size=vp.vector(court_length,0.25,singles_court_width), color = vp.color.green)
# Doubles court
#ground = vp.box(pos=vp.vector(0,0,0), size=vp.vector(court_length,0.25,doubles_court_width), color = vp.color.green)

##### 




# CENTER

# Top spin 
ellipsoid.rotate(angle=-mt.radians(80), axis=vp.vector(0,0,1), origin=ellipsoid.pos)
# The angle defines the rotation in radians where a positive value indicates a counterclockwise rotation and
#  the bigger the value, the faster it rotates.

title = 'Ellipsoid <b><i>positions</i></b> and trajectory'
title += '<br>(Choose the graph by clicking 2 times on its label)'
gmix = vp.graph(title=title, xtitle='t[s]', ytitle='distance[m]', fast = False, width = 800)
f1 = vp.gcurve(graph = gmix, color=vp.color.purple, markers = True, label = 'Y', width=2)
f2 = vp.gcurve(graph = gmix, color=vp.color.red, markers = True, label = 'X', width=2)
f3 = vp.gcurve(graph = gmix, color=vp.color.blue, markers = True, label = 'Z', width=2)
graph2 = vp.graph(title='Ball torque', xtitle='t[s]', ytitle='Torque[Nm]')
f4 = vp.gcurve(graph=graph2, color=vp.color.green)
g = vp.vector(0,-9.8,0)
v0 = initial_velocity
theta = mt.radians(initial_angle_ground) #initial_angle
rho = 1.2 #kg/m^3


# Ellipsoid
ellipsoid.m = ball_mass_kg
ellipsoid.v = v0*vp.vector(mt.cos(theta),mt.sin(theta),0)
ellipsoid.p = ellipsoid.m*v0*vp.vector(mt.cos(theta),mt.sin(theta),0) 


# Reference frame
arrow_scale_factor = 0.5
axis_x = vp.arrow(pos=ellipsoid.pos, axis=arrow_scale_factor*vp.vector(1, 0, 0), color=vp.color.red, shaftwidth=0.025)
axis_y = vp.arrow(pos=ellipsoid.pos, axis=arrow_scale_factor*vp.vector(0, 1, 0), color=vp.color.purple, shaftwidth=0.025)
axis_z = vp.arrow(pos=ellipsoid.pos, axis=arrow_scale_factor*vp.vector(0, 0, 1), color=vp.color.blue, shaftwidth=0.025)
reference_frame = vp.compound([axis_x, axis_y, axis_z])


t = 0
dt = 0.001
torques = []
max_z = []


while ellipsoid.pos.y >= ground.pos.y+ellipsoid.radius+ground.size.y/2: 
    vp.rate(10) #put at 10 to see the rotation of the ball | put at 1000 to see the trajectory of the ball along with 'make_trail=True' in the Ball
         
    # Force on the ellipsoid
    F  = ellipsoid.m*g
    
    ellipsoid.rotate(angle=-mt.radians(80), axis=vp.vector(0,0,1), origin=ellipsoid.pos) 
    
    ellipsoid.p = ellipsoid.p + F *dt
    ellipsoid.pos = ellipsoid.pos + ellipsoid.p*dt/ellipsoid.m
    
    a = F/ellipsoid.m
    ellipsoid.v = ellipsoid.v + a*dt

    reference_frame.pos = ellipsoid.pos
    
    max_z.append([t, ellipsoid.pos.y])
    
    if t == 0:
        # When the raquet hits the ellipsoid | Contact with the racket
        torque_force = ellipsoid_compute_torque_force(F)
        w0 = ellipsoid_initial_angular_velocity()
        w_dot0 = ellipsoid_initial_angular_acceleration(w0)
        torque_angular = ellipsoid_compute_torque_angular(w0, w_dot0)
        torque = compute_net_torque(torque_angular, torque_force)
        #print("Initial torque: ", torque, "Nm")
        
    elif t>0:
        # Ellipsoid in the air (in time)
        w = ellipsoid_compute_angular_velocity(ellipsoid.v.y)
        w_dot = ellipsoid_compute_angular_acceleration(w, w0, t)
        torque_angular = ellipsoid_compute_torque_angular(w, w_dot)
        torque = compute_net_torque(torque_angular, torque_force)
        #print("Torque: ", torque, "Nm", "Time: ", t, "s")

    torques.append(torque)
    sum_torques = sum_of_torques(torques)
    
    # Time update
    t = t + dt
    t = round(t, 3)
    
    # Plots
    f1.plot(t, ellipsoid.pos.y)
    f2.plot(t, ellipsoid.pos.x)
    f3.plot(t, ellipsoid.pos.z)
    f4.plot(t, torque)


d_time = t    
print("Time to hit the ground: ", round(d_time, 3), "s")
print("Distance y: ", round(ellipsoid.pos.x - pos_init.x, 3), "m")
print("Distance x: ", round(ellipsoid.pos.z - pos_init.z, 3), "m")
t_max, max_z = find_max(max_z)
print("Max z: ", round(max_z, 3), "m at t = ", round(t_max, 3), "s")
print("Final velocity: ", round(ellipsoid.v.y, 3), "m/s")
print("Initial torque: ", round(torques[0], 3), "Nm")
print("Final torque: ", round(torques[-1], 3), "Nm") 