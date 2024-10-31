# Rotation of the ball test
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

# Court variables
court_length = 23.77 #m
doubles_court_width = 10.97 #m  # | --->
singles_court_width = 8.23 #m # | ->
net_height_middle = 0.915 #m
net_height_sides = 1.065 #m


#####

def ball_inertia():
    """Calculates the inertia of the ball (a thin spherical shell)"""
    inertia = 2/3 * ball_mass_kg * ball_radius_m**2 
    return inertia #kg*m^2

def initial_angular_velocity():
    """Calculates the initial angular velocity of the ball when it is hit by the raquet"""
    w0 = initial_velocity / ball_radius_m 
    return w0 #rad/s

def initial_angular_acceleration(w0):
    """Calculates the initial angular acceleration of the ball when it is hit by the raquet"""
    w_initial = 0
    w_dot0 = (w0-w_initial) / ball_racket_contact_time_s
    return w_dot0 #rad/s^2

def compute_angular_velocity(v_linear):
    """Calculates the angular velocity of the ball"""
    w = v_linear / ball_radius_m 
    return w #rad/s

def compute_angular_acceleration(w, w0, t):
    """Calculates the angular velocity of the ball"""
    w_dot = (w - w0) / t
    return w_dot #rad/s^2

def compute_torque_force(force):
    """Calculates the initial torque applied to the ball when it is hit by the raquet"""
    l = ball_radius_m * mt.cos(mt.radians(initial_angle_ground))
    torque = l * (force*(mt.sin(mt.radians(initial_angle_ground))*ball_radius_m))   #The force is negative
    torque = round(torque.y, 9) 
    return torque #Nm

def compute_torque_angular(w, w_dot):
    """Calculates the torque when the ball is spinning"""
    inertia = ball_inertia()
    torque = inertia * w_dot + w * (inertia * w)
    torque = round(torque, 9) 
    return torque #Nm

def compute_net_torque(torque_angular, torque_force):
    """Calculates the net torque on the ball"""
    # Sum of the torques due to both angular acceleration and external forces
    net_torque = torque_angular + torque_force
    return net_torque #Nm

def sum_of_torques(torques):
    """Calculates the sum of all the torques applied on the ball"""
    sum_torques = np.sum(torques)
    return sum_torques #Nm

#####

# BALL 
# In "origin": 
#ball = vp.sphere(pos=vp.vector(-round(court_length/2, 3),avg_dist_cp,-singles_court_width/2), radius=0.08, color=vp.color.yellow, make_trail=True)
# In "middle":
x0 = -round(court_length/2, 3)
ball = vp.sphere(pos=vp.vector(x0,avg_dist_cp,0), radius=ball_radius_m, texture=vp.textures.earth, make_trail=False) 
# In "right":
#ball = vp.sphere(pos=vp.vector(-round(court_length/2, 3),avg_dist_cp,singles_court_width/2), radius=0.08, color=vp.color.yellow, make_trail=True)

# COURT
# Singles court
ground = vp.box(pos=vp.vector(0,-0.25/2,0), size=vp.vector(court_length,0.25,singles_court_width), color = vp.color.green)
# Doubles court
#ground = vp.box(pos=vp.vector(0,0,0), size=vp.vector(court_length,0.25,doubles_court_width), color = vp.color.green)

##### 


# Top spin 
ball.rotate(angle=-mt.radians(80), axis=vp.vector(0,0,1), origin=ball.pos)
# The angle defines the rotation in radians where a positive value indicates a counterclockwise rotation and
#  the bigger the value, the faster it rotates.


graph1 = vp.graph(title='Ball trajectory', xtitle='t[s]', ytitle='y[m]')
graph2 = vp.graph(title='Ball torque', xtitle='t[s]', ytitle='Torque[Nm]')
f1 = vp.gcurve(graph=graph1, color=vp.color.yellow)
f2 = vp.gcurve(graph=graph2, color=vp.color.green)
g = vp.vector(0,-9.8,0)
v0 = initial_velocity
theta = mt.radians(initial_angle_ground) #initial_angle

# Ball 
ball.m = ball_mass_kg
ball.v = v0*vp.vector(mt.cos(theta),mt.sin(theta),0) #Ball velocity
ball.p = ball.m*v0*vp.vector(mt.cos(theta),mt.sin(theta),0) #Ball momentum


t = 0
dt = 0.001
torques = []

while ball.pos.y >= ground.pos.y+ball.radius+ground.size.y/2: 
    vp.rate(10) #put at 10 to see the rotation of the ball | put at 1000 to see the trajectory of the ball along with 'make_trail=True' in the Ball
    
    # Force on the ball
    F = ball.m*g
    
    ball.rotate(angle=-mt.radians(80), axis=vp.vector(0,0,1), origin=ball.pos) 
    
    ball.p = ball.p + F*dt 
    ball.pos = ball.pos + ball.p*dt/ball.m 
    
    a = F/ball.m
    ball.v = ball.v + a*dt

    if t == 0:
        # When the raquet hits the ball | Contact with the racket :
        torque_force = compute_torque_force(F)
        w0 = initial_angular_velocity()
        w_dot0 = initial_angular_acceleration(w0)
        torque_angular = compute_torque_angular(w0, w_dot0)
        torque = compute_net_torque(torque_angular, torque_force)
        #print("Initial torque: ", torque, "Nm")
        
    elif t>0:
        # Ball in the air (in time)
        w = compute_angular_velocity(ball.v.y)
        w_dot = compute_angular_acceleration(w, w0, t)
        torque_angular = compute_torque_angular(w, w_dot)
        torque = compute_net_torque(torque_angular, torque_force)
        #print("Torque: ", torque, "Nm", "Time: ", t, "s")

    torques.append(torque)
    sum_torques = sum_of_torques(torques)
    
    # Time update
    t = t + dt
    t = round(t, 3)
    
    # Plots
    f1.plot(t, ball.pos.y)
    f2.plot(t, torque)
    
print("Sum of the torques: ", sum_torques, "Nm")