"""

Here the computation of the trajectory of one simple dot is done, 
NOT considering the mass, deformation and air friction of a real ball
To be a point and not a ball, we consider the mass unitary and the deformation 0

First approach in 2D (x,y)
Second approach in 3D (x,y,z)

|---|----------.----------|---|
|   |    A     |     B    |   |
|   |          |   p      |   |
|   |---------------------|   |
|   |          |          |   |
|   |    c     |     D    |   |
|   |          |          |   |
|   |=====================|   |
|   |          |          |   |
|   |    E     |     F    |   |
|   |          |          |   |
|   |---------------------|   |
|   |          |          |   |
|   |    G     |     H    |   |
|-- Z -------- O -------- W --|

Example of a trajectory: From O to p passing by H-F-D-B

"""


import json
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
# from vpython import *
import sys
from mplcursors import cursor
from matplotlib.animation import FuncAnimation

import time
################################

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

# Court variables
court_length = 23.77 #m
doubles_court_width = 10.97 #m  # | --->
singles_court_width = 8.23 #m # | ->
net_height_middle = 0.915 #m
net_height_sides = 1.065 #m



# Functions *******************************

def initial_coordinates(position="origin", unit="cm"):
    """ Chooses the initial coordinates of the point of the ball depending on the position of the player """
    
    coordinates = []

    if position == "origin":
        x = 0
        y = 0
        z = avg_dist_cp
    elif position == "middle":
        if unit == "mm":
            x = round(10.97/2, 3)
        else:
            x = round(10.97/2, 2)
        y = 0
        z = avg_dist_cp
    elif position == "right":
        x = 10.97
        y = 0
        z = avg_dist_cp
        
    coordinates.append((x,y,z))
    
    return coordinates

# Field functions
def cm_mm_var(unit="cm"):
    """ Cols and rows in cm or mm for the cm or mm version of the court, respectively """
    if unit == "mm":
        rows = int(court_length * 1000)            # = 23770
        columns = int(doubles_court_width * 1000)  # = 10970
    else:
        rows = int(court_length * 100)            # = 2377
        columns = int(doubles_court_width * 100)  # = 1097
    return rows, columns


def create_2D_tennis_court_matrix(rows, cols):
    """Creates a blank 2D matrix for the tennis court"""
    return [[' ' for _ in range(cols+1)] for _ in range(rows+1)]


def iterate_over_court_matrix_cm(matrix_cm, rows, cols):
    """ Iterates over the court matrix in cm"""
    # Time to compute: > 5s
    
    origin_start = 0.00
    step = 0.01
    
    for i in range(int(origin_start * 100), rows + 1, int(step * 100)):
        for j in range(int(origin_start * 100), cols + 1, int(step * 100)):
            x_coordinate = j / 100.0
            y_coordinate = i / 100.0
            matrix_cm[i][j] = (x_coordinate, y_coordinate)
    
    return matrix_cm
    
    
def iterate_over_court_matrix_mm(matrix_mm, rows, cols):
    """ The iteration in mm is more precise and avoid computation errors | for example: half of 10.97 is 5.485 but in cm is 5.48 """
    # Time to compute: 
    
    origin_start = 0.000
    step = 0.001
    
    for i in range(int(origin_start * 1000), rows + 1, int(step * 1000)):
        for j in range(int(origin_start * 1000), cols + 1, int(step * 1000)):
            x_coordinate = j / 1000.0
            y_coordinate = i / 1000.0
            matrix_mm[i][j] = (x_coordinate, y_coordinate)
    
    return matrix_mm


def compute_court_matrix_cm_mm(matrix, rows, cols, unit="cm"):
    """ Computes the court matrix in cm or mm """
        
    if unit == "mm":
        matrix = iterate_over_court_matrix_mm(matrix, rows, cols)
    else:
        matrix = iterate_over_court_matrix_cm(matrix, rows, cols)
    return matrix
    
    
def create_tennis_field(unit="cm"):
    """ Creates the field in 2D. Map coordinates of each point of the field"""
    
    rows, cols = cm_mm_var(unit)    
    
    # Create a 2D matrix for the tennis court
    tennis_court_matrix = create_2D_tennis_court_matrix(rows, cols)

    # Iterate over the matrix for each point and map the coordinates
    tennis_court_matrix = compute_court_matrix_cm_mm(tennis_court_matrix, rows, cols, unit)
    #print(np.shape(tennis_court_matrix))
    
    return tennis_court_matrix
    
def get_coordinates(coordinates, initial = True, final = False):
    """ Return the coordinates x, y, z of the point of the ball """
    if initial == True:
        x = coordinates[0][0]
        y = coordinates[0][1]
        z = coordinates[0][2]
    elif final == True:
        x = coordinates[-1][0]
        y = coordinates[-1][1]
        z = coordinates[-1][2]
    return x, y, z

def decide_judge_call(coordinates, tennis_court):
    """ Judge call. In or out. Depending on the coordinates of the point of the ball and the tennis court"""
    
    judge_decisions = ("IN", "OUT")
    x, y, z = get_coordinates(coordinates, False, True)
    
    # Check if the point is in the court
    if (float(x) >= 0 and float(x) <= singles_court_width) and (float(y) >= 0 and float(y) <= court_length):
        judge_call = judge_decisions[0]
    else:
        judge_call = judge_decisions[1]
    
    return judge_call

# Trajectory computation functions
def compute_trajectory_values(x0, y0, z0):
    """ Computes the equations for the trajectory of the ball in 3D """
    
    v_x, v_y, v_z, v_w = compute_axis_velocities() #m/s
    t = compute_time(v_z, z0) #seconds
    disp = v_w * t #displacement in the direction of the impact
    dx = compute_dx(disp)
    dy = compute_dy(disp)
    print_results(v_x, v_y, v_z, v_w, x0, y0, z0, t, disp, dx, dy)
    compute_error(v_x, t, dy)
    
    return dx, dy, t, v_w, v_z
    
    
def print_results(v_x, v_y, v_z, v_w, x0, y0, z0, t, s, dx, dy):
    print("v_x: ", v_x)
    print("v_y: ", v_y)
    print("v_z: ", v_z)
    print("v_w: ", v_w)
    print("x0: ", x0, "| y0: ", y0, "| z0: ", z0)
    print("t: ", t)
    print("s: ", s)
    print("dx: ", dx)
    print("dy: ", dy)
    
def compute_axis_velocities(theta_x = initial_angle_x, theta_y = initial_angle_y, theta_z = 90-initial_angle_ground, 
                            theta_w = initial_angle_ground, v0 = initial_velocity):
    """ Computes the velocities of the ball in the x, y and z axis as well as in direction w """
    
    # Velocities in the x, y and z axis
    # x
    if theta_x > 0 and theta_x < 90:
        v_x = v0 * mt.cos(mt.radians(theta_x))
    elif theta_x == 90:
        v_x = 0
    else:
        if theta_x == 0 and theta_y > 0 and theta_y < 90: 
            # quando nao tenho o theta_x mas tenho o theta_y
            v_x = v0 * mt.sin(mt.radians(theta_y))
        elif theta_x == 0 or theta_y == 90: 
            #quando nao tenho o theta_x e sei que o theta_y é 90º 
            v_x = v0

    # y
    if theta_y > 0 and theta_y < 90:
        v_y = v0 * mt.cos(mt.radians(theta_y))
    elif theta_y == 90:
        v_y = 0
    else:
        if theta_y == 0 and theta_x > 0: 
            # quando nao tenho o theta_y mas tenho o theta_x
            v_y = v0 * mt.sin(mt.radians(theta_x))
        elif theta_y == 0 or theta_x == 90: 
            #quando nao tenho o theta_y e sei que o theta_x é 90º 
            v_y = v0

    # z
    if theta_z > 0 and theta_z < 90:
        v_z = v0 * mt.cos(mt.radians(theta_z))
    elif theta_z == 0:
        v_z = v0
    elif theta_z == 90:
        v_z = 0
        
    # Velocity in the direction w
    if theta_w > 0 and theta_w < 90:
        v_w = v0 * mt.cos(mt.radians(theta_w))
    elif theta_w == 0:
        v_w = v0
    elif theta_w == 90:
        v_w = 0
    
    return v_x, v_y, v_z, v_w

def compute_time(v_z, z0 = avg_dist_cp, g = a_gravity):
    """ Computes the time of the trajectory in seconds """
    
    #t = (v_z + mt.sqrt(v_z**2 + 2*g*z0)) / g
    #t = equationroots(a=-g/2, b=v_z, c=z0) 
    t = (2*v_z) / g 
    
    if t > 0:      
        return t

def equationroots(a, b, c): 
    """ Solve the quadratic equation ax**2 + bx + c = 0 """
    
    if a > 0:
        a = a 
        b = b 
        c = c
    elif a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0:
        print("Input correct quadratic equation") 
        pass
    
    # calculating discriminant using formula
    dis = b * b - 4 * a * c 
    sqrt_val = mt.sqrt(abs(dis)) 
    
    # checking condition for discriminant
    if dis > 0: 
        #Real and different roots
        print((-b + sqrt_val)/(2 * a)) 
        print((-b - sqrt_val)/(2 * a)) 
        if (-b + sqrt_val)/(2 * a) > 0:
            value = (-b + sqrt_val)/(2 * a)
        elif (-b - sqrt_val)/(2 * a) > 0:
            value = (-b - sqrt_val)/(2 * a)
        else:
            value = 0
    
    elif dis == 0: 
        #Real and same roots 
        value = -b / (2 * a)
    
    # when discriminant is less than 0
    else:
        #Complex Roots
        # print(- b / (2 * a), + i, sqrt_val) 
        # print(- b / (2 * a), - i, sqrt_val) 
        pass

    return value

def compute_dx(s):
    """ Compute the displacement in x coordinates """
    
    if ((initial_angle_x >= 0 and initial_angle_x <= 90) and ((initial_angle_x + initial_angle_y) <= 90)) or ((initial_angle_x >= 90 and initial_angle_x <= 180) and 
        ((initial_angle_x + initial_angle_y) <= 180)):
        # If we have theta_x and not theta_y -> use theta_x for the computation, vice-versa, else we can't compute
        if (initial_angle_x > 0 and initial_angle_x < 180) or (initial_angle_y > 0 and initial_angle_y < 180):
            if initial_angle_x == 90 and initial_angle_y == 0:
                dx = 0
            else:
                if initial_angle_y == 0 or initial_angle_y < 0:
                    #compute using theta_x
                    dx = s * mt.cos(mt.radians(initial_angle_x))
                elif initial_angle_x == 0 or initial_angle_x < 0:
                    #compute using theta_y
                    dx = s * mt.sin(mt.radians(initial_angle_y))
                else: 
                    #use one of them
                    dx = s * mt.cos(mt.radians(initial_angle_x))
    
    return dx

def compute_dy(s):
    """ Compute the displacement in y coordinates """
    if ((initial_angle_x >= 0 and initial_angle_x <= 90) and ((initial_angle_x + initial_angle_y) <= 90)) or ((initial_angle_x >= 90 and initial_angle_x <= 180) and 
        ((initial_angle_x + initial_angle_y) <= 180)):
        if (initial_angle_x > 0 and initial_angle_x < 180) or (initial_angle_y > 0 and initial_angle_y < 180):
            if initial_angle_y == 90 and initial_angle_x == 0:
                dy = 0
            else:
                if initial_angle_y == 0 or initial_angle_y < 0:
                    #compute using theta_x
                    dy = s * mt.sin(mt.radians(initial_angle_x))
                elif initial_angle_x == 0 or initial_angle_x < 0:
                    #compute using theta_y
                    dy = s * mt.cos(mt.radians(initial_angle_y))
                else: 
                    #use one of them
                    dy = s * mt.cos(mt.radians(initial_angle_y))
    
    return dy

def compute_error(v_x, t, dy):
    """ Computes the error by calculating the precision of the above computation """
    
    if v_x != 0 and t != 0 and dy != 0:
        
        if initial_angle_x > 0:
            cos_x = mt.cos(mt.radians(initial_angle_x))
            cos2t_x = mt.pow(cos_x,2)
        else:
            cos2t_x = (mt.pow(v_x,2))/(mt.pow(initial_velocity,2))
            
        if initial_angle_y > 0:
            cos_y = mt.cos(mt.radians(initial_angle_y))
            cos2t_y = mt.pow(cos_y,2)
        else:
            cos2t_y = (mt.pow(dy,2))/(mt.pow(initial_velocity,2)*mt.pow(t,2))

        theta_z = 90-initial_angle_ground
        if theta_z > 0:
            cos_z = mt.cos(mt.radians(theta_z))
            cos2t_z = mt.pow(cos_z,2)
        else:
            cos2t_z = (mt.pow(a_gravity,2)*mt.pow(t,2))/(mt.pow(2,2)*mt.pow(initial_velocity,2))
        
        cos_sum = cos2t_x + cos2t_y + cos2t_z 
        error = 1 - cos_sum
        error_perc = abs(error)*100 #%
        precision = 100-error_perc #%
        
        if cos_sum == 1 and error == 0:
            print("Computation precision is perfect, there is no calculation errors")
        else:
            print("Computation Error:", error_perc, "%")
            print("Computation Precision:", precision, "%")
    
    

# Plot/graph functions - all of them should have a hovering effect/annotations to show the values at that hover point
# def update(frame, ax):
#     ax.clear()
#     ax.plot(coordinates[:frame, 0], coordinates[:frame, 1], label="Displacement X|Y", color="blue")
#     ax.set_xlabel('dx')
#     ax.set_ylabel('dy')
#     ax.set_title('Trajectory: View from the top')
#     ax.legend()
    
# Position|Trajectory plots
def plot_disp_2d_top(coordinates):
    """ Plot the displacement graph by dx and dy | view from the top """
    
    initial_coordinates = coordinates[0]
    final_coordinates = coordinates[-1]
    xi = initial_coordinates[0]
    xf = final_coordinates[0]
    yi = initial_coordinates[1]
    yf = final_coordinates[1]
    x = [xi, xf]
    y = [yi, yf]
    print("X:", x)
    print("Y:", y)
    
    plt.plot(x,y, label = "Displacement X|Y", color = "blue")
    cursor(hover=True)
    plt.xlabel('dx') 
    plt.ylabel('dy') 
    plt.title('Trajectory: View from the top') 
    plt.legend() 
    plt.show()
    
    # x = np.arange(float(xi), float(xf), 0.01)
    # y = np.arange(float(yi), float(yf), 0.01)
    
    # polifit = np.polyfit(1, 3, 2) # -> to create a polynomial function | if I want to keep track of the points in the graph | DO LATER

    # fig, ax = plt.subplots()
    # ani = FuncAnimation(fig, update, fargs=(ax,), frames=len(coordinates), interval=100, repeat=False)
    


def plot_disp_2d_side_y(coordinates):
    """ Plot the displacement graph in Y/Z axis | view from the side zy """
    
    initial_coordinates = coordinates[0]
    final_coordinates = coordinates[-1]
    yi = initial_coordinates[1]
    yf = final_coordinates[1]
    zi = initial_coordinates[2]
    zf = final_coordinates[2]
    y = [yi, yf]
    z = [zi, zf]
    
    plt.plot(y,z, label = "Displacement Y|Z", color = "blue")
    cursor(hover=True)
    plt.xlabel('dy') 
    plt.ylabel('dz') 
    plt.title('Trajectory: View from the side ZY') 
    plt.legend() 
    plt.show()
    
    
def plot_disp_2d_side_x(coordinates):
    """ Plot the displacement graph in Z/X axis | view from the side zx """
 
    initial_coordinates = coordinates[0]
    final_coordinates = coordinates[-1]
    xi = initial_coordinates[0]
    xf = final_coordinates[0]
    zi = initial_coordinates[2]
    zf = final_coordinates[2]
    x = [xi, xf]
    z = [zi, zf]
    
    plt.plot(x,z, label = "Displacement X|Z", color = "blue")
    cursor(hover=True)
    plt.xlabel('dx') 
    plt.ylabel('dz') 
    plt.title('Trajectory: View from the side ZX') 
    plt.legend() 
    plt.show()
    

def plot_trajectory_3d(coordinates):
    """ Plot the trajectory graph by x, y and z using vpython"""
    #!DONE IN ANOTHER FILES!
    pass

# Position|Time graphs
def graph_dx_in_time(coordinates, time):
    """ Graph of dx in respect to time """
    
    # !UNFINISHED!
    
    initial_coordinates = coordinates[0]
    final_coordinates = coordinates[-1]
    xi = initial_coordinates[0]
    xf = final_coordinates[0]
    x = [xi, xf]
    
    n = 0
    while n < time:
        t=n
        
        # Do the polid
        # find a function for x(t)
        
        # Plot
        # plt.plot(n,x, label = "Position in X in respect to time", color = "green")
        # cursor(hover=True)
        # plt.xlabel('t') 
        # plt.ylabel('x') 
        # plt.title('Position: X in respect to time') 
        # #plt.legend() 
        # plt.show()
        
        n = n + 0.01
    
    

def graph_dy_in_time(coordinates):
    """ Graph of dy in respect to time """
    pass

def graph_dw_in_time(v_w, final_time):
    """ Graph of dw in respect to time | should be the same result as s"""
    
    # !UNFINISHED!
    
    # Function for w(t) = v0 * cos(theta_w) * t = Vw0 * t
    w = np.poly1d([0, v_w, 0])
    
    t_values = np.linspace(0, final_time)
    w_t = w(t_values)
    
    plt.plot(t_values, w_t, color='green')
    cursor(hover=True)
    plt.xlabel('t(s)')
    plt.ylabel('position in w(m)')
    plt.title('Position: W in respect to time')
    plt.grid(True)
    plt.show()
    
    
    #n = 0
    #img_render = 10
    # while n < final_time:
    #     t=n
    #     w_t = w(t)
        
    #     if n % img_render == 0:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(t, w_t, color='green')
    #         plt.xlabel('t') 
    #         plt.ylabel('position in w')
    #         plt.title('Position: W in respect to time') 
    #         cursor(hover=True)
    #         time.sleep(0.2)
    #         plt.show()
        
    #     n = n + 0.01
    
    
        
    

def graph_dz_in_time(coordinates, v_z, final_time):
    """ Graph of z in respect to time """
    
    initial_coordinates = coordinates[0]
    final_coordinates = coordinates[-1]
    zi = initial_coordinates[2]
    zf = final_coordinates[2]
    z = [zi, zf]

        
    # Function for z(t) = v0 * sin(theta_z) * t -1/2 * g *t^2 = Vz0*t -4.*9t^2
    z = np.poly1d([-4.9, v_z, zi])
    
    t_values = np.linspace(0, final_time)
    z_t = z(t_values)
    
    plt.plot(t_values, z_t, color='green')
    cursor(hover=True)
    plt.xlabel('t(s)')
    plt.ylabel('position in z(m)')
    plt.title('Position: Z in respect to time')
    plt.grid(True)
    plt.show()
    
    # There is no way to end at 0, only if I use a loop for continuous time and plot until 
    #   the final_time-0.001 whereas at the final time Z should be 0
        


# Velocity|Time graphs - Maybe not so important
def graph_vx_in_time(coordinates):
    """ Graph of velocity in x axis in respect to time """
    pass

def graph_vy_in_time(coordinates):
    """ Graph of velocity in y axis in respect to time """
    pass

def graph_vz_in_time(coordinates):
    """ Graph of velocity in z axis in respect to time """
    pass

def graph_vw_in_time(coordinates):
    """ Graph of velocity in impact direction in respect to time """
    pass

# Angle|Position graph
def graph_angle_position(coordinates):
    """ Ground angle graph in respect to the position in dw """
    pass




# Draw functions - This are done in another files where the trajectory and the field are drawn in 3D
def draw_field():
    pass

def draw_trajectory():
    pass

# Main *******************************
def main():
    """ Main function """
    
    unit = "cm" #can be mm or cm | mm is more precise but takes more time to compute
    coordinates = initial_coordinates("middle", unit) #print(init_coordinates)
    x0, y0, z0 = get_coordinates(coordinates) #m
    court = create_tennis_field(unit) #print(court)
    dx, dy, t, v_w, v_z = compute_trajectory_values(x0, y0, z0)
    x_f = x0 + dx
    y_f = y0 + dy

    if unit == "cm":
        x_f = "{:.2f}".format(x_f)
        y_f = "{:.2f}".format(y_f)
    elif unit == "mm":
        x_f = "{:.3f}".format(x_f)
        y_f = "{:.3f}".format(y_f)
        
    end_point = (x_f, y_f, 0)
    coordinates.append(end_point)

    print("Coordinates: ", coordinates)
    judge_call = decide_judge_call(coordinates, court)
    print("Judge call: ", judge_call)

    plot_disp_2d_top(coordinates)
    plot_disp_2d_side_y(coordinates)
    plot_disp_2d_side_x(coordinates) 

    graph_dz_in_time(coordinates, v_z, t)


if __name__ == "__main__":
    main()