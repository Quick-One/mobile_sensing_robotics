from math import atan2, hypot, sqrt, pi
import numpy as np

def inverse_motion_model(X1, X2):
    x_, y_, theta_ = X1
    x, y, theta = X2
    delta_rot1 = atan2((y - y_), (x - x_)) - theta_
    delta_trans = hypot(x - x_, y - y_)
    delta_rot2 = theta - theta_ - delta_rot1

    return (delta_rot1, delta_trans, delta_rot2)

def motion_model_odometry(inital_pose, query_pose, odometry_reading, alpha, grid = False):
    if grid == False:
        x0, x1 = odometry_reading
        odelta_rot11, odelta_trans, odelta_rot2 = inverse_motion_model(x0, x1)
        hdelta_rot11, hdelta_trans, hdelta_rot2 = inverse_motion_model(inital_pose, query_pose)
        a1, a2, a3, a4 = alpha
        p1 = triangular_distribution(odelta_rot11 - hdelta_rot11, a1* abs(odelta_rot11) + a2 * odelta_trans)
        p2 = triangular_distribution(odelta_trans - hdelta_trans, a3 * odelta_trans + a4 *(abs(odelta_rot11) + abs(odelta_rot2)))
        p3 = triangular_distribution(odelta_rot2 - hdelta_rot2, a1 * abs(odelta_rot2) + a2 * odelta_trans)    
        return p1 * p2 * p3
    else:
        x0, x1 = odometry_reading
        odelta_rot11, odelta_trans, odelta_rot2 = inverse_motion_model(x0, x1)
        hdelta_rot11, hdelta_trans, hdelta_rot2 = inverse_motion_model(inital_pose, query_pose)
        a1, a2, a3, a4 = alpha
        p1 = triangular_distribution(odelta_rot11 - hdelta_rot11, a1* abs(odelta_rot11) + a2 * odelta_trans)
        p2 = triangular_distribution(odelta_trans - hdelta_trans, a3 * odelta_trans + a4 *(abs(odelta_rot11) + abs(odelta_rot2)))
        p3 = 1    
        return p1 * p2 * p3

def triangular_distribution(a,b):
    return max(0, (1/(sqrt(6)*b) - (abs(a)/(6 * b*b))))

def coord(x,y):
    a = (x-75) * 0.01 + 2
    b = (y-75) * 0.01 + 3
    return (a,b,0)

def sample_motion_model_odometry(initial_pose, odometry, noise_alpha):
    a1, a2, a3, a4 = noise_alpha
    x0, x1 = odometry
    x,y,theta = initial_pose

    odelta_rot1, odelta_trans, odelta_rot2 = inverse_motion_model(x0, x1)
    hdelta_rot1 = odelta_rot1 + sample_triangle(  a1* abs(odelta_rot1) + a2 * odelta_trans )
    hdelta_trans = odelta_trans + sample_triangle(a3 * odelta_trans + a4 *(abs(odelta_rot1) + abs(odelta_rot2)))
    hdelta_rot2 = odelta_rot2 + sample_triangle( a1 * abs(odelta_rot2) + a2 * odelta_trans)

    x_ = x + np.cos(theta + hdelta_rot1) * hdelta_trans
    y_  = y + np.sin(theta + hdelta_rot1) * hdelta_trans
    theta_ = theta + hdelta_rot1 + hdelta_rot2
    return (x_, y_, theta_)


def sample_triangle(b):
    return sqrt(6)/2 * (np.random.uniform(-b,b) + np.random.uniform(-b,b))


def inver_coord(x,y):
    a = (x-2) * 100 + 75
    b = (y-3) * 100 + 75
    return (a,b)

