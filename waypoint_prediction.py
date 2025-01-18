import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import math

def normalize(v):
    # print("norm:", v)
    # print(v)
    norm = np.linalg.norm(v,axis=0) + 0.00001
    # print(norm)
    # print(norm.shape, norm)
    # return v / norm.reshape(1, v.shape[1])
    return norm

def curvature(waypoints, ugh = False):
    '''
    ##### TODO #####
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    # print(waypoints)
    num_waypoints = waypoints.shape[1]
    sum = 0
    for i in range(1, num_waypoints - 1):
        # print("sdsd")
        futureDiff = waypoints[:, i+1] - waypoints[:, i]
        # print(waypoints[:, i+1], waypoints[:, i])
        pastDiff = waypoints[:, i] - waypoints[:, i-1]

        dot = np.dot(futureDiff, pastDiff)

        futureDiff = futureDiff.reshape(2, -1)
        pastDiff = pastDiff.reshape(2, -1)
        # print(futureDiff)
        # print(pastDiff)
        normFuture = normalize(futureDiff)
        normPast = normalize(pastDiff)
        # print("dot:", dot)
        # if ugh:
        #     print("sum: ", sum)
        #     print("dot:", dot)
        #     print("f: ", futureDiff)
        #     print("p: ", pastDiff)
        #     print("nf: ", normFuture)
        #     print("np: ", normPast)
        if normFuture.any() != 0 and normPast.any() != 0:
            sum += dot/(normFuture[0] * normPast[0])

    return sum


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    # print(waypoints.shape, waypoints_center.shape)
    # print("wp:", waypoints)
    # print("wpc:", waypoints_center)
    ls_tocenter = np.mean((waypoints_center - waypoints.reshape(2, -1))**2)
    # print("ls:", ls_tocenter)
    # derive curvature
    curv = curvature(waypoints.reshape(2, -1))
    # print("cruv:", curv)
    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    sections = np.linspace(0, 1, num_waypoints)
    if isinstance(roadside1_spline, int):
        return []
    lspline_points = np.array(splev(sections, roadside1_spline[0]))
    rspline_points = np.array(splev(sections, roadside2_spline[0]))

    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments

        # derive roadside points from spline            

        # derive center between corresponding roadside points
        # way_points = np.zeros(2, num_waypoints)
        center_list = []
        # for l, r in zip(lspline_points, rspline_points):
        #     lx = l[0]
        #     ly = l[1]

        #     rx = r[0]
        #     ry = r[1]

        #     center_list.append([int(lx + abs(rx-lx)/2), int(ly + abs(ry-ly)/2)])

        way_points = (lspline_points + rspline_points)/2

        way_points = np.array(center_list)
        # output way_points with shape(2 x Num_waypoints)
        return way_points
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points
        # center_list = []
        # for l, r in zip(lspline_points, rspline_points):
        #     lx = l[0]
        #     ly = l[1]

        #     rx = r[0]
        #     ry = r[1]

        #     center_list.append([int(lx + abs(rx-lx)/2), int(ly + abs(ry-ly)/2)])

        # way_points_center = np.array(center_list)        
        way_points_center = np.array((lspline_points + rspline_points)/2)#.reshape(1, -1)
        # print(way_points_center.shape)
        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
   
    if isinstance(waypoints, list) and len(waypoints) == 0:
        return offset_speed
    
    vRange = max_speed - offset_speed

    waypoints = np.array(waypoints)
    # print(curvature(waypoints[:num_waypoints_used]))
    curv =  abs(num_waypoints_used - 2 - curvature(waypoints[:num_waypoints_used], True))
    # print(waypoints[:num_waypoints_used])
    # print("tgp: ", curvature(waypoints[:num_waypoints_used], True))
    target_speed = vRange * math.exp(-1 * exp_constant * curv) + offset_speed
    # print("targetspeed: ", target_speed)
    return target_speed