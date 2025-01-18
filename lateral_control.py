import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=0.05, damping_constant=0.7):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation
        if isinstance(waypoints, list) and len(waypoints) == 0:
            return 0.0
        firstW = waypoints[:, 0]
        secondW = waypoints[:, 1]
        # print(secondW, firstW)
        xDiff = secondW[0] - 155.5
        yDiff = secondW[1] - 0

        orientationError = np.arctan2(xDiff, yDiff)
        # print("OE: ", orientationError)
        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        crossTrackError = firstW[0] - 155.5
        # print("CTE: ", crossTrackError)
        # print("Speed: ", speed)

        # derive stanley control law
        # prevent division by zero by adding as small epsilon
        # print(np.arctan2(self.gain_constant * crossTrackError,(speed + 1)))
        stanley = (orientationError + np.arctan2(self.gain_constant * crossTrackError,(speed + 1)))
        # derive damping term   
        # print("stan: ", stanley, (stanley - self.previous_steering_angle))
        steering_angle = stanley - self.damping_constant * (stanley - self.previous_steering_angle)
        self.previous_steering_angle = steering_angle
        # print("steer: ", steering_angle)
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4)# / 0.4






