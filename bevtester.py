import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

import cv2 as cv

src = cv.imread("dagtest.png", cv.IMREAD_COLOR)
# src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

cv.imshow("src", src)

cv.waitKey(0)

print(src.shape)

# sourcePnts = np.float32([[144, 140], [144, 192], 
#                               [230, 302], [230, 69]])

# destPnts = np.float32([[144, 69], [144, 230],
#                             [230, 302], [230, 69]])

sourcePnts = np.float32([
        [140, 144],  # Top-left (near horizon, left lane line)
        [192, 144],  # Top-right (near horizon, right lane line)
        [302, 230],  # Bottom-right (close to car, right lane line)
        [69, 230]    # Bottom-left (close to car, left lane line)
    ])

    # Define destination points (rectangle for bird's-eye view)
destPnts = np.float32([
        [69, 0],     # Top-left of bird's eye view
        [302, 0],    # Top-right of bird's eye view
        [302, 240],  # Bottom-right of bird's eye view
        [69, 240]    # Bottom-left of bird's eye view
    ])

h = cv.getPerspectiveTransform(sourcePnts, destPnts)
bird_eye_view = cv.warpPerspective(src, h, (320, 240), flags=cv.INTER_LINEAR)

cv.imshow("src", bird_eye_view)

cv.waitKey(0)