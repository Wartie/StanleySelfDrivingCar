import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import math
import statistics
import random

import cv2 as cv

PI = 3.14159265358

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=50, gradient_threshold=10, distance_maxima_gradient=5):
        self.car_position = np.array([156,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0

        self.bev_image_gray = None

        self.roi_mask = cv.imread("roiMask.png", cv.IMREAD_GRAYSCALE)
        self.left = 140
        self.right = 180
        self.offset_left = 136
        self.offset_right = 176
        # self.roi_mask = self.roi_mask.swapaxes(0, 1)
    

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)320x240x3

        output:
            bev_image 320x240x3

        '''
        toCVFront = front_view_image.swapaxes(0, 1).astype(np.uint8)
        # if np.sum(toCVFront) != 0:
        #     shadows_removed = self.remove_shadows_front(toCVFront)
        # else:
        #     shadows_removed = toCVFront
        sourcePnts = np.float32([[142, 144], [190, 144],  
                                 [302, 230], [69, 230]])

        # destPnts = np.float32([[80, 150], [261, 150],
        #                        [261, 230], [80, 230] ])
        
        # sourcePnts = np.float32([[155, 121], [164, 123],  
        #                          [311, 240], [56, 240]])

        destPnts = np.float32([[self.left, 160], [self.right, 160],
                               [self.right, 240], [self.left, 240]])

        h = cv.getPerspectiveTransform(sourcePnts, destPnts)
        bird_eye_view = cv.warpPerspective(toCVFront, h, (320, 240), flags=cv.INTER_LINEAR)

        # cv.imshow("disp", bird_eye_view)

        return bird_eye_view

    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car
        and translate to grey scale

        input:
            state_image_full 320x240x3

        output:
            gray_state_image 320x120x1

        '''
        toCVFull = state_image_full
        cropped = toCVFull[self.cut_size - 10:-10, :, :]
        gray_state_image = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY) # or is it bgr?

        croppedHLS = cv.cvtColor(cropped, cv.COLOR_BGR2HLS)
        croppedH = croppedHLS[:,:,0]
        croppedL = croppedHLS[:,:,1]
        croppedS = croppedHLS[:,:,2]
        satBin = np.zeros_like(croppedS)
        # Detect pixels that have a high saturation value
        satBin[(croppedS > 80) & (croppedS < 256)] = 1

        ligBin = np.zeros_like(croppedL)

        ligBin[(croppedL > 80) & (croppedL < 256)] = 1

        hueBin =  np.zeros_like(croppedH)
        # Detect pixels that are yellow using the hue component
        hueBin[(croppedH > 90) & (croppedH <= 115)] = 1

        # Combine all pixels detected above
        hslFilter = cv.bitwise_and(cv.bitwise_and(satBin, hueBin), ligBin)

        # gray_state_image = cv.normalize(gray_state_image.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # print(np.unique(hueBin))
        gray_state_image[(hslFilter == 1)] = 0

        # gray_state_image[(gray_state_image > 125)] = 255
        # gray_state_image[(gray_state_image <= 125)] = 0

        return gray_state_image 

    # Show the image with parallel lines

    def hough_lines(self, edge_image, segment):
        if segment:
            linesP = cv.HoughLinesP(edge_image, 1, np.pi/180, 40, 40, 5)
            if linesP is not None:
                blank_image = np.zeros_like(edge_image)
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv.line(blank_image, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1, cv.LINE_8)
                return linesP, blank_image
            else:
                return linesP, edge_image
        
        else:
            lines = cv.HoughLines(edge_image, 1, np.pi/180, 25)
            # lines = cv.HoughLines(edge_image, 1, np.pi/180, 60)
            if lines is not None:
                # print(lines[:3])
                blank_image = np.zeros_like(edge_image)
                lines_as_coords = []
                for r_theta in lines:
                    arr = np.array(r_theta[0], dtype=np.float64)
                    r, theta = arr

                    a = np.cos(theta)
                    b = np.sin(theta)

                    x0 = a * r
                    y0 = b * r

                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    lines_as_coords.append([[x1, y1, x2, y2]])

                    cv.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv.LINE_8)
                return np.array(lines_as_coords), blank_image
            else:
                return lines, edge_image
        
    def calculate_slope(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:  # Vertical line
            return np.inf
        else:
            return (y2 - y1) / (x2 - x1)
        
    def calculate_line_parameters(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:  
            return np.inf, np.inf
        m = self.calculate_slope(line)  # Slope
        b = y1 - m * x1  # Intercept
        return m, b

    def distance_between_lines(self, line1, line2):
        m1, b1 = self.calculate_line_parameters(line1)
        m2, b2 = self.calculate_line_parameters(line2)
        
        if m1 == np.inf or m2 == np.inf:
            return np.inf if m1 != m2 else 0
        
        slope_diff = np.abs(m1 - m2)
        intercept_diff = np.abs(b1 - b2)
        
        return slope_diff + intercept_diff
        
    def ransac_remove_outliers(self, lines, threshold=0.1, iterations=100):
        best_inliers = []

        if lines is not None:
            for i in range(iterations):
                # Randomly sample 1 line
                sample_line = random.choice(lines)
                # Find the inliers: lines that have similar slopes and intercepts
                inliers = []
                for line in lines:
                    distance = self.distance_between_lines(sample_line[0], line[0])
                    
                    # Check if the line is within the threshold for distance
                    if distance < threshold:
                        inliers.append(line)

                # Keep track of the largest set of inliers
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

        return best_inliers

    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 320x120x1

        output:
            gradient_sum 320x120x1

        '''

        rowGrad, colGrad = np.gradient(gray_image.astype(np.float32))
        # grads = np.sqrt(colGrad**2 + rowGrad**2)

        absRowGrad = np.abs(rowGrad)
        absColGrad = np.abs(colGrad)

        grads = absRowGrad + absColGrad

        grads[grads < self.gradient_threshold] = 0

        # thresh, grads = cv.threshold(grads, self.gradient_threshold, 255, cv.THRESH_BINARY)
        grads[self.roi_mask < 255] = 0
        final_grads = grads
        # normalizedGrads = cv.normalize(grads.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # bigger = cv.resize(grads, (640, 240))

        # cv.imshow("grads", grads)
        # cv.imwrite("masking.png", grads)


        lines, hough_grads = self.hough_lines(grads.astype(np.uint8), True)
        cv.imshow("phough", hough_grads)

        inlier_lines = self.ransac_remove_outliers(lines)

        img_inliers = np.zeros_like(hough_grads)
        for line in inlier_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_inliers, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_8)

        cv.imshow("ransac", img_inliers)

        lines, hough_grads = self.hough_lines(img_inliers, False)

        # lines, hough_grads = self.hough_lines(grads.astype(np.uint8), False)
        # inlier_lines = self.ransac_remove_outliers(lines)

        # img_inliers = np.zeros_like(hough_grads)
        # for line in inlier_lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(img_inliers, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_8)
        # cv.imshow("phough", hough_grads)

        # cv.imshow("ransac", img_inliers)

        if lines is not None:
            final_grads = hough_grads
    
        # cv.imshow("hough", final_grads)
        return final_grads


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 320x120x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        maxima = []
        for row in gradient_sum:
            peaks, props = find_peaks(row, distance=self.distance_maxima_gradient)
            maxima.append(peaks)
        arrmaxima = np.array(maxima, dtype=object)[::-1]
        return arrmaxima
        # return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row], distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                print("1")
                if argmaxima[0] <= self.car_position[0]:
                    lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                    lane_boundary2_startpoint = np.array([[self.offset_right,  row]])
                else: 
                    lane_boundary1_startpoint = np.array([[self.offset_left,  row]])
                    lane_boundary2_startpoint = np.array([[argmaxima[0],  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                print("2")
                if abs(argmaxima[0] - self.car_position[0]) < 7:
                    lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                else:
                    lane_boundary1_startpoint = np.array([[self.offset_left, row]])
                if abs(argmaxima[1] - self.car_position[0]) < 7:
                    lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                else:
                    lane_boundary2_startpoint = np.array([[self.offset_right, row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                print("multi")
                distsA = abs(argmaxima - (self.offset_left))
                distsB = abs(argmaxima - (self.offset_right))
                closeA = np.argsort(distsA)
                closeB = np.argsort(distsB)

                argA = argmaxima[closeA[0]]
                distA = distsA[closeA[0]]

                argB = argmaxima[closeB[0]]
                distB = distsB[closeB[0]]

                if argA == argB and distA <= 7 and distB <= 7:
                    if distA <= distB:
                        lane_boundary1_startpoint = np.array([[argA, row]])
                        lane_boundary2_startpoint = np.array([[self.offset_right,  row]])
                    else:
                        lane_boundary1_startpoint = np.array([[self.offset_left, row]])
                        lane_boundary2_startpoint = np.array([[argB,  row]])
                else:
                    if distA <= 7:
                        lane_boundary1_startpoint = np.array([[argA, row]])
                    else: 
                        lane_boundary1_startpoint = np.array([[self.offset_left,  row]])

                    if distB <= 7:
                        lane_boundary2_startpoint = np.array([[argB, row]])
                    else:
                        lane_boundary2_startpoint = np.array([[self.offset_right,  row]])
                # print(lane_boundary1_startpoint, lane_boundary2_startpoint)
                
                lanes_found = True
            
            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[self.offset_left,  0]])
                lane_boundary2_startpoint = np.array([[self.offset_right,  0]])
                print("no beginning")
                break
        
        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [320, 240, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        gray_state = self.cut_gray(state_image_full)
        # print("sds", gray_state.shape)
        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        # print("grads:", gradient_sum.shape)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        # print("maxima:", maxima.shape)

        # first lane_boundary points
        # gradient_sum = gradient_sum.swapaxes(0, 1)
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
   
            startingPnt = lane_boundary1_points[0][1]
            
            testImage = np.zeros_like(gradient_sum)
            for row, rowPeaks in enumerate(maxima):
                for val in rowPeaks:
                    testImage[119-row][val] = 255
                if row <= startingPnt:
                    continue

                curLb1 = lane_boundary1_points[-1]
                curLb2 = lane_boundary2_points[-1]
                # print(curLb1, curLb2)
                
                l1xlist = []
                l2xlist = []

                while rowPeaks.size > 0:
                    # print(rowPeaks.size)
                    dists1 = abs(rowPeaks - curLb1[0])
                    dists2 = abs(rowPeaks - curLb2[0])
                    sortedArgLb1 = np.argmin(dists1)
                    sortedArgLb2 = np.argmin(dists2)
                    # print(sortedArgLb1[0], sortedArgLb2[0])
                    # print(rowPeaks[sortedArgLb1[0]], rowPeaks[sortedArgLb2[0]])
                    if dists1[sortedArgLb1] >= 5 and dists2[sortedArgLb2] >= 5:
                        # print("hmmmmmm")
                        break
                    else:
                        # print("new row")
                        # print([[rowPeaks[sortedArgLb1[0]], row]])
                        # print([[rowPeaks[sortedArgLb2[0]], row]])

                        if dists1[sortedArgLb1] <= dists2[sortedArgLb2]:
                            if dists1[sortedArgLb1] <= 5:
                                # lane_boundary1_points = np.append(lane_boundary1_points, [[rowPeaks[sortedArgLb1], row]], axis=0)
                                l1xlist.append(rowPeaks[sortedArgLb1])

                                rowPeaks = np.delete(rowPeaks, sortedArgLb1)
                        else:
                            if dists2[sortedArgLb2] <= 5:
                                # lane_boundary2_points = np.append(lane_boundary2_points, [[rowPeaks[sortedArgLb2], row]], axis=0)
                                l2xlist.append(rowPeaks[sortedArgLb2])

                                rowPeaks = np.delete(rowPeaks, sortedArgLb2)

                if len(l1xlist) > 0:
                    lane_boundary1_points = np.append(lane_boundary1_points, [[int(sum(l1xlist)/len(l1xlist)), row]], axis=0)
                
                if len(l2xlist) > 0:
                    lane_boundary2_points = np.append(lane_boundary2_points, [[int(sum(l2xlist)/len(l2xlist)), row]], axis=0)
                    # print(rowPeaks.size)
                # if len(rowPeaks) > 0:
                #     dists1 = abs(rowPeaks - curLb1[0])
                #     dists2 = abs(rowPeaks - curLb2[0])
                #     sortedArgLb1 = np.argsort(dists1)
                #     sortedArgLb2 = np.argsort(dists2)
                #     # print(sortedArgLb1[0], sortedArgLb2[0])
                #     # print(rowPeaks[sortedArgLb1[0]], rowPeaks[sortedArgLb2[0]])

                #     if dists1[sortedArgLb1[0]] >= 20.0 and dists2[sortedArgLb2[0]] >= 20.0:
                #         print("hmmmmmm")
                #         print(dists1, dists2)
                #         break
                #     else:
                #         # print("new row")
                #         # print([[rowPeaks[sortedArgLb1[0]], row]])
                #         # print([[rowPeaks[sortedArgLb2[0]], row]])
                #         if len(rowPeaks) > 1:
                #             lane_boundary1_points = np.append(lane_boundary1_points, [[rowPeaks[sortedArgLb1[0]], row]], axis=0)
                #             lane_boundary2_points = np.append(lane_boundary2_points, [[rowPeaks[sortedArgLb2[0]], row]], axis=0)
                #         else:
                #             if dists1[sortedArgLb1[0]] <= dists2[sortedArgLb2[0]]:
                #                 lane_boundary1_points = np.append(lane_boundary1_points, [[rowPeaks[sortedArgLb1[0]], row]], axis=0)
                #             else:
                #                 lane_boundary2_points = np.append(lane_boundary2_points, [[rowPeaks[sortedArgLb2[0]], row]], axis=0)

            cv.imshow("Peaks", testImage)
            # cv.imwrite("Peaks.png", testImage)
            # print(lane_boundary1_points)
            # print(lane_boundary2_points)
            # lane_boundary 1

            # lane_boundary 2

            ################
            

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                # print("l1: ", lane_boundary1_points)
                # print("l2: ", lane_boundary2_points)
                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                x1 = lane_boundary1_points[:, 0]
                y1 = lane_boundary1_points[:, 1]
                lane_boundary1 = splprep([x1, y1], s=self.spline_smoothness)
                # print(y1)


                # lane_boundary 2
                x2 = lane_boundary2_points[:, 0]
                y2 = lane_boundary2_points[:, 1]
                lane_boundary2 = splprep([x2, y2], s=self.spline_smoothness)
                # print(lane_boundary1_points[-1], lane_boundary2_points[-1])
                # cv.circle(state_image_full, lane_boundary1_points[-1], 3, (255, 0, 0), -1)
                # cv.circle(state_image_full, lane_boundary2_points[-1], 3, (255, 0, 0), -1)

            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return self.lane_boundary1_old, self.lane_boundary2_old, state_image_full


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        if isinstance(self.lane_boundary1_old, int):
            return
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old[0]))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old[0]))
        
        plt.figure(fig)
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])

        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1] + 10, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1] + 10, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1] + 10, color='white')

        plt.axis('off')
        plt.xlim((0,320))
        plt.ylim((0,240))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
