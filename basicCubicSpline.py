#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 00:48:22 2017

@author: Quintus
"""

import numpy as np

def cubicSpline(f):
    ''' calculate coeffients of each constained cubic curve
    args: 
        f is a 2-by-n numpy array that stores nodes (xi, yi)
    return:
        paraList is a (n-1)-by-4 numpy array that stores coeffients of each cubic curve
    '''
    xDiff = np.diff(f[0])
    yDiff = np.diff(f[1])
    rSlopeLeft = xDiff[0:-1] / yDiff[0:-1] # reciprocal of the slope on the leftside 
    rSlopeRight = xDiff[1:] / yDiff[1:]
    x_i = f[0][1:]
    x_im1 = f[0][0:-1]
    y_im1 = f[1][0:-1]
    
    # calculate first derivatives of f
    # first derivative of f is a harmonic average of slope on each side if slope keep same sign at point, otherwise 0
    f1 = 2 / (rSlopeLeft + rSlopeRight) * (np.sign(rSlopeLeft) == np.sign(rSlopeRight)) 
    f1_0 = 1.5 * yDiff[0] / xDiff[0] - 0.5 * f1[0]
    f1_n = 1.5 * yDiff[-1] / xDiff[-1] - 0.5 * f1[-1]
    f1 = np.insert(f1, 0, f1_0)
    f1 = np.append(f1, f1_n)
    
    # calculate second derivatives of f
    f2_im1 = -2 * (f1[1:] + 2*f1[0:-1]) / xDiff + 6 * yDiff / xDiff**2 
    f2_i = 2 * (2*f1[1:] + f1[0:-1]) / xDiff - 6 * yDiff / xDiff**2 
    
    # calculate parameters for each segment of the curve
    d = (f2_i - f2_im1) / (6*xDiff)
    c = (x_i*f2_im1 - x_im1*f2_i) / (2*xDiff)
    b = (yDiff - c*(x_i**2-x_im1**2) - d*(x_i**3-x_im1**3)) / xDiff
    a = y_im1 - b*x_im1 - c*x_im1**2 - d*x_im1**3
    coeffList = np.array(list(zip(a, b, c, d)))
    
    return coeffList

# for test
#f = np.array([[0, 10, 30, 50, 70, 90, 100], [30, 130, 150, 150, 170, 220, 320]])
