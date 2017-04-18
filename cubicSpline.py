#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 02:01:14 2017

@author: Quintus
"""

import numpy as np

def cubicSpline(f, x):
    ''' calculate coefficients of each cubic curve and output a constrained cubic spline passing given node points
    args:
        f is a 2-by-n numpy array that stores nodes (xi, yi)
        x is a numpy array that stores time in years
    return:
        y is a numpy array that stores instantaneous forward rate corresponding to given time(x)
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
    
    # calculate coefficients for each segment of the curve
    d = (f2_i - f2_im1) / (6*xDiff)
    c = (x_i*f2_im1 - x_im1*f2_i) / (2*xDiff)
    b = (yDiff - c*(x_i**2-x_im1**2) - d*(x_i**3-x_im1**3)) / xDiff
    a = y_im1 - b*x_im1 - c*x_im1**2 - d*x_im1**3
    coeffList = np.array(list(zip(a, b, c, d)))
    
    # for each x, find the interval it belongs to
    pos = np.searchsorted(f[0], x, side = 'right')
    y = np.zeros(len(x))
    for i in range(len(x)):
        # Use constant forward rates before the first node and after the last node
        if pos[i] == 0:
            y[i] = np.sum(coeffList[0] * np.array([1, f[0][0], f[0][0]**2, f[0][0]**3]))
        elif pos[i] == len(f[0]):
            y[i] = np.sum(coeffList[-1] * np.array([1, f[0][-1], f[0][-1]**2, f[0][-1]**3]))
        else:
            y[i] = np.sum(coeffList[pos[i]-1] * np.array([1, x[i], x[i]**2, x[i]**3]))

    return y  

def cubicSplineIntg(f, x, var, ind):
    ''' calculate coefficients of each cubic curve and integral of instantaneous forward curve between any given point and origin
    args:
        f is a 2-by-n numpy array that stores nodes (xi, yi)
        x is a numpy array that stores time in years
    return:
        Y is a numpy array that stores integral of instantaneous forward curve between any given point and origin
    '''
    f[1][ind] = var

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
    
    # calculate coefficients for each segment of the curve
    d = (f2_i - f2_im1) / (6*xDiff)
    c = (x_i*f2_im1 - x_im1*f2_i) / (2*xDiff)
    b = (yDiff - c*(x_i**2-x_im1**2) - d*(x_i**3-x_im1**3)) / xDiff
    a = y_im1 - b*x_im1 - c*x_im1**2 - d*x_im1**3
    coeffList = np.array(list(zip(a, b, c, d)))
    
    
    # integral between given nodes(f)
    yIntg = np.zeros(f.shape[1])
    for i in range(len(f[0])):
        if i == 0:
            yIntg[0] = np.sum(coeffList[0] * np.array([1, f[0][0], f[0][0]**2, f[0][0]**3])) * f[0][0]
        else:    
            yIntg[i] = np.sum(coeffList[i-1] * np.array([f[0][i], f[0][i]**2/2, f[0][i]**3/3, f[0][i]**4/4])) \
            - np.sum(coeffList[i-1] * np.array([f[0][i-1], f[0][i-1]**2/2, f[0][i-1]**3/3, f[0][i-1]**4/4]))
    # integral before each node
    yIntg = np.cumsum(yIntg)
    
    # integral of instantaneous forward curve between any given point(x) and origin
    pos = np.searchsorted(f[0], x, side = 'right')
    Y = np.zeros(len(x))
    for i in range(len(x)):
        # x is on the left-hand side of x0
        if pos[i] == 0:
            Y[i] = yIntg[0] * x[i] / f[0][0]
        # x is on the right-hand side of xn
        elif pos[i] == len(f[0]):
            Y[i] = yIntg[-1] + \
                 np.sum(coeffList[-1] * np.array([1, f[0][-1], f[0][-1]**2, f[0][-1]**3])) * (x[i] - f[0][-1])
        else:
            Y[i] = yIntg[pos[i]-1] + \
            np.sum(coeffList[pos[i]-1] * np.array([x[i], x[i]**2/2, x[i]**3/3, x[i]**4/4])) \
            - np.sum(coeffList[pos[i]-1] * np.array([f[0][pos[i]-1], f[0][pos[i]-1]**2/2, f[0][pos[i]-1]**3/3, f[0][pos[i]-1]**4/4]))
    return Y
    