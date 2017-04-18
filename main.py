#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:17:25 2017

@author: Quintus
"""

import pandas as pd
import numpy as np
from datetime import date
from scipy import optimize as opt
import matplotlib.pyplot as plt

from cubicSpline import cubicSpline, cubicSplineIntg

#==============================================================================
# process the data
#==============================================================================
def euroDollarDaysCount(ticker, today):
    symbSet = ['F','G','H','J','K','M','N','Q','U','V','X','Z']
    monthSet = list(range(1,13))
    futuresSymb = dict(zip(symbSet, monthSet))
    month = futuresSymb[ticker[-2]]
    year = 2010 + int(ticker[-1]) if int(ticker[-1]) >= 6 else 2020 + int(ticker[-1])
    settleDay = date(year, month, 16)
    return (settleDay - today).days + 90

def swapDaysCount(ticker, today):
    year = int(ticker.split('W')[1]) + 2016
    maturity = date(year, 1, 11) # ?
    return (maturity - today).days

data = pd.read_excel('/Users/Quintus/Dropbox/python code/Projects/inputs.xlsx', header=2)
today = date(2016, 1, 7)
data['Ticker'] = data['Ticker'].str.strip()
data['DaysCount'] = 0 # create new column called DayCount

data.loc[0, 'DaysCount'] = 2 # US libor US0003M setteled 2 business days from today
posED = data['Ticker'].str.startswith('ED')
data.loc[posED, 'DaysCount'] = data.loc[posED, 'Ticker'].apply(euroDollarDaysCount, args=(today,))
posSW = data['Ticker'].str.startswith('USSW')
data.loc[posSW, 'DaysCount'] = data.loc[posSW, 'Ticker'].apply(swapDaysCount, args=(today,))

# select
dataSelected = data.loc[data['Selected'] == 1].copy()
dataSelected.reset_index(drop=True, inplace=True)

# calculate nodes(in years)
l = dataSelected['DaysCount'].iloc[0:-1].values
l = np.insert(l, 0, 0)
r = dataSelected['DaysCount'].iloc[0:].values
dataSelected['Nodes'] = pd.Series((l + r) / 2 / 360)

#==============================================================================
# fitting the forward curve
#==============================================================================
MaxNumLoops = 20
Sigma = 0.005
numNodes = dataSelected.shape[0]

time = dataSelected['Nodes'].values
initialFR = np.ones(numNodes)*0.02
f = np.vstack((time, initialFR))


for n in range(MaxNumLoops):
    for i in range(numNodes):
        maturity = dataSelected.loc[i, 'DaysCount']/360
        # US Treasury
        if dataSelected.loc[i, 'Ticker'].endswith('M'):
            firstCF = 1
            secondCF = 1 + dataSelected.loc[i, 'Quote']/100*(30/360)
            dcf = lambda instFR: firstCF * np.exp(-cubicSplineIntg(f, np.array([2/360]), instFR, i)) \
            - secondCF * np.exp(-cubicSplineIntg(f, np.array([32/360]), instFR, i))
            
        elif dataSelected.loc[i, 'Ticker'].startswith('ED'):
            firstCF = 0.75 + 0.25*dataSelected.loc[i, 'Quote']/100 + 0.125*Sigma**2*maturity*(maturity-0.25)
            secondCF = 1
            dcf = lambda instFR: firstCF * np.exp(-cubicSplineIntg(f, np.array([maturity-92/360]), instFR, i)) \
            - secondCF * np.exp(-cubicSplineIntg(f, np.array([maturity]), instFR, i))
            
        else:
            periods = dataSelected.loc[i, 'DaysCount']//180
            quote = dataSelected.loc[i, 'Quote']/100
            dcf = lambda instFR: np.exp(-cubicSplineIntg(f, np.array([2/360]), instFR, i)) \
            - np.sum(0.5*quote * np.exp(-cubicSplineIntg(f, (2+180*np.arange(1,periods))/360, instFR, i))) \
            -(1+0.5*quote) * np.exp(-cubicSplineIntg(f, np.array([maturity]), instFR, i))    
        f[1][i] = opt.fsolve(dcf, 0)         
            
#==============================================================================
# plotting
#==============================================================================
instFR = cubicSpline(f, np.arange(0,30,0.001))
zeroR = cubicSplineIntg(f, np.arange(1,30,0.001), f[1][0], 0) / np.arange(1,30,0.001)

plt.figure()
plt.plot(np.arange(0,30,0.001), instFR, '-', c='dodgerblue')
plt.scatter(f[0], f[1], c='orangered', marker='o', s=30, edgecolors='orangered')
plt.plot(np.arange(1,30,0.001), zeroR, '-', c='orange')

axes = plt.gca()
axes.set_xlim([0,35])
axes.set_ylim([0.005,0.030001])

plt.grid(True, which='major', c='lightslategrey', linestyle='-', linewidth=0.5)
plt.xlabel('Time in  years')
plt.ylabel('Rate')
plt.title('Instantaneous Forward Curve')
plt.legend(['Instantaneous Rate', 'Zero Rate', 'Nodes'], loc=4, frameon=True, fontsize='medium')
plt.show()

