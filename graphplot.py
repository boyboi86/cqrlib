# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:50:20 2020

@author: Wei_X
"""

# Import necesary libraries
import matplotlib.pyplot as plt

p = print

#periods = [df1['period90'], df1['period180'], df1['period252'], df1['period360'], df1['period504'], df1['period600']]
#labels = ['period90', 'period180', 'period252', 'period360', 'period504', 'period600']

def multipleplotkde(periods, labels, title, xlabel, ylabel):
    periods = periods
    plt.figure(figsize=(15,8))
    for period in periods:
        period.plot.kde()
        
    # Plot formatting
    plt.legend(labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

def multipleplot(periods, labels, title, xlabel, ylabel):
    periods = periods
    plt.figure(figsize=(15,8))
    for period in periods:
        period.plot()
        
    # Plot formatting
    plt.legend(labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    