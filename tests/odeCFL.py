#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:21:59 2020

@author: rahul
"""

def odeCFL1(data, tSpan, rhsFunc, grid, rhsData, options):
    
    t = tSpan[0]
    
    while t < tSpan[1]:
        rhs, stepBound = rhsFunc(data, grid, rhsData)
        deltaT = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)
        t = t + deltaT
        print('time inside odeCFL1: ', t)
        data_new = data + deltaT * rhs
        data = data_new
    
    return t, data
