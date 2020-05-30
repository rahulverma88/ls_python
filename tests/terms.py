#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:21:13 2020

@author: rahul
"""

import numpy as np
from spatialDerivative import upwindFirstFirst

def getVelocityTerm(data, grid, vel):
    delta = np.zeros(data.shape)
    for dim in range(grid.dim):
        d_minus, d_plus = upwindFirstFirst(data, dim)
        v = vel[dim]
        deriv = d_minus * (v > 0) + d_plus * (v < 0)
        delta += deriv * v
        
    return -1 * delta
    