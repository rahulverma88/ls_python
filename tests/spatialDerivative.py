#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:11:35 2020

@author: rahul
"""

import numpy as np

def upwindFirstFirst(data, dim):
    # dim 0 is x -> in data it is axis 1
    # dim 1 is y -> in data it is axis 0
    # dim 2 is z -> same
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D_minus = np.diff(data,prepend=1,axis=axis)
    D_plus = np.diff(data,append=1,axis=axis)
    
    return D_minus, D_plus


