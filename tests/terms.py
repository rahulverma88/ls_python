#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:21:13 2020

@author: rahul
"""

import numpy as np
from spatialDerivative import upwindFirstFirst

class schemeData:
    def __init__(self, grid, **kwargs):
        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        
        if 'normal_vel' in kwargs:
            self.normal_vel = kwargs['normal_vel']
        
        if 'curv_coef' in kwargs:
            self.curv_coef = kwargs['curv_coef']

def fill_grid(grid, vel):
    '''
    Parameters
    ----------
    grid : TYPE
        DESCRIPTION.
    vel : velocity field
        could be entered at single value array (ex: [1]), 
        array with each dimension having single velocity (ex: [1,2,3], for 3D case)
        or array with each dimension having full velocity field set
        (ex: [[1,2,3]])
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if np.shape(vel) == (1,):
        print('Assuming velocity assigned in x direction')
        return [np.ones(grid.ghostShape)*vel, np.zeros(grid.ghostShape)]
    if len(vel) > 1:
        vel_field = np.array([np.ones(grid.ghostShape) * vel_comp for vel_comp in vel])
        
        return vel_field
        
def velocityTerm(data, grid, schemeData):
    vel = schemeData.velocity
    delta = np.zeros(data.shape)
    for dim in range(grid.dim):
        d_minus, d_plus = upwindFirstFirst(data, dim, grid)
        v = vel[dim]
        deriv = d_minus * (v > 0) + d_plus * (v < 0)
        delta += deriv * v#/grid.dx
    
    # calculate inverse of bound on time step
    step_inv = np.sum([np.max(np.abs(v))/grid.dx for v in vel])
    
    # -1 is for transferring term to RHS
    return -1 * delta, 1/step_inv

