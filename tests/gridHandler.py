#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:53:35 2020

@author: rahul
"""
import numpy as np

class Grid:
    ''' class structure for managing all grids'''
    
    def __init__(self, dim, gmin, gmax, dx, bdry = 'ghostExtrapolate'):
        self.dim = dim
        self.gmin = gmin
        self.gmax = gmax
        self.dx = dx
        self.bdry = bdry
    
    def getGridVals(self):
        if self.dim == 2:
            x = np.arange(self.gmin, self.gmax + self.dx, self.dx)
            y = np.arange(self.gmin, self.gmax + self.dx, self.dx)
            
            xv, yv = np.meshgrid(x,y)
            
            return xv, yv
        elif self.dim == 3:
            x = np.arange(self.gmin, self.gmax + self.dx, self.dx)
            y = np.arange(self.gmin, self.gmax + self.dx, self.dx)
            z = np.arange(self.gmin, self.gmax + self.dx, self.dx)
            
            xv, yv, zv = np.meshgrid(x,y,z)
            
            return xv, yv, zv

'''
Function for adding extrapolated boundary conditions.
Currently 2D only. Must add functionality for 3D
'''
def ghostExtrapolate(g, data, stencil):
    ny, nx = np.shape(data)
    
    # at lower x-end:
    sign = np.sign(data[:,0])
    abs_diff = np.abs(data[:,0]-data[:,1])
    slope = sign * abs_diff
        
    lower_bdry_x = np.array([data[:,0] + slope * (sten + 1) for sten in np.arange(stencil)])[::-1].transpose()
    
    # at upper x-end:
    sign = np.sign(data[:, nx - 1])
    abs_diff = np.abs(data[:, nx - 1]-data[:, nx - 2])
    slope = sign * abs_diff
    
    upper_bdry_x = np.array([data[:,nx - 1] + slope * (sten + 1) for sten in np.arange(stencil)]).transpose()
    
    
    data = np.concatenate((lower_bdry_x, data, upper_bdry_x),axis=1)
    
    # at lower y-end:
    sign = np.sign(data[0])
    abs_diff = np.abs(data[0]-data[1])
    slope = sign * abs_diff
    
    lower_bdry_y = np.array([data[0] + slope * (sten + 1) for sten in np.arange(stencil)])[::-1]
    
    # at upper y-end:
    sign = np.sign(data[ny - 1])
    abs_diff = np.abs(data[ny - 1]-data[ny - 2])
    slope = sign * abs_diff
    
    upper_bdry_y = np.array([data[ny - 1] + slope * (sten + 1) for sten in np.arange(stencil)])
    
    data = np.concatenate((lower_bdry_y, data, upper_bdry_y), axis=0)

    return data

