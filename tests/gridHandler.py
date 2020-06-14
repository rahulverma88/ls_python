#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:53:35 2020

@author: rahul
"""
import numpy as np
from decimal import Decimal

class Grid:
    ''' class structure for managing all grids'''
    
    def __init__(self, dim, gmin, gmax, dx, bdry = 'ghostExtrapolate'):
        self.dim = dim
        self.gmin = np.array(gmin)
        actual_gmax = []
        for dim_gmax in gmax:
            mod_val = Decimal(str(dim_gmax)) % Decimal(str(dx))
            if mod_val == 0:
                actual_gmax.append(dim_gmax)
            else:
                actual_gmax.append(dim_gmax + dx - float(mod_val))
        self.gmax = np.array(actual_gmax)
        self.dx = np.array(dx)
        self.bdry = bdry
        self.gridShape = (len(np.arange(self.gmin[1],self.gmax[1] + self.dx, self.dx)),
                                len(np.arange(self.gmin[0],self.gmax[0] + self.dx, self.dx))
                                )
    
    def getGhostBounds(self, stencil):
        self.gmin_ghost = self.gmin - self.dx * stencil
        self.gmax_ghost = self.gmax + self.dx * stencil
        self.ghostShape = (self.gridShape[0] + 2 * stencil,
                                self.gridShape[1] + 2 * stencil)
        
    def getGridVals(self):
        if self.dim == 2:
            x = np.arange(self.gmin[1], self.gmax[1] + self.dx, self.dx)
            y = np.arange(self.gmin[0], self.gmax[0] + self.dx, self.dx)
            
            xv, yv = np.meshgrid(x,y)
            
            return xv, yv
        elif self.dim == 3:
            x = np.arange(self.gmin[1], self.gmin[1] + self.dx, self.dx)
            y = np.arange(self.gmin[0], self.gmin[0] + self.dx, self.dx)
            z = np.arange(self.gmin[2], self.gmax[2]+ self.dx, self.dx)
            
            xv, yv, zv = np.meshgrid(x,y,z)
            
            return xv, yv, zv

    def getGridValsGhost(self):
        if self.dim == 2:
            x = np.arange(self.gmin_ghost[1], self.gmax_ghost[1] + self.dx, self.dx)
            y = np.arange(self.gmin_ghost[0], self.gmax_ghost[0] + self.dx, self.dx)
            
            xv, yv = np.meshgrid(x,y)
            
            return xv, yv
        elif self.dim == 3:
            x = np.arange(self.gmin_ghost[1], self.gmin_ghost[1] + self.dx, self.dx)
            y = np.arange(self.gmin_ghost[0], self.gmin_ghost[0] + self.dx, self.dx)
            z = np.arange(self.gmin_ghost[2], self.gmin_ghost[2]+ self.dx, self.dx)
            
            xv, yv, zv = np.meshgrid(x,y,z)
            
            return xv, yv, zv
    '''
    Function for adding extrapolated boundary conditions.
    Currently 2D only. Must add functionality for 3D
    '''
    def ghostExtrapolate(self, data, stencil):
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

