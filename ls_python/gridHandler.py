#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:53:35 2020

@author: rahul
"""
import numpy as np
from decimal import Decimal

class Grid:
    ''' class structure for managing all grids
    
        grid spacing dx is assumed to be constant in all dimensions for now.
        doesn't seem to make much sense complicating code without having full
        unstructured grids, or at least adaptive grids.
    
    '''
    
    def __init__(self, dim, gmin, gmax, dx, bdry = 'ghostExtrapolate'):
        '''
        Constructor assuming grid spacing dx is passed
        Passing both grid spacing dx and grid bounds is inherently inconsistent,
        because of floating point precision issues
        
        Here, the convention is taken of adjusting grid_max to remain consistent
        with the passed grid spacing. This is judged to be better as numerical
        algorithms are usually analyzed for error in terms of grid spacing and 
        it is likely a user would care more about that
        
        TO-DO:
        A different constructor which respects grid_max

        Parameters
        ----------
        dim : TYPE
            DESCRIPTION.
        gmin : TYPE
            DESCRIPTION.
        gmax : TYPE
            DESCRIPTION.
        dx : TYPE
            DESCRIPTION.
        bdry : TYPE, optional
            DESCRIPTION. The default is 'ghostExtrapolate'.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.gmin = np.array(gmin)
        actual_gmax = []
        self.gridShape = []
        for dim_gmax in gmax:
            # trying to determine grid max, while respecting grid spacing
            mod_val = Decimal(str(dim_gmax)) % Decimal(str(dx))
            if mod_val == 0:
                actual_gmax.append(dim_gmax)
            else:
                actual_gmax.append(dim_gmax + dx - float(mod_val))
        self.gmax = np.array(actual_gmax)
        self.dx = np.array(dx)
        self.bdry = bdry
        
        self.gridShape = [int((self.gmax[1]-self.gmin[1])/self.dx),
                          int((self.gmax[0]-self.gmin[0])/self.dx)
                          ]
        if dim == 3:
            self.gridShape.append(int((self.gmax[2]-self.gmin[2])/self.dx))
    
    def getGhostBounds(self, stencil):
        self.gmin_ghost = self.gmin - self.dx * stencil
        self.gmax_ghost = self.gmax + self.dx * stencil
        self.ghostShape = [self.gridShape[0] + 2 * stencil,
                                self.gridShape[1] + 2 * stencil]
        if self.dim == 3:
            self.ghostShape.append(self.gridShape[2] + 2 * stencil)
        
    def getGridVals(self):
        x = np.linspace(self.gmin[1], self.gmax[1], self.gridShape[1])
        y = np.linspace(self.gmin[0], self.gmax[0], self.gridShape[0])

        if self.dim == 2:                 
            xv, yv = np.meshgrid(x,y, indexing='xy')
            
            return xv, yv
        elif self.dim == 3:
            z = np.linspace(self.gmin[2], self.gmax[2], self.gridShape[2])
            
            xv, yv, zv = np.meshgrid(x,y,z, indexing='xy')
            
            return xv, yv, zv

    def getGridValsGhost(self):
        x = np.linspace(self.gmin_ghost[1], self.gmax_ghost[1], self.ghostShape[1])
        y = np.linspace(self.gmin_ghost[0], self.gmax_ghost[0], self.ghostShape[0])
 
        if self.dim == 2:
           
            xv, yv = np.meshgrid(x,y, indexing='xy')
            
            return xv, yv
        elif self.dim == 3:
            z = np.linspace(self.gmin_ghost[2], self.gmax_ghost[2], self.ghostShape[2])
            
            xv, yv, zv = np.meshgrid(x,y,z, indexing='xy')
            
            return xv, yv, zv
    '''
    Function for adding extrapolated boundary conditions.
    3D functionality to be verified
    '''
    def ghostExtrapolate(self, data, stencil):
        if self.dim == 2:
            ny, nx = np.shape(data)
            
            # at lower x-end:
            sign = np.sign(data[:,0])
            abs_diff = np.abs(data[:,0]-data[:,1])
            slope = sign * abs_diff
                
            lower_bdry_x = np.array([data[:,0] + slope * (sten + 1) for sten in np.arange(stencil)], ndmin = 2)[::-1].transpose()
            
            # at upper x-end:
            sign = np.sign(data[:, nx - 1])
            abs_diff = np.abs(data[:, nx - 1]-data[:, nx - 2])
            slope = sign * abs_diff
            
            upper_bdry_x = np.array([data[:,nx - 1] + slope * (sten + 1) for sten in np.arange(stencil)], ndmin = 2).transpose()
            
            data = np.concatenate((lower_bdry_x, data, upper_bdry_x),axis=1)
            
            # at lower y-end:
            sign = np.sign(data[0])
            abs_diff = np.abs(data[0]-data[1])
            slope = sign * abs_diff
            
            lower_bdry_y = np.array([data[0] + slope * (sten + 1) for sten in np.arange(stencil)], ndmin = 2)[::-1]
            
            # at upper y-end:
            sign = np.sign(data[ny - 1])
            abs_diff = np.abs(data[ny - 1]-data[ny - 2])
            slope = sign * abs_diff
            
            upper_bdry_y = np.array([data[ny - 1] + slope * (sten + 1) for sten in np.arange(stencil)], ndmin = 2)
            
            data = np.concatenate((lower_bdry_y, data, upper_bdry_y), axis=0)
        
            return data
        elif self.dim == 3:
            ny, nx, nz = np.shape(data)
            
            # at lower x-end:
            sign = np.sign(data[:,0,:])
            abs_diff = np.abs(data[:,0,:]-data[:,1,:])
            slope = sign * abs_diff
                
            lower_bdry_x = np.moveaxis(np.array([data[:,0,:] + slope * (sten + 1) for sten in np.arange(stencil)])[::-1],0,1)
            
            # at upper x-end:
            sign = np.sign(data[:, nx - 1, :])
            abs_diff = np.abs(data[:, nx - 1, :]-data[:, nx - 2, :])
            slope = sign * abs_diff
            
            upper_bdry_x = np.moveaxis(np.array([data[:,nx - 1, :] + slope * (sten + 1) for sten in np.arange(stencil)]),0,1)
            
            data = np.concatenate((lower_bdry_x, data, upper_bdry_x),axis=1)
            
            # at lower y-end:
            sign = np.sign(data[0,:,:])
            abs_diff = np.abs(data[0,:,:]-data[1,:,:])
            slope = sign * abs_diff
                
            lower_bdry_y = np.array([data[0,:,:] + slope * (sten + 1) for sten in np.arange(stencil)])[::-1]
            
            # at upper y-end:
            sign = np.sign(data[ny - 1, :,:])
            abs_diff = np.abs(data[ny - 1, :,:]-data[ny - 2, :,:])
            slope = sign * abs_diff
            
            upper_bdry_y = np.array([data[ny - 1, :, :] + slope * (sten + 1) for sten in np.arange(stencil)])
            
            data = np.concatenate((lower_bdry_y, data, upper_bdry_y),axis=0)
            
            # at lower z-end:
            sign = np.sign(data[:,:,0])
            abs_diff = np.abs(data[:,:,0]-data[:,:,1])
            slope = sign * abs_diff
                
            lower_bdry_z = np.moveaxis(np.array([data[:,:,0] + slope * (sten + 1) for sten in np.arange(stencil)])[::-1],0,2)
            
            # at upper y-end:
            sign = np.sign(data[:, :, nz - 1])
            abs_diff = np.abs(data[:, :, nz - 1] - data[:, :, nz - 2])
            slope = sign * abs_diff
            
            upper_bdry_z = np.moveaxis(np.array([data[:, :, nz - 1] + slope * (sten + 1) for sten in np.arange(stencil)]),0,2)
            
            data = np.concatenate((lower_bdry_z, data, upper_bdry_z),axis=2)
            
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

