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
        
g = Grid(2,-1,1,1/100)
        