#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:11:35 2020

@author: rahul
"""

import numpy as np

def upwindFirstFirst(data, dim, grid):
    # dim 0 is x -> in data it is axis 1
    # dim 1 is y -> in data it is axis 0
    # dim 2 is z -> same
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D_minus = np.diff(data,prepend=1,axis=axis)/grid.dx
    D_plus = np.diff(data,append=1,axis=axis)/grid.dx
    
    
    return D_minus, D_plus


def upwindFirstENO2(data, dim, grid):
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D1_dim = np.diff(data, prepend=1,axis=axis)#/grid.dx#phi(i,j) - phi(i-1,j)
    D1_dim_plus = np.diff(data,append=1,axis=axis)#/grid.dx#phi(i+1,j) - phi(i,j)
    D2_dim = D1_dim_plus - D1_dim #phi(i+1,j) + phi(i-1,j) - 2*phi(i,j)
    D2_dim_plus = np.roll(D2_dim, -1, axis=axis)#phi(i+2,j) + phi(i,j) - 2*phi(i+1,j)
    D2_dim_minus = np.diff(D2_dim, 1, axis=axis)#phi(i-2,j) + phi(i,j) - 2*phi(i-1,j)
    
    cond_dim_plus = (np.abs(D2_dim) < np.abs(D2_dim_plus))
    cond_dim_minus = (np.abs(D2_dim_minus) < np.abs(D2_dim))
    
    phi_dim_plus = ((D1_dim_plus - 0.5*D2_dim)*cond_dim_plus + (D1_dim_plus - 0.5*D2_dim_plus)*(1-cond_dim_plus))/grid.dx
    
    phi_dim_minus = ((D1_dim + 0.5*D2_dim_minus)*cond_dim_minus + (D1_dim + 0.5*D2_dim)*(1-cond_dim_minus))/grid.dx

    return phi_dim_minus, phi_dim_plus

def upwindFirstENO3(data,dim,grid):
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    
