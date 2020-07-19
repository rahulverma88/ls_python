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
        
    D1_dim = np.diff(data, prepend=1,axis=axis) #phi(i,j) - phi(i-1,j)
    D1_dim_plus = np.roll(D1_dim, -1, axis=axis) #phi(i+1,j) - phi(i,j)
    D1_dim_minus = np.roll(D1_dim, 1, axis=axis) #phi(i-1,j) - phi(i-2,j)
    D1_dim_minus_minus = np.roll(D1_dim, 2, axis=axis)#phi(i-2,j) - phi(i-3,j)
    D1_dim_plus_plus = np.roll(D1_dim, -2, axis=axis) #phi(i+2,j) - phi(i+1,j)
    D1_dim_plus_plus_plus = np.roll(D1_dim, -3, axis=axis) #phi(i+3,j) - phi(i+2,j)
            
    D2_dim = D1_dim_plus - D1_dim
    D2_dim_plus = D1_dim_plus_plus - D1_dim_plus
    D2_dim_minus = D1_dim - D1_dim_minus
    D2_dim_plus_plus = D1_dim_plus_plus_plus - D1_dim_plus_plus
    D2_dim_minus_minus = D1_dim_minus - D1_dim_minus_minus
            
    D3_dim = D2_dim - D2_dim_minus
    D3_dim_plus = D2_dim_plus - D2_dim
    D3_dim_plus_plus = D2_dim_plus_plus - D2_dim_plus
    D3_dim_minus = D2_dim_minus - D2_dim_minus_minus  

    cond_dim_plus = (np.abs(D2_dim) < np.abs(D2_dim_plus))
    cond_dim_D3_plus = (np.abs(D3_dim) < np.abs(D3_dim_plus))
    cond_dim_D3_plus_plus = (np.abs(D3_dim) < np.abs(D3_dim_plus_plus))
    
    cond_dim_minus = (np.abs(D2_dim_minus) < np.abs(D2_dim))
    cond_dim_D3_minus = (np.abs(D3_dim_minus) < np.abs(D3_dim))
    cond_dim_D3_minus_minus = (np.abs(D3_dim) < np.abs(D3_dim_plus))
    
    phi_dim_plus = ((D1_dim_plus + cond_dim_D3_plus * (-1/6*D3_dim)+ 
                     (1-cond_dim_D3_plus) * (-1/6*D3_dim_plus) - 1/2*D2_dim)*cond_dim_plus +
                    (D1_dim_plus + cond_dim_D3_plus_plus *(1/3*D3_dim_plus) +
                     (1-cond_dim_D3_plus_plus) *(1/3*D3_dim_plus_plus)- 0.5*D2_dim_plus)*(1-cond_dim_plus))/grid.dx

    phi_dim_minus = ((D1_dim + cond_dim_D3_minus * (1/3*D3_dim_minus)+ 
                     (1-cond_dim_D3_minus) * (1/3*D3_dim) + 1/2*D2_dim_minus)*cond_dim_minus +
                    (D1_dim + cond_dim_D3_minus_minus *(-1/6*D3_dim) +
                     (1-cond_dim_D3_minus_minus) *(-1/6*D3_dim_plus)+ 0.5*D2_dim)*(1-cond_dim_minus))/grid.dx
    
    return phi_dim_minus, phi_dim_plus

def upwindFirstENO3_new(data,dim,grid):
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D1_minus_half = np.diff(data, prepend=1,axis=axis)/grid.dx #phi(i,j) - phi(i-1,j)
    D1_minus_3_2 = np.roll(D1_minus_half, 1, axis=axis) #phi(i-1,j) - phi(i-2,j)
    D1_minus_5_2 = np.roll(D1_minus_half, 2, axis=axis)#phi(i-2,j) - phi(i-3,j)
    
    D1_plus_half = np.roll(D1_minus_half, -1, axis=axis) #phi(i+1,j) - phi(i,j)
    D1_plus_3_2 = np.roll(D1_minus_half, -2, axis=axis) #phi(i+2,j) - phi(i+1,j)
    D1_plus_5_2 = np.roll(D1_minus_half, -3, axis=axis) #phi(i+3,j) - phi(i+2,j)
            
    D2 = (D1_plus_half - D1_minus_half)/(2*grid.dx)
    D2_minus_1 = (D1_minus_half - D1_minus_3_2)/(2*grid.dx)
    D2_minus_2 = (D1_minus_3_2 - D1_minus_5_2)/(2*grid.dx)
    D2_plus_1 = (D1_plus_3_2- D1_plus_half)/(2*grid.dx)
    D2_plus_2 = (D1_plus_5_2-D1_plus_3_2)/(2*grid.dx)
    #D2_dim_minus_minus = D1_dim_minus - D1_dim_minus_minus
            
    D3_ = D2_dim - D2_dim_minus
    D3_dim_plus = D2_dim_plus - D2_dim
    D3_dim_plus_plus = D2_dim_plus_plus - D2_dim_plus
    D3_dim_minus = D2_dim_minus - D2_dim_minus_minus  

    cond_dim_plus = (np.abs(D2_dim) < np.abs(D2_dim_plus))
    cond_dim_D3_plus = (np.abs(D3_dim) < np.abs(D3_dim_plus))
    cond_dim_D3_plus_plus = (np.abs(D3_dim) < np.abs(D3_dim_plus_plus))
    
    cond_dim_minus = (np.abs(D2_dim_minus) < np.abs(D2_dim))
    cond_dim_D3_minus = (np.abs(D3_dim_minus) < np.abs(D3_dim))
    cond_dim_D3_minus_minus = (np.abs(D3_dim) < np.abs(D3_dim_plus))
    
    phi_dim_plus = ((D1_dim_plus + cond_dim_D3_plus * (-1/6*D3_dim)+ 
                     (1-cond_dim_D3_plus) * (-1/6*D3_dim_plus) - 1/2*D2_dim)*cond_dim_plus +
                    (D1_dim_plus + cond_dim_D3_plus_plus *(1/3*D3_dim_plus) +
                     (1-cond_dim_D3_plus_plus) *(1/3*D3_dim_plus_plus)- 0.5*D2_dim_plus)*(1-cond_dim_plus))/grid.dx

    phi_dim_minus = ((D1_dim + cond_dim_D3_minus * (1/3*D3_dim_minus)+ 
                     (1-cond_dim_D3_minus) * (1/3*D3_dim) + 1/2*D2_dim_minus)*cond_dim_minus +
                    (D1_dim + cond_dim_D3_minus_minus *(-1/6*D3_dim) +
                     (1-cond_dim_D3_minus_minus) *(-1/6*D3_dim_plus)+ 0.5*D2_dim)*(1-cond_dim_minus))/grid.dx
    
    return phi_dim_minus, phi_dim_plus
