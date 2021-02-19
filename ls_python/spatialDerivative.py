#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:11:35 2020

@author: rahul
"""

import numpy as np

def getWENOApprox(v1,v2,v3,v4,v5):
    phi_x_1 = v1/3 - 7*v2/6 + 11*v3/6
    phi_x_2 = -v2/6 + 5*v3/6 + v4/3
    phi_x_3 = v3/3 + 5*v4/6 - v5/6
    
    S1 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - 4*v2+3*v3)**2
    S2 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(v2 - v4)**2
    S3 = (13/12)*(v3 - 2*v4 + v5)**2 + (1/4)*(3*v3 -4*v4 + v5)**2
    
    eps = 1e-6*max(v1**2, v2**2, v3**2, v4**2, v5**2) + 10e-99
    
    alpha_1 = 0.1/(S1 + eps)**2
    alpha_2 = 0.6/(S2 + eps)**2
    alpha_3 = 0.3/(S3 + eps)**2
    
    sum_alpha = alpha_1 + alpha_2 + alpha_3
    omega_1 = alpha_1/sum_alpha
    omega_2 = alpha_2/sum_alpha
    omega_3 = alpha_3/sum_alpha
    
    return omega_1*phi_x_1 + omega_2*phi_x_2 + omega_3*phi_x_3

def upwindFirstFirst(data,dim,grid):
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D1_minus_half = np.diff(data, prepend=1,axis=axis)/grid.dx #phi(i,j) - phi(i-1,j)
 
    D1_plus_half = np.roll(D1_minus_half, -1, axis=axis) #phi(i+1,j) - phi(i,j)
    
    return D1_minus_half, D1_plus_half

    
def upwindFirstENO2(data,dim,grid):
    '''
    Second order accurate upwind derivatives using ENO2
    
    data: 2d or 3d Numpy array
    dim: dimension on which gradients are to be calculated
        I assume dim 0 is x -> so axis 1 in a numpy array
                 dim 1 is y -> axis 0 in numpy array
                 dim 3 is z (same)
        
    grid: grid object - here is used mainly for storing grid spacing, dx
    
    returns: upwind derivative phi_{dim}_plus, downwind derivative phi_{dim}_minus
    '''
    if dim == 0:
        axis = 1
    elif dim == 1:
        axis = 0
    else:
        axis = dim
        
    D1_minus_half = np.diff(data, prepend=1,axis=axis)/grid.dx #phi(i,j) - phi(i-1,j)
    D1_minus_3_2 = np.roll(D1_minus_half, 1, axis=axis) #phi(i-1,j) - phi(i-2,j)
    
    D1_plus_half = np.roll(D1_minus_half, -1, axis=axis) #phi(i+1,j) - phi(i,j)
    D1_plus_3_2 = np.roll(D1_minus_half, -2, axis=axis) #phi(i+2,j) - phi(i+1,j)
            
    D2 = (D1_plus_half - D1_minus_half)/(2*grid.dx) # defined at original grid point i
    D2_minus_1 = (D1_minus_half - D1_minus_3_2)/(2*grid.dx)
    D2_plus_1 = (D1_plus_3_2- D1_plus_half)/(2*grid.dx)

    cond_D2_plus = (np.abs(D2) < np.abs(D2_plus_1))
    cond_D2_minus = (np.abs(D2_minus_1) < np.abs(D2))
    
    phi_dim_plus = D1_plus_half + \
        (cond_D2_plus * D2 + (1 - cond_D2_plus) * D2_plus_1) * grid.dx
                
    phi_dim_minus = D1_minus_half + \
                    (cond_D2_minus * D2_minus_1 + (1-cond_D2_minus) * D2)*grid.dx         
    return phi_dim_minus, phi_dim_plus

def upwindFirstENO3(data,dim,grid):
    '''
    Third order accurate upwind derivatives
    
    data: 2d or 3d Numpy array
    dim: dimension on which gradients are to be calculated
        I assume dim 0 is x -> so axis 1 in a numpy array
                 dim 1 is y -> axis 0 in numpy array
                 dim 3 is z (same)
        
    grid: grid object - here is used mainly for storing grid spacing, dx
    
    returns: upwind derivative phi_{dim}_plus, downwind derivative phi_{dim}_minus
    '''
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
            
    D2 = (D1_plus_half - D1_minus_half)/(2*grid.dx) # defined at original grid point i
    D2_minus_1 = (D1_minus_half - D1_minus_3_2)/(2*grid.dx)
    D2_minus_2 = (D1_minus_3_2 - D1_minus_5_2)/(2*grid.dx)
    D2_plus_1 = (D1_plus_3_2- D1_plus_half)/(2*grid.dx)
    D2_plus_2 = (D1_plus_5_2-D1_plus_3_2)/(2*grid.dx)
            
    D3_minus_half = (D2 - D2_minus_1)/(3*grid.dx)
    D3_minus_3_2 = (D2_minus_1 - D2_minus_2)/(3*grid.dx)  

    D3_plus_half = (D2_plus_1 - D2)/(3*grid.dx)
    D3_plus_3_2 = (D2_plus_2 - D2_plus_1)/(3*grid.dx)

    cond_D2_plus = (np.abs(D2) < np.abs(D2_plus_1))
    cond_D2_minus = (np.abs(D2_minus_1) < np.abs(D2))

    cond_D3_plus_half_3_2 = (np.abs(D3_plus_half) < np.abs(D3_plus_3_2))    
    cond_D3_minus_3_2_half = (np.abs(D3_minus_3_2) < np.abs(D3_minus_half))
    cond_D3_minus_plus_half = (np.abs(D3_minus_half) < np.abs(D3_plus_half))
    
    
    phi_dim_plus = D1_plus_half + \
        (cond_D2_plus * D2 + (1 - cond_D2_plus) * D2_plus_1) * grid.dx + \
        (cond_D2_plus * (cond_D3_minus_plus_half * D3_minus_half + (1-cond_D3_minus_plus_half) * D3_plus_half) * (-1) + \
         (1-cond_D2_plus) * (cond_D3_plus_half_3_2 * D3_plus_half + (1-cond_D3_plus_half_3_2) * D3_plus_3_2) * 2) * (grid.dx ** 2)    
                
    phi_dim_minus = D1_minus_half + \
                    (cond_D2_minus * D2_minus_1 + (1-cond_D2_minus) * D2)*grid.dx + \
                    (cond_D2_minus * (cond_D3_minus_3_2_half * D3_minus_3_2 + (1-cond_D3_minus_3_2_half) * D3_minus_half)*2 + \
                    (1-cond_D2_minus) * (cond_D3_minus_plus_half * D3_minus_half + (1-cond_D3_minus_plus_half) * D3_plus_half) * (-1))*(grid.dx ** 2)
        
    return phi_dim_minus, phi_dim_plus

def upwindFirstWENO5(data, dim, grid):
    '''
    Fifth order accurate weighted ENO derivatives
    
    data: 2d or 3d Numpy array
    dim: dimension on which gradients are to be calculated
        I assume dim 0 is x -> so axis 1 in a numpy array
                 dim 1 is y -> axis 0 in numpy array
                 dim 3 is z (same)
        
    grid: grid object - here is used mainly for storing grid spacing, dx
    
    returns: upwind derivative phi_{dim}_plus, downwind derivative phi_{dim}_minus
    '''
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
    
     
    # phi_minus
    v1 = D1_minus_5_2
    v2 = D1_minus_3_2
    v3 = D1_minus_half
    v4 = D1_plus_half
    v5 = D1_plus_3_2
    
    phi_dim_minus = getWENOApprox(v1, v2, v3, v4, v5)
    
    # phi_plus
    v1 = D1_plus_5_2
    v2 = D1_plus_3_2
    v3 = D1_plus_half
    v4 = D1_minus_half
    v5 = D1_minus_3_2
    
    phi_dim_plus = getWENOApprox(v1, v2, v3, v4, v5)
    
    return phi_dim_minus, phi_dim_plus

def curvatureSecond(data, grid):
    '''
    Parameters
    ----------
    data
    grid

    Returns
    -------

    Calculates: phi_x, phi_y, (and phi_z, if 3d). Then for Hessian, phi_xx, phi_yy
    (and phi_zz for 3d), phi_xy, (and phi_xz, phi_yz for 3d). All center-difference
    '''

    phi_x = np.gradient(data, grid.dx, axis=1)
    phi_y = np.gradient(data, grid.dx, axis=0)

    phi_xy = np.gradient(phi_x, grid.dx, axis=0)

    phi_xx = np.gradient(data, grid.dx, axis=1)
    phi_yy = np.gradient(data, grid.dx, axis=0)

    if grid.dim == 3:
        phi_z = np.gradient(data, grid.dx, axis=2)
        phi_zz = np.gradient(data, grid.dx, axis=2)
        phi_xz = np.gradient(phi_x, grid.dx, axis=2)
        phi_yz = np.gradient(phi_y, grid.dx, axis=2)

    gradMag2 = phi_x ^ 2 + phi_y ^ 2

    if grid.dim == 3:
        gradMag2 += phi_z ^ 2

    gradMag = np.sqrt(gradMag2)

    curv_num = phi_x ^ 2 * phi_yy - 2 * phi_x * phi_y * phi_xy + phi_y ^ 2 * phi_xx

    if grid.dim == 3:
        curv_num += phi_x ^ 2 * phi_zz - 2 * phi_x * phi_y * phi_xz + phi_z ^ 2 * phi_xx + \
                    phi_y ^ 2 * phi_zz - 2 * phi_y * phi_z * phi_yz + \
                    phi_z ^ 2 * phi_yy

    curv = curv_num/gradMag ^ 3

    return curv


def upwindFirstFirst_old(data, dim, grid):
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


def upwindFirstENO2_old(data, dim, grid):
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

def upwindFirstENO3_old(data,dim,grid):
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
