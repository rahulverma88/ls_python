#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:21:59 2020

@author: rahul
"""
import numpy as np

def clipData(data, grid):
    stencil = grid.stencil

    if grid.dim == 2:
        data = data[stencil:grid.ghostShape[0]-stencil, 
                    stencil:grid.ghostShape[1]-stencil].copy()
    else:
        data = data[stencil:grid.ghostShape[0]-stencil,
                    stencil:grid.ghostShape[1]-stencil,
                    stencil:grid.ghostShape[2]-stencil].copy()
    
    return data

def odeCFL1(data, tSpan, rhsFunc, grid, rhsData, spatDerivFunc, options):
    
    t = tSpan[0]
    
    while t < tSpan[1]:
        rhs, stepBound = rhsFunc(data, grid, rhsData, spatDerivFunc)
        deltaT = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)
        t = t + deltaT
        print('time inside odeCFL1: ', t)
        data_new = data + deltaT * rhs
        data = data_new
        
        data = clipData(data, grid)
        data = grid.ghostExtrapolate(data, grid.stencil)
        
    return t, data

def odeCFL2(data, tSpan, rhsFunc, grid, rhsData, spatDerivFunc, options):
    
    t = tSpan[0]
    
    while t < tSpan[1]:
        rhs, stepBound = rhsFunc(data, grid, rhsData, spatDerivFunc)
        deltaT = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)
        print('time inside odeCFL2: ', t)
        data_n_plus_1 = data + deltaT * rhs
        
        data_n_plus_1 = clipData(data_n_plus_1, grid)
        data_n_plus_1 = grid.ghostExtrapolate(data_n_plus_1, grid.stencil)
        
        rhs, stepBound = rhsFunc(data_n_plus_1, grid, rhsData, spatDerivFunc)
        deltaT_int = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)

        data_n_plus_2 = data_n_plus_1 + deltaT_int * rhs
        data = 0.5*data + 0.5*data_n_plus_2
        
        data = clipData(data, grid)
        data = grid.ghostExtrapolate(data, grid.stencil)
        
        t = t + deltaT

    return t, data

def odeCFL3(data, tSpan, rhsFunc, grid, rhsData, spatDerivFunc, options):
    
    t = tSpan[0]
    
    while t < tSpan[1]:
        rhs, stepBound = rhsFunc(data, grid, rhsData, spatDerivFunc)
        deltaT = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)
        print('time inside odeCFL3: ', t)
        data_n_plus_1 = data + deltaT * rhs
        
        data_n_plus_1 = clipData(data_n_plus_1, grid)
        data_n_plus_1 = grid.ghostExtrapolate(data_n_plus_1, grid.stencil)
        
        rhs, stepBound = rhsFunc(data_n_plus_1, grid, rhsData, spatDerivFunc)
        deltaT_int = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)

        data_n_plus_2 = data_n_plus_1 + deltaT_int * rhs
        data_n_plus_half = (3/4)*data + (1/4)*data_n_plus_2
        
        data_n_plus_half = clipData(data_n_plus_half, grid)
        data_n_plus_half = grid.ghostExtrapolate(data_n_plus_half, grid.stencil)
        
        rhs, stepBound = rhsFunc(data_n_plus_half, grid, rhsData, spatDerivFunc)
        deltaT_int = min(options.factorCFL * stepBound, tSpan[1] - t, options.maxStep)

        data_n_plus_3_2 = data_n_plus_half + deltaT_int * rhs
        
        data = (1/3)*data + (2/3)*data_n_plus_3_2
        
        data = clipData(data, grid)
        data = grid.ghostExtrapolate(data, grid.stencil)
        
        #if (data > 1e3).any():
           # print('boom')
        
        #if np.isnan(data).any():
           # pass
        
        t = t + deltaT

    
    return t, data
