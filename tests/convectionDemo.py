#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:31:36 2020

@author: rahul

Inspired from Ian Mitchell's examples
level set with only convection velocity term

"""
import numpy as np
from gridHandler import Grid
from terms import schemeData, velocityTerm, fill_grid
from odeCFL import odeCFL1
from options import Options

# Integration parameters.
tMax = 1.0                 # End time.
plotSteps = 9              # How many intermediate plots to produce?
t0 = 0                     # Start time.
singleStep = 0             # Plot at each timestep (overrides tPlot).

# Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (plotSteps - 1)

# How close (relative) do we need to get to tMax to be considered finished?
eps = 1e-4
small = 100 * eps

# What level set should we view?
level = 0

# Pause after each plot?
pauseAfterPlot = 0

# Delete previous plot before showing next?
deleteLastPlot = 0

# Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 1

# Create the grid.
g_dim = 2
# minimum in each direction: x, y, and (if present) z
# Note the convention: always x, y and z except for gridShape
g_min = [-1, -2]#,-1]
g_max = [1, 4]#,1]
g_dx = 1/100

grid = Grid(g_dim, g_min, g_max, g_dx)


# set accuracy
accuracy = 'high'

if accuracy == 'low':
    stencil = 1
elif accuracy == 'medium':
    stencil = 2
elif accuracy == 'high':
    stencil = 3

grid.getGhostBounds(stencil)
#%%
# Create flow field
constV = np.zeros(2)
constV[1] = 1

# velocity is also defined with the same convention:
# [vx, vy [, vz]]
# However, note that inside each of vx, vy and vz, the shapes are corresponding 
# actual matrix dimensions
vel = fill_grid(grid, constV)
flowType = 'constant'

# Create initial conditions (a circle/sphere)

center = [0, 0.1, 0] # (x,y,z)
radius = 0.35
gridvals = grid.getGridVals()
data = np.zeros(grid.gridShape)

for i in range(0, grid.dim):
  data = data + np.power(gridvals[i] - center[i],2)

data = np.sqrt(data) - radius

#%%
data = grid.ghostExtrapolate(data, stencil)

data0 = data

schemeConvection = schemeData(grid, velocity=vel)

t = t0
opts = Options()

#%%
while t < tMax:
    tSpan = [ t, min(tMax, t + tPlot) ]
    t, data_next = odeCFL1(data, tSpan, velocityTerm, grid, schemeConvection, opts)
    print(t)
    data = data_next


    