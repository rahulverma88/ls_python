#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:31:36 2020

@author: rahul

Inspired from Ian Mitchell's examples
level set with only normal velocity term
"""
import numpy as np
from gridHandler import Grid
from terms import schemeData, normalTerm, fill_grid
from odeCFL import odeCFL1, odeCFL2, odeCFL3
from spatialDerivative import upwindFirstFirst, upwindFirstENO2, upwindFirstENO3, upwindFirstWENO5
from options import Options

# Speed of motion normal to the interface.
aValue = 0.25

# Integration parameters.
tMax = 1.0  # End time.
plotSteps = 9  # How many intermediate plots to produce?
t0 = 0  # Start time.
singleStep = 0  # Plot at each timestep (overrides tPlot).

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
g_min = [-1, -2]  # ,-1]
g_max = [1, 4]  # ,1]
g_dx = 1 / 100

grid = Grid(g_dim, g_min, g_max, g_dx)

# set accuracy
accuracy = 'high'

if accuracy == 'low':
    stencil = 1
    timeInt = odeCFL1
    spatDeriv = upwindFirstFirst
elif accuracy == 'medium':
    stencil = 2
    timeInt = odeCFL2
    spatDeriv = upwindFirstENO2
elif accuracy == 'high':
    stencil = 3
    timeInt = odeCFL3
    spatDeriv = upwindFirstENO3
elif accuracy == 'veryHigh':
    stencil = 3
    timeInt = odeCFL3
    spatDeriv = upwindFirstWENO5

grid.getGhostBounds(stencil)

# Create initial conditions (a circle/sphere)
center = [0, 0.1, 0]  # (x,y,z)
radius = 0.35
gridvals = grid.getGridVals()
data = np.zeros(grid.gridShape)

for i in range(0, grid.dim):
    data = data + np.power(gridvals[i] - center[i], 2)

data = np.sqrt(data) - radius

# %%
data = grid.ghostExtrapolate(data, stencil)

data0 = data

schemeNormal = schemeData(grid, normal_vel=aValue)

t = t0
opts = Options()

# %%
while t < tMax:
    tSpan = [t, min(tMax, t + tPlot)]
    t, data_next = timeInt(data, tSpan, normalTerm, grid, schemeNormal, spatDeriv, opts)
    print(t)
    data = data_next


