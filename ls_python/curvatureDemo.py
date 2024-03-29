#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:50:50 2021

@author: rahul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:31:36 2020

@author: rahul

Inspired from Ian Mitchell's examples
level set with only normal velocity term
"""
import numpy as np
import importlib

from gridHandler import Grid
from terms import schemeData, fill_grid_norm, curvatureTerm
import odeCFL
importlib.reload(odeCFL)

from odeCFL import odeCFL1, odeCFL2, odeCFL3
from spatialDerivative import curvatureSecond#, upwindFirstFirst, upwindFirstENO2, upwindFirstENO3, upwindFirstWENO5
from options import Options

# Curvature coefficient
bValue = 0.5

# Integration parameters.
tMax = 0.1  # End time.
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
g_min = [-1, -1]  # ,-1]
g_max = [1, 1]  # ,1]
g_dx = 1 / 100

# set accuracy
accuracy = 'high'

if accuracy == 'low':
    stencil = 1
    timeInt = odeCFL1
    spatDeriv = curvatureSecond
elif accuracy == 'medium':
    stencil = 2
    timeInt = odeCFL2
    spatDeriv = curvatureSecond
elif accuracy == 'high':
    stencil = 3
    timeInt = odeCFL3
    spatDeriv = curvatureSecond
elif accuracy == 'veryHigh':
    stencil = 3
    timeInt = odeCFL3
    spatDeriv = curvatureSecond

grid = Grid(g_dim, g_min, g_max, g_dx, stencil)

#grid.getGhostBounds(stencil)

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

b_vel = fill_grid_norm(grid, [bValue])

schemeCurvature = schemeData(grid, curv_coef=b_vel)

t = t0
opts = Options()
#opts.factorCFL=0.9
# %%
while t < tMax:
    tSpan = [t, min(tMax, t + tPlot)]
    t, data_next = timeInt(data, tSpan, curvatureTerm, grid, schemeCurvature, spatDeriv, opts)
    print(t)
    data = data_next


