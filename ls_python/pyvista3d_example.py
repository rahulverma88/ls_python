#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:49:52 2020

@author: rahul

pyvista 3d plotting

"""

from numpy import cos, pi, mgrid
import pyvista as pv
from pyvistaqt import BackgroundPlotter

#%% Data
x, y, z = pi*mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
vol = cos(x) + cos(y) + cos(z)
grid = pv.StructuredGrid(x, y, z)
grid["vol"] = vol.flatten()
contours = grid.contour([0])

#%% Visualization
pv.set_plot_theme('document')
p = pv.PlotterITK()
p.add_mesh(contours, scalars=contours.points[:, 2])#, show_scalar_bar=False)
p.show()