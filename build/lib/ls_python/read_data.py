#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:18:57 2020

@author: rahul
"""

import numpy as np
import matplotlib.pyplot as plt

data_folder = '/Users/rahul/Desktop/NewProjects/ls_python/data/bentheimer_sandstone/'

img = np.fromfile(data_folder + 'block00000000.nc',dtype='uint16')
#%%
