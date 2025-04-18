#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess results for pressure bands in Figure 2

@author: chloe
"""
import numpy as np
import sys
import os

os.makedirs('/scratch/users/chloe1/postproc_uq/pressures/N_avg/max_pres', exist_ok=True)
os.makedirs('/scratch/users/chloe1/postproc_uq/pressures/N_avg/avg_exp', exist_ok=True)

N_tot         = int(2**9)
exp           = int(sys.argv[1]) 
loaded        = np.load('/scratch/users/chloe1/postproc_uq/pressures/pressure_in_exp_'+str(exp)+'.npy', allow_pickle=True)
time          = loaded.item().get('time')
time_steps    = len(time)
all_pressures = np.zeros((time_steps, N_tot))

for iter in range(0, N_tot):

    pressure = loaded.item().get('aorta:pressure_in:exp_'+str(exp)+'_iter'+str(iter))

    all_pressures[:,iter] = pressure

# Compute mean and std
N_range = [2**3, 2**5, 2**7, 2**9]
for N in N_range:
    mean_data = np.mean(all_pressures[:,:N], axis=1)
    std_data  = np.std(all_pressures[:,:N], axis=1)
    np.save('/scratch/users/chloe1/postproc_uq/pressures/N_avg/mean_pressure_in_exp_'+str(exp)+'_N_'+str(N)+'.npy', mean_data)
    np.save('/scratch/users/chloe1/postproc_uq/pressures/N_avg/std_pressure_in_exp_'+str(exp)+'_N_'+str(N)+'.npy', std_data)

#%% Obtain mean pressures
import numpy as np

mean_pressures = np.zeros((10000,100))

for N in [2**3, 2**5, 2**7, 2**9]:
    for exp in np.arange(0, 100):
        pressure         = np.load('/scratch/users/chloe1/postproc_uq/pressures/N_avg/mean_pressure_in_exp_'+str(exp)+'_N_'+str(N)+'.npy')
        mean_pressures[:,exp] = pressure/1333.22
    dict = {}
    dict['mean_pressure']   = np.mean(mean_pressures, axis=1)
    dict['median_pressure'] = np.median(mean_pressures, axis=1)
    dict['std_pressure']    = np.std(mean_pressures, axis=1)
    dict['time']            = np.linspace(0, 1.087, 10000)
    np.save('/scratch/users/chloe1/postproc_uq/pressures/N_avg/avg_exp/avg_pressure_in_N_'+str(N)+'.npy', dict)


#%% Obtain max pressures

import numpy as np
import matplotlib.pyplot as plt

max_pressure = np.zeros((100,))
for N in [2**3, 2**5, 2**7, 2**9]:
    for exp in np.arange(0, 100):
        pressure         = np.load('/scratch/users/chloe1/postproc_uq/pressures/N_avg/mean_pressure_in_exp_'+str(exp)+'_N_'+str(N)+'.npy')
        max_pressure[exp] = np.max(pressure)/1333.22
    
    dict = {}
    dict['max_pressure_mean'] = np.mean(max_pressure)
    dict['max_pressure_std']  = np.std(max_pressure)
    dict['max_pressure']      = max_pressure

    np.save('/scratch/users/chloe1/postproc_uq/pressures/N_avg/max_pres/max_pressure_in_N_'+str(N)+'.npy', dict)
