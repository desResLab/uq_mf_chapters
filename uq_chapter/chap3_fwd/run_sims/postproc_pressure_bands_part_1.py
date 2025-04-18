#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess results for pressure bands in Figure 2

@author: chloe
"""
import numpy as np
import sys

N   = int(2**9)
exp = int(sys.argv[1])

zerod_data = {}

for iter in range(0, N):

    filename = '/oak/stanford/groups/amarsden/chloe1/scratch_backup_10_29_2024/uq_chapter/uq_chapter/zerod_sims/uni_50std/uni_exp_'+str(exp)+'/sim_'+str(iter)+'/aobif_'+str(iter)+'.csv'

    vessel_names = ['aorta']

    var_names = np.genfromtxt(filename, dtype = 'U', usecols=0, skip_header=1, delimiter=',')
    time      = np.genfromtxt(filename, usecols=1, skip_header=1, delimiter=',')
    data      = np.genfromtxt(filename, usecols=range(2,6), skip_header=1, delimiter=',')

    num_0d_cycles = 1
    num_0d_tsteps = len(np.nonzero(var_names == var_names[0])[0])
    steps_per_cycle = int((num_0d_tsteps+num_0d_cycles-1)/num_0d_cycles)

    # store the last cardiac cycle pressure waveform
    for name in vessel_names:
        idxs = np.nonzero(var_names == name)[0]
        zerod_data[name+':pressure_in:exp_'+str(exp)+'_iter'+str(iter)] = (data[idxs,2])[-steps_per_cycle:]

zerod_data['time'] = time[-steps_per_cycle:]
np.save('/scratch/users/chloe1/postproc_uq/pressures/pressure_in_exp_'+str(exp)+'.npy', zerod_data)
