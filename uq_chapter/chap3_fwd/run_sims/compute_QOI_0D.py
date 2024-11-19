#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute QoI from svZeroDSolver output file.

@author: chloe
"""
import numpy as np

def get_QOI_0D(filename, QOI_name=None):

    vessel_names = ['aorta', 'left_iliac', 'right_iliac']

    var_names = np.genfromtxt(filename, dtype = 'U', usecols=0, skip_header=1, delimiter=',')
    time      = np.genfromtxt(filename, usecols=1, skip_header=1, delimiter=',')
    data      = np.genfromtxt(filename, usecols=range(2,6), skip_header=1, delimiter=',')

    num_0d_cycles = 1
    num_0d_tsteps = len(np.nonzero(var_names == var_names[0])[0])
    steps_per_cycle = int((num_0d_tsteps+num_0d_cycles-1)/num_0d_cycles)

    zerod_data = {}
    for name in vessel_names:
        idxs = np.nonzero(var_names == name)[0]
        zerod_data[name+':flow_out'] = (data[idxs,1])[-steps_per_cycle:]
        zerod_data[name+':pressure_out'] = (data[idxs,3])[-steps_per_cycle:]

    flow_aorta = zerod_data['aorta:flow_out']
    pressure_aorta = zerod_data['aorta:pressure_out']

    flow_left     = zerod_data['left_iliac:flow_out']
    pressure_left = zerod_data['left_iliac:pressure_out']

    flow_right     = zerod_data['right_iliac:flow_out']
    pressure_right = zerod_data['right_iliac:pressure_out']
        
    QOI = {}

    QOI['flow:aorta:min'] = np.min(flow_aorta)
    QOI['flow:aorta:max'] = np.max(flow_aorta)
    QOI['flow:aorta:avg'] = np.mean(flow_aorta)

    QOI['pressure:aorta:min'] = np.min(pressure_aorta)
    QOI['pressure:aorta:max'] = np.max(pressure_aorta)
    QOI['pressure:aorta:avg'] = np.mean(pressure_aorta)
 
    QOI['flow:left:min'] = np.min(flow_left)
    QOI['flow:left:max'] = np.max(flow_left)
    QOI['flow:left:avg'] = np.mean(flow_left)

    QOI['pressure:left:min'] = np.min(pressure_left)
    QOI['pressure:left:max'] = np.max(pressure_left)
    QOI['pressure:left:avg'] = np.mean(pressure_left)

    QOI['flow:right:min'] = np.min(flow_right)
    QOI['flow:right:max'] = np.max(flow_right)
    QOI['flow:right:avg'] = np.mean(flow_right)

    QOI['pressure:right:min'] = np.min(pressure_right)
    QOI['pressure:right:max'] = np.max(pressure_right)
    QOI['pressure:right:avg'] = np.mean(pressure_right)

    if QOI_name is None:
        return QOI
    else:
        return QOI[QOI_name]
