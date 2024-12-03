# One-time generation of random observation

#%% Before running svZeroDSolver

import numpy as np
import scipy.io as sio
import torch

dirpath = '/Users/chloe/Documents/Stanford/ME398_Spring/marsden_uq/uq_mf_chapters/uq_chapter/chap6_inv/'

Rp_bounds = 0.5
C_bounds  = 0.5
Rd_bounds = 0.5

# load the RCR values
Rp = 6.8123e2
C  = 3.6664e-5
Rd = 3.1013e4

Rp = np.random.uniform(Rp-Rp_bounds*Rp, Rp+Rp_bounds*Rp)
C  = np.random.uniform(C-C_bounds*C,    C+C_bounds*C)
Rd = np.random.uniform(Rd-Rd_bounds*Rd, Rd+Rd_bounds*Rd)

# save the RCR values
sio.savemat(dirpath+'data/x_obs.mat', {'Rp':Rp, 'C':C, 'Rd':Rd})

#%% Generate svZeroDSolver input file
import pandas as pd

# create the json file and open it
svzerod_json = open(dirpath+'data/aobif_obs.json','w')

# distal pressure used for RCR boundary conditions
P_dist = 4000 # dynes/cm^2

# define an indent
indent = '    '

# begin dictionary
svzerod_json.write('{' + '\n')

# write simulation parameters -----------------
svzerod_json.write(indent+'"simulation_parameters": {'+'\n')
svzerod_json.write(indent+indent+'"number_of_cardiac_cycles": 10,\n')
svzerod_json.write(indent+indent+'"number_of_time_pts_per_cardiac_cycle": 10000\n')
svzerod_json.write(indent+'},'+'\n')

# write boundary conditions --------------------
svzerod_json.write(indent+'"boundary_conditions": ['+'\n')

# load the inflow data -------------------------
inflowFile = pd.read_csv('aobif_inflow.csv', sep=' ')
t_inflow   = inflowFile['time'].values
Q_inflow   = inflowFile['flow'].values

# inflow boundary condition --------------------
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_name": "INFLOW",'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_type": "FLOW",'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_values": {'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Q": ['+'\n')
for q in Q_inflow[0:-1]:
    svzerod_json.write(indent+indent+indent+indent+indent+'{0:.10f},'.format(q)+'\n')
svzerod_json.write(indent+indent+indent+indent+indent+'{0:.10f}'.format(Q_inflow[-1])+'\n')
svzerod_json.write(indent+indent+indent+indent+'],'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"t": ['+'\n')
for t in t_inflow[0:-1]:
    svzerod_json.write(indent+indent+indent+indent+indent+'{0:.10f},'.format(t)+'\n')
svzerod_json.write(indent+indent+indent+indent+indent+'{0:.10f}'.format(t_inflow[-1])+'\n')
svzerod_json.write(indent+indent+indent+indent+']'+'\n')
svzerod_json.write(indent+indent+indent+'}'+'\n')
svzerod_json.write(indent+indent+'},'+'\n')

# RCR boundary conditions ----------------------
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_name": "BC_RCR0",\n')
svzerod_json.write(indent+indent+indent+'"bc_type": "RCR",'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_values": {'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Rp": {0:.10f},'.format(Rp)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Rd": {0:.10f},'.format(Rd)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"C": {0:.10f},'.format(C)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Pd": {0:.10f}'.format(P_dist)+'\n')
svzerod_json.write(indent+indent+indent+'}'+'\n')
svzerod_json.write(indent+indent+'},'+'\n')
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_name": "BC_RCR1",\n')
svzerod_json.write(indent+indent+indent+'"bc_type": "RCR",'+'\n')
svzerod_json.write(indent+indent+indent+'"bc_values": {'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Rp": {0:.10f},'.format(Rp)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Rd": {0:.10f},'.format(Rd)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"C": {0:.10f},'.format(C)+'\n')
svzerod_json.write(indent+indent+indent+indent+'"Pd": {0:.10f}'.format(P_dist)+'\n')
svzerod_json.write(indent+indent+indent+'}'+'\n')
svzerod_json.write(indent+indent+'}'+'\n')
svzerod_json.write(indent+'],\n')

# Write junctions ------------------------------
svzerod_json.write(indent+'"junctions": ['+'\n')
# only one junction
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"junction_name": "J0",'+'\n')
svzerod_json.write(indent+indent+indent+'"junction_type": "NORMAL_JUNCTION",'+'\n')
# inlet vessel is the aorta
svzerod_json.write(indent+indent+indent+'"inlet_vessels": ['+'\n')
svzerod_json.write(indent+indent+indent+indent+'0\n')
svzerod_json.write(indent+indent+indent+'],'+'\n')
# outlet vessels are the left common iliac artery and the right common iliac artery
svzerod_json.write(indent+indent+indent+'"outlet_vessels": ['+'\n')
svzerod_json.write(indent+indent+indent+indent+'1,\n')
svzerod_json.write(indent+indent+indent+indent+'2\n')
svzerod_json.write(indent+indent+indent+']'+'\n')
svzerod_json.write(indent+indent+'}'+'\n')
svzerod_json.write(indent+'],'+'\n')

# Write vessels -------------------------------
svzerod_json.write(indent+'"vessels": ['+'\n')
# aorta
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"boundary_conditions": { \n')
svzerod_json.write(indent+indent+indent+indent+'"inlet": "INFLOW" \n')
svzerod_json.write(indent+indent+indent+'},\n')
svzerod_json.write(indent+indent+indent+'"vessel_id": 0,\n')
svzerod_json.write(indent+indent+indent+'"vessel_length": 17.549720037726747,\n')
svzerod_json.write(indent+indent+indent+'"vessel_name": "aorta", \n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",\n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 1.6548448521166672,\n')
svzerod_json.write(indent+indent+indent+indent+'"L": 5.697307730623212,\n')
svzerod_json.write(indent+indent+indent+indent+'"C": 9.93691840089246e-05,\n')
svzerod_json.write(indent+indent+indent+indent+'"stenosis_coefficient": 0.061271872722872926\n')
svzerod_json.write(indent+indent+indent+'}\n')
svzerod_json.write(indent+indent+'},\n')

# left common iliac artery
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"boundary_conditions": { \n')
svzerod_json.write(indent+indent+indent+indent+'"outlet": "BC_RCR0"\n')
svzerod_json.write(indent+indent+indent+'},\n')
svzerod_json.write(indent+indent+indent+'"vessel_id": 1,\n')
svzerod_json.write(indent+indent+indent+'"vessel_length": 5.691099654626812,\n')
svzerod_json.write(indent+indent+indent+'"vessel_name": "left_iliac",\n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",\n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {\n')
svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 1.663383207632923,\n')
svzerod_json.write(indent+indent+indent+indent+'"L": 3.2527462989629985,\n')
svzerod_json.write(indent+indent+indent+indent+'"C": 1.830302617712935e-05,\n')
svzerod_json.write(indent+indent+indent+indent+'"stenosis_coefficient": 0.280553698089018\n')
svzerod_json.write(indent+indent+indent+'}\n')
svzerod_json.write(indent+indent+'},\n')

# right common iliac artery
svzerod_json.write(indent+indent+'{'+'\n')
svzerod_json.write(indent+indent+indent+'"boundary_conditions": { \n')
svzerod_json.write(indent+indent+indent+indent+'"outlet": "BC_RCR1"\n')
svzerod_json.write(indent+indent+indent+'},\n')
svzerod_json.write(indent+indent+indent+'"vessel_id": 2, \n')
svzerod_json.write(indent+indent+indent+'"vessel_length": 4.462860340574513,\n')
svzerod_json.write(indent+indent+indent+'"vessel_name": "right_iliac",\n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",'+'\n')
svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {'+'\n')
svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 0.5374196224863078, \n')
svzerod_json.write(indent+indent+indent+indent+'"L": 1.6372663939202936, \n')
svzerod_json.write(indent+indent+indent+indent+'"C": 2.236084739578155e-05,\n')
svzerod_json.write(indent+indent+indent+indent+'"stenosis_coefficient": 0.06518004916414315\n')
svzerod_json.write(indent+indent+indent+'}\n')
svzerod_json.write(indent+indent+'}\n')
svzerod_json.write(indent+']\n')

# end dictionary
svzerod_json.write('}'+'\n')
svzerod_json.close()

#%% Run svZeroDSolver

import os

json_file = dirpath+'data/aobif_obs.json'
csv_file  = dirpath+'data/aobif_obs.csv'

# run the simulation
os.system('/Users/chloe/Documents/Stanford/ME398_Winter/Release/svzerodsolver '+json_file+' '+csv_file)

#%% After running svZeroDSolver

filename = dirpath+'data/aobif_obs.csv'

def get_QOI_0D(filename, QOI_name=None):

    # radius of the aorta
    mu = 0.04         # viscosity of blood
    r  = 1.019478928  # radius of the aorta

    vessel_names = ['aorta','left_iliac','right_iliac']

    var_names = np.genfromtxt(filename, dtype = 'U', usecols=0, skip_header=1, delimiter=',')
    time      = np.genfromtxt(filename, usecols=1, skip_header=1, delimiter=',')
    data      = np.genfromtxt(filename, usecols=range(2,6), skip_header=1, delimiter=',')

    num_0d_cycles = 1
    num_0d_tsteps = len(np.nonzero(var_names == var_names[0])[0])
    steps_per_cycle = int((num_0d_tsteps+num_0d_cycles-1)/num_0d_cycles)

    zerod_data = {}
    for name in vessel_names:
        idxs = np.nonzero(var_names == name)[0]
        zerod_data[name+':pressure_in'] = (data[idxs,2])[-steps_per_cycle:]

    pressure_aorta = zerod_data['aorta:pressure_in']

    return torch.tensor([[min(pressure_aorta)], [max(pressure_aorta)]])

pressure = get_QOI_0D(filename)
eps      = np.random.normal(0, 2000) # noise
pressure = pressure + eps # with added noise

sio.savemat('../data/y_obs.mat', {'y_obs':pressure, 'epsilon':eps})

# %%
