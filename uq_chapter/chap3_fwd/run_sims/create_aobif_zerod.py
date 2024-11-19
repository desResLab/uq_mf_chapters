#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create input files for 0D simulation

@author: chloe
"""
import scipy.io as sio
import pandas as pd
import sys, os

method = sys.argv[3]

def create_aobif_zerod(iter, s, method):

    file_path = '/scratch/users/chloe1/aobif_sims/'
    
    # distal pressure used for RCR boundary conditions
    P_dist = 4000 # dynes/cm^2

    # define an indent
    indent = '    '

    # write the aobif.json file
    if not os.path.exists(file_path+'zerod_sims/'+ method+'_exp_'+str(s)+'/sim_'+str(iter)+'/'):
        os.makedirs(file_path+'zerod_sims/'+method+'_exp_'+str(s)+'/sim_'+str(iter)+'/')

    # create the json file and open it
    svzerod_json = open(file_path+'zerod_sims/'+method+'_exp_'+str(s)+'/sim_'+str(iter)+'/'+'aobif_'+str(iter)+'.json','w')

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
    inflowFile = pd.read_csv(file_path+'aobif_inflow.csv', sep=' ')
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
    loadedFile = sio.loadmat(file_path+method+'_RCR_vals.mat')
    Rp = loadedFile['Rp']
    C  = loadedFile['C']
    Rd = loadedFile['Rd']
    svzerod_json.write(indent+indent+'{'+'\n')
    svzerod_json.write(indent+indent+indent+'"bc_name": "BC_RCR0",\n')
    svzerod_json.write(indent+indent+indent+'"bc_type": "RCR",'+'\n')
    svzerod_json.write(indent+indent+indent+'"bc_values": {'+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"Rp": {0:.10f},'.format(Rp[s][iter])+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"Rd": {0:.10f},'.format(Rd[s][iter])+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"C": {0:.10f},'.format(C[s][iter])+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"Pd": {0:.10f}'.format(P_dist)+'\n')
    svzerod_json.write(indent+indent+indent+'}'+'\n')
    svzerod_json.write(indent+indent+'},'+'\n')
    svzerod_json.write(indent+indent+'{'+'\n')
    svzerod_json.write(indent+indent+indent+'"bc_name": "BC_RCR1",\n')
    svzerod_json.write(indent+indent+indent+'"bc_type": "RCR",'+'\n')
    svzerod_json.write(indent+indent+indent+'"bc_values": {'+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"Rp": {0:.10f},'.format(Rp[s][iter])+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"Rd": {0:.10f},'.format(Rd[s][iter])+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"C": {0:.10f},'.format(C[s][iter])+'\n')
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
    svzerod_json.write(indent+indent+indent+'"vessel_length": 8.6,\n')
    svzerod_json.write(indent+indent+indent+'"vessel_name": "aorta", \n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",\n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {'+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 4.403899969,\n')
    svzerod_json.write(indent+indent+indent+indent+'"L": 3.923354411,\n')
    svzerod_json.write(indent+indent+indent+indent+'"C": 4.99557214e-5\n')
    svzerod_json.write(indent+indent+indent+'}\n')
    svzerod_json.write(indent+indent+'},\n')

    # left common iliac artery
    svzerod_json.write(indent+indent+'{'+'\n')
    svzerod_json.write(indent+indent+indent+'"boundary_conditions": { \n')
    svzerod_json.write(indent+indent+indent+indent+'"outlet": "BC_RCR0"\n')
    svzerod_json.write(indent+indent+indent+'},\n')
    svzerod_json.write(indent+indent+indent+'"vessel_id": 1,\n')
    svzerod_json.write(indent+indent+indent+'"vessel_length": 8.5,\n')
    svzerod_json.write(indent+indent+indent+'"vessel_name": "left_iliac",\n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",\n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {\n')
    svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 18.37158911,\n')
    svzerod_json.write(indent+indent+indent+indent+'"L": 7.96689096,\n')
    svzerod_json.write(indent+indent+indent+indent+'"C": 1.71665699e-5\n')
    svzerod_json.write(indent+indent+indent+'}\n')
    svzerod_json.write(indent+indent+'},\n')

    # right common iliac artery
    svzerod_json.write(indent+indent+'{'+'\n')
    svzerod_json.write(indent+indent+indent+'"boundary_conditions": { \n')
    svzerod_json.write(indent+indent+indent+indent+'"outlet": "BC_RCR1"\n')
    svzerod_json.write(indent+indent+indent+'},\n')
    svzerod_json.write(indent+indent+indent+'"vessel_id": 2, \n')
    svzerod_json.write(indent+indent+indent+'"vessel_length": 8.5,\n')
    svzerod_json.write(indent+indent+indent+'"vessel_name": "right_iliac",\n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_type": "BloodVessel",'+'\n')
    svzerod_json.write(indent+indent+indent+'"zero_d_element_values": {'+'\n')
    svzerod_json.write(indent+indent+indent+indent+'"R_poiseuille": 18.37158911, \n')
    svzerod_json.write(indent+indent+indent+indent+'"L": 7.96689096, \n')
    svzerod_json.write(indent+indent+indent+indent+'"C": 1.71665699e-5\n')
    svzerod_json.write(indent+indent+indent+'}\n')
    svzerod_json.write(indent+indent+'}\n')
    svzerod_json.write(indent+']\n')

    # end dictionary
    svzerod_json.write('}'+'\n')
    svzerod_json.close()

if __name__ == '__main__':
    create_aobif_zerod(int(sys.argv[1]), int(sys.argv[2]), method)
