from __future__ import print_function, absolute_import
from builtins import range
import matplotlib.pyplot as plt
import h5py
import numpy as np
import csv
import os
from split_cycles import split_signal_diastole_auto
import pprint

# set font for figures
plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 24,
                     'legend.frameon': False,
                     'legend.fontsize': 24,
                     'savefig.transparent': True})

m3_to_ml     = 1e6
unit_mmhg_pa = 133.3
unit_pa_mmhg = 1./unit_mmhg_pa

lines   = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed']
markers = ['s','o','p']
colors  = ['tab:blue', 'tab:orange','tab:green']

# MAIN 
if __name__ == "__main__":

    figure_dir = '../Figures/'

    ####################################################################################################################
    # load 1D data
    file_name_1D = '../Simulation_Folder/model_1D/SolutionData_000/Multifidelity_1D_0D_SolutionData_000.hdf5'

    data_dict_1D = {}
    with h5py.File(file_name_1D, "r") as f:
        data_dict_1D['t'] = np.array(f['VascularNetwork']['simulationTime'])
        data_dict_1D['P'] = np.array(f['vessels']['vessel_0  -  0']['Psol'])
        data_dict_1D['Q_1D'] = np.array(f['vessels']['vessel_0  -  0']['Qsol'])
        data_dict_1D['A_1D'] = np.array(f['vessels']['vessel_0  -  0']['Asol'])

    # extract the last cycle for all variables of the 1D model
    cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(data_dict_1D['t'],
                                                                                  data_dict_1D['Q_1D'][:, 0])
    data_dict_1D['t'] = data_dict_1D['t'][cycle_indices[-1][0]:cycle_indices[-1][1]] - \
                                data_dict_1D['t'][cycle_indices[-1][0]]
    data_dict_1D['P'] = data_dict_1D['P'][cycle_indices[-1][0]:cycle_indices[-1][1], 2]
    data_dict_1D['Q'] = data_dict_1D['Q_1D'][cycle_indices[-1][0]:cycle_indices[-1][1], 2]
    data_dict_1D['A_1D'] = data_dict_1D['A_1D'][cycle_indices[-1][0]:cycle_indices[-1][1], :]

    data_dict_1D['Delta_R'] = (
                np.sqrt(data_dict_1D['A_1D'][:, 2] / np.pi) - np.min(np.sqrt(data_dict_1D['A_1D'][:, 2] / np.pi)))
    data_dict_1D['V'] = data_dict_1D['Q'] / data_dict_1D['A_1D'][:,2]

    ####################################################################################################################
    # load 0D data
    file_name_0D = '../Simulation_Folder/model_0D/Two_vessels_baseline/CCA_0D_45years_twoVessels.csv'

    data_dict_0D = {}

    P_in_0D = np.array([])
    P_out_0D = np.array([])
    Q_in_0D = np.array([])
    Q_out_0D = np.array([])
    t_0D = np.array([])
    with open(file_name_0D) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        next(csv_read, None)
        for row in csv_read:
            t_0D = np.append(t_0D, float(row[1]))
            P_in_0D = np.append(P_in_0D, float(row[4]))
            P_out_0D = np.append(P_out_0D, float(row[5]))
            Q_in_0D = np.append(Q_in_0D, float(row[2]))
            Q_out_0D = np.append(Q_out_0D, float(row[3]))

    # extract data for mid-point
    data_dict_0D['t'] = t_0D[:int(len(t_0D) / 2)]
    data_dict_0D['P_in_0D'] = P_in_0D[:int(len(t_0D) / 2)]
    data_dict_0D['P'] = P_in_0D[int(len(t_0D) / 2) + 1:]
    data_dict_0D['P_out_0D'] = P_out_0D[int(len(t_0D) / 2) + 1:]
    data_dict_0D['Q_in_0D'] = Q_in_0D[:int(len(t_0D) / 2)]
    data_dict_0D['Q_out_0D'] = Q_out_0D[int(len(t_0D) / 2) + 1:]
    data_dict_0D['Q'] = Q_in_0D[int(len(t_0D) / 2) + 1:]

    # extract the last cycle for all variables of the 0D model
    cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(data_dict_0D['t'],
                                                                                  data_dict_0D['Q_in_0D'])
    data_dict_0D['t'] = data_dict_0D['t'][cycle_indices[-1][0]:cycle_indices[-1][1]] - data_dict_0D['t'][
        cycle_indices[-1][0]]
    data_dict_0D['P_in_0D'] = data_dict_0D['P_in_0D'][cycle_indices[-1][0]:cycle_indices[-1][1]]
    data_dict_0D['P'] = data_dict_0D['P'][cycle_indices[-1][0]:cycle_indices[-1][1]]
    data_dict_0D['P_out_0D'] = data_dict_0D['P_out_0D'][cycle_indices[-1][0]:cycle_indices[-1][1]]
    data_dict_0D['Q_in_0D'] = data_dict_0D['Q_in_0D'][cycle_indices[-1][0]:cycle_indices[-1][1]]
    data_dict_0D['Q'] = data_dict_0D['Q'][cycle_indices[-1][0]:cycle_indices[-1][1]]
    data_dict_0D['Q_out_0D'] = data_dict_0D['Q_out_0D'][cycle_indices[-1][0]:cycle_indices[-1][1]]

    # compute radius change, radius, and fluid velocity
    data_dict_0D['Delta_R'] = (data_dict_0D['P'] - np.min(data_dict_0D['P'])) * 3 / 4 * (
        0.003289) ** 2 / (393130 * 0.000785)
    data_dict_0D['R'] = (data_dict_0D['P'] - np.min(data_dict_0D['P'])) * 3 / 4 * (
        0.003289) ** 2 / (393130 * 0.000785) + 0.003289
    data_dict_0D['V'] = (data_dict_0D['Q'] / (data_dict_0D['R'] ** 2 * np.pi))

    ####################################################################################################################
    # load 3D data
    file_name_3D = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/FSI_Simulations/temp/Prestress_nHK_fine/Postprocess/Integrated_Data.csv'
    area_file    = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/FSI_Simulations/temp/Prestress_nHK_fine/Postprocess/Area.csv'
    displ_file   = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/FSI_Simulations/temp/Prestress_nHK_fine/Postprocess/Displacement_nHK_fine.csv'

    data_dict_3D = {}
    P_3D = np.array([])
    V_3D = np.array([])
    Delta_R_3D = np.array([])
    t_3D = np.linspace(0, 7.27, 727)

    with open(area_file) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        next(csv_read, None)
        for row in csv_read:
            area = float(row[4])

    with open(file_name_3D) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        next(csv_read, None)
        for row in csv_read:
            P_3D = np.append(P_3D, float(row[6]))
            V_3D = np.append(V_3D, float(row[9]))

    with open(displ_file) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        next(csv_read, None)
        for row in csv_read:
            Delta_R_3D = np.append(Delta_R_3D, float(row[6]))

    cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(t_3D,V_3D / area,prominence = 0.01)

    data_dict_3D['P'] = (P_3D[cycle_indices[-2][0]:cycle_indices[-2][1]] / area) * 1e-1
    data_dict_3D['V'] = (V_3D[cycle_indices[-2][0]:cycle_indices[-2][1]] / area) * 1e-2
    data_dict_3D['t'] = t_3D[cycle_indices[-2][0]:cycle_indices[-2][1]] - \
                        t_3D[cycle_indices[-2][0]]
    data_dict_3D['Delta_R'] = (Delta_R_3D[cycle_indices[-2][0]:cycle_indices[-2][1]] - Delta_R_3D[cycle_indices[-2][0]]) * 1e-2
    data_dict_3D['Q'] = (V_3D[cycle_indices[-2][0]:cycle_indices[-2][1]]) * 1e-6

    # plot the three models for pressure, displacement and velocity
    fig, ax = plt.subplots()
    fig_name = 'FidelityValidation_pressure.svg'
    plt.plot(data_dict_0D['t'], data_dict_0D['P'] / 133.3, label='0D', linewidth=3,linestyle=lines[0])
    plt.plot(data_dict_1D['t'], data_dict_1D['P'] / 133.3, label='1D', linewidth=3,linestyle=lines[1])
    plt.plot(data_dict_3D['t'], data_dict_3D['P'] / 133.2, label='3D', linewidth=3, linestyle=lines[2])
    plt.ylim([75, 140])
    plt.xlabel('time [s]')
    plt.ylabel('pressure [mmHg]')
    #plt.legend()
    plt.subplots_adjust(left=0.175,
                        bottom=0.175,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    fig_name = 'FidelityValidation_displacement.svg'
    plt.plot(data_dict_0D['t'], data_dict_0D['Delta_R'] * 1e3, label='0D', linewidth=3, linestyle=lines[0])
    plt.plot(data_dict_1D['t'], data_dict_1D['Delta_R'] * 1e3, label='1D', linewidth=3, linestyle=lines[1])
    plt.plot(data_dict_3D['t'], data_dict_3D['Delta_R'] * 1e3, label='3D', linewidth=3, linestyle=lines[2])
    plt.xlabel('time [s]')
    plt.ylabel('displacement [mm]')
    #plt.legend()
    plt.subplots_adjust(left=0.175,
                        bottom=0.175,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    fig_name = 'FidelityValidation_velocity.svg'
    plt.plot(data_dict_0D['t'], data_dict_0D['V'] * 1e2, label='0D', linewidth=3, linestyle=lines[0])
    plt.plot(data_dict_1D['t'], data_dict_1D['V'] * 1e2, label='1D', linewidth=3, linestyle=lines[1])
    plt.plot(data_dict_3D['t'], data_dict_3D['V'] * 1e2, label='3D', linewidth=3, linestyle=lines[2])
    plt.xlabel('time [s]')
    plt.ylabel('velocity [cm/s]')
    #plt.legend()
    plt.subplots_adjust(left=0.175,
                        bottom=0.175,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots()
    fig_name = 'FidelityValidation_flowrate.svg'
    plt.plot(data_dict_0D['t'], data_dict_0D['Q'] * 1e6, label='0D', linewidth=3, linestyle=lines[0])
    plt.plot(data_dict_1D['t'], data_dict_1D['Q'] * 1e6, label='1D', linewidth=3, linestyle=lines[1])
    plt.plot(data_dict_3D['t'], data_dict_3D['Q'] * 1e6, label='3D', linewidth=3, linestyle=lines[2])
    plt.xlabel('time [s]')
    plt.ylabel('flow rate [ml/s]')
    #plt.legend()
    plt.subplots_adjust(left=0.175,
                        bottom=0.175,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()
    
    # compute error measures
    # following error measures from Pfaller et al. 2022
    # flow rate difference is only considered at the outlet since it is prescribed at the inlet

    difference_dict = {}

    for comp in ['3D-1D', '3D-0D', '1D-0D']:
        difference_dict.update({comp: {}})

        #choose which model is the high fidelity model and which one is the low fidelity model
        if comp == '3D-1D':
            high_model = data_dict_3D
            low_model = data_dict_1D
        elif comp == '3D-0D':
            high_model = data_dict_3D
            low_model = data_dict_0D
        else:
            high_model = data_dict_1D
            low_model = data_dict_0D

        for QoI in ['P', 'Q', 'Delta_R']:
            difference_dict[comp].update({QoI: {'avg': {}, 'max': {}, 'sys': {}, 'dia': {}}})

            difference_dict[comp][QoI]['max'] = np.abs(np.max(
                    ((high_model[QoI] - np.interp(high_model['t'], low_model['t'], low_model[QoI])) ))/ np.mean(high_model[QoI]))


            # difference_dict[comp][QoI]['max'] = np.max(
            #         ((high_model[QoI] - np.interp(high_model['t'], low_model['t'], low_model[QoI])) / high_model[QoI]))

            difference_dict[comp][QoI]['sys'] = np.abs((np.max(high_model[QoI]) - np.max(low_model[QoI])) / np.max(
                    high_model[QoI]))

            if QoI != 'P':
                difference_dict[comp][QoI]['avg'] = 1/high_model[QoI].size *  np.sum((np.abs(high_model[QoI]- np.interp(high_model['t'], low_model['t'],low_model[QoI])))) / (np.max(high_model[QoI]) - np.min(high_model[QoI]))
                difference_dict[comp][QoI]['dia'] = np.abs(np.min(high_model[QoI]) - np.min(low_model[QoI])) / (np.max(high_model[QoI]) - np.min(high_model[QoI]))
            else:
                #difference_dict[comp][QoI]['avg'] = np.sqrt((np.mean(high_model[QoI]- np.interp(high_model['t'], low_model['t'],low_model[QoI])) / np.mean(high_model[QoI])) ** 2 )
                difference_dict[comp][QoI]['avg'] = (np.sum(np.abs(high_model[QoI] - np.interp(high_model['t'], low_model['t'], low_model[QoI])))) / np.sum(high_model[QoI])
                difference_dict[comp][QoI]['dia'] = np.abs(np.min(high_model[QoI]) - np.min(low_model[QoI])) / (np.mean(high_model[QoI]))

    with open(
            '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Figures/Differences_Fidelity_Validation.txt',
            'w') as file_name:
        pprint.pprint(difference_dict, file_name)

print('Done with comparison')