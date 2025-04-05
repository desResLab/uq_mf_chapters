from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import chaospy as cp
import subprocess
import os
from separate_vol_surf import separate_vol_surf
#from split_cycles import split_signal_diastole_auto

# set font for figures
plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 18,
                     'legend.frameon': False,
                     'legend.fontsize': 13,
                     'savefig.transparent': True})



def mesh_gen(path_bf, path_sample, sample, idx):

    # ####################################################################################################################
    # ####################################################################################################################
    # # Parameters to set
    # uncertaintyType = 'pm10percent'                         # uncertainty type: physiological ranges or ±10% around mean value
    # # path_bf = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/FSI_mesh/Trial_mesh_gen/base_file.geo'
    # #                                                         # path to base file of the mesh
    # # N = 150                                                 # number of samples = number of meshes
    #
    # ####################################################################################################################
    # ####################################################################################################################
    #
    #
    # # definition of uncertain input parameters
    # if uncertaintyType == 'physio':
    #     uncertainties = {'IMT': {'mean': 0.785 * 1e-3, 'lower': 0.478 * 1e-3, 'upper': 1.092 * 1e-3},
    #                      'E': {'mean': 440043, 'lower': 207782, 'upper': 672304},
    #                      'R': {'mean': 3.289 * 1e-3, 'lower': 2.530 * 1e-3, 'upper': 4.048 * 1e-3}
    #                      }
    # elif uncertaintyType == 'pm10percent':
    #     uncertainties = {'IMT': {'mean': 0.785 * 1e-3, 'lower': 0.785 * 1e-3 * 0.9, 'upper': 0.785 * 1e-3 * 1.1},
    #                      'E': {'mean': 440043, 'lower': 440043 * 0.9, 'upper': 440043 * 1.1},
    #                      'R': {'mean': 3.289 * 1e-3, 'lower': 3.289 * 0.9 * 1e-3, 'upper': 3.289 * 1.1 * 1e-3}
    #                      }
    #
    # h = cp.Uniform(uncertainties['IMT']['lower'],uncertainties['IMT']['upper'])
    # E = cp.Uniform(uncertainties['E']['lower'], uncertainties['E']['upper'])
    # r = cp.Uniform(uncertainties['R']['lower'], uncertainties['R']['upper'])
    # jpdf = cp.J(h,E,r)
    #
    # # generate samples
    #
    # samples = jpdf.sample(N,'S')

    #for idx, sample in enumerate(samples.T):
    # define new name of file according to sample
    os.makedirs(path_sample + '/00-mesh_' + str(idx).zfill(3), exist_ok=True)
    fn = path_sample + '/00-mesh_' + str(idx).zfill(3) + '/sample_' + str(idx).zfill(3)+ '.geo'

    # update radius, wall thickness, and Young's modulus according to the sample
    with open(path_bf, 'r') as input_file, open(fn, 'w') as output_file:
        for line in input_file:
            if line.strip() == 'radius    = 0.3289; // CM':
                output_file.write('radius    = ' + str(round(sample[2],6)) + '; // CM\n')
            else:
                if line.strip() == 'thickness = 0.0785; // CM':
                    output_file.write('thickness = ' + str(round(sample[0], 6)) + '; // CM\n')
                else:
                    output_file.write(line)

    # generate mesh with gmesh
    subprocess.run(['gmsh', fn, '-3', '-format', 'vtk'])

    separate_vol_surf(path_sample + '/00-mesh_'+ str(idx).zfill(3) + '/sample_' + str(idx).zfill(3)+ '.vtk', path_sample)



    print('Done mesh generation')
