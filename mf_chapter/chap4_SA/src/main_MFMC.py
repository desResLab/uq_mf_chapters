import numpy as np
import chaospy as cp
import os
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
from ModelStatisticsEstimation import model_statistics
from multifidelity_Models import Model0D, Model1D, Model3D
from multifidelityEstimation import mfmc_sa, opt_allocation
from ishigami_functions import Ishigami_func_level0, Ishigami_func_level1, Ishigami_func_level2
from write_output_file import write_output_file

m3_to_ml = 1e6
unit_mmhg_pa = 133.3
unit_pa_mmhg = 1./unit_mmhg_pa
mm_to_m = 1e-3

# set font for figures
plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 18,
                     'legend.frameon': False,
                     'legend.fontsize': 13,
                     'savefig.transparent': True})

if __name__ == '__main__':

    ####################################################################################################################
    # parameters to set
    ####################################################################################################################
    #
    # ####################################################################################################################
    # # paramerters Ishigami function
    # budget_array = np.array([200])                              # affordable computational budget
    # d = 3                                                       # number of uncertainties
    # case_description = 'Ishigami'                               # string describing the case to be run
    # levels = 3                                                  # numbers of levels for the multifidelity estimation
    # statistics_file = "Estimating_statistics/MFMC_Ishigami_Model_Statistics.pkl"      # path to estimated statistics file
    # costs = np.array([1, 0.03, 0.01])                   # computational costs of models
    # QoIs = ['scalar']                                    # QoIs
    # uncertainties = {'Z_1': {'mean': 0, 'lower': -np.pi, 'upper': np.pi},
    #                  'Z_2': {'mean': 0, 'lower': -np.pi, 'upper': np.pi},
    #                  'Z_3': {'mean': 0, 'lower': -np.pi, 'upper': np.pi}}  # uncertain input parameters
    # sampling = 'S'                                              # 'S' Sobol or 'R' random sampling
    # base_param = np.array([0,0,0])                              # base parameters for generation of models
    # model_0 = Ishigami_func_level0(base_param)                  # high fidelity model
    # model_1 = Ishigami_func_level1(base_param)                  # model of level 1 fidelity
    # model_2 = Ishigami_func_level2(base_param)                  # model of level 2 fidelity
    #
    # models = np.array([model_0 ,model_1, model_2])


    ####################################################################################################################
    # paramerters artery model
    ####################################################################################################################
    budget_array = np.array([500, 1000, 2000, 4000, 6000, 8000, 10000])                              # affordable computational budget
    repetitions = 2
    d = 3                                                       # number of uncertainties
    case_description = 'Artery'                                 # string describing the case to be run
    levels = 2                                                  # numbers of levels for the multifidelity estimation

    QoIs = QoIs = ['Psys', 'PP', 'Delta_R_max']                 # QoIs
    uncertainties = {'IMT': {'mean': 0.785 * 1e-3, 'lower': 0.785 * 1e-3 * 0.9, 'upper': 0.785 * 1e-3 * 1.1},
                     'E': {'mean': 440043, 'lower': 440043*0.9, 'upper': 440043*1.1},
                     'R': {'mean': 3.289 * 1e-3, 'lower': 3.289 * 0.9 * 1e-3, 'upper': 3.289 * 1.1 * 1e-3}
                     }                                          # uncertain input parameters
    sampling = 'R'                                              # 'S' Sobol or 'R' random sampling

    # 3D model specifications
    base_folder_3D = 'Simulation_Folder'
    # base_folder_3D = 'FSI_Simulations/Statistics_Estimation/Base_folder'
    # solution_folder_3D = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/FSI_Simulations/Statistics_Estimation/Data'
    solution_folder_3D = '/media/friedees/LaCie/MFMC_CCA/01_MFMC_solution/budget_' + str(budget_array[0]) + '_S_Artery/'
    QoI_3D_file = 'Simulation_Folder/QoIs_3D_FSI_StatisticsEstimation_150.pkl'


    # 1D model specifications
    networkName = 'MFMC_1D_model_statistics_estimate'
    description = 'MFMC 1D mixed 40s'
    base_dataNumber = '999'
    case_dataNumber = base_dataNumber


    # 0D model specifications
    base_file_0D = 'Simulation_Folder/MFMC_0D_base_mixed_40s.json'
    L = 0.126 / 2  # length of vessel in meters
    mu = 0.00465  # blood viscosity
    with open(base_file_0D) as ff:
        config_0D = json.load(ff)
    alpha_perturb = 1  # perturbation factor of the 0D model solution -> need for correlation adjustment
    perturb_0D_model = True

    # base parameters for model generation
    base_param = np.array([[uncertainties['IMT']['mean'], uncertainties['E']['mean'], uncertainties['R']['mean']],
                              [uncertainties['IMT']['lower'], uncertainties['E']['lower'], uncertainties['R']['lower']],
                              [uncertainties['IMT']['upper'], uncertainties['E']['upper'], uncertainties['R']['upper']]])



    if levels == 3:
        # generate models
        model_0 = Model3D(base_folder_3D, solution_folder_3D, QoI_3D_file, base_param, 0, QoIs)
        model_1 = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs)
        model_1.generate_network()  # high fidelity model
        model_2 = Model0D(config_0D, base_file_0D, base_param, 0, QoIs, perturb_0D_model, L,
                          mu)  # model of level 1 fidelity
        models = np.array([model_0 ,model_1, model_2])
        statistics_file = "Estimating_statistics/MFMC_statistics_estimation_Artery_3D_1D_0D.pkl"  # path to estimated statistics file
        costs = np.array([1, 9e-5, 3e-5])  # computational costs of models for 3 levels
    else:
        model_0 = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs)
        model_0.generate_network()  # high fidelity model
        model_1 = Model0D(config_0D, base_file_0D, base_param, 0, QoIs, perturb_0D_model, L, mu)
        models = np.array([model_0, model_1])
        if perturb_0D_model == False:                               # path to estimated statistics file
            statistics_file = "Estimating_statistics/MFMC_statistics_estimation_Artery_1D_0D_unperturbed.pkl"
        else:
            statistics_file = "Estimating_statistics/MFMC_statistics_estimation_Artery_1D_0D_perturbed.pkl"
        costs = np.array([1, 0.3])                                  # computational costs of models for 2 levels

    ####################################################################################################################

    # load model statistics
    # check model statistics file already exists or if it needs to be generated
    if os.path.isfile(statistics_file):
        # load statistics
        f = open(statistics_file, 'rb')
        mf_statistics = pickle.load(f)
        print('Loaded statistics')
    else:
        # compute statistics
        N = 150
        mf_statistics = model_statistics(models, N, uncertainties, QoIs, case_description)
        print('Estimated statistics')


    d = uncertainties.__len__()                             # number of uncertain parameters
    uqs =list(uncertainties.keys())                         # list of uncertain quantities
    uq_1 = cp.Uniform(uncertainties[uqs[0]]['lower'], uncertainties[uqs[0]]['upper'])
    uq_2 = cp.Uniform(uncertainties[uqs[1]]['lower'], uncertainties[uqs[1]]['upper'])
    uq_3 = cp.Uniform(uncertainties[uqs[2]]['lower'], uncertainties[uqs[2]]['upper'])

    jpdf = cp.J(uq_1, uq_2, uq_3)                           # jpdf of uncertain model parameters

    source_dir = Path('Simulation_Folder')
    N = 150                                              # number of samples to be used to estimate model statistics
    MF_Sensitvities = {}                                    # definition of solution dictionary of the multifidelity sensitivity indices
    model_solutions = {}                                    # dictionary for QoI solution of each model
    working_dir = os.path.dirname(os.path.realpath(__file__))

    # load least squares fit for the 0D model perturbation
    temp = ''
    for idx in range(len(models)):
        temp = temp + '_' + str(models[idx].__class__.__name__[-2:])
    f = open('Estimating_statistics/0D_Model_lstsq_perturbation' + temp + ".pkl", "rb")
    lstsq = pickle.load(f)

    for budget in budget_array:
        # compute effective budget
        eff_budget = budget / (d + 2)

        for rep in range(1,repetitions):
            start_sim = time.time()
            # define network name for 1D model model evaluations
            case_definition = 'budget_' + str(budget) + '_' + sampling + '_' + case_description + '_' + str(rep)

            m = {}
            alpha = {}
            for QoI in QoIs:
                # evaluate optimal number of evaluations and weights per fidelity level
                m[QoI], alpha[QoI] = opt_allocation(eff_budget, costs, mf_statistics[QoI])

            #     # simulate only MC on the highest fidelity level
            #     m[QoI]= np.array([int(budget/(d+2)), 0])
            # case_definition = 'budget_' + str(budget) + '_' + sampling + '_MC_single_fidelity' + '_' + str(rep)

            # estimate the main and total Sobol indices through the multifidelity approach
            sensitivity_dict = mfmc_sa(models, jpdf, sampling,  m, alpha, case_definition, mf_statistics, rep)

            end_time = time.time() - start_sim

            # save the result as an output file
            txt_file = 'Simulation_Folder/Output_files/sensitivities_' + str(len(models)) + '_levels_' + case_definition +'.txt'
            write_output_file(sensitivity_dict, txt_file, m, alpha, sampling, costs, end_time)

            # save sensitivity_dict
            f = open('Simulation_Folder/Sensitivity_dicts/MFMC_sensitivity_dict_' + case_definition + '.pkl', "wb")
            pickle.dump(sensitivity_dict, f)
            f.close()

    print('Multifidelity sensitivity analysis ran sucessfully')

