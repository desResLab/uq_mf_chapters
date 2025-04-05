import numpy as np
import chaospy as cp
import os
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
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
    # Li, Chenzhao, and Sankaran Mahadevan.
    # "An efficient modularized sample-based method to estimate the first-order Sobol’index."
    # Reliability Engineering & System Safety (2016)
    # following Algorithm 2
    ####################################################################################################################

    # definition of the Ishigami function
    # budget_array = np.array([200])                              # affordable computational budget
    # repetitions = 100
    # d = 3                                                       # number of uncertainties
    # case_description = 'Ishigami'                               # string describing the case to be run
    # levels = 3                                                  # numbers of levels for the multifidelity estimation
    # statistics_file = "Estimating_statistics/MFMC_Ishigami_Model_Statistics.pkl"      # path to estimated statistics file
    # costs = np.array([1, 0.05, 0.001])                   # computational costs of models
    # QoIs = ['scalar']                                    # QoIs
    uncertainties = {'Z_1': {'mean': 0, 'lower': -np.pi, 'upper': np.pi},
                     'Z_2': {'mean': 0, 'lower': -np.pi, 'upper': np.pi},
                     'Z_3': {'mean': 0, 'lower': -np.pi, 'upper': np.pi}}  # uncertain input parameters
    sampling = 'R'                                              # 'S' Sobol or 'R' random sampling
    N_array = [64,81,100,400,900,1600,2500,3600]                                                   # number of samples
    # base_param = np.array([0,0,0])                              # base parameters for generation of models
    # model_0 = Ishigami_func_level0(base_param)                  # high fidelity model
    # model_1 = Ishigami_func_level1(base_param)                  # model of level 1 fidelity
    # model_2 = Ishigami_func_level2(base_param)                  # model of level 2 fidelity
    #
    # models = np.array([model_0 ,model_1, model_2])

    for N in N_array:
    ####################################################################################################################
        # 1. random samples of x
        dim = uncertainties.__len__()                             # number of uncertain parameters
        uqs =list(uncertainties.keys())                         # list of uncertain quantities
        uq_1 = cp.Uniform(uncertainties[uqs[0]]['lower'], uncertainties[uqs[0]]['upper'])
        uq_2 = cp.Uniform(uncertainties[uqs[1]]['lower'], uncertainties[uqs[1]]['upper'])
        uq_3 = cp.Uniform(uncertainties[uqs[2]]['lower'], uncertainties[uqs[2]]['upper'])

        jpdf = cp.J(uq_1, uq_2, uq_3)

        np.random.seed(42)
        Z = jpdf.sample(N, sampling).transpose()


        ####################################################################################################################
        # 2. evaluate model y=f(x) and estimate Var(y)

        y = np.zeros(N)

        for run in range(N):
            y[run] = Ishigami_func_level0(Z[run,:]).run_simulation()

        ####################################################################################################################
        # 3. divide the domain of x_i into M equally probable intervals
        # M = sqrt(n)
        M = int(np.sqrt(N))

        intervals = np.zeros((M,2,dim))

        for d, uq in enumerate(uqs):
            tot_interval = uncertainties[uq]['upper'] - uncertainties[uq]['lower']
            sub_interval = tot_interval/M

            lower = uncertainties[uq]['lower']
            for m in range(M):
                intervals[m,:,d] = [lower, lower + sub_interval]
                lower = lower + sub_interval

        ####################################################################################################################
        # 4. assign the samples of y into divided intervals

        #temp = np.array(sorted(Z[:, 0]))
        assigned_samples = {}                       # sample values assigned to the different intervals
        indices_samples = {}                        # sample indices assigned to the different intervals
        assigned_y = {}                             # y values assigned to the different intervals
        for d, uq in enumerate(uqs):
            assigned_samples.update({d: {}})
            indices_samples.update({d: {}})
            assigned_y.update({d: {}})
            interval = intervals[:, :, d]
            for i in range(interval.shape[0]):
                assigned_samples[d].update({i: []})
                assigned_y[d].update({i: []})
                indices_samples[d].update({i: []})
                for idx, sample in enumerate(Z[:, d]):
                    if sample > interval[i, 0] and sample < interval[i, 1]:
                        assigned_samples[d][i].append(sample)
                        indices_samples[d][i].append(idx)
                assigned_y[d][i] = y[indices_samples[d][i]]



        ####################################################################################################################
        # 5. estimate Var_phi' (y) as the sampling variance of each interval
        var_interval = {}
        for d, uq in enumerate(uqs):
            var_interval.update({d: {}})
            interval = intervals[:, :, d]
            for i in range(interval.shape[0]):
                var_interval[d].update({i: np.var(assigned_y[d][i])})

        ####################################################################################################################
        # 6. estimate Exp_phi (Var_phi' (y)) as the sampling mean

        mean_interval = {}
        for d, uq in enumerate(uqs):
            mean_interval.update({d: np.mean(np.array(list(var_interval[d].items()))[:,1])})

        ####################################################################################################################
        # 7. compute main Sobol' index

        S = np.zeros(dim)
        for d, uq in enumerate(uqs):
            S[d] = 1 - ((mean_interval[d]) / (np.var(y)))

        print(N,M,S)

    print('Done')