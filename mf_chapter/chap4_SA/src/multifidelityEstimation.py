import numpy as np
import os.path
import pickle
import os
import math
from sample_matrix_helper import replace_inital_samples

####################################################################################################################
# wrapper function for the computation of the MFMC Sobol sensitivity indices
#
# inputs:
# fcts                  array of models described as classes
# jpdf                  jpdf of the uncertain model parameters
# sampling              sampling method to be used for generating sample matrices
# m                     dict with model evaluations per fidelity level for each QoI
# alpha                 dict with control variate coefficients per fidelity level for each QoI
# case_definition       string describing the case
# rep                   counter of repetition which is currently computed
#
# output:
# sensitivity_dict      dict with the MFMC sensitivities and the intermediate results for each QoI
####################################################################################################################
# main function for mfmc
def mfmc_sa(fcts, jpdf, sampling, m, alpha,case_definition, mf_statistics, rep):

    # determine QoIs
    QoIs = list(m.keys())

    # case description for model solutions
    temp = ''
    for idx in range(len(fcts)):
        temp = temp + '_' + str(fcts[idx].__class__.__name__[-2:])

    # evaluate maximal number of evaluations and weights needed per fidelity level
    max_eval = np.zeros(len(m[QoIs[0]]))
    for model in range(len(m[QoIs[0]])):
        for QoI in m.keys():
            # evaluate maximal number of evaluations
            if m[QoI][model] > max_eval[model]: max_eval[model] = int(m[QoI][model])

    if max_eval[-1] == 0: max_eval[-1] = max_eval[0]

    ####################################
    # 1. generate two independent sets of inputs - as many as needed for lowest fidelity model evaluations
    A, B, C_BA = generate_samples(int(max_eval[-1]),jpdf, sampling, rep)

    ####################################

    # replace inital samples with FSI simulation samples
    A = replace_inital_samples(A, 'A', case_definition[7:9], m['Psys'][0])
    B = replace_inital_samples(B, 'B', case_definition[7:9], m['Psys'][0])
    C_BA = replace_inital_samples(C_BA, 'BA', case_definition[7:9], m['Psys'][0])
    ####################################

    # save sampling matrices
    f = open('Simulation_Folder/Sample_matrices/MFMC' + temp + '_A_' + case_definition + '.pkl', "wb")
    pickle.dump(A, f)
    f.close()
    f = open('Simulation_Folder/Sample_matrices/MFMC' + temp + '_B_' + case_definition + '.pkl', "wb")
    pickle.dump(B, f)
    f.close()
    f = open('Simulation_Folder/Sample_matrices/MFMC' + temp + '_BA_' + case_definition + '.pkl', "wb")
    pickle.dump(C_BA, f)
    f.close()
    ####################################
    # 2. evaluate all models
    # number of levels in multifidelity analysis
    levels = fcts.__len__()
    Y = {}

    solutions_file_name = 'MFMC' + temp + '_Model_Solutions_' + case_definition + '.pkl'

    # compute all model evaluations
    # check if model solution file alreads exists -> load only data
    if os.path.isfile('Simulation_Folder/' + solutions_file_name):
        file = 'Simulation_Folder/' + solutions_file_name
        Y_dict = open(file, 'rb')
        Y = pickle.load(Y_dict)

    # evaluate all models
    else:
        for level in range(levels):
            # initialize solution dictionary of the high fidelity model
            Y.update({level: {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}})
            model = fcts[level]
            Y[level] = model.mfmc(A[: int(max_eval[level]),:], B[: int(max_eval[level]),:], C_BA[: , :int(max_eval[level]),:], case_definition, QoIs)

        # save model solutions to file
        f = open('Simulation_Folder/MFMC' + temp + '_Model_Solutions_' + case_definition + '.pkl', "wb")
        pickle.dump(Y, f)
        f.close()

    ####################################
    # 3. reshape QoIs solutions to be passed on to mfmc sesitivity estimation

    # allocate output dictionary
    sensitivity_dict = {}

    # extract all QoIs
    for QoI in QoIs:
        Y_interest = {}

        for level in range(levels):
            Y_interest.update({level: {}})
            for mat in Y[level].keys():
                # extract QoI of interest and normalize data
                #Y_interest[level][mat] = (Y[level][mat][QoI] - mf_statistics[QoI]['mean'][0]) / mf_statistics[QoI]['sigma'][0]
                Y_interest[level][mat] = Y[level][mat][QoI]

    ####################################
        # 4. estimate sensitivity indices for each QoI
        sensitivity_dict.update({QoI: mf_sensitivities(Y_interest, alpha[QoI], m[QoI])})

    return sensitivity_dict


####################################################################################################################
# function for the computation of the MFMC Sobol sensitivity indices
#
# inputs:
# Y                     dict containing the solution matrices for each sample matrix, QoI, and fidelity level
# m                     dict with model evaluations per fidelity level for each QoI
# alpha                 dict with control variate coefficients per fidelity level for each QoI
#
# output:
# sensitivity_dict      dict with the MFMC sensitivities and the intermediate results for each QoI
####################################################################################################################
def mf_sensitivities(Y, alpha, m):

    # compute mean and variance of high fidelity model evaluations
    level = list(Y.keys())[0]  # model level = 0 -> high fidelity model
    sensitivity_dict = {'mf_mu': {}, 'mf_var': {}, 'mf_sm': {}, 'mf_st': {},
                        'sf_mu': {level: {}}, 'sf_var': {level: {}}, 'sf_sm': {level: {}}, 'sf_st': {level: {}},
                        'hl_sm': {}, 'hl_st': {}, 'hl_mu': {}, 'hl_var': {}}
    sensitivity_dict['mf_mu'] = np.mean(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]])), axis=0)
    sensitivity_dict['sf_mu'][level] = np.mean(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]])),
                                               axis=0)
    sensitivity_dict['mf_var'] = np.var(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]])), axis=0)
    sensitivity_dict['sf_var'][level] = np.var(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]])),
                                               axis=0)

    # estimate Sobol indices from high fidelity model evaluations
    sensitivity_dict['mf_sm'], sensitivity_dict['mf_st'], sensitivity_dict['mf_var'] = estimate_sobol(Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]], Y[level]['Y_BA'][:m[level],:].T)
    sensitivity_dict['sf_sm'][level], sensitivity_dict['sf_st'][level], sensitivity_dict['sf_var'][level] =  sensitivity_dict['mf_sm'], sensitivity_dict['mf_st'], sensitivity_dict['mf_var']

    if m[-1] != 0:

        # loop over all lower fidelity levels
        for level in np.arange(1, Y.__len__()):

            for key in ['sf_mu', 'sf_var', 'sf_sm', 'sf_st', 'hl_sm', 'hl_st', 'hl_mu', 'hl_var']:
                sensitivity_dict[key].update({level: {}})

            # compute single fidelity mean, variance and sensitivity indices and mean, variance and sensitivity indices
            # estimated on low fidelity model with high fidelity model samples
            sensitivity_dict['sf_mu'][level] = np.mean(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]])))
            sensitivity_dict['hl_mu'][level] = np.mean(np.hstack((Y[level]['Y_A'][:m[level - 1]], Y[level]['Y_B'][:m[level - 1]])))
            sensitivity_dict['mf_mu'] = sensitivity_dict['mf_mu'] + alpha[level] * (
                    np.mean(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]]))) - np.mean(
                np.hstack((Y[level]['Y_A'][:m[level - 1]], Y[level]['Y_B'][:m[level - 1]]))))
            sensitivity_dict['mf_var'] = sensitivity_dict['mf_var'] + alpha[level] ** 2 * (
                    np.var(np.hstack((Y[level]['Y_A'][:m[level]], Y[level]['Y_B'][:m[level]]))) - np.var(
                np.hstack((Y[level]['Y_A'][:m[level - 1]], Y[level]['Y_B'][:m[level - 1]]))))
            sensitivity_dict['sf_sm'][level], sensitivity_dict['sf_st'][level], sensitivity_dict['sf_var'][level] = estimate_sobol(Y[level]['Y_A'][:int(m[level])],
                                                                       Y[level]['Y_B'][:int(m[level])],
                                                                       Y[level]['Y_BA'][:int(m[level]),
                                                                       :].T)
            sensitivity_dict['hl_sm'][level], sensitivity_dict['hl_st'][level], sensitivity_dict['hl_var'][level] = estimate_sobol(
                Y[level]['Y_A'][:int(m[level - 1])], Y[level]['Y_B'][:int(m[level - 1])],
                Y[level]['Y_BA'][:int(m[level - 1]), :].T)

            # update the estimated Sobol indices from all higher fidelity models evaluations with the current lower level
            # fidelity estimate
            sensitivity_dict['mf_sm'] = sensitivity_dict['mf_sm'] + alpha[level] * (sensitivity_dict['sf_sm'][level] - sensitivity_dict['hl_sm'][level])
            sensitivity_dict['mf_st'] = sensitivity_dict['mf_st'] + alpha[level] * (sensitivity_dict['sf_st'][level] - sensitivity_dict['hl_st'][level])

    return sensitivity_dict


####################################################################################################################
# function estimating Sobol sensitivity indices following un-biased estimators by Owen
#
# inputs:
# y_a                  array with model solutions of one fidelity level of sample matrix A
# y_b                  array with model solutions of one fidelity level of sample matrix B
# y_ba                 array with model solutions of one fidelity level of sample matrix C_BA
#
# output:
# sm                   array of main Sobol sensitivity indices estimates
# st                   array of total Sobol sensitivity indices estimates
# var_est              array of model output variance estimates
####################################################################################################################
def estimate_sobol(y_a, y_b, y_ba):

    N = y_a.shape[0]                                        # number of samples
    var_est = np.var(np.hstack((y_a,y_b)), axis=0)         # estimation of the model variance
    mean_est = np.mean(np.hstack((y_a,y_b)), axis=0)

    # allocate variables for the sensitivity indices
    sm = np.zeros(y_ba.shape[0])
    st = np.zeros(y_ba.shape[0])
    #
    # y_a_center = y_a - mean_est
    # y_b_center = y_b - mean_est

    # loop over all uncertain parameters and computation of the sensitivity indices
    for i in range(y_ba.shape[0]):
        #y_ab_i = y_ab[i, :]
        y_ba_i = y_ba[i,:]
        # y_ba_i_center = y_ba[i,:] - mean_est
        #
        # sm[i] = (2 * N) / (2 * N - 1) * (1 / N * np.sum((y_a_center * y_ba_i_center), axis=0) -
        #                                         (1 / 2 * (np.mean(y_a_center, axis=0) + np.mean(y_ba_i_center, axis=0))) ** 2 +
        #                                         (1 / (4 * N) * (np.var(y_a_center, axis=0) + np.var(y_ba_i_center,
        #                                                                                      axis=0)))) / var_est  # Owen
        # st[i] =  1 / (2 * N) * np.sum((y_b_center - y_ba_i_center) ** 2, axis=0) / var_est

        # main Sobol index following Owen
        sm[i] = (2 * N) / (2 * N - 1) * (1 / N * np.sum((y_a * y_ba_i), axis=0) -
                                                (1 / 2 * (np.mean(y_a, axis=0) + np.mean(y_ba_i, axis=0))) ** 2 +
                                                (1 / (4 * N) * (np.var(y_a, axis=0) + np.var(y_ba_i,
                                                                                             axis=0)))) / var_est  # Owen
        # total Sobol index following Owen
        st[i] =  1 / (2 * N) * np.sum((y_b - y_ba_i) ** 2, axis=0) / var_est

    return sm, st, var_est

####################################################################################################################
# function to generate sample matrices for two sets of realizations
#
# inputs:
# N                   number of samples per set
# jpdf                jpdf of the uncertain model parameters
# sample_method       sampling_method used to generate;
# rep                 counter of repetition which is currently computed
#
# output:
# A                   array with samples for the first set of model evaluations
# B                   array with samples for the second set of model evaluations
# C_BA                3-dimensional array with samples from the second set where the jth component is replaced with the
#                     jth component of the first sample set
####################################################################################################################
def generate_samples(N,jpdf,sample_method, rep):
    # sample model inputs
    #np.random.seed(420)
    Z = jpdf.sample(2*N, sample_method).transpose()
    np.random.seed(42)
    np.random.shuffle(Z)

    # generate A and B matrices
    A = Z[0:N]
    B = Z[N:]

    C_BA = np.empty((len(jpdf), N, len(jpdf)))

    # create C_BA sample matrices
    for i in range(len(jpdf)):
        C_BA[i, :, :] = B.copy()
        C_BA[i, :, i] = A[:, i].copy()

    return A, B,C_BA


########################################################################################################################
# Function to compute the optimal model evaluations and weights per fidelity level
# details in Qian et al. (2018) Multifidelity MC Estimation
# following Theorem 3.5
#
# Inputs:
# budget            int        value of total computational budget available for all fidelity levels
# budget_QoI        str        QoI used for optimal budget allocation
# costs             array      array of costs size(number models), where costs[0] is the high fidelity model
# statistics        dictionary dictionary containing the model statistics for each fidelity level
#
# Outputs:
# m                 array      array containing the number of model evaluations per fidelity level
# alpha             array      array containing the control variate coefficients size(number models)
########################################################################################################################
def opt_allocation(budget, costs, statistics):
    # number of models
    N_models = len(costs)
    temp = statistics['rho'] ** 2 - (np.hstack((statistics['rho'][1:], np.array([0]))) ** 2)
    r = np.sqrt(costs[0] * temp / (costs * (1 - statistics['rho'][1] ** 2)))
    m1 = budget / (np.dot(costs,r))
    # computation of the model evaluations per fidelity leveldgng
    m =  [math.floor(m1 * r[model]) for model in range(N_models)]
    # computation of control variate coefficients
    alpha = statistics['rho'] * statistics['sigma'][0] / statistics['sigma']

    return m, alpha
########################################################################################################################
