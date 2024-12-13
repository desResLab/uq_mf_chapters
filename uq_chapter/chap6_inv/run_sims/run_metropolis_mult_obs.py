#%%
import numpy as np
import os
import scipy.io as sio
from get_QOI_0D import get_QOI_0D
import sys

#%%

# Next steps:
# (0) Run finer grid? (yes for 2D)
# (1) Run MH on Sherlock, now that we fixed numerical issues
    # send to Daniele and get his feedback
# (2) Make the 2D version: now we have a ratio (Rp/Rd), and so we have Rd and C only (one 2D, two 1D plots)

file_path   = '/scratch/users/chloe1/new_mult_obs/chain_1/'
solver      = '/home/users/chloe1/svZeroDSolver/Release/svzerodsolver'
#solver      = '/oak/stanford/groups/amarsden/chloe1/svZeroDSolver/Release/svzerodsolver'
total_size  = int(sys.argv[1])
burnin_size = int(sys.argv[2])
dim = 3

# Load observation with noise (min pressure, max pressure)
s               = sio.loadmat('./data/y_obs_mult.mat')
sigma_noise_max = s['sigma_noise_max'][0][0]
sigma_noise_min = s['sigma_noise_min'][0][0]
sigma_noise_mean= s['sigma_noise_mean'][0][0]
y_obs           = s['y_obs']

# [Rp, Rd, C] mean values
Rp                             = 6.8123e2
Rd                             = 3.1013e4
C                              = 3.6664e-5
x_avg                          = np.array([Rp, Rd, C])
Rp_low, Rp_high                = Rp - Rp*0.5, Rp + Rp*0.5
Rd_low, Rd_high                = Rd - Rd*0.5, Rd + Rd*0.5
C_low, C_high                  = C - C*0.5, C + C*0.5
length_Rp, length_Rd, length_C = Rp_high - Rp_low, Rd_high - Rd_low, C_high - C_low
x_init = np.array([np.random.uniform(Rp-0.5*Rp, Rp+0.5*Rp), np.random.uniform(Rd-0.5*Rd, Rd+0.5*Rd), np.random.uniform(C-0.5*C, C+0.5*C)])

# uniform prior
def p_prior(x):
    Rp, Rd, C = x[0], x[1], x[2]
    if  (Rp >= Rp_low and Rp <= Rp_high) and \
        (C >= C_low and C <= C_high) and \
        (Rd >= Rd_low and Rd <= Rd_high):
        p_prior = 1 #/(Rp_high-Rp_low)/(C_high-C_low)/(Rd_high-Rd_low)
    else:
        p_prior = 0
    return p_prior

def p_likelihood(y):
    cov_matrix           = np.array([[(sigma_noise_min)**2, 0, 0], [0, (sigma_noise_max)**2, 0], [0, 0, (sigma_noise_mean)**2]])
    inv_cov, det_cov, k  = np.linalg.inv(cov_matrix), np.linalg.det(cov_matrix), np.shape(cov_matrix)[0]
    p_log_likelihood     = 0
    if np.shape(y_obs)[1]>1:
        for i in np.arange(np.shape(y_obs)[1]):
            # add the log likelihoods for the observations
            y_obs_idx          = np.array([[y_obs[0][i]], [y_obs[1][i]], [y_obs[2][i]]])
            p_log_likelihood  += -0.5*np.matmul(np.matmul(np.transpose(y-y_obs_idx),inv_cov),y-y_obs_idx)
        p_likelihood = np.exp(p_log_likelihood)
    else:
        p_likelihood = np.exp(-0.5*np.matmul(np.matmul(np.transpose(y-y_obs),inv_cov),y-y_obs))
    return p_likelihood

def posterior_distribution(x, y):
    p_posterior = p_likelihood(y) * p_prior(x)
    return p_posterior

# def sample_candidate_point(xt,var):
#     # sample point and ensure it is within the bounds
#     xt_candidate = np.random.multivariate_normal(xt, var)
#     if xt_candidate[0] < Rp_low or xt_candidate[0] > Rp_high:
#         xt_candidate[0] = Rp_low + (xt_candidate[0] - Rp_low) % length_Rp
#     if xt_candidate[1] < Rd_low or xt_candidate[1] > Rd_high:
#         xt_candidate[1] = Rd_low + (xt_candidate[1] - Rd_low) % length_Rd
#     if xt_candidate[2] < C_low or xt_candidate[2] > C_high:
#         xt_candidate[2] = C_low + (xt_candidate[2] - C_low) % length_C
#     return xt_candidate

def metropolis_hastings(file_path, target_density, dim, var, burnin_size):

    xt = x_init # x_avg
    os.system("python " + file_path + "create_input.py 0 " + str(xt[0]) + ' ' + str(xt[1]) + ' ' + str(xt[2]))
    os.system(solver + ' ' + file_path + 'sims/sim_0/aobif_0.json ' + file_path + 'sims/sim_0/aobif_0.csv')
    
    # now we change this to get both max and min pressure
    yt          = get_QOI_0D(file_path + 'sims/sim_0/aobif_0.csv')
    n_accepted  = 0
    samples     = np.array([])
    
    for i in range(1, total_size):

        xt_candidate = np.random.multivariate_normal(xt, var)
        # xt_candidate = sample_candidate_point(xt, var)
        
        if p_prior(xt_candidate) == 0:
            # do not run 0D solver if the xt_candidate is outside the bounds
            mh_ratio = 0
        else:
            os.system("python " + file_path + "create_input.py " + str(i) + ' ' + str(xt_candidate[0]) + ' ' + str(xt_candidate[1]) + ' ' + str(xt_candidate[2]))
            os.system(solver + ' ' + file_path + 'sims/sim_' + str(i) + '/aobif_' + str(i) + '.json ' + file_path + 'sims/sim_' + str(i) + '/aobif_' + str(i) + '.csv')
            yt_candidate = get_QOI_0D(file_path + 'sims/sim_' + str(i) + '/aobif_' + str(i) +'.csv')
            mh_ratio = (target_density(xt_candidate, yt_candidate))/(target_density(xt, yt))

        accept_prob = min(1, mh_ratio)
        if np.random.uniform(0, 1) < accept_prob:
            xt_new = xt_candidate
            yt_new = yt_candidate
            n_accepted += 1
        else:
            xt_new = xt
            yt_new = yt
        samples = np.append(samples, xt_new)
        
        xt = xt_new
        yt = yt_new

        sio.savemat(file_path+'mh_sim.mat', {'samples':samples, 'n_accepted':n_accepted})

var = np.array([[(x_avg[0]/10)**2, 0, 0], [0, (x_avg[1]/10)**2, 0], [0, 0, (x_avg[2]/10)**2]])
metropolis_hastings(file_path, posterior_distribution, dim, var, burnin_size)
