import numpy as np
import os
import scipy.io as sio
from get_QOI_0D import get_QOI_0D
import sys

file_path   = '/scratch/users/chloe1/chain_6/'
solver = '/home/users/chloe1/svZeroDSolver/Release/svzerodsolver'
total_size  = int(sys.argv[1])
burnin_size = int(sys.argv[2])
dim = 3

# Observation with noise (wall shear stress)
obs_file    = sio.loadmat(file_path + 'y_obs.mat')
y_obs       = obs_file['y_obs']

# [Rp, Rd, C]
Rp = 6.8123e2
Rd = 3.1013e4
C  = 3.6664e-5
#x_avg = np.array([6.8123e2, 3.1013e4, 3.6664e-5])
x_avg = np.array([Rp, Rd, C])
x_init = np.array([np.random.uniform(Rp-0.5*Rp, Rp+0.5*Rp), np.random.uniform(Rd-0.5*Rd, Rd+0.5*Rd), np.random.uniform(C-0.5*C, C+0.5*C)])

# Prior distribution

sigma_prior = 0.3
sigma_noise = 0.0005 # this is chosen for the average wss

x_mean       = np.array([x_avg[0],x_avg[1], x_avg[2]])
cov_matrix   = np.array([[(x_avg[0]/6)**2, 0, 0], [0, (x_avg[1]/6)**2, 0], [0, 0, (x_avg[2]/6)**2]])
inv_cov      = np.linalg.inv(cov_matrix)
det_cov      = np.linalg.det(cov_matrix)

p_prior      = lambda x: (2*np.pi)**(-3/2) * det_cov**(-1/2) * np.exp(-0.5*np.matmul(np.matmul(np.transpose(x-x_mean),inv_cov),x-x_mean))

def posterior_distribution(x, y):
    p_likelihood = np.exp(-(y_obs-y)**2/(2*sigma_noise**2))
    p = p_prior(x)
    p_posterior = p_likelihood * p
    return p_posterior

def metropolis_hastings(file_path, target_density, dim, var, burnin_size):

    xt          = x_init #x_avg
    os.system("python " + file_path + "create_input.py 0 " + str(xt[0]) + ' ' + str(xt[1]) + ' ' + str(xt[2]))
    os.system(solver + ' ' + file_path + 'sims/sim_0/aobif_0.json ' + file_path + 'sims/sim_0/aobif_0.csv')
    yt = get_QOI_0D(file_path + 'sims/sim_0/aobif_0.csv')
    n_accepted  = 0
    samples     = np.array([])
    
    for i in range(1, total_size):

        xt_candidate = np.random.multivariate_normal(xt, var)
        
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
