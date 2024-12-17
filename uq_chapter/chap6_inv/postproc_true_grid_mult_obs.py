# -*- coding: utf-8 -*-
"""
Postprocessing the grid files to create a ground truth for the inverse problem.
This script creates ground truth data needed for Figure 9 in the UQ chapter.

This one will take as input the Rp/Rd ratio and the total resistance Rp+Rd

@author: chloe
"""
#%% Import the modules
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.io as sio

# %% Choose the QOI for parsing the grid files

n_points = 32

# properties of blood/vessel
mu      = 0.04
r_aorta = 1.019478928

# quantity of interest
vessel = 'aorta'
what   = 'pressure'
# prober = 'min'
# name   = what+':'+vessel+':'+prober

# currently we are choosing the interpolated data
# zerod_filename  = '/Users/chloe/Desktop/grid_0D/QoI_0D_grid.csv'
zerod_filename  = './grid_0D/QoI_0D_grid.csv'
zerod_data      = pd.read_csv(zerod_filename)

# list all the different types of QOI headers
zerod_var_names = np.genfromtxt(zerod_filename, dtype='U', usecols=0, skip_header=1, delimiter=',')

# load the data
zerod_data      = np.genfromtxt(zerod_filename, usecols=range(1,n_points**3+1), skip_header=1, delimiter=',')

# pull out the QOI of interest
idxs = np.nonzero(zerod_var_names == what+':'+vessel+':min')[0]
qoi_min  = zerod_data[idxs,:]
qoi_min  = qoi_min.reshape(n_points, n_points, n_points)

idxs = np.nonzero(zerod_var_names == what+':'+vessel+':max')[0]
qoi_max  = zerod_data[idxs,:]
qoi_max  = qoi_max.reshape(n_points, n_points, n_points)

idxs = np.nonzero(zerod_var_names == what+':'+vessel+':avg')[0]
qoi_mean = zerod_data[idxs,:]
qoi_mean = qoi_mean.reshape(n_points, n_points, n_points)

#%% Create relevant meshgrids for posterior plots

# original [Rp, C, Rd] values
Rp_orig  = 6.8123e2
C_orig   = 3.6664e-5
Rd_orig  = 3.1013e4

# with corresponding bounds
RCR_bounds_grid = np.array([0.5, 0.5, 0.5])

# # Create linspace of [Rp, C, Rd] values to pull out the first 20 points
Rp_low  = Rp_orig - Rp_orig*RCR_bounds_grid[0]
Rp_high = Rp_orig + Rp_orig*RCR_bounds_grid[0]
C_low   = C_orig - C_orig*RCR_bounds_grid[1]
C_high  = C_orig + C_orig*RCR_bounds_grid[1]
Rd_low  = Rd_orig - Rd_orig*RCR_bounds_grid[2]
Rd_high = Rd_orig + Rd_orig*RCR_bounds_grid[2]

Rp_grid = np.linspace(Rp_low, Rp_high, n_points)
C_grid  = np.linspace(C_low, C_high, n_points)
Rd_grid = np.linspace(Rd_low, Rd_high, n_points)

# Create meshgrid
Rp_mesh, C_mesh, Rd_mesh = np.meshgrid(Rp_grid, C_grid, Rd_grid)

#%% Construct the posterior distribution

# s                = sio.loadmat('./data/y_obs.mat')
# y_no_noise       = s['y_no_noise']

x_coord = 16 # Rp
y_coord = 16 # C
z_coord = 16 # Rd

print(np.array([[Rp_mesh[x_coord, y_coord, z_coord]], [C_mesh[x_coord, y_coord, z_coord]], [Rd_mesh[x_coord, y_coord, z_coord]]]))
y_no_noise = np.array([[qoi_min[x_coord, y_coord, z_coord]], [qoi_max[x_coord, y_coord, z_coord]], [qoi_mean[x_coord, y_coord, z_coord]]])

# sigma_noise_min  = y_no_noise[0][0] * 0.01
# sigma_noise_max  = y_no_noise[1][0] * 0.01
# sigma_noise_mean = y_no_noise[2][0] * 0.01

# epsilon = [[np.random.normal(0, sigma_noise_min)], [np.random.normal(0, sigma_noise_max)], [np.random.normal(0, sigma_noise_mean)]]
# y_obs = np.array([[y_no_noise[0][0] + epsilon[0][0]], [y_no_noise[1][0] + epsilon[1][0]], [y_no_noise[2][0] + epsilon[2][0]]])

num_of_obs = 1

scaling_factor   = [0.01, 0.01, 0.01]

sigma_noise_min  = y_no_noise[0][0]*scaling_factor[0]
sigma_noise_max  = y_no_noise[1][0]*scaling_factor[1]
sigma_noise_mean = y_no_noise[2][0]*scaling_factor[2]

for i in range(num_of_obs):

    epsilon = [[np.random.normal(0, sigma_noise_min)], [np.random.normal(0, sigma_noise_max)], [np.random.normal(0, sigma_noise_mean)]]

    if i == 0:
        y_min  = y_no_noise[0] + epsilon[0]
        y_max  = y_no_noise[1] + epsilon[1]
        y_mean = y_no_noise[2] + epsilon[2]
    else:
        y_min  = np.append(y_min, y_no_noise[0][0] + epsilon[0][0])
        y_max  = np.append(y_max,  y_no_noise[1][0] + epsilon[1][0])
        y_mean = np.append(y_mean, y_no_noise[2][0] + epsilon[2][0])

y_obs = [y_min, y_max, y_mean]

#%%
# construct prior
x_mean          = np.array([[Rp_orig],[C_orig],[Rd_orig]])

def p_prior(x):
    # p_prior = 1
    Rp, C, Rd = x[0][0], x[1][0], x[2][0]
    if  (Rp > Rp_low and Rp < Rp_high) and \
        (C > C_low and C < C_high) and \
        (Rd > Rd_low and Rd < Rd_high):
        p_prior = 1 #/(Rp_high-Rp_low)/(C_high-C_low)/(Rd_high-Rd_low)
    else:
        p_prior = 0
    # cov_matrix   = np.array([[(Rp_orig/8)**2, 0, 0], [0, (C_orig/8)**2, 0], [0, 0, (Rd_orig/8)**2]])
    # inv_cov      = np.linalg.inv(cov_matrix)
    # det_cov      = np.linalg.det(cov_matrix)
    # p_prior      = (2*np.pi)**(-3/2) * det_cov**(-1/2) * np.exp(-0.5*np.matmul(np.matmul(np.transpose(x-x_mean),inv_cov),x-x_mean))
    return p_prior

def p_likelihood(y):
    cov_matrix  = np.array([[(sigma_noise_min)**2, 0, 0], [0, (sigma_noise_max)**2, 0], [0, 0, (sigma_noise_mean)**2]])
    inv_cov     = np.linalg.inv(cov_matrix)
    p_log_likelihood     = 0
    if np.shape(y_obs)[1]>1:
        for i in np.arange(np.shape(y_obs)[1]):
            # add the log likelihoods for the observations
            y_obs_idx          = np.array([[y_obs[0][i]], [y_obs[1][i]], [y_obs[2][i]]])
            p_log_likelihood   += -0.5*np.matmul(np.matmul(np.transpose(y-y_obs_idx),inv_cov),y-y_obs_idx)
        p_likelihood = np.exp(p_log_likelihood)
    else:
        p_likelihood = np.exp(-0.5*np.matmul(np.matmul(np.transpose(y-y_obs),inv_cov),y-y_obs))
    return p_likelihood

# construct posterior
P_posterior    = lambda x,y: p_likelihood(y)*p_prior(x)

grid_posterior = np.zeros((qoi_min.shape[0], qoi_min.shape[1], qoi_min.shape[2]))
grid_prior     = np.zeros((qoi_min.shape[0], qoi_min.shape[1], qoi_min.shape[2]))

for i in tqdm(np.arange(qoi_min.shape[0])):
    for j in np.arange(qoi_min.shape[1]):
        for k in np.arange(qoi_min.shape[2]):
            # theta = np.array([[Rp_mesh[i,j,k]], [C_mesh[i,j,k]], [Rd_mesh[i,j,k]]])
            # grid_posterior[i,j,k] = P_posterior(theta, np.array([[qoi_min[i,j,k]], [qoi_max[i,j,k]], [qoi_mean[i,j,k]]]))
            # grid_prior[i,j,k] = p_prior(theta)
            theta = np.array([[Rp_mesh[i,j,k]], [C_mesh[i,j,k]], [Rd_mesh[i,j,k]]])
            grid_posterior[i,j,k] = P_posterior(theta, np.array([[qoi_min[i,j,k]], [qoi_max[i,j,k]], [qoi_mean[i,j,k]]]))
            grid_prior[i,j,k] = p_prior(theta)

posterior = grid_posterior / np.sum(grid_posterior.flatten()*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])*(Rd_grid[1]-Rd_grid[0]))
prior     = grid_prior / np.sum(grid_prior.flatten()*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])*(Rd_grid[1]-Rd_grid[0]))

posterior = np.reshape(posterior, (n_points, n_points, n_points))
prior     = np.reshape(prior, (n_points, n_points, n_points))

# %% Compute the marginal distributions
# dim = 1 is Rp, dim = 0 is C, dim = 2 is Rd

# marginal distributions along 2 dims
marginal_CRd  = np.sum(posterior, axis=1)*(Rp_grid[1]-Rp_grid[0])
marginal_RpRd = np.sum(posterior, axis=0)*(C_grid[1]-C_grid[0])
marginal_CRp  = np.sum(posterior, axis=2)*(Rd_grid[1]-Rd_grid[0])

# marginal distribution along 1 dim
marginal_Rp   = np.sum(posterior, axis=(0,2))*(Rd_grid[1]-Rd_grid[0])*(C_grid[1]-C_grid[0])
marginal_Rd   = np.sum(posterior, axis=(1,0))*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])
marginal_C    = np.sum(posterior, axis=(1,2))*(Rp_grid[1]-Rp_grid[0])*(Rd_grid[1]-Rd_grid[0])

#%% Compute the prior distributions

# prior distribution along 2 dims
prior_CRd     = np.sum(prior, axis=1)*(Rp_grid[1]-Rp_grid[0])
prior_RpRd    = np.sum(prior, axis=0)*(C_grid[1]-C_grid[0])
prior_CRp     = np.sum(prior, axis=2)*(Rd_grid[1]-Rd_grid[0])

# prior distribution along 1 dim
prior_Rp      = np.sum(prior, axis=(0,2))*(Rd_grid[1]-Rd_grid[0])*(C_grid[1]-C_grid[0])
prior_Rd      = np.sum(prior, axis=(1,0))*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])
prior_C       = np.sum(prior, axis=(1,2))*(Rp_grid[1]-Rp_grid[0])*(Rd_grid[1]-Rd_grid[0])

#%% Plot the posterior distributions for the marginal distributions along 2 dims

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18
})

fig, ax = plt.subplots(1, 3, figsize=(12,3), gridspec_kw={'width_ratios': [1, 1, 1],
                                          'height_ratios': [1]})

slice_idx = 0

cbar_a = ax[0].contourf(Rp_grid, C_grid, marginal_CRp.T)
# ax[0].contour(Rp_grid, C_grid, prior_CRp.T, colors='y', linestyles='--')
ax[0].set_xlabel('$R_p$')
ax[0].set_ylabel('$C$')

cbar_b = ax[1].contourf(Rp_grid, Rd_grid, marginal_RpRd.T)
# ax[1].contour(Rp_grid, Rd_grid, prior_RpRd.T, colors='y', linestyles='--')
ax[1].set_xlabel('$R_p$')
ax[1].set_ylabel('$R_d$')
# ax[1].set_ylim([25000, 28000])
# ax[1].set_xlim([Rp_low, 1000])

cbar_c = ax[2].contourf(Rd_grid, C_grid, marginal_CRd)
# ax[2].contour(Rd_grid, C_grid, prior_CRd, colors='y', linestyles='--')
ax[2].set_xlabel('$R_d$')
ax[2].set_ylabel('$C$')
# ax[2].set_ylim([3.5e-5, 5e-5])
# ax[2].set_xlim([25000, 30000])

nbins = 6
cbar = fig.colorbar(cbar_a)
cbar.ax.tick_params(labelsize=12)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cbar.locator = tick_locator
cbar.update_ticks()

cbar2 = fig.colorbar(cbar_b)
cbar2.ax.tick_params(labelsize=12)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cbar2.locator = tick_locator
cbar2.update_ticks()

cbar3 = fig.colorbar(cbar_c)
cbar3.ax.tick_params(labelsize=12)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cbar3.locator = tick_locator
cbar3.update_ticks()

fig.suptitle('Posterior distributions')

fig.tight_layout()

# %% Plot the one dimensional marginals along 1 dim

fig, ax = plt.subplots(1, 3, figsize=(12,3), gridspec_kw={'width_ratios': [1, 1, 1],
                                          'height_ratios': [1]})

slice_idx = 0

cbar_a = ax[0].plot(Rp_grid, marginal_Rp)
# ax[0].plot(Rp_grid, prior_Rp, 'k--')
ax[0].set_xlabel('$R_p$')
# ax[0].legend(['Posterior', 'Prior'])

cbar_b = ax[1].plot(C_grid, marginal_C)
# ax[1].plot(C_grid, prior_C, 'k--')
# ax[1].legend(['Posterior', 'Prior'])
ax[1].set_xlabel('$C$')

cbar_c = ax[2].plot(Rd_grid, marginal_Rd)
# ax[2].plot(Rd_grid, prior_Rd, 'k--')
# ax[2].legend(['Posterior', 'Prior'])
ax[2].set_xlabel('$R_d$')

fig.tight_layout()

#%% Save the marginal distributions

# zerod_filepath  = '/Users/chloe/Desktop/grid_0D/'

# zerod_filepath  = './grid_0D/'

# sio.savemat(zerod_filepath+'marginal_CRp.mat', 
#             {'Rp_grid': Rp_grid,
#              'C_grid': C_grid, 
#              'Rd_grid': Rd_grid,
#              'marginal_CRp': marginal_CRp.T.detach().numpy(),
#              'marginal_RpRd': marginal_RpRd.T.detach().numpy(),
#              'marginal_CRd': marginal_CRd.detach().numpy(),
#              'marginal_Rp': marginal_Rp.detach().numpy(),
#              'marginal_C': marginal_C.detach().numpy(),
#              'marginal_Rd': marginal_Rd.detach().numpy(),
#              'prior_CRp': prior_CRp.T.detach().numpy(),
#              'prior_RpRd': prior_RpRd.T.detach().numpy(),
#              'prior_CRd': prior_CRd.detach().numpy(),
#              'prior_Rp': prior_Rp.detach().numpy(),
#              'prior_C': prior_C.detach().numpy(),
#              'prior_Rd': prior_Rd.detach().numpy()})

# %%
