# -*- coding: utf-8 -*-
"""
Postprocessing the grid files to create a ground truth for the inverse problem.
This script creates ground truth data needed for Figure 9 in the UQ chapter.

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
prober = 'avg'
what   = 'flow'
name   = what+':'+vessel+':'+prober

# currently we are choosing the interpolated data
# zerod_filename  = '/Users/chloe/Desktop/grid_0D/QoI_0D_grid.csv'
zerod_filename  = './grid_0D/QoI_0D_grid.csv'
zerod_data      = pd.read_csv(zerod_filename)

# list all the different types of QOI headers
zerod_var_names = np.genfromtxt(zerod_filename, dtype='U', usecols=0, skip_header=1, delimiter=',')

# load the data
zerod_data      = np.genfromtxt(zerod_filename, usecols=range(1,n_points**3+1), skip_header=1, delimiter=',')

# pull out the QOI of interest
idxs = np.nonzero(zerod_var_names == name)[0]
qoi  = zerod_data[idxs,:]
qoi  = 4 * mu * qoi / (np.pi * r_aorta**3)
qoi  = qoi.reshape(n_points, n_points, n_points)

#%% Create relevant meshgrids for posterior plots

# original [Rp, C, Rd] values
Rp_orig  = 6.8123e2
C_orig   = 3.6664e-5
Rd_orig  = 3.1013e4

# with corresponding bounds
RCR_bounds_grid = np.array([0.5, 0.5, 0.5])

# # Create linspace of [Rp, C, Rd] values to pull out the first 20 points
Rp_grid = np.linspace(Rp_orig-Rp_orig*RCR_bounds_grid[0], Rp_orig+Rp_orig*RCR_bounds_grid[0], n_points)
C_grid  = np.linspace(C_orig-C_orig*RCR_bounds_grid[1], C_orig+C_orig*RCR_bounds_grid[1], n_points)
Rd_grid = np.linspace(Rd_orig-Rd_orig*RCR_bounds_grid[2], Rd_orig+Rd_orig*RCR_bounds_grid[2], n_points)

# Create meshgrid
Rp_mesh, C_mesh, Rd_mesh = np.meshgrid(Rp_grid, C_grid, Rd_grid)

#%% Construct the posterior distribution

s        = sio.loadmat('/Users/chloe/Desktop/invprob/y_obs.mat')
y_obs    = s['y_obs']
y_obs    = torch.Tensor(y_obs)

# construct prior
x_mean       = torch.tensor([[Rp_orig],[C_orig],[Rd_orig]]).float()
cov_matrix   = torch.tensor([[(Rp_orig/6)**2, 0, 0], [0, (C_orig/6)**2, 0], [0, 0, (Rd_orig/6)**2]])
inv_cov      = torch.inverse(cov_matrix).float()
det_cov      = torch.det(cov_matrix).float()
P_prior      = lambda x: (2*np.pi)**(-3/2) * det_cov**(-1/2) * \
                    torch.exp(-0.5*torch.mm(torch.mm(torch.transpose((x-x_mean),1,0).float(),inv_cov),(x-x_mean).float()))

# construct likelihood
sigma_noise    = 0.0005
P_likelihood   = lambda y: torch.exp(-(y_obs - y)**2/(2*sigma_noise**2))

# construct posterior
P_posterior    = lambda x,y: P_likelihood(y)*P_prior(x)

grid_posterior = torch.zeros((qoi.shape[0], qoi.shape[1], qoi.shape[2]))
grid_prior     = torch.zeros((qoi.shape[0], qoi.shape[1], qoi.shape[2]))

for i in tqdm(np.arange(qoi.shape[0])):
    for j in np.arange(qoi.shape[1]):
        for k in np.arange(qoi.shape[2]):
            theta = torch.tensor([[Rp_mesh[i,j,k]], [C_mesh[i,j,k]], [Rd_mesh[i,j,k]]])
            grid_posterior[i,j,k] = P_posterior(theta, torch.Tensor([qoi[i,j,k]]))
            grid_prior[i,j,k] = P_prior(theta)

posterior = grid_posterior / torch.sum(grid_posterior.flatten()*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])*(Rd_grid[1]-Rd_grid[0]))
prior     = grid_prior / torch.sum(grid_prior.flatten()*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])*(Rd_grid[1]-Rd_grid[0]))

posterior = np.reshape(posterior, (n_points, n_points, n_points))
prior = np.reshape(prior, (n_points, n_points, n_points))

# %% Compute the marginal distributions
# dim = 1 is Rp, dim = 0 is C, dim = 2 is Rd

# marginal distributions along 2 dims
marginal_CRd  = torch.sum(posterior, dim=1)*(Rp_grid[1]-Rp_grid[0])
marginal_RpRd = torch.sum(posterior, dim=0)*(C_grid[1]-C_grid[0])
marginal_CRp  = torch.sum(posterior, dim=2)*(Rd_grid[1]-Rd_grid[0])

# marginal distribution along 1 dim
marginal_Rp   = torch.sum(posterior, dim=(0,2))*(Rd_grid[1]-Rd_grid[0])*(C_grid[1]-C_grid[0])
marginal_Rd   = torch.sum(posterior, dim=(1,0))*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])
marginal_C    = torch.sum(posterior, dim=(1,2))*(Rp_grid[1]-Rp_grid[0])*(Rd_grid[1]-Rd_grid[0])

#%% Compute the prior distributions

# prior distribution along 2 dims
prior_CRd     = torch.sum(prior, dim=1)*(Rp_grid[1]-Rp_grid[0])
prior_RpRd    = torch.sum(prior, dim=0)*(C_grid[1]-C_grid[0])
prior_CRp     = torch.sum(prior, dim=2)*(Rd_grid[1]-Rd_grid[0])

# prior distribution along 1 dim
prior_Rp      = torch.sum(prior, dim=(0,2))*(Rd_grid[1]-Rd_grid[0])*(C_grid[1]-C_grid[0])
prior_Rd      = torch.sum(prior, dim=(1,0))*(Rp_grid[1]-Rp_grid[0])*(C_grid[1]-C_grid[0])
prior_C       = torch.sum(prior, dim=(1,2))*(Rp_grid[1]-Rp_grid[0])*(Rd_grid[1]-Rd_grid[0])

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
ax[0].contour(Rp_grid, C_grid, prior_CRp.T, colors='y', linestyles='--')
ax[0].set_xlabel('$R_p$')
ax[0].set_ylabel('$C$')

cbar_b = ax[1].contourf(Rp_grid, Rd_grid, marginal_RpRd.T)
ax[1].contour(Rp_grid, Rd_grid, prior_RpRd.T, colors='y', linestyles='--')
ax[1].set_xlabel('$R_p$')
ax[1].set_ylabel('$R_d$')

cbar_c = ax[2].contourf(Rd_grid, C_grid, marginal_CRd)
ax[2].contour(Rd_grid, C_grid, prior_CRd, colors='y', linestyles='--')
ax[2].set_xlabel('$R_d$')
ax[2].set_ylabel('$C$')

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
ax[0].plot(Rp_grid, prior_Rp, 'k--')
ax[0].set_xlabel('$R_p$')
ax[0].legend(['Posterior', 'Prior'])

cbar_b = ax[1].plot(C_grid, marginal_C)
ax[1].plot(C_grid, prior_C, 'k--')
ax[1].legend(['Posterior', 'Prior'])
ax[1].set_xlabel('$C$')

cbar_c = ax[2].plot(Rd_grid, marginal_Rd)
ax[2].plot(Rd_grid, prior_Rd, 'k--')
ax[2].legend(['Posterior', 'Prior'])
ax[2].set_xlabel('$R_d$')

fig.tight_layout()

#%% Save the marginal distributions

# zerod_filepath  = '/Users/chloe/Desktop/grid_0D/'

zerod_filepath  = './grid_0D/'

sio.savemat(zerod_filepath+'marginal_CRp.mat', 
            {'Rp_grid': Rp_grid,
             'C_grid': C_grid, 
             'Rd_grid': Rd_grid,
             'marginal_CRp': marginal_CRp.T.detach().numpy(),
             'marginal_RpRd': marginal_RpRd.T.detach().numpy(),
             'marginal_CRd': marginal_CRd.detach().numpy(),
             'marginal_Rp': marginal_Rp.detach().numpy(),
             'marginal_C': marginal_C.detach().numpy(),
             'marginal_Rd': marginal_Rd.detach().numpy(),
             'prior_CRp': prior_CRp.T.detach().numpy(),
             'prior_RpRd': prior_RpRd.T.detach().numpy(),
             'prior_CRd': prior_CRd.detach().numpy(),
             'prior_Rp': prior_Rp.detach().numpy(),
             'prior_C': prior_C.detach().numpy(),
             'prior_Rd': prior_Rd.detach().numpy()})

# %%
