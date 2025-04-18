"""
Postprocess MH results then calculate Gelman-Rubin statistics.
This script plots Figures 7, 8, 9 in the UQ chapter.

@author: chloe
"""
#%% Import modules
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm

#%% Plot Figure 9 in the UQ chapter

Rp_obs = 670.242419
Rd_obs = 32513.629
C_obs  = 3.13418065e-5

burnin_size = 1000
dim         = 3
# file_path   = '/Users/chloe/Desktop/'
file_path = './data/'

fs = 21

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": fs,
    "axes.titlesize": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs-2,
    "figure.titlesize": fs,
})

num_obs = 'one_obs'

for i in range(1,6+1): #range(6,6+1) for the paper

    print(i)

    filename = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
    n_accepted = filename['n_accepted']
    samples    = filename['samples']
    samples    = np.squeeze(samples)

    size       = np.shape(samples)[0]/3

    samples    = np.array(samples[3*burnin_size:])
    samples    = np.reshape(samples, [int(samples.shape[0]/dim), dim])
    acceptance_rate = n_accepted / size

    print('accepted='+str(n_accepted))
    print('size='+str(int(size)))
    mhratio = n_accepted/size
    print('acceptance rate='+str(mhratio))

    fig, ax = plt.subplots(2, 3, figsize=(15,7), gridspec_kw={'width_ratios': [1, 1, 1],
                                            'height_ratios': [1, 1]}, layout='constrained')
    ax = ax.flatten()

    # original [Rp, C, Rd] values
    Rp_orig  = 6.8123e2
    C_orig   = 3.6664e-5
    Rd_orig  = 3.1013e4

    # with corresponding bounds
    RCR_bounds_grid = np.array([0.5, 0.5, 0.5])

    # Create linspace of [Rp, C, Rd] values to pull out the first 20 points
    n_points = 32
    Rp_grid = np.linspace(Rp_orig-Rp_orig*RCR_bounds_grid[0], Rp_orig+Rp_orig*RCR_bounds_grid[0], n_points)
    C_grid  = np.linspace(C_orig-C_orig*RCR_bounds_grid[1], C_orig+C_orig*RCR_bounds_grid[1], n_points)
    Rd_grid = np.linspace(Rd_orig-Rd_orig*RCR_bounds_grid[2], Rd_orig+Rd_orig*RCR_bounds_grid[2], n_points)

    # dim=0 is Rp, dim=2 is C
    cbar_a = ax[0].hexbin(samples[:,0], samples[:,2], mincnt=1, cmap='rainbow', label='_nolegend_')
    ax[0].set_xlabel(r'$R_p$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    ax[0].set_ylabel(r'$C$ ($\mathrm{cm}^5/\mathrm{dyne}$)')
    x_obs = ax[0].scatter(Rp_obs, C_obs, s=150, marker=(5,1), color='red', zorder=2)
    ax[0].legend(['Obs'], loc='lower left')
    ax[0].set_xlim(Rp_grid[0], Rp_grid[-1])
    ax[0].set_ylim(C_grid[0], C_grid[-1])

    # dim=0 is Rp, dim=1 is Rd
    cbar_b = ax[1].hexbin(samples[:,0],samples[:,1], mincnt=1, cmap='rainbow', label='_nolegend_')
    ax[1].set_xlabel(r'$R_p$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    ax[1].set_ylabel(r'$R_d$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    x_obs = ax[1].scatter(Rp_obs, Rd_obs, s=150, marker=(5,1), color='red', zorder=2)
    ax[1].legend(['Obs'], loc='lower left')
    ax[1].set_xlim(Rp_grid[0], Rp_grid[-1])
    ax[1].set_ylim(Rd_grid[0], Rd_grid[-1])

    # dim=1 is Rd, dim=2 is C
    cbar_c = ax[2].hexbin(samples[:,1],samples[:,2], mincnt=1, cmap='rainbow', label='_nolegend_')
    ax[2].set_xlabel(r'$R_d$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    ax[2].set_ylabel(r'$C$ ($\mathrm{cm}^5/\mathrm{dyne}$)')
    x_obs = ax[2].scatter(Rd_obs, C_obs, s=150, marker=(5,1), color='red', zorder=2)
    ax[2].legend(['Obs'], loc='lower left')
    ax[2].set_xlim(Rd_grid[0], Rd_grid[-1])
    ax[2].set_ylim(C_grid[0], C_grid[-1])

    fig.colorbar(cbar_a)
    fig.colorbar(cbar_b)
    fig.colorbar(cbar_c)

    # One-dimensional histograms
    ax[3].hist(samples[:,0], density=True, bins=40)
    ax[3].axvline(Rp_obs, color='r', linestyle='--')
    ax[3].set_xlabel(r'$R_p$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    ax[3].set_xlim(Rp_grid[0], Rp_grid[-1])
    ax[3].legend(['Obs','MH'])

    ax[4].hist(samples[:,2], density=True, bins=40)
    ax[4].axvline(C_obs, color='r', linestyle='--')
    ax[4].set_xlabel(r'$C$ ($\mathrm{cm}^5/\mathrm{dyne}$)')
    ax[4].set_xlim(C_grid[0], C_grid[-1])
    ax[4].legend(['Obs','MH'])

    ax[5].hist(samples[:,1], density=True, bins=40)
    ax[5].axvline(Rd_obs, color='r', linestyle='--')
    ax[5].set_xlabel(r'$R_d$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
    ax[5].set_xlim(Rd_grid[0], Rd_grid[-1])
    ax[5].legend(['Obs','MH'])

    fig.get_layout_engine().set(hspace=0.2)

    # if i == 6:
    #     plt.savefig('./figs/hexbin_and_hist.png', dpi=300)

#%% Compute the MAP using various methods

# using the MH samples
print('Obtaining MAP values...')
print(' -----------------')
print('From MH samples: ')
print('Rp = '+str(statistics.mode(samples[:,0])))
print('C = '+str(statistics.mode(samples[:,2])))
print('Rd = '+str(statistics.mode(samples[:,1])))

# the original mean values
print(' -----------------')
print('Original mean values: ')
print('Rp = '+str(Rp_orig))
print('C = '+str(C_orig))
print('Rd = '+str(Rd_orig))

# the observation
print(' -----------------')
print('Observation: ')
print('Rp = '+str(Rp_obs))
print('C = '+str(C_obs))
print('Rd = '+str(Rd_obs))

#%% Calculate Gelman-Rubin statistics

# number of parameters
num_params = 3

# number of chains
J = 6

# burn-in size
D = 1000

# remaining samples
L = 18999

chain_mean = np.zeros((J, num_params))
sj_squared = np.zeros((J, num_params))

for i in range(1,J+1):

    # filename   = sio.loadmat(file_path+'chains_var_div10/chain_'+str(i)+'.mat')
    filename = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
    n_accepted = filename['n_accepted']
    samples    = filename['samples']
    samples    = np.squeeze(samples)
    size       = np.shape(samples)[0]/3
    samples    = np.array(samples[3*D:])
    samples    = np.reshape(samples, [int(samples.shape[0]/dim), dim])

    # mean of each chain
    chain_mean[i-1,0] = np.mean(samples[:,0])
    chain_mean[i-1,1] = np.mean(samples[:,1])
    chain_mean[i-1,2] = np.mean(samples[:,2])

    # within-chain variance
    #sj_squared[i-1] = 1/(L-1)*np.sum((np.mean(samples.flatten())-chain_mean)**2)  
    sj_squared[i-1,0] = np.var(samples[:,0])
    sj_squared[i-1,1] = np.var(samples[:,1])
    sj_squared[i-1,2] = np.var(samples[:,2])

for i in range(num_params):
    # mean over all chains
    grand_mean = np.mean(chain_mean[:,i])

    # between-chain variance
    B = L/(J-1)*np.sum((chain_mean[:,i]-grand_mean)**2)
    W = np.mean(sj_squared[:,i])

    # Compute the Gelman-Rubin statistic
    R = ((L-1)/L*W + 1/L*B)/W

    if i == 0:
        print('The Gelman-Rubin statistic for Rp is '+str(R))
    elif i == 1:
        print('The Gelman-Rubin statistic for Rd is '+str(R))
    elif i == 2:
        print('The Gelman-Rubin statistic for C is '+str(R))

# %% Gelman Rubin convergence

GR_evolution = np.empty((L,num_params))

for j in range(num_params):

    for k in tqdm(range(1,L+1)):
        
        chain_mean = np.zeros((J,num_params))
        sj_squared = np.zeros((J,num_params))
        
        for i in range(1,J+1):
        
            filename   = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
            # filename = sio.loadmat('./'+num_obs+'/chain_'+str(i)+'/mh_sim.mat')
            n_accepted = filename['n_accepted']
            samples    = filename['samples']
            samples    = np.squeeze(samples)
            size       = np.shape(samples)[0]/3
            samples    = np.array(samples[3*D:3*(L+D)])
            samples    = np.reshape(samples, [int(samples.shape[0]/dim), dim])[:k]
        
            # mean of each chain
            chain_mean[i-1,0] = np.mean(samples[:,0])
            chain_mean[i-1,1] = np.mean(samples[:,1])
            chain_mean[i-1,2] = np.mean(samples[:,2])

            # within-chain variance
            #sj_squared[i-1] = 1/(L-1)*np.sum((np.mean(samples.flatten())-chain_mean)**2)  
            sj_squared[i-1,0] = np.var(samples[:,0])
            sj_squared[i-1,1] = np.var(samples[:,1])
            sj_squared[i-1,2] = np.var(samples[:,2])
        
        # mean over all chains
        grand_mean = np.mean(chain_mean[:,j])

        # between-chain variance
        B = L/(J-1)*np.sum((chain_mean[:,j]-grand_mean)**2)
        W = np.mean(sj_squared[:,j])

        # Compute the Gelman-Rubin statistic
        R = ((L-1)/L*W + 1/L*B)/W

        GR_evolution[k-1,j] = R

#%% New plots

fig, ax = plt.subplots(1, 4, figsize=(15,3), gridspec_kw={'width_ratios': [1, 1, 1, 1],
                                            'height_ratios': [1]}, layout='constrained')
ax = ax.flatten()

mh_iters = np.arange(1000, 18999)

ax[0].plot(mh_iters, samples[1000:,0])
ax[0].set_xlabel('MH iterations')
ax[0].set_title(r'$R_p$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
ax[0].set_xticks([1000, 5000, 10000, 15000])
# ax[0].set_ylim(Rp_grid[0], Rp_grid[-1])

ax[1].plot(mh_iters, samples[1000:,1])
ax[1].set_xlabel('MH iterations')
ax[1].set_title(r'$R_d$ ($\mathrm{dyne} \cdot \mathrm{s}/\mathrm{cm}^2$)')
ax[1].set_xticks([1000, 5000, 10000, 15000])
ax[1].set_ylim(Rd_grid[0], Rd_grid[-1])

ax[2].plot(mh_iters, samples[1000:,2])
ax[2].set_xlabel('MH iterations')
ax[2].set_title(r'$C$ ($\mathrm{cm}^5/\mathrm{dyne}$)')
ax[2].set_xticks([1000, 5000, 10000, 15000])
ax[2].set_ylim(C_grid[0], C_grid[-1])

ax[3].plot(mh_iters, GR_evolution[1000:,0])
ax[3].plot(mh_iters, GR_evolution[1000:,1])
ax[3].plot(mh_iters, GR_evolution[1000:,2])
ax[3].set_xlabel('MH iterations')
ax[3].set_title('$GR$')
ax[3].set_xticks([1000, 5000, 10000, 15000])
ax[3].legend(['$R_p$','$R_d$','$C$'])

# plt.savefig('./figs/traces_and_GR.png', dpi=300)
# %%
