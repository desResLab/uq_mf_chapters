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

#%% Load the truth data for comparison

#zerod_filepath  = '/Users/chloe/Desktop/grid_0D/'
zerod_filepath  = './grid_0D/'
load_truth      = sio.loadmat(zerod_filepath+'marginal_CRp.mat')
marginal_CRp    = load_truth['marginal_CRp']
marginal_RpRd   = load_truth['marginal_RpRd']
marginal_CRd    = load_truth['marginal_CRd']
marginal_Rp     = np.squeeze(load_truth['marginal_Rp'])
marginal_C      = np.squeeze(load_truth['marginal_C'])
marginal_Rd     = np.squeeze(load_truth['marginal_Rd'])
prior_Rp        = np.squeeze(load_truth['prior_Rp'])
prior_C         = np.squeeze(load_truth['prior_C'])
prior_Rd        = np.squeeze(load_truth['prior_Rd'])

#%% Plot Figure 9 in the UQ chapter

burnin_size = 1000
dim         = 3
# file_path   = '/Users/chloe/Desktop/'
file_path = './data/'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18
})

for i in range(6,6+1): #range(1,6+1):

    print(i)

    # filename   = sio.loadmat(file_path+'chains_var_div10/chain_'+str(i)+'.mat')
    filename = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
    n_accepted = filename['n_accepted']
    samples    = filename['samples']
    samples    = np.squeeze(samples)

    size       = np.shape(samples)[0]/3

    samples    = np.array(samples[3*burnin_size:3*6000])
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

    # MAP computed using the true posterior 2D
    map_CRp = np.unravel_index(np.argmax(marginal_CRp), marginal_CRp.shape)
    map_RpRd = np.unravel_index(np.argmax(marginal_RpRd), marginal_RpRd.shape)
    map_CRd = np.unravel_index(np.argmax(marginal_CRd), marginal_CRd.shape)

    # dim=0 is Rp, dim=2 is C
    cbar_a = ax[0].hexbin(samples[:,0], samples[:,2], mincnt=1, cmap='rainbow')
    ax[0].contour(Rp_grid, C_grid, marginal_CRp, cmap='afmhot')
    # CHOOSE WHICH MAP FROM THE SECTION BELOW
    map0=ax[0].scatter(Rp_grid[map_CRp[1]], C_grid[map_CRp[0]], s=150, marker=(5,1), color='red', zorder=2)    
    # ax[0].legend([map0],['MAP'], loc='lower left')
    ax[0].set_xlabel('$R_p$')
    ax[0].set_ylabel('$C$')
    ax[0].set_xlim(Rp_grid[0], Rp_grid[-1])
    ax[0].set_ylim(C_grid[0], C_grid[-1])

    # dim=0 is Rp, dim=1 is Rd
    cbar_b = ax[1].hexbin(samples[:,0],samples[:,1], mincnt=1, cmap='rainbow')
    ax[1].contour(Rp_grid, Rd_grid, marginal_RpRd, cmap='afmhot')
    map1=ax[1].scatter(Rp_grid[map_RpRd[1]], Rd_grid[map_RpRd[0]], s=150, marker=(5,1), color='red', zorder=2)
    # ax[1].legend([map1],['MAP'], loc='lower left')
    ax[1].set_xlabel('$R_p$')
    ax[1].set_ylabel('$R_d$')
    ax[1].set_xlim(Rp_grid[0], Rp_grid[-1])
    ax[1].set_ylim(Rd_grid[0], Rd_grid[-1])

    # dim=1 is Rd, dim=2 is C
    cbar_c = ax[2].hexbin(samples[:,1],samples[:,2], mincnt=1, cmap='rainbow')
    ax[2].contour(Rd_grid, C_grid, marginal_CRd, cmap='afmhot')
    map2=ax[2].scatter(Rd_grid[map_CRd[1]], C_grid[map_CRd[0]], s=150, marker=(5,1), color='red', zorder=2)
    ax[2].legend([map2],['MAP'], loc='lower left')
    ax[2].set_xlabel('$R_d$')
    ax[2].set_ylabel('$C$')
    ax[2].set_ylim()
    ax[2].set_xlim(Rd_grid[0], Rd_grid[-1])
    ax[2].set_ylim(C_grid[0], C_grid[-1])

    fig.colorbar(cbar_a)
    fig.colorbar(cbar_b)
    fig.colorbar(cbar_c)

    # One-dimensional histograms
    ax[3].hist(samples[:,0], density=True, bins=40)
    ax[3].plot(Rp_grid, marginal_Rp, color='r')
    ax[3].plot(Rp_grid, prior_Rp, linestyle='--', color='k')
    ax[3].set_xlabel('$R_p$')
    ax[3].set_xlim(Rp_grid[0], Rp_grid[-1])

    ax[4].hist(samples[:,2], density=True, bins=40)
    ax[4].plot(C_grid, marginal_C, color='r')
    ax[4].plot(C_grid, prior_C, linestyle='--', color='k')
    ax[4].set_xlabel('$C$')
    ax[4].set_xlim(C_grid[0], C_grid[-1])

    ax[5].hist(samples[:,1], density=True, bins=40)
    ax[5].plot(Rd_grid, marginal_Rd, color='r')
    ax[5].plot(Rd_grid, prior_Rd, linestyle='--', color='k')
    ax[5].set_xlabel('$R_d$')
    ax[5].legend(['Posterior', 'Prior', 'MH'])
    ax[5].set_xlim(Rd_grid[0], Rd_grid[-1])

    fig.get_layout_engine().set(hspace=0.2)

    if i == 6:
        plt.savefig('./figs/hexbin_and_hist.png', dpi=300)

#%% Compute the MAP using various methods

# using the MH samples
print('Obtaining MAP values...')
print(' -----------------')
print('From MH samples: ')
print('Rp = '+str(statistics.mode(samples[:,0])))
print('C = '+str(statistics.mode(samples[:,2])))
print('Rd = '+str(statistics.mode(samples[:,1])))

# using the true posterior
print(' -----------------')
print('From true posterior: ')
print('Rp = '+str(Rp_grid[np.argmax(marginal_Rp)]))
print('C = '+str(C_grid[np.argmax(marginal_C)]))
print('Rd = '+str(Rd_grid[np.argmax(marginal_Rd)]))

# the original mean values
print(' -----------------')
print('Original mean values: ')
print('Rp = '+str(Rp_orig))
print('C = '+str(C_orig))
print('Rd = '+str(Rd_orig))

#%% Calculate Gelman-Rubin statistics

# number of chains
J = 6

# burn-in size
D = 1000

# remaining samples
L = 5000

chain_mean = np.zeros(J)
sj_squared = np.zeros(J)

for i in range(1,J+1):

    # filename   = sio.loadmat(file_path+'chains_var_div10/chain_'+str(i)+'.mat')
    filename = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
    n_accepted = filename['n_accepted']
    samples    = filename['samples']
    samples    = np.squeeze(samples)
    size       = np.shape(samples)[0]/3
    samples    = np.array(samples[3*D:3*(L+D)])
    samples    = np.reshape(samples, [int(samples.shape[0]/dim), dim])

    # mean of each chain
    chain_mean[i-1] = np.mean(samples.flatten())

    # within-chain variance
    sj_squared[i-1] = 1/(L-1)*np.sum((np.mean(samples.flatten())-chain_mean)**2)    

# mean over all chains
grand_mean = np.mean(chain_mean)

# between-chain variance
B          = L/(J-1)*np.sum((chain_mean-grand_mean)**2)

W = 1/J*np.sum(sj_squared)

# Compute the Gelman-Rubin statistic
R = ((L-1)/L*W + 1/L*B)/W

print('The Gelman-Rubin statistic is '+str(R))

# %% Gelman Rubin convergence

GR_evolution = np.empty(L)

for k in tqdm(range(1,L+1)):
    
    chain_mean = np.zeros(J)
    sj_squared = np.zeros(J)
    
    for i in range(1,J+1):
    
        filename   = sio.loadmat(file_path+'chain_'+str(i)+'.mat')
        n_accepted = filename['n_accepted']
        samples    = filename['samples']
        samples    = np.squeeze(samples)
        size       = np.shape(samples)[0]/3
        samples    = np.array(samples[3*D:3*(L+D)])
        samples    = np.reshape(samples, [int(samples.shape[0]/dim), dim])[:k]
    
        # mean of each chain
        chain_mean[i-1] = np.mean(samples.flatten())
    
        # within-chain variance
        sj_squared[i-1] = 1/(L-1)*np.sum((np.mean(samples.flatten())-chain_mean)**2)    
    
    # mean over all chains
    grand_mean = np.mean(chain_mean)
    
    # between-chain variance
    B          = L/(J-1)*np.sum((chain_mean-grand_mean)**2)
    
    W = 1/J*np.sum(sj_squared)
    
    # Compute the Gelman-Rubin statistic
    R = ((L-1)/L*W + 1/L*B)/W
    
    GR_evolution[k-1] = R

#%% New plots

plt.figure(figsize=(5,4.5))
plt.plot(samples[:,0])
plt.xlabel('Iteration MH')
plt.ylabel('$R_p$')
plt.ylim(Rp_grid[0], Rp_grid[-1])
plt.savefig('./figs/trace_Rp.png', dpi=300)

plt.figure(figsize=(5,4.5))
plt.plot(samples[:,1])
plt.xlabel('Iteration MH')
plt.ylabel('$R_d$')
plt.ylim(Rd_grid[0], Rd_grid[-1])
plt.savefig('./figs/trace_Rd.png', dpi=300)

plt.figure(figsize=(5,4.5))
plt.plot(samples[:,2])
plt.xlabel('Iteration MH')
plt.ylabel('$C$')
plt.ylim(C_grid[0], C_grid[-1])
plt.savefig('./figs/trace_C.png', dpi=300)

plt.figure(figsize=(10,5))
plt.plot(GR_evolution)
plt.xlabel('iteration MH')
plt.ylabel('$GR$')
plt.savefig('./figs/convergence_GR.png', dpi=300)
# %%
