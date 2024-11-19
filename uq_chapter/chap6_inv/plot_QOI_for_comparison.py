#%% import modules and the x- and y-axes
import numpy as np
import pandas as pd

# which QOI to plot
vessel    = 'aorta' # 'aorta' or 'left_iliac' or 'right_iliac'
prober    = 'avg'   # 'max' or 'min' or 'avg'
what      = 'flow'  # 'flow' or 'pressure'

# number of points in N x N x N grid
N         = 32
n_points  = N

# multiplied to the mean value
Rp_bounds = 0.5
C_bounds  = 0.5
Rd_bounds = 0.5

# mean values
Rp_orig   = 6.8123e2
C_orig    = 3.6664e-5
Rd_orig   = 3.1013e4

# create meshgrid of Rp, C, Rd values
Rp = np.linspace(Rp_orig-Rp_bounds*Rp_orig, Rp_orig+Rp_bounds*Rp_orig, N)
C  = np.linspace(C_orig-C_bounds*C_orig,    C_orig+C_bounds*C_orig, N)
Rd = np.linspace(Rd_orig-Rd_bounds*Rd_orig, Rd_orig+Rd_bounds*Rd_orig, N)
Rp_mesh, C_mesh, Rd_mesh = np.meshgrid(Rp, C, Rd)

#%% Load the QOI of interest

# which qoi to plot
name = what+':'+vessel+':'+prober

filename  = './grid_0D/QoI_0D_grid.csv'
oned_data = pd.read_csv(filename)

# list all the different types of QOI headers
var_names = np.genfromtxt(filename, dtype='U', usecols=0, skip_header=1, delimiter=',')

# load the data
data     = np.genfromtxt(filename, usecols=range(1,N**3+1), skip_header=1, delimiter=',')

# properties of blood/vessel
mu      = 0.04
r_aorta = 1.019478928

# pull out the QOI of interest
idxs     = np.nonzero(var_names == name)[0]
qoi      = data[idxs,:]

qoi  = 4 * mu * qoi / (np.pi * r_aorta**3)
qoi  = qoi.reshape(n_points, n_points, n_points)

#%% Plot Rp and C

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

qoi_3d = qoi.reshape(N,N,N)

fig, ax = plt.subplots(4,8)
ax = ax.flatten()

for slice_idx in range(0, N):

    #slice_idx = 0
    sliced_qoi = np.squeeze(qoi_3d[:,slice_idx, :])

    cbar = ax[slice_idx].contourf(Rp, C, sliced_qoi) #, cmap='viridis')
    ax[slice_idx].set_xlabel('$R_p$')
    ax[slice_idx].set_ylabel('$C$')
    ax[slice_idx].set_title('$R_d$ = '+str(round(Rd[slice_idx],2)))
    fig.colorbar(cbar)

fig.set_figwidth(50)
fig.set_figheight(20)
fig.suptitle(name+' from $R_p=$'+str(Rp_orig)+', $C$='+str(C_orig)+', $R_d$='+str(Rd_orig), fontsize=20)

# %% Plot Rp and Rd

fig, ax = plt.subplots(4,8)
ax = ax.flatten()

for slice_idx in range(0, N):
    
    sliced_qoi = np.squeeze(qoi_3d[:,:,slice_idx])

    cbar = ax[slice_idx].contourf(Rp, Rd, sliced_qoi, cmap='viridis')
    ax[slice_idx].set_xlabel('$R_p$')
    ax[slice_idx].set_ylabel('$R_d$')
    ax[slice_idx].set_title('$C$ = '+str(round(C[slice_idx],8)))
    fig.colorbar(cbar)

fig.set_figwidth(50)
fig.set_figheight(20)
fig.suptitle(name+' from $R_p=$'+str(Rp_orig)+', $C$='+str(C_orig)+', $R_d$='+str(Rd_orig), fontsize=20)

# %% Plot C and Rd

fig, ax = plt.subplots(4,8)
ax      = ax.flatten()

for slice_idx in range(0, N):
    
    sliced_qoi = np.squeeze(qoi_3d[slice_idx,:,:])
    cbar = ax[slice_idx].contourf(Rd, C, sliced_qoi, cmap='viridis')
    ax[slice_idx].set_xlabel('$R_d$')
    ax[slice_idx].set_ylabel('$C$')
    ax[slice_idx].set_title('$R_p$ = '+str(round(Rp[slice_idx],2)))
    fig.colorbar(cbar)

fig.set_figwidth(50)
fig.set_figheight(20)
fig.suptitle(name+' from $R_p=$'+str(Rp_orig)+', $C$='+str(C_orig)+', $R_d$='+str(Rd_orig), fontsize=20)
