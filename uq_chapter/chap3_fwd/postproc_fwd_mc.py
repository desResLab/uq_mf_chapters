"""
Postprocess simulation results and plot Figure 2 in the UQ chapter

@author: chloe
"""
#%%
import numpy as np
import scipy.io as sio

s_tot           = 100                   # total number of experiments
d               = '0'                   # dimension of solver
steps_per_cycle = 10000                 # number of steps per cycle in the solver

name            = 'pressure:aorta:max'      # quantity of interest
mu              = 0.04                  # blood viscosity
r_poiseuille    = 1.019478928           # radius of aorta

Q_mc = np.zeros(s_tot)
var  = np.zeros(4)

vidx = 0

for N in [2**3, 2**5, 2**7, 2**9]:

    for s in np.arange(0, s_tot):

        # load file
        filename  = "./data/QOI_0D/" + "QOI_" + d + "D_exp_" + str(s) + ".csv"
        var_names = np.genfromtxt(filename, dtype = 'U', usecols=0, skip_header=1, delimiter=',')
        
        # Load the data
        data    = np.genfromtxt(filename, usecols=range(1,N+1), skip_header=1, delimiter=',')

        data    = data/1333.22

        # Compute Q_mc of the QOI
        idxs    = np.nonzero(var_names == name)[0]
        Q_mc[s] = np.sum(data[idxs,:])/N

    # Compute the associated variance
    var[vidx] = np.var(Q_mc)
    vidx += 1

#     # Save the results to a new file
    sio.savemat("./data/QOI_" + d + "D_N" + str(N) + ".mat", {'Q_mc': Q_mc})

sio.savemat("./data/vars_" + d + "D.mat", {'var': var})

#%% Plot the results
import numpy as np
import scipy.io as sio
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

d         = '0'                   # dimension of solver
figs_path = "./data/"

fs = 22
plt.rc('font',  family='serif', size=fs)
plt.rc('text',  usetex=True) # set to True later if tex is on Sherlock?
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['axes.titlesize']  = fs+2
plt.rcParams['axes.labelsize']  = fs+3
plt.rcParams['figure.dpi']      = 300

# default python colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

fig, ax    = plt.subplots(1,3)

N_tot      = [2**3, 2**5, 2**7, 2**9]

# plot histograms for each experiment
for N in N_tot:
    s = sio.loadmat(figs_path + "QOI_" + d + "D_N" + str(N) + ".mat")
    Q_mc = s['Q_mc']
    (_, bins,_) = ax[0].hist(np.transpose(Q_mc), density=True, alpha=0.3)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

xmin, xmax = ax[0].get_xlim()
x = np.linspace(xmin-0.001, xmax+0.001, 1000)
for i in range(0,4):
    s = sio.loadmat(figs_path + "QOI_" + d + "D_N" + str(N_tot[i]) + ".mat")
    Q_mc = s['Q_mc']
    ax[0].plot(x, stats.norm.pdf(x, np.mean(Q_mc), np.std(Q_mc)),linewidth=2.0, color=colors[i % len(colors)])
ax[0].set_xlabel(r'$I_N = P_{\mathrm{sys}}$ (mmHg)', labelpad=8)
ax[0].set_title('PDF', pad=20)
ax[0].legend(['$N=8$', '$N=32$', '$N=128$', '$N=512$'], fontsize=fs-2)

s = sio.loadmat(figs_path + "vars_" + d + "D.mat")
var = s['var']
ax[1].loglog(N_tot, np.transpose(var), 'k--', label='_nolegend_')
for var_idx in np.arange(0,4):
    # plot the variance reduction
    ax[1].loglog(N_tot[var_idx], np.transpose(var)[var_idx], 'o', ms=10)
ax[1].loglog(N_tot, 1/np.array(N_tot)*1e3, color='purple')

ax[1].legend(['$N=8$', '$N=32$', '$N=128$', '$N=512$', r'$\sim 1/N$'], fontsize=fs-2, loc='upper right')
ax[1].minorticks_off()
ax[1].set_xticks([8, 32, 128, 512])
ax[1].set_xticklabels([r'$8.0 \times 10^0$', r'$3.2 \times 10^1$', r'$1.28 \times 10^2$', r'$5.12 \times 10^2$'])
ax[1].set_xlabel(r'$\log(N)$ samples', labelpad=12)
ax[1].set_title(r'$\log(\sigma^2)$', pad=20)

N_range    = [2**3, 2**5, 2**7, 2**9]
time_steps = 10000
time       = np.linspace(0, 1.087, time_steps)
default_colors = ['C0', 'C1', 'C2', 'C3']

for i, N in enumerate(N_range):
    loaded = np.load('./data/avg_exp_pressure_in_N_'+str(N)+'.npy', allow_pickle=True)
    mean_pressure = loaded.item()['mean_pressure']
    std_pressure  = loaded.item()['std_pressure']
    time          = loaded.item()['time']
    mean_pressure = mean_pressure
    std_pressure  = std_pressure
    # find point at which there is a max pressure
    max_pressure_index = np.argmax(mean_pressure)
    max_pressure = mean_pressure[max_pressure_index]
    ax[2].plot(time, mean_pressure, label='$N = $ '+str(N))
    ax[2].fill_between(time, mean_pressure - 2*std_pressure, mean_pressure + 2*std_pressure, alpha=0.2)

for i, N in enumerate(N_range):
    loaded = np.load('./data/avg_exp_pressure_in_N_'+str(N)+'.npy', allow_pickle=True)
    mean_pressure = loaded.item()['mean_pressure']
    std_pressure  = loaded.item()['std_pressure']
    time          = loaded.item()['time']
    mean_pressure = mean_pressure
    std_pressure  = std_pressure
    max_pressure_index = np.argmax(mean_pressure)
    max_pressure = mean_pressure[max_pressure_index]
    rectangle = ax[2].errorbar(time[max_pressure_index+600*i], max_pressure, yerr=2*std_pressure[max_pressure_index], capsize=4, elinewidth=2, markersize=5, fmt='o', color=default_colors[N_range.index(N)], label=None)

# ax[2].text(0.05, 115, r'$P_{\mathrm{sys}}$:', fontsize=fs)
ax[2].annotate(r'$P_{\mathrm{sys}}$', xy=(time[max_pressure_index-500], max_pressure), xytext=(0.05, max_pressure),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[2].legend(loc='lower center', fontsize=fs-2) #'center left', bbox_to_anchor=(1, 0.5), fontsize=fs-2)
ax[2].set_xlabel('Time (s)', labelpad=12)
ax[2].set_title('Pressure (mmHg) \n with 95\% confidence intervals')

#fig.tight_layout()
fig.set_figwidth(20)
fig.set_figheight(6)

# plt.savefig(figs_path+ "mc_" + d + "D_aobif.png", dpi=300)
# plt.close()
# %%
