# -*- coding: utf-8 -*-
"""
Plot Figure 6 from Section 5.3 of the UQ chapter.

@author: andre
"""

#%% Modules

import numpy as np
import matplotlib.pyplot as plt

from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import TotalDegreeBasis, LeastSquareRegression, PolynomialChaosExpansion

from SALib.analyze import sobol
from UQpy.sensitivity.PceSensitivity import PceSensitivity

from tqdm import tqdm
import random
import pandas as pd
import scipy.io as sio
import os

from sklearn.metrics import r2_score

#%% Data

save    = False
only_SA = False

QoI = 'pressure:aorta:max' #min, max, avg, pulse
# QoI = 'flow:aorta:max'
if QoI[-3:] == 'avg':
    end = 'mean'
elif QoI[-3:] == 'min':
    end = 'diastolic'
elif QoI[-3:] == 'max':
    end = 'systolic'
elif QoI[-5:] == 'pulse':
    end = 'pulse'
if QoI[:4] == 'flow':
    title = "WSS at " + end + " flow rate"
elif QoI[:8] == 'pressure':
    title = end.capitalize() + " pressure"

# QOI: average WSS, maximum WSS, minimum WSS, average pressure, maximum pressure, minimum pressure 

Rp = 6.8123e2
C  = 3.6664e-5
Rd = 3.1013e4

Rp_bounds = 0.5
C_bounds  = 0.5
Rd_bounds = 0.5

Rp_a = Rp - Rp_bounds*Rp
Rp_b = Rp + Rp_bounds*Rp

C_a = C - C_bounds*C
C_b = C + C_bounds*C

Rd_a = Rd - Rd_bounds*Rd
Rd_b = Rd + Rd_bounds*Rd

mu = 0.04
r_aorta = 1.019478928

#%% Plot settings 

fs = 16
plt.rc('font',  family='serif', size=fs)
plt.rc('text',  usetex=True)
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rcParams['xtick.labelsize'] = fs+2
plt.rcParams['ytick.labelsize'] = fs+2
plt.rcParams['axes.titlesize'] = fs+2
plt.rcParams['figure.dpi'] = 300


#%% Sensitivity analysis using SALib (Section 5.3, Figure 6)

problem = {
    'num_vars': 3,
    'names': ['Rp', 'C', 'Rd'],
    'bounds': [[Rp_a, Rp_b],
               [C_a, C_b],
               [Rd_a, Rd_b]]
}

values = sio.loadmat('./data/sobol_RCR_vals.mat')
samples = np.array([values['Rp'][0], values['C'][0], values['Rd'][0]]).T
n_samples = samples.shape[0]

QOI_table = pd.read_table('./data/QOI_0D_sobol_exp_0.csv', delimiter=',', index_col=0)
if QoI[:4] == 'flow':
    flow_aorta = np.array(list(QOI_table.loc[QoI]))
    wss_aorta = (4*mu*flow_aorta)/(np.pi*(r_aorta**3))
    quantity = wss_aorta
elif QoI[:8] == 'pressure':
    if QoI[15:] == 'pulse':
        pressure_max = np.array(list(QOI_table.loc[QoI[:15]+'max']))
        pressure_min = np.array(list(QOI_table.loc[QoI[:15]+'min']))
        quantity = pressure_max - pressure_min
    else:
        pressure_aorta = np.array(list(QOI_table.loc[QoI]))
        quantity = pressure_aorta


Si = sobol.analyze(problem, quantity)

num_vars = 3
_idx = np.arange(num_vars)
variable_names = ["$R_p$", "$C$", "$R_d$"]
# variable_names = ["$R_p$ \n (dyne$\cdot$s/cm$^5$)", "$C$ \n (cm$^5$/dyne)", "$R_d$ \n (dyne$\cdot$s/cm$^5$)"]

# round to 2 decimal places
indices_1 = np.around(Si['S1'], decimals=2)
indices_T = np.around(Si['ST'], decimals=2)

#%%

fig, ax = plt.subplots(figsize=(2.4,1.6))
width = 0.2
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_indices_1 = ax.bar(
    _idx-width*3/2,  # x-axis
    indices_1,  # y-axis
    yerr=Si['S1_conf'],  # error bar
    width=width,  # bar width
    color="C0",  # bar color
    label="First order",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

bar_indices_T = ax.bar(
    _idx-width/2,  # x-axis
    indices_T,  # y-axis
    yerr=Si['ST_conf'],  # error bar
    width=width,  # bar width
    color="C1",  # bar color
    label="Total effect",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)
    
#% PCE sensitivity analysis using UQpy (Section 5.3, Figure 6)

degree = 9

marg = [Uniform(loc=Rp_a, scale=Rp_b-Rp_a), Uniform(loc=C_a, scale=C_b-C_a), Uniform(loc=Rd_a, scale=Rd_b-Rd_a)]
dist = JointIndependent(marginals=marg)

values = sio.loadmat('./data/uni_RCR_vals.mat')
samples = np.array([values['Rp'][0], values['C'][0], values['Rd'][0]]).T
n_samples = samples.shape[0]

QOI_table = pd.read_table('./data/QOI_0D_uni_exp_0.csv', delimiter=',', index_col=0)
if QoI[:4] == 'flow':
    flow_aorta = np.array(list(QOI_table.loc[QoI]))
    wss_aorta = (4*mu*flow_aorta)/(np.pi*(r_aorta**3))
    quantity = wss_aorta
elif QoI[:8] == 'pressure':
    if QoI[15:] == 'pulse':
        pressure_max = np.array(list(QOI_table.loc[QoI[:15]+'max']))
        pressure_min = np.array(list(QOI_table.loc[QoI[:15]+'min']))
        quantity = pressure_max - pressure_min
    else:
        pressure_aorta = np.array(list(QOI_table.loc[QoI]))
        quantity = pressure_aorta

polynomial_basis = TotalDegreeBasis(dist, degree)
regression = LeastSquareRegression()

leave_out = 10
repetitions = 100
elements = np.arange(n_samples).tolist()

first_order_PCE_vec = np.empty((repetitions, 3))
total_order_PCE_vec = np.empty((repetitions, 3))

for i in tqdm(range(repetitions)):
    
    selected_elements = random.sample(elements, n_samples - leave_out)
    samples_fit = samples[selected_elements]
    quantity_fit = quantity[selected_elements]
    
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=regression)
    pce.fit(samples_fit, quantity_fit)

    PCE_SA = PceSensitivity(pce)
    
    first_order_PCE_vec[i,:] = np.squeeze(PCE_SA.calculate_first_order_indices())
    total_order_PCE_vec[i,:] = np.squeeze(PCE_SA.calculate_total_order_indices())
    
mask_first = (np.prod((first_order_PCE_vec > 0)*(first_order_PCE_vec < 1), axis=1)).astype(bool)
mask_total = (np.prod((total_order_PCE_vec > 0)*(total_order_PCE_vec < 1), axis=1)).astype(bool)

first_order_PCE = np.mean(first_order_PCE_vec[mask_first], axis=0)
total_order_PCE = np.mean(total_order_PCE_vec[mask_total], axis=0)

first_order_PCE_std = np.std(first_order_PCE_vec[mask_first], axis=0)*np.sqrt(20)
total_order_PCE_std = np.std(total_order_PCE_vec[mask_total], axis=0)*np.sqrt(20)

bar_indices_1_PCE = ax.bar(
    _idx+width/2,  # x-axis
    first_order_PCE,  # y-axis
    yerr=first_order_PCE_std, # error bar
    width=width,  # bar width
    color="C2",  # bar color
    label="First order PCE",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

bar_indices_T_PCE = ax.bar(
    _idx+width*3/2,  # x-axis
    total_order_PCE,  # y-axis
    yerr=total_order_PCE_std, # error bar
    width=width,  # bar width
    color="C3",  # bar color
    label="Total effect PCE",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

ax.set_xticks(_idx, variable_names)
ax.set_yticks(np.array([0, 0.5, 1]))
ax.set_xlabel("RCR BCs")
ticks =  ax.get_yticks()
ax.set_yticklabels([abs(tick) for tick in ticks])
plt.title(title)
#ax.legend(ncol=4)
#plt.show()
if save:
    plt.savefig("figures/SA_QOI_" + QoI.replace(':','_') + "_horizontal.png")
# %%
