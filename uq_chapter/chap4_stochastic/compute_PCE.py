# -*- coding: utf-8 -*-
"""
Plot figures 5, 4, and 3 from Section 4.3 of UQ chapter.
@author: andre
"""

#%% Modules

import numpy as np
import matplotlib.pyplot as plt

from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import TotalDegreeBasis, LeastSquareRegression, PolynomialChaosExpansion

from SALib.analyze import sobol
from UQpy.sensitivity.PceSensitivity import PceSensitivity

import pandas as pd
import scipy.io as sio
import os

from sklearn.metrics import r2_score

#%% Data

save = False
only_SA = False

QoI = 'flow:aorta:avg'
if QoI[-3:] == 'avg':
    end = 'average'
elif QoI[-3:] == 'min':
    end = 'minimum'
elif QoI[-3:] == 'max':
    end = 'maximum'
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

fs = 10
plt.rc('font',  family='serif', size=fs)
plt.rc('text',  usetex=True)
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rcParams['figure.dpi'] = 300

#%% Plot Figure 5 from Section 4.3: Polynomial Chaos expansion using UQpy

if not only_SA:

    for degree in [1,10]:
    
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
            pressure_aorta = np.array(list(QOI_table.loc[QoI]))
            quantity = pressure_aorta
        
        polynomial_basis = TotalDegreeBasis(dist, degree)
        regression = LeastSquareRegression()
        pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=regression)
        pce.fit(samples, quantity)
        
        samples_test = np.array([values['Rp'][1], values['C'][1], values['Rd'][1]]).T
        
        QOI_table_test = pd.read_table('./data/QOI_0D_uni_exp_1.csv', delimiter=',', index_col=0)
        if QoI[:4] == 'flow':
            flow_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
            wss_aorta_test = (4*mu*flow_aorta_test)/(np.pi*(r_aorta**3))
            quantity_test = wss_aorta_test
        elif QoI[:8] == 'pressure':
            pressure_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
            quantity_test = pressure_aorta_test
        quantity_test_pce = pce.predict(samples_test).flatten()
        
        r2 = r2_score(quantity_test, quantity_test_pce)
        plt.figure(figsize=(3.6,2.4))
        plt.scatter(quantity_test, quantity_test_pce, label='samples', s=8)
        plt.plot(quantity_test, quantity_test, color='red', label='identity', linewidth=1)
        plt.xlabel('Exact')
        plt.ylabel('PCE')
        plt.title(title + "\nDegree " + str(degree) + ", $R^2 = " + str(round(r2, 3)) + "$")
        plt.legend()
        #plt.show()
        if save:
            if not os.path.exists("./figures"):
                os.mkdir("./figures")
            plt.savefig("./figures/PCE_scatter_degree" + str(degree) + "_QOI_" + QoI.replace(':','_') + ".png", bbox_inches = 'tight')

#%% Plot Figure 4 from Section 4.3: Polynomial Chaos expansion using UQpy

if not only_SA:

    n_list = np.arange(2, 513)
    error_list_train = np.zeros(len(n_list))
    error_list_test = np.zeros(len(n_list))
    
    for degree in [1,10]:
    
        for i, t in enumerate(n_list):
        
            marg = [Uniform(loc=Rp_a, scale=Rp_b-Rp_a), Uniform(loc=C_a, scale=C_b-C_a), Uniform(loc=Rd_a, scale=Rd_b-Rd_a)]
            dist = JointIndependent(marginals=marg)
            
            values = sio.loadmat('./data/uni_RCR_vals.mat')
            samples = np.array([values['Rp'][0], values['C'][0], values['Rd'][0]]).T[:t]
            n_samples = samples.shape[0]
            
            QOI_table = pd.read_table('./data/QOI_0D_uni_exp_0.csv', delimiter=',', index_col=0)
            if QoI[:4] == 'flow':
                flow_aorta = np.array(list(QOI_table.loc[QoI]))[:t]
                wss_aorta = (4*mu*flow_aorta)/(np.pi*(r_aorta**3))
                quantity = wss_aorta
            elif QoI[:8] == 'pressure':
                pressure_aorta = np.array(list(QOI_table.loc[QoI]))[:t]
                quantity = pressure_aorta
            
            polynomial_basis = TotalDegreeBasis(dist, degree)
            regression = LeastSquareRegression()
            pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=regression)
            pce.fit(samples, quantity)
            
            quantity_pce = pce.predict(samples).flatten()
            error_train = np.sum(np.abs(quantity_pce - quantity)/np.abs(quantity))/n_samples
            
            samples_test = np.array([values['Rp'][1], values['C'][1], values['Rd'][1]]).T#[:t]
            
            QOI_table_test = pd.read_table('./data/QOI_0D_uni_exp_1.csv', delimiter=',', index_col=0)
            if QoI[:4] == 'flow':
                flow_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
                wss_aorta_test = (4*mu*flow_aorta_test)/(np.pi*(r_aorta**3))
                quantity_test = wss_aorta_test
            elif QoI[:8] == 'pressure':
                pressure_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
                quantity_test = pressure_aorta_test
            quantity_test_pce = pce.predict(samples_test).flatten()
            
            error_test = np.sum(np.abs(quantity_test_pce - quantity_test)/np.abs(quantity_test))/n_samples
            
            error_list_train[i] = error_train
            error_list_test[i] = error_test
            
        plt.figure(figsize=(3.6,2.4))
        plt.semilogy(n_list[pce.coefficients.size:], error_list_train[pce.coefficients.size:], '--', label='Train')
        plt.semilogy(n_list[pce.coefficients.size:], error_list_test[pce.coefficients.size:], '-', label='Test')
        plt.title(title  + "\nDegree " + str(degree))
        plt.xlabel("Number of samples")
        plt.ylabel("Approximation error")
        plt.legend()
        #plt.show()
        if save:
            if not os.path.exists("./figures"):
                os.mkdir("./figures")
            plt.savefig("./figures/PCE_varyingSamples_degree" + str(degree) + "_QOI_" + QoI.replace(':','_') + ".png", bbox_inches = 'tight')

#%% Plot Figure 3 from Section 4.3: Polynomial Chaos expansion using UQpy

if not only_SA:

    n_list = np.arange(1, 13)
    error_list_train = np.zeros(len(n_list))
    error_list_test = np.zeros(len(n_list))
    n_coefficients = np.zeros(len(n_list))
    
    for i, t in enumerate(n_list):
        
        degree = t
    
        marg = [Uniform(loc=Rp_a, scale=Rp_b-Rp_a), Uniform(loc=C_a, scale=C_b-C_a), Uniform(loc=Rd_a, scale=Rd_b-Rd_a)]
                
        values = sio.loadmat('./data/uni_RCR_vals.mat')
        samples = np.array([values['Rp'][0], values['C'][0], values['Rd'][0]]).T
        n_samples = samples.shape[0]
        
        QOI_table = pd.read_table('./data/QOI_0D_uni_exp_0.csv', delimiter=',', index_col=0)
        if QoI[:4] == 'flow':
            flow_aorta = np.array(list(QOI_table.loc[QoI]))
            wss_aorta = (4*mu*flow_aorta)/(np.pi*(r_aorta**3))
            quantity = wss_aorta
        elif QoI[:8] == 'pressure':
            pressure_aorta = np.array(list(QOI_table.loc[QoI]))
            quantity = pressure_aorta
        
        polynomial_basis = TotalDegreeBasis(dist, degree)
        regression = LeastSquareRegression()
        pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=regression)
        pce.fit(samples, quantity)
        
        quantity_pce = pce.predict(samples).flatten()
        error_train = np.sum(np.abs(quantity_pce - quantity)/np.abs(quantity))/n_samples
        
        samples_test = np.array([values['Rp'][1], values['C'][1], values['Rd'][1]]).T
        
        QOI_table_test = pd.read_table('./data/QOI_0D_uni_exp_1.csv', delimiter=',', index_col=0)
        if QoI[:4] == 'flow':
            flow_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
            wss_aorta_test = (4*mu*flow_aorta_test)/(np.pi*(r_aorta**3))
            quantity_test = wss_aorta_test
        elif QoI[:8] == 'pressure':
            pressure_aorta_test = np.array(list(QOI_table_test.loc[QoI]))
            quantity_test = pressure_aorta_test
        quantity_test_pce = pce.predict(samples_test).flatten()
        
        error_test = np.sum(np.abs(quantity_test_pce - quantity_test)/np.abs(quantity_test))/n_samples
        
        n_coefficients[i] = pce.coefficients.size
        error_list_train[i] = error_train
        error_list_test[i] = error_test
        
    plt.figure(figsize=(3.6,2.4))
    plt.semilogy(n_list, error_list_train, '--x', label='Train')
    plt.semilogy(n_list, error_list_test, ':', marker='o', mfc='none', label='Test')
    plt.xlabel("Degree")
    plt.ylabel("Approximation error")
    plt.title(title)
    plt.legend()
    #plt.show()
    if save:
        plt.savefig("./figures/PCE_varyingDegree_QOI_" + QoI.replace(':','_') + ".png", bbox_inches = 'tight')

    plt.figure(figsize=(3.6,2.4))
    plt.plot(n_list, n_coefficients, '--x')
    plt.xlabel("Degree")
    plt.ylabel("Number of coefficients")
    plt.title(title)
    #plt.show()
    if save:
        if not os.path.exists("./figures"):
            os.mkdir("./figures")
        plt.savefig("./figures/PCE_Ncoeff_varyingDegree_QOI_" + QoI.replace(':','_') + ".png", bbox_inches = 'tight')
# %%
