# -*- coding: utf-8 -*-
"""
Save QoI to tables for each experiment

@author: chloe
"""
from compute_QOI_0D import get_QOI_0D
# from compute_QOI_1D import get_QOI_1D
# from compute_QOI_3D import get_QOI_3D
# from tqdm import tqdm
import pandas as pd
import numpy as np

method = 'sobol'
N_start = 0
N_end   = 2048 # total number of simulations
s_begin = 0
s_tot   = 1 # total number of experiments
d_range = ['0'] # for zerod simulations

for d in d_range:

    for s in np.arange(s_begin, s_tot):

        QOI = {}

        QOI['flow:aorta:min'] = []
        QOI['flow:aorta:max'] = []
        QOI['flow:aorta:avg'] = []

        QOI['pressure:aorta:min'] = []
        QOI['pressure:aorta:max'] = []
        QOI['pressure:aorta:avg'] = []

        QOI['flow:left:min'] = []
        QOI['flow:left:max'] = []
        QOI['flow:left:avg'] = []

        QOI['pressure:left:min'] = []
        QOI['pressure:left:max'] = []
        QOI['pressure:left:avg'] = []

        QOI['flow:right:min'] = []
        QOI['flow:right:max'] = []
        QOI['flow:right:avg'] = []

        QOI['pressure:right:min'] = []
        QOI['pressure:right:max'] = []
        QOI['pressure:right:avg'] = []

        print("Importing " + d + "D simulations results ...")

        for i in np.arange(N_start, N_end):

            if d == '0':
                filename = "/scratch/users/azanoni/aobif_sims/zerod_sims/"+method+"_exp_"+str(s)+"/sim_"+str(i)+"/aobif_"+str(i)+".csv"
                QOI_data = get_QOI_0D(filename)

            # elif d == '1':
            #     filename = "/scratch/users/chloe1/aobif_sims/uni_1D/uni_exp_"+str(s)+"/sim_"+str(i)+"/"
            #     QOI_data = get_QOI_1D(filename)

            # elif d == '3':
            #     filename = "../results/3D/" + str(n)
            #     QOI_data = get_QOI_3D(filename)

            for key, value in QOI_data.items():
                QOI[key].append(value)

        QOI_DF = pd.DataFrame.from_dict(QOI, orient='index', columns=range(N_start, N_end))
        QOI_DF.to_csv("/scratch/users/azanoni/aobif_sims/QOI_" + d + "D_"+method+"_exp_" + str(s) + ".csv")
