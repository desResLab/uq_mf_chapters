import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 18,
                     'legend.frameon': False,
                     'legend.fontsize': 13,
                     'savefig.transparent': True})

# define budget to read in
budget = 30

# QoIs
QoIs = ['Psys', 'PP', 'Delta_R_max']

# case definition
case_definition =  'budget_' + str(budget) + '_S_pm10percent'

plot_dict = {'Psys': {'conversion': 1/133.32, 'label': '$P_{sys}$ [mmHg]'},
             'PP': {'conversion': 1/133.32, 'label': 'PP [mmHg]'},
             'Delta_R_max': {'conversion': 1000, 'label': '$\Delta R_{max}$ [mmHg]'}}

# read in data
file_3D_Y_data = open('Simulation_Folder/Y_3D_' + case_definition + '.pkl', 'rb')
Y_3D = pickle.load(file_3D_Y_data)

file_1D_Y_data = open('Simulation_Folder/Y_1D_' + case_definition + '.pkl', 'rb')
Y_1D = pickle.load(file_1D_Y_data)

file_0D_Y_data = open('Simulation_Folder/Y_0D_' + case_definition + '.pkl', 'rb')
Y_0D = pickle.load(file_0D_Y_data)

# max evaluations in high-fidelity model
m = Y_3D['Y_A']['PP'].size

for mat in ['Y_A', 'Y_B', 'Y_BA']:
    for QoI in QoIs:
        if mat != 'Y_BA':
            Y_1D[mat][QoI] = Y_1D[mat][QoI][:m]
            Y_0D[mat][QoI] = Y_0D[mat][QoI][:m]
        else:
            Y_1D[mat][QoI] = Y_1D[mat][QoI][:m,:]
            Y_0D[mat][QoI] = Y_0D[mat][QoI][:m,:]

for QoI in QoIs:
    print(QoI)
    print(np.var(Y_3D['Y_A'][QoI]), np.var(Y_3D['Y_B'][QoI]), np.var(Y_3D['Y_A'][QoI]) - np.var(Y_3D['Y_B'][QoI]))
    print(np.var(Y_1D['Y_A'][QoI]), np.var(Y_1D['Y_B'][QoI]), np.var(Y_1D['Y_A'][QoI]) - np.var(Y_1D['Y_B'][QoI]))
    print(np.var(Y_0D['Y_A'][QoI]), np.var(Y_0D['Y_B'][QoI]), np.var(Y_0D['Y_A'][QoI]) - np.var(Y_0D['Y_B'][QoI]))

for QoI in QoIs:
    plt.figure()
    for mat in ['Y_A', 'Y_B']:
        plt.scatter(Y_3D[mat][QoI]*plot_dict[QoI]['conversion'], Y_1D[mat][QoI]*plot_dict[QoI]['conversion'],marker='s', s=60,label=mat + ' 3D-1D')
        plt.scatter(Y_3D[mat][QoI]*plot_dict[QoI]['conversion'], Y_0D[mat][QoI]*plot_dict[QoI]['conversion'],marker='o', s=60,label=mat + ' 3D-0D')
    for d in range(Y_3D['Y_BA'][QoI].shape[1]):
        mat = 'Y_BA'
        plt.scatter(Y_3D[mat][QoI][:,d] * plot_dict[QoI]['conversion'], Y_1D[mat][QoI][:,d] * plot_dict[QoI]['conversion'],
                    marker='s', s=60, label=mat + ' 3D-1D')
        plt.scatter(Y_3D[mat][QoI][:,d] * plot_dict[QoI]['conversion'], Y_0D[mat][QoI][:,d] * plot_dict[QoI]['conversion'],
                    marker='o', s=60, label=mat + ' 3D-0D')
    plt.axline([0,0],[1,1],color='tab:gray',linestyle='--')
    plt.xlim(( 0.95 * np.min((np.min(Y_3D[mat][QoI]), np.min(Y_1D[mat][QoI]), np.min(Y_0D[mat][QoI]))) * plot_dict[QoI]['conversion'], 1.05 * np.max((np.max(Y_3D[mat][QoI]), np.max(Y_1D[mat][QoI]), np.max(Y_0D[mat][QoI]))) * plot_dict[QoI]['conversion']))
    plt.ylim((0.95 * np.min((np.min(Y_3D[mat][QoI]), np.min(Y_1D[mat][QoI]), np.min(Y_0D[mat][QoI]))) * plot_dict[QoI]['conversion'],
              1.05 * np.max((np.max(Y_3D[mat][QoI]), np.max(Y_1D[mat][QoI]), np.max(Y_0D[mat][QoI]))) * plot_dict[QoI]['conversion']))
    plt.legend()
    plt.xlabel(QoI + ' 3D')
    plt.ylabel(QoI + ' 1D/0D')
    plt.tight_layout()
    plt.show()


print('Done with sampling comparison')