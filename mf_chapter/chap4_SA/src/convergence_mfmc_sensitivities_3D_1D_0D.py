import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

m3_to_ml = 1e6
unit_mmhg_pa = 133.3
unit_pa_mmhg = 1./unit_mmhg_pa
mm_to_m = 1e-3

# boxplot specifications
# def setBoxColors(bp, color):
#     setp(bp['boxes'], color=color)
#     setp(bp['medians'], color='red')

# set font for figures
plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 20,
                     'legend.frameon': False,
                     'legend.fontsize': 20,
                     'savefig.transparent': True})



if __name__ == '__main__':

    # definition of the budget
    lines = ['solid', 'dotted', 'dashed']
    markers = ['s','o','p']
    colors = ['tab:blue', 'tab:orange','tab:green']
    S_labels = ['$S_h$', '$S_E$', '$S_r$']
    ST_labels = ['$ST_h$', '$ST_E$', '$ST_r$']

    uncertainties = {'IMT': {'mean': 0.785 * 1e-3, 'lower': 0.785 * 1e-3 * 0.9, 'uPsyser': 0.785 * 1e-3 * 1.1},
                     'E': {'mean': 440043, 'lower': 440043 * 0.9, 'upper': 440043 * 1.1},
                     'R': {'mean': 3.289 * 1e-3, 'lower': 3.289 * 0.9 * 1e-3, 'upper': 3.289 * 1.1 * 1e-3}
                     }

    PC_4 = {'Psys': {'mf_sm': np.array([0.06322077, 0.09015489, 0.83738448]), 'mf_st': np.array([0.070942, 0.09658003, 0.84527663])},
            'PP': {'mf_sm': np.array([0.06635052, 0.09167611, 0.83023649]), 'mf_st': np.array([0.07555913, 0.09965463, 0.84056052])},
            'Delta_R_max': {'mf_sm': np.array([0.14199701, 0.21575985, 0.55411664]), 'mf_st': np.array([0.21677841, 0.26200743, 0.62771489])}}

    # sensitivity indices Sobol sampling plus/minus 10%

    solution_folder = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/'

    QoIs = ['Psys', 'PP', 'Delta_R_max']
    budgets = np.array([20, 30, 40, 50])
    levels = 3


    results = ['mf_sm', 'mf_st', 'mf_mu', 'mf_var', 'sf_sm', 'sf_st', 'sf_mu', 'sf_var', 'hl_sm', 'hl_st', 'hl_mu', 'hl_var']
    Psys = {'perturbed': {}}
    PP = {'perturbed': {}}
    Delta_R_max = {'perturbed': {}}
    # Psys = {'unperturbed': {}, 'perturbed': {}}
    # PP = {'unperturbed': {}, 'perturbed': {}}
    # Delta_R_max = {'unperturbed': {},'perturbed': {}}

    for pert in ['perturbed']:
        for result in results:
            if 'hl_' in result:
                if '_var' in result or '_mu' in result:
                    Psys[pert].update({result: np.zeros((budgets.__len__(), levels - 1))})
                    PP[pert].update({result: np.zeros((budgets.__len__(), levels - 1))})
                    Delta_R_max[pert].update({result: np.zeros((budgets.__len__(), levels - 1))})
                else:
                    Psys[pert].update({result: np.zeros((budgets.__len__(), 3, levels - 1))})
                    PP[pert].update({result: np.zeros((budgets.__len__(), 3, levels - 1))})
                    Delta_R_max[pert].update({result: np.zeros((budgets.__len__(), 3, levels - 1))})
            elif 'sf_' in result or 'sf_' in result:
                if '_var' in result or '_mu' in result:
                    Psys[pert].update({result: np.zeros((budgets.__len__(),levels))})
                    PP[pert].update({result: np.zeros((budgets.__len__(),levels))})
                    Delta_R_max[pert].update({result: np.zeros((budgets.__len__(),levels))})
                else:
                    Psys[pert].update({result: np.zeros((budgets.__len__(), 3, levels))})
                    PP[pert].update({result: np.zeros((budgets.__len__(), 3, levels))})
                    Delta_R_max[pert].update({result: np.zeros((budgets.__len__(), 3, levels))})
            else:
                Psys[pert].update({result: np.zeros((budgets.__len__(), 3))})
                PP[pert].update({result: np.zeros((budgets.__len__(), 3))})
                Delta_R_max[pert].update({result: np.zeros((budgets.__len__(), 3))})

        for idx, budget in enumerate(budgets):
            # load data
            file = solution_folder + 'MFMC_sensitivity_dict_budget_'+ str(budget) +'_S_Artery_'+pert+'.pkl'
            f = open(file, "rb")
            temp = pickle.load(f)

            # # determine number of levels
            # levels = temp['PP']['sf_sm'].__len__()

            # save data in correct format
            for result in results:
                if 'hl_' in result:
                    if '_var' in result or '_mu' in result:
                        for level in range(1,levels):
                            Psys[pert][result][idx, level-1] = temp['Psys'][result][level]
                            PP[pert][result][idx, level-1] = temp['PP'][result][level]
                            Delta_R_max[pert][result][idx, level-1] = temp['Delta_R_max'][result][level]
                    else:
                        for level in range(1,levels):
                            Psys[pert][result][idx, :,level-1] = temp['Psys'][result][level]
                            PP[pert][result][idx, :,level-1] = temp['PP'][result][level]
                            Delta_R_max[pert][result][idx, :,level-1] = temp['Delta_R_max'][result][level]
                elif 'sf_' in result:
                    if '_var' in result or '_mu' in result:
                        for level in range(levels):
                            Psys[pert][result][idx,level] = temp['Psys'][result][level]
                            PP[pert][result][idx,level] = temp['PP'][result][level]
                            Delta_R_max[pert][result][idx,level] = temp['Delta_R_max'][result][level]
                    else:
                        for level in range(levels):
                            Psys[pert][result][idx, :,level] = temp['Psys'][result][level]
                            PP[pert][result][idx, :,level] = temp['PP'][result][level]
                            Delta_R_max[pert][result][idx, :,level] = temp['Delta_R_max'][result][level]
                else:
                    Psys[pert][result][idx, :] = temp['Psys'][result]
                    PP[pert][result][idx, :] = temp['PP'][result]
                    Delta_R_max[pert][result][idx, :] = temp['Delta_R_max'][result]

    figure_dir = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Figures'



    plt.figure()
    fig_name = 'MainSensitivity_vs_budget_Psys_3D_1D_0D.svg'
    for idx, sampling in enumerate(Psys.keys()):
        for S in range(Psys[sampling]['mf_sm'].shape[1]):
            plt.plot(budgets, Psys[sampling]['mf_sm'][:, S], linewidth=3, color=colors[S], marker=markers[S], linestyle=lines[idx],
                     label=S_labels[S],
                     markersize=10)
            plt.axhline(y=PC_4['Psys']['mf_sm'][S], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$S_i$ ')
    plt.ylim((-0.1, 1))
    #plt.legend(frameon=False)
    plt.grid(True, linestyle='--')
    plt.yticks((0, 0.25, 0.5, 0.75, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    plt.figure()
    fig_name = 'TotalSensitivity_vs_budget_Psys_3D_1D_0D.svg'
    for idx, sampling in enumerate(Psys.keys()):
        for ST in range(Psys[sampling]['mf_st'].shape[1]):
            plt.plot(budgets, Psys[sampling]['mf_st'][:, ST], linewidth=3, color=colors[ST], marker=markers[ST], linestyle=lines[idx],
                     label=ST_labels[ST],
                     markersize=10)
            plt.axhline(y=PC_4['Psys']['mf_st'][ST], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$ST_i$ ')
    #plt.legend(frameon=False)
    plt.ylim((-0.1, 1))
    plt.grid(True, linestyle='--')
    plt.yticks((0, 0.25, 0.5, 0.75, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    plt.figure()
    fig_name = 'MainSensitivity_vs_budget_PP_3D_1D_0D.svg'
    for idx, sampling in enumerate(PP.keys()):
        for S in range(PP[sampling]['mf_sm'].shape[1]):
            plt.plot(budgets, PP[sampling]['mf_sm'][:, S], linewidth=3, color=colors[S], marker=markers[S], linestyle=lines[idx],
                     label=S_labels[S],
                     markersize=10)
            plt.axhline(y=PC_4['PP']['mf_sm'][S], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$S_i$ ')
    #plt.legend(frameon=False)
    plt.ylim((-0.1, 1))
    plt.grid(True, linestyle='--')
    plt.yticks((0, 0.25, 0.5, 0.75, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    plt.figure()
    fig_name = 'TotalSensitivity_vs_budget_PP_3D_1D_0D.svg'
    for idx, sampling in enumerate(PP.keys()):
        for ST in range(PP[sampling]['mf_st'].shape[1]):
            plt.plot(budgets, PP[sampling]['mf_st'][:, ST], linewidth=3, color=colors[ST], marker=markers[ST], linestyle=lines[idx],
                     label=ST_labels[ST],
                     markersize=10)
            plt.axhline(y=PC_4['PP']['mf_st'][ST], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$ST_i$ ')
    plt.ylim((-0.1, 1))
    #plt.legend(frameon=False)
    plt.grid(True, linestyle='--')
    plt.yticks((0, 0.25, 0.5, 0.75, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    plt.figure()
    fig_name = 'MainSensitivity_vs_budget_DeltaR_3D_1D_0D.svg'
    for idx, sampling in enumerate(Delta_R_max.keys()):
        for S in range(Delta_R_max[sampling]['mf_sm'].shape[1]):
            plt.plot(budgets, Delta_R_max[sampling]['mf_sm'][:, S], linewidth=3, color=colors[S], marker=markers[S], linestyle=lines[idx],
                     label=S_labels[S],
                     markersize=10)
            plt.axhline(y=PC_4['Delta_R_max']['mf_sm'][S], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$S_i$ ')
    plt.ylim((-0.1, 1))
    #plt.legend(frameon=False)
    plt.grid(True, linestyle='--')
    plt.yticks((0, 0.25, 0.5, 0.75, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    plt.figure()
    fig_name = 'TotalSensitivity_vs_budget_DeltaR_3D_1D_0D.svg'
    for idx, sampling in enumerate(Delta_R_max.keys()):
        for ST in range(Delta_R_max[sampling]['mf_st'].shape[1]):
            plt.plot(budgets, Delta_R_max[sampling]['mf_st'][:, ST], linewidth=3, color=colors[ST], marker=markers[ST], linestyle=lines[idx],
                     label=ST_labels[ST],
                     markersize=10)
            plt.axhline(y=PC_4['Delta_R_max']['mf_st'][ST], color='tab:gray', linewidth=3, linestyle='dashed')
    #plt.xticks([2000, 6000, 10000])
    plt.xlabel('computational budget')
    plt.ylabel('$ST_i$ ')
    plt.ylim((-0.1, 1))
    #plt.legend(frameon=False)
    plt.grid(True, linestyle='--')
    plt.yticks((0,0.25,0.5,0.75,1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
    plt.show()

    # bar plots of the intermediate data

    s_array = ['mf_sm', 'sf_sm', 'hl_sm']
    st_array = ['mf_st', 'sf_st', 'hl_st']

    patterns = [" ", "..", "//", "||", "\\", "--", "++", "xx", "oo", "OO", "**"]

    plot_dict = {'s_array': {'y_label': [r'$S_h$', r'$S_E$', r'$S_r$'], 'fig_name': 'Main_Intermediate_'},
                 'st_array': {'y_label': [r'$ST_h$', r'$ST_E$', r'$ST_r$'], 'fig_name': 'Total_Intermediate_'}}

    for pert in Psys.keys():
        for S in [s_array, st_array]:
            for idx, uncertain in enumerate(['h', 'E', 'r']):
                x_loc = np.linspace(0, budgets.__len__()-1, budgets.__len__())
                w = 0.1
                fig, ax = plt.subplots()
                if '_sm' in S[0]:
                    fig_name = plot_dict['s_array']['fig_name'] + 'Psys_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                else:
                    fig_name = plot_dict['st_array']['fig_name'] + 'Psys_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                plt.bar(x_loc, Psys[pert][S[0]][:, idx], width=w, alpha=0.5, label='mf', hatch=patterns[0])
                x_loc = x_loc + 0.05
                for level in range(levels):
                    x_loc = x_loc + w
                    plt.bar(x_loc, Psys[pert][S[1]][:, idx, level], width=w, alpha=0.5, label='sf',
                            hatch=patterns[1 + level])
                x_loc = x_loc + 0.05
                for level in range(1, levels):
                    x_loc = x_loc + w
                    plt.bar(x_loc, Psys[pert][S[2]][:, idx, level - 1], width=w, alpha=0.5, label='hl',
                            hatch=patterns[3 + level])
                ax.hlines(PC_4['Psys']['mf_sm'][idx], -0.5, 8, color='tab:grey', linestyle='--')
                plt.subplots_adjust(left=0.175, bottom=0.3, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
                ax.set_xticks(np.linspace(0.4, budgets.__len__()-0.6, budgets.__len__()), budgets)
                ax.hlines(0, -0.5, 8, color='tab:grey')
                plt.xlabel('computational budget')
                if '_sm' in S[0]:
                    plt.ylabel(plot_dict['s_array']['y_label'][idx])
                else:
                    plt.ylabel(plot_dict['st_array']['y_label'][idx])
                plt.xticks(rotation=45, ha='center')
                plt.ylim([-0.3, 1])
                plt.xlim([-0.5, 4.2])
                ax.set_yticks([0, 0.5, 1])
                # plt.xlim([4., 6.8])
                # plt.legend()
                plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
                plt.show()

    for pert in PP.keys():
        for S in [s_array, st_array]:
            for idx, uncertain in enumerate(['h', 'E', 'r']):
                x_loc = np.linspace(0, budgets.__len__()-1, budgets.__len__())
                w = 0.1
                fig, ax = plt.subplots()
                if '_sm' in S[0]:
                    fig_name = plot_dict['s_array']['fig_name'] + 'PP_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                else:
                    fig_name = plot_dict['st_array']['fig_name'] + 'PP_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                plt.bar(x_loc, PP[pert][S[0]][:, idx], width=w, alpha=0.5, label='mf', hatch=patterns[0])
                x_loc = x_loc + 0.05
                for level in range(levels):
                    x_loc = x_loc +  w
                    plt.bar(x_loc, PP[pert][S[1]][:, idx, level], width=w, alpha=0.5, label='sf',
                            hatch=patterns[1 + level])
                x_loc = x_loc + 0.05
                for level in range(1, levels):
                    x_loc = x_loc + w
                    plt.bar(x_loc, PP[pert][S[2]][:, idx, level - 1], width=w, alpha=0.5, label='hl',
                            hatch=patterns[3 + level])
                ax.hlines(PC_4['Psys']['mf_sm'][idx], -0.5, 8, color='tab:grey', linestyle='--')
                plt.subplots_adjust(left=0.175, bottom=0.3, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
                ax.set_xticks(np.linspace(0.4, budgets.__len__()-0.6, budgets.__len__()), budgets)
                ax.hlines(0, -0.5, 8, color='tab:grey')
                plt.xlabel('computational budget')
                if '_sm' in S[0]:
                    plt.ylabel(plot_dict['s_array']['y_label'][idx])
                else:
                    plt.ylabel(plot_dict['st_array']['y_label'][idx])
                plt.xticks(rotation=45, ha='center')
                plt.ylim([-0.3, 1])
                plt.xlim([-0.5, 4.2])
                ax.set_yticks([0, 0.5, 1])
                # plt.xlim([4., 6.8])
                # plt.legend()
                plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
                plt.show()

    for pert in Delta_R_max.keys():
        for S in [s_array, st_array]:
            for idx, uncertain in enumerate(['h', 'E', 'r']):
                x_loc = np.linspace(0, budgets.__len__()-1, budgets.__len__())
                w = 0.1
                fig, ax = plt.subplots()
                if '_sm' in S[0]:
                    fig_name = plot_dict['s_array']['fig_name'] + 'Delta_R_max_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                else:
                    fig_name = plot_dict['st_array']['fig_name'] + 'Delta_R_max_3D_1D_0D_' + pert + '_' + uncertain + '.svg'
                plt.bar(x_loc, Delta_R_max[pert][S[0]][:, idx], width=w, alpha=0.5, label='mf', hatch=patterns[0])
                x_loc = x_loc + 0.05
                for level in range(levels):
                    x_loc = x_loc +  w
                    plt.bar(x_loc, Delta_R_max[pert][S[1]][:, idx, level], width=w, alpha=0.5, label='sf',
                            hatch=patterns[1 + level])
                x_loc = x_loc + 0.05
                for level in range(1, levels):
                    x_loc = x_loc +  w
                    plt.bar(x_loc, Delta_R_max[pert][S[2]][:, idx, level - 1], width=w, alpha=0.5, label='hl',
                            hatch=patterns[3 + level])
                ax.hlines(PC_4['Psys']['mf_sm'][idx], -0.5, 8, color='tab:grey', linestyle='--')
                plt.subplots_adjust(left=0.175, bottom=0.3, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
                ax.set_xticks(np.linspace(0.4, budgets.__len__()-0.6, budgets.__len__()), budgets)
                plt.xticks(rotation=45, ha='center')
                ax.hlines(0, -0.5, 4, color='tab:grey')
                plt.xlabel('computational budget')
                if '_sm' in S[0]:
                    plt.ylabel(plot_dict['s_array']['y_label'][idx])
                else:
                    plt.ylabel(plot_dict['st_array']['y_label'][idx])
                plt.ylim([-0.3, 1])
                ax.set_yticks([0, 0.5, 1])
                plt.xlim([-0.5, 4.2])
                # plt.xlim([4., 6.8])
                # plt.legend()
                plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
                plt.show()


    print('Done')

