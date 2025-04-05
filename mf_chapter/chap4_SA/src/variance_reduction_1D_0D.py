import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from pylab import setp
import os, os.path


# boxplot specifications
def setBoxColors(bp, color):
    setp(bp['boxes'], color=color)
    setp(bp['medians'], color='red')

# set font for figures
plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'font.size': 20,
                     'legend.frameon': False,
                     'legend.fontsize': 13,
                     'savefig.transparent': True})

# specifications for plotting
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['s', 'o', 'p']
lineTypes = ['solid','dashed', 'dotted']
hatches = [" ", "..", "//", "||", "\\", "--", "++", "xx", "oo", "OO", "**"]


if __name__ == '__main__':

    # folder with saved sensitivity dicts from the MFMC SA
    path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction/budget_500/'
    QoIs = ['Psys', 'PP', 'Delta_R_max']
    budgets = np.array([500, 1000, 2000, 4000])
    levels = 2
    patterns = [" ", "...", "///", "||", "\\", "--", "++", "xx", "oo", "OO", "**"]

    results = ['mf_sm', 'mf_st', 'mf_mu', 'mf_var', 'sf_sm', 'sf_st', 'sf_mu', 'sf_var', 'hl_sm', 'hl_st', 'hl_mu', 'hl_var']
    Psys = {'MFMC': {}, 'MC': {}}
    PP = {'MC': {}, 'MFMC': {}}
    Delta_R_max = {'MC': {}, 'MFMC': {}}

    #PC expansion of order 4
    PC_4 = {'Psys': {'mf_sm': np.array([0.09109461, 0.09092129, 0.81677972]), 'mf_st': np.array([0.09174154, 0.0915708, 0.81789865])},
            'PP': {'mf_sm': np.array([0.09429512, 0.09419727, 0.80964605]), 'mf_st': np.array([0.09528499, 0.0951839, 0.81139427])},
            'Delta_R_max': {'mf_sm': np.array([0.21764474,0.21774116, 0.56449991]), 'mf_st': np.array([0.21774362, 0.21784011, 0.5645357])}}

    m = {'Psys': np.array([[6, 310], [9, 635], [18, 1271], [37, 2542], [55,3818], [74,5085],[93,6356]]),
         'PP': np.array([[4, 318],[6, 646], [12, 1292], [24,2584], [37,3876], [49,5168],[61,6460]]),
         'Delta_R_max': np.array([[4, 317],[3,655], [6, 1311], [13,2622], [19,3933], [26,5244], [33,6556]])}
    alpha = {'Psys': np.array([1, 0.98079162]),
             'PP': np.array([1, 0.99950055]),
             'Delta_R_max': np.array([1, 0.98390381])}

    ####################################################################################################################
    # Ishigami function
    path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction_Ishigami/budget_200'
    QoIs = ['Psys','PP','Delta_R_max']
    budgets = np.array([200, 2000, 20000])
    levels = 3
    Psys = {'MFMC': {}}
    PP = {'MFMC': {}}
    Delta_R_max = {'MFMC': {}}

    PC_4 = {'Psys': {'mf_sm': np.array([0.401, 0.288, 0]), 'mf_st': np.array([0.712, 0.288, 0.311])},
            'PP': {'mf_sm': np.array([0.401, 0.288, 0]), 'mf_st': np.array([0.712, 0.288, 0.311])},
            'Delta_R_max': {'mf_sm': np.array([0.401, 0.288, 0]), 'mf_st': np.array([0.712, 0.288, 0.311])}}

    m = {'scalar': np.array([[7, 455, 9473], [77, 4554, 94737], [775, 45547, 947372]]),
        'Psys': np.array([[7, 455, 9473], [77, 4554, 94737], [775, 45547, 947372]]),
         'PP': np.array([[7, 455, 9473], [77, 4554, 94737], [775, 45547, 947372]]),
         'Delta_R_max': np.array([[5, 429, 2178], [53, 4299, 21789], [531, 42991, 217890]])}
    alpha = {'scalar': np.array([1, 1.0140279, 0.8812344]), 'Psys': np.array([1, 1.0140279, 0.8812344]),
             'PP': np.array([1, 1.0140279, 0.8812344]),
             'Delta_R_max': np.array([1, 1.0140279, 0.8812344])}

    ####################################################################################################################

    for method in Psys.keys():
        #path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction/budget_500/'

        # if method == 'MFMC':
        #     budgets = np.array([500, 1000, 2000, 4000])
        # else:
        #     budgets = np.array([500, 1000])

        runs = len([name for name in os.listdir(path)])
        for result in results:
            if 'hl_' in result:
                if '_var' in result or '_mu' in result:
                    Psys[method].update({result: np.zeros((budgets.__len__(), levels - 1, runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), levels - 1, runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), levels - 1, runs))})
                else:
                    Psys[method].update({result: np.zeros((budgets.__len__(), 3, levels - 1, runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), 3, levels - 1, runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), 3, levels - 1, runs))})
            elif 'sf_' in result or 'sf_' in result:
                if '_var' in result or '_mu' in result:
                    Psys[method].update({result: np.zeros((budgets.__len__(), levels, runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), levels, runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), levels, runs))})
                else:
                    Psys[method].update({result: np.zeros((budgets.__len__(), 3, levels, runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), 3, levels, runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), 3, levels, runs))})
            else:
                if '_var' in result or '_mu' in result:
                    Psys[method].update({result: np.zeros((budgets.__len__(), runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), runs))})
                else:
                    Psys[method].update({result: np.zeros((budgets.__len__(), 3, runs))})
                    PP[method].update({result: np.zeros((budgets.__len__(), 3, runs))})
                    Delta_R_max[method].update({result: np.zeros((budgets.__len__(), 3, runs))})

        for idx, budget in enumerate(budgets):
            # folder with saved seneitivity dicts from the MFMC SA
            # if method == 'MFMC':
            #     path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction/budget_' + str(budget) + '/'
            # else:
            #     path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction/MC_budget_' + str(
            #         budget) + '/'
            path = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Simulation_Folder/Sensitivity_dicts/Variance_reduction_Ishigami/budget_' + str(
                budget) + '/'

            runs = len([name for name in os.listdir(path)])

            for run in range(runs):
                # load data
                file = path + 'MFMC_sensitivity_dict_budget_'+ str(budget) +'_R_Ishigami_'+ str(run) +'.pkl'
                # if method == 'MFMC':
                #     file = path + 'MFMC_sensitivity_dict_budget_' + str(budget) + '_R_Artery_' + str(run) + '.pkl'
                # else:
                #     file = path + 'MFMC_sensitivity_dict_budget_' + str(budget) + '_R_MC_single_fidelity_' + str(run) + '.pkl'
                f = open(file, "rb")
                temp = pickle.load(f)
                temp['Psys'] = temp['scalar']
                temp['PP'] = temp['scalar']
                temp['Delta_R_max'] = temp['scalar']
                temp.pop('scalar')

                # determine number of levels
                levels = temp['PP']['sf_sm'].__len__()

                # save data in correct format
                for result in results:
                    if 'hl_' in result:
                        if '_var' in result or '_mu' in result:
                            for level in range(1,levels):
                                Psys[method][result][idx, level-1, run] = temp['Psys'][result][level]
                                PP[method][result][idx, level-1, run] = temp['PP'][result][level]
                                Delta_R_max[method][result][idx, level-1, run] = temp['Delta_R_max'][result][level]
                        else:
                            for level in range(1,levels):
                                Psys[method][result][idx, :,level-1, run] = temp['Psys'][result][level]
                                PP[method][result][idx, :,level-1, run] = temp['PP'][result][level]
                                Delta_R_max[method][result][idx, :,level-1, run] = temp['Delta_R_max'][result][level]
                    elif 'sf_' in result:
                        if '_var' in result or '_mu' in result:
                            for level in range(levels):
                                Psys[method][result][idx,level, run] = temp['Psys'][result][level]
                                PP[method][result][idx,level, run] = temp['PP'][result][level]
                                Delta_R_max[method][result][idx,level, run] = temp['Delta_R_max'][result][level]
                        else:
                            for level in range(levels):
                                Psys[method][result][idx, :,level, run] = temp['Psys'][result][level]
                                PP[method][result][idx, :,level, run] = temp['PP'][result][level]
                                Delta_R_max[method][result][idx, :,level, run] = temp['Delta_R_max'][result][level]
                    else:
                        if '_var' in result or '_mu' in result:
                            Psys[method][result][idx, run] = temp['Psys'][result]
                            PP[method][result][idx, run] = temp['PP'][result]
                            Delta_R_max[method][result][idx, run] = temp['Delta_R_max'][result]
                        else:
                            Psys[method][result][idx, :, run] = temp['Psys'][result]
                            PP[method][result][idx, :, run] = temp['PP'][result]
                            Delta_R_max[method][result][idx, :, run] = temp['Delta_R_max'][result]


    for method in Psys.keys():
        # make a box plot with the data
        figure_dir = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Figures/'
        # Psys
        # if method == 'MFMC':
        #     budgets = np.array([500, 1000, 2000, 4000])
        # else:
        #     budgets = np.array([500, 1000])
        for S in ['mf_sm', 'mf_st']:
            pos = 0
            fig, ax = plt.subplots()
            fig_name = 'boxplot_Psys_'+ method + '_' + S + '.svg'
            for idx, budget in enumerate(budgets):
                bp = ax.boxplot(Psys[method][S][idx, :, :].T, positions=np.array([pos, pos + 0.5, pos + 1]), widths=0.5,
                                patch_artist=True, medianprops=dict(linewidth=2, color='tab:red'),whis=(0, 100))
                for patch, color, hatch in zip(bp['boxes'], colors,hatches):
                    patch.set_facecolor(color)
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.5)
                pos = pos + 2
            ax.hlines(PC_4['Psys'][S], -0.5, 7.5, linestyles='dashed', colors='tab:grey')
            plt.subplots_adjust(left=0.175,
                                bottom=0.175,
                                right=0.95,
                                top=0.95,
                                wspace=0.1,
                                hspace=0.1)
            if S == 'mf_sm':
                plt.ylabel(r'$\hat{S}_i$')
            else:
                plt.ylabel(r'$\widehat{ST}_i$')
            plt.ylim([-0.2,1.4])
            plt.xlabel('computational budget')
            plt.xticks(np.arange(0.5, 7.5, 2), [500, 1000, 2000, 4000])
            plt.xticks(np.arange(0.5, 7.5, 2), budgets)
            #plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
            plt.show()

        # PP
        for S in ['mf_sm', 'mf_st']:
            pos = 0
            fig, ax = plt.subplots()
            fig_name = 'boxplot_PP'+ method + '_' + S + '.svg'
            for idx, budget in enumerate(budgets):
                bp = ax.boxplot(PP[method][S][idx, :, :].T, positions=np.array([pos, pos + 0.5, pos + 1]), widths=0.5,
                                patch_artist=True, medianprops=dict(linewidth=2, color='tab:red'),showfliers=True,whis=(0, 100))
                for patch, color,hatch in zip(bp['boxes'], colors, hatches):
                    patch.set_facecolor(color)
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.5)
                pos = pos + 2
            ax.hlines(PC_4['PP'][S], -0.5, 7.5, linestyles='dashed', colors='tab:grey')
            plt.subplots_adjust(left=0.175,
                                bottom=0.175,
                                right=0.95,
                                top=0.95,
                                wspace=0.1,
                                hspace=0.1)
            if S == 'mf_sm':
                plt.ylabel(r'$\hat{S}_i$')
            else:
                plt.ylabel(r'$\widehat{ST}_i$')
            plt.xlabel('computational budget')
            plt.ylim([-0.2,1.4])
            plt.xticks(np.arange(0.5, 7.5, 2), [500, 1000, 2000, 4000])
            #plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
            plt.show()

        # Delta_R_max
        for S in ['mf_sm', 'mf_st']:
            pos = 0
            fig, ax = plt.subplots()
            fig_name = 'boxplot_Delta_R_max'+ method + '_' + S+ '.svg'
            for idx, budget in enumerate(budgets):
                bp = ax.boxplot(Delta_R_max[method][S][idx, :, :].T, positions=np.array([pos, pos + 0.5, pos + 1]), widths=0.5,
                                patch_artist=True, medianprops=dict(linewidth=2, color='tab:red'),showfliers=True,whis=(0, 100))
                for patch, color, hatch in zip(bp['boxes'], colors,hatches):
                    patch.set_facecolor(color)
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.5)
                pos = pos + 2
            ax.hlines(PC_4['Delta_R_max'][S], -0.5, 7.5, linestyles='dashed', colors='tab:grey')
            plt.subplots_adjust(left=0.175,
                                bottom=0.175,
                                right=0.95,
                                top=0.95,
                                wspace=0.1,
                                hspace=0.1)
            if S == 'mf_sm':
                plt.ylabel(r'$\hat{S}_i$')
            else:
                plt.ylabel(r'$\widehat{ST}_i$')
            #plt.ylim([-0.05, 1])
            plt.xlabel('computational budget')
            plt.xticks(np.arange(0.5, 7.5, 2), [500, 1000, 2000, 4000])
            plt.ylim([-0.2,1.4])
            #plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
            plt.show()



    # make a table of the mean and standard deviations of the sensitivity indices
    data_repetitions = {}
    for method in Psys.keys():
        data_repetitions.update({method: {}})
        if method == 'MFMC':
            budgets = np.array([500, 1000, 2000, 4000])
        else:
            budgets = np.array([500, 1000])
        for idx, budget in enumerate(budgets):
            data_repetitions[method].update({budget: {}})
            for QoI in QoIs:
                data_repetitions[method][budget].update({QoI: {'mean': {}, 'std': {}}})

            data_repetitions[method][budget]['Psys']['mean'] = np.hstack((np.mean(Psys[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.mean(Psys[method]['mf_st'][idx,:,:],axis=1)))
            data_repetitions[method][budget]['PP']['mean'] = np.hstack((np.mean(PP[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.mean(PP[method]['mf_st'][idx,:,:],axis=1)))
            data_repetitions[method][budget]['Delta_R_max']['mean'] = np.hstack((np.mean(Delta_R_max[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.mean(Delta_R_max[method]['mf_st'][idx,:,:],axis=1)))

            data_repetitions[method][budget]['Psys']['std'] = np.hstack((np.std(Psys[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.std(Psys[method]['mf_st'][idx,:,:],axis=1)))
            data_repetitions[method][budget]['PP']['std'] = np.hstack((np.std(PP[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.std(PP[method]['mf_st'][idx,:,:],axis=1)))
            data_repetitions[method][budget]['Delta_R_max']['std'] = np.hstack((np.std(Delta_R_max[method]['mf_sm'][idx,:,:],axis=1),
                                                                          np.std(Delta_R_max[method]['mf_st'][idx,:,:],axis=1)))

            data_repetitions[method][budget]['Psys'] = np.ravel(
                (data_repetitions[method][budget]['Psys']['mean'], data_repetitions[method][budget]['Psys']['std']),
                'F')
            data_repetitions[method][budget]['PP'] = np.ravel(
                (data_repetitions[method][budget]['PP']['mean'], data_repetitions[method][budget]['PP']['std']),
                'F')
            data_repetitions[method][budget]['Delta_R_max'] = np.ravel(
                (data_repetitions[method][budget]['Delta_R_max']['mean'], data_repetitions[method][budget]['Delta_R_max']['std']),
                'F')

        headers = ['budget', r'$\hat{S}_h$', r'$\hat{S}_E$', r'$\hat{S}_r$', r'$\widehat{ST}_h$', r'$\widehat{ST}_E$', r'$\hat{ST}_r$']

        # print table
        print(method)
        textabular = f"l|{'c' * len(headers)}"
        texheader = " & " + " & ".join(headers) + "\\\\"
        texdata = "\\hline\n"
        print("\\begin{tabular}{" + textabular + "}")
        print(texheader)
        for QoI in QoIs:
            for budget in sorted(budgets):
                texdata += f"{QoI} & {budget} & {' & '.join(map(str, np.round(data_repetitions[method][budget][QoI],3)))} \\\\\n"
        print(texdata, end="")
        print("\\end{tabular}")



    # load mf_statistics file
    statistics_file = "/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Estimating_statistics/MFMC_statistics_estimation_Artery_1D_0D_unperturbed.pkl"
    #statistics_file = "/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Estimating_statistics/MFMC_statistics_estimation_Ishigami_l0_l1_l2.pkl"
    f = open(statistics_file, "rb")
    mf_statistics = pickle.load(f)
    # mf_statistics['Psys'] = {'mean': np.array([2.5, 2.375, 1.5]), 'sigma': np.array([3.29, 3.25, 3.53]), 'rho': np.array([1, 0.9997, 0.9465]),
    #                          'tau': np.array([19.34, 19.06, 19.31]), 'q': np.array([1,0.9997, 0.9442]), 'delta': np.array([492, 475, 528])}
    # mf_statistics['PP'] = mf_statistics['Psys']
    # mf_statistics['Delta_R_max'] = mf_statistics['Psys']

    # variance of the MFMC variance estimator
    #budgets = np.array([500, 1000, 2000])
    var_mf_theory = {'Psys': np.zeros((budgets.__len__())), 'PP': np.zeros((budgets.__len__())), 'Delta_R_max': np.zeros((budgets.__len__()))}
    var_mf_sim = {'MFMC': {'Psys': np.zeros((budgets.__len__())), 'PP': np.zeros((budgets.__len__())), 'Delta_R_max': np.zeros((budgets.__len__()))},
                  'MC': {'Psys': np.zeros((budgets.__len__())), 'PP': np.zeros((budgets.__len__())), 'Delta_R_max': np.zeros((budgets.__len__()))}}
    # Theorem 3.3
    #budgets = np.array([200, 2000, 20000])
    for idx, budget in enumerate(budgets):
        for QoI in QoIs:
            t_1 = 1/m[QoI][idx, 0] * (mf_statistics[QoI]['delta'][0] - (m[QoI][idx,0] - 3)/(m[QoI][idx, 0] - 1) * mf_statistics[QoI]['sigma'][0] ** 4)
            t_2 = 0
            t_3 = 0
            for model in np.arange(1, len(m[QoI][idx,:])):
                t_2 = t_2 + alpha[QoI][model] ** 2 * (1 / m[QoI][idx,model - 1] * (
                            mf_statistics[QoI]['delta'][model] - (m[QoI][idx,model - 1] - 3) / (m[QoI][idx,model - 1] - 1) * mf_statistics[QoI][
                        'sigma'][model] ** 4) - 1 / m[QoI][idx,model] * (mf_statistics[QoI]['delta'][model] - (m[QoI][idx,model] - 3) / (m[QoI][idx,model] - 1) *
                                                         mf_statistics[QoI]['sigma'][model] ** 4))
                t_3 = t_3 + alpha[QoI][model] * (1 / m[QoI][idx,model] * (
                            mf_statistics[QoI]['q'][model] * mf_statistics[QoI]['tau'][0] * mf_statistics[QoI]['tau'][model] + 2 / (m[QoI][idx,model] - 1) *
                            mf_statistics[QoI]['rho'][model] ** 2 * mf_statistics[QoI]['sigma'][0] ** 2 * mf_statistics[QoI]['sigma'][model] ** 2) -
                                            1 / m[QoI][idx,model - 1] * (
                                                        mf_statistics[QoI]['q'][model] * mf_statistics[QoI]['tau'][0] * mf_statistics[QoI]['tau'][
                                                    model] + 2 / (m[QoI][idx,model - 1] - 1) * mf_statistics[QoI]['rho'][model] ** 2 *
                                                        mf_statistics[QoI]['sigma'][0] ** 2 * mf_statistics[QoI]['sigma'][model] ** 2))

            var_mf_theory[QoI][idx] = t_1 + t_2 + 2 * t_3

    # compute variance from simmulations
    for method in var_mf_sim.keys():
        for idx, budget in enumerate(budgets):
            var_mf_sim[method]['Psys'][idx] = np.mean((Psys[method]['mf_var'][idx, :] - var_mf_theory['Psys'][idx]) ** 2)
            var_mf_sim[method]['PP'][idx] = np.mean((PP[method]['mf_var'][idx, :] - var_mf_theory['PP'][idx]) ** 2)
            var_mf_sim[method]['Delta_R_max'][idx] = np.mean((Delta_R_max[method]['mf_var'][idx, :] - var_mf_theory['Delta_R_max'][idx]) ** 2)

    var_dict = {'Psys': np.var(Psys[method]['mf_var'],axis=1), 'PP': np.var(PP[method]['mf_var'],axis=1), 'Delta_R_max': np.var(Delta_R_max[method]['mf_var'],axis=1)}



    for idx, QoI in enumerate(QoIs):
        fig, ax = plt.subplots()
        fig_name = 'varianceReduction' + QoI + '.svg'
        ax.loglog(budgets, var_mf_theory[QoI], linewidth=2)
        ax.scatter(budgets, var_dict[QoI], s=80)
        plt.subplots_adjust(left=0.2,
                            bottom=0.175,
                            right=0.95,
                            top=0.95,
                            wspace=0.1,
                            hspace=0.1)
        plt.ylabel(r'$\mathbb{V}\,[\hat{V}_{mf}]$')
        # plt.ylim([-0.05, 1])
        plt.xlabel('computational budget')
        ax.set_xticks(budgets, budgets)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.grid(True, which="both", ls="-")
        #plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
        plt.show()

    # compute MSE of the sensitvity indices
    # dimensions: 0=budget, 1=2 for Si and STi, 2=uncertain parameters
    MSE_sensitivity = {}
    for method in Psys.keys():
        if method == 'MFMC':
            budgets = np.array([500, 1000, 2000, 4000])
        else:
            budgets = np.array([500, 1000])
        MSE_sensitivity.update({method: {'Psys': np.zeros((budgets.__len__(),2,3)), 'PP': np.zeros((budgets.__len__(),2,3)),
                                      'Delta_R_max': np.zeros((budgets.__len__(),2,3))}})

    for method in MSE_sensitivity.keys():
        if method == 'MFMC':
            budgets = np.array([500, 1000, 2000, 4000])
        else:
            budgets = np.array([500, 1000])
        for idx, budget in enumerate(budgets):
            MSE_sensitivity[method]['Psys'][idx,0,:] = np.mean((Psys[method]['mf_sm'][idx,:,:].T - np.mean(Psys[method]['mf_sm'][idx,:,:],axis=1)) ** 2,axis=0)
            MSE_sensitivity[method]['Psys'][idx, 1, :] = np.mean(
                (Psys[method]['mf_st'][idx, :, :].T - np.mean(Psys[method]['mf_st'][idx, :, :], axis=1)) ** 2, axis=0)
            MSE_sensitivity[method]['PP'][idx,0,:] = np.mean((PP[method]['mf_sm'][idx,:,:].T - np.mean(PP[method]['mf_sm'][idx,:,:],axis=1)) ** 2,axis=0)
            MSE_sensitivity[method]['PP'][idx, 1, :] = np.mean(
                (PP[method]['mf_st'][idx, :, :].T - np.mean(PP[method]['mf_st'][idx, :, :], axis=1)) ** 2, axis=0)
            MSE_sensitivity[method]['Delta_R_max'][idx,0,:] = np.mean((Delta_R_max[method]['mf_sm'][idx,:,:].T - np.mean(Delta_R_max[method]['mf_sm'][idx,:,:],axis=1)) ** 2,axis=0)
            MSE_sensitivity[method]['Delta_R_max'][idx, 1, :] = np.mean(
                (Delta_R_max[method]['mf_st'][idx, :, :].T - np.mean(Delta_R_max[method]['mf_st'][idx, :, :], axis=1)) ** 2, axis=0)

    for QoI in QoIs:
        # loop over main and total sensitivity indices
        for S_idx, S in enumerate(['$S$', '$ST$']):
            fig, ax = plt.subplots()
            fig_name = 'varianceReduction_sensitivities_' + S + '_' + QoI + '.svg'
            for method_idx, method in enumerate(MSE_sensitivity.keys()):
                if method == 'MFMC':
                    budgets = np.array([500, 1000, 2000, 4000])
                else:
                    budgets = np.array([500, 1000])
                for d in range(MSE_sensitivity[method][QoI].shape[2]):
                    ax.loglog(budgets, MSE_sensitivity[method][QoI][:, S_idx, d], linewidth=2, linestyle=lineTypes[method_idx], color=colors[d], marker=markers[d],markersize=10)
            plt.subplots_adjust(left=0.2,
                                bottom=0.175,
                                right=0.95,
                                top=0.95,
                                wspace=0.1,
                                hspace=0.1)
            if S_idx == 0:
                plt.ylabel(r'$MSE\,[\hat{S}_{j,mf}]$')
            else: plt.ylabel(r'$MSE\,[\widehat{ST}_{j,mf}]$')
            plt.xlabel('computational budget')
            ax.set_xticks([500, 1000, 2000, 4000], [500, 1000, 2000, 4000], minor=True)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            plt.ylim([0.000007,0.03])
            plt.grid(True, which="both", ls="-")
            #plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
            plt.show()

    print('Done')