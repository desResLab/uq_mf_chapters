import numpy as np
import pickle
import chaospy as cp
import matplotlib.pyplot as plt
import os.path

########################################################################################################################
# Function to estimate model statistics
#
# Inputs:
# fcts          dict        dictionary of model functions
# N             int         number of samples used for the model statistics estimation
# jpdf          jpdf        jpdf of the uncertain input parameters of the model functions
# a             int         a value for the Ishigami function; default set to a=5
# b             int         b value for the Ishigami function; default set to b=0.1
#
# Outputs:
# statistics    dict        dictionary with the statistics of the models
#                           mean, sigma, delta, tau, rho
#                           details in Qian et al. (2018) Multifidelity MC Estimation
########################################################################################################################
def model_statistics(fcts, N, uncertainties, QoIs, case_description):

    # case description for model solutions
    temp = ''
    for idx in range(len(fcts)):
        temp = temp + '_' + str(fcts[idx].__class__.__name__[-2:])

    case_definition = 'MFMC_statistics_estimation_' + case_description

    # generate samples to propagate to the models
    d = uncertainties.__len__()                             # number of uncertain parameters
    uqs =list(uncertainties.keys())                         # list of uncertain quantities
    uq_1 = cp.Uniform(uncertainties[uqs[0]]['lower'], uncertainties[uqs[0]]['upper'])
    uq_2 = cp.Uniform(uncertainties[uqs[1]]['lower'], uncertainties[uqs[1]]['upper'])
    uq_3 = cp.Uniform(uncertainties[uqs[2]]['lower'], uncertainties[uqs[2]]['upper'])

    jpdf = cp.J(uq_1, uq_2, uq_3)

    samples = jpdf.sample(N, 'S').transpose()

    # evaluate all models for all samples
    model_solutions = {}
    for QoI in QoIs:
        model_solutions.update({QoI: np.zeros((N, len(fcts)))})

    for level in range(len(fcts)):
        # initialize solution dictionary of the high fidelity model
        model = fcts[level]
        Y = model.solutions_estimating_statistics(samples, QoIs, case_definition)
        for QoI in QoIs:
            model_solutions[QoI][:,level] = Y[QoI]

    # estimate model statistics
    mf_statistics = {}
    for QoI in QoIs:
        mf_statistics[QoI] = compute_statistics(fcts, N, model_solutions[QoI])

    # perturbe lowest fidelity model solution if necessary
    alpha_perturb = 1
    if hasattr(fcts[-1], 'perturb_model') == True:
        low_fidelity_model = fcts[-1]
        if low_fidelity_model.perturb_model == True:
            lstsq = {}
            for QoI in QoIs:
                lstsq.update({QoI: {}})
                # check if 0D correlation is greater than 1D model correlation
                # if so reduce the correlation by 0.001
                #if mf_statistics[QoI]['rho'][1] < mf_statistics[QoI]['rho'][2]:
                # compute difference between 3D and 0D model
                diff_models = model_solutions[QoI][:, 0] - model_solutions[QoI][:, -1]

                # normalize samples
                for dim in range(samples.shape[1]):
                    samples[:, dim] = 2 * (samples[:, dim] - np.min(samples[:, dim])) / (
                                np.max(samples[:, dim]) - np.min(samples[:, dim])) - 1

                # compute linear discrepancy between the solutions
                lstsq[QoI] = np.linalg.lstsq(samples, diff_models)

                # compute 0D_solution - alpha * discrepancy
                perturbed_0D = model_solutions[QoI][:, -1] - alpha_perturb *(np.dot(lstsq[QoI][0],samples.T))
                if model_solutions[QoI].shape[-1] == 3:
                    perturbed_solutions = np.array(
                        [model_solutions[QoI][:, 0], model_solutions[QoI][:, 1], perturbed_0D])
                else:
                    perturbed_solutions = np.array([model_solutions[QoI][:, 0], perturbed_0D])
                # recompute statistics with perturbed 0D-model solutions
                mf_statistics[QoI] = compute_statistics(fcts, N, perturbed_solutions.T)

            # save lstq 0D_Model_lstsq_perturbation.pkl
            f = open('Estimating_statistics/0D_Model_lstsq_perturbation' + temp + ".pkl", "wb")
            pickle.dump(lstsq, f)
            f.close()

    # save estimated statistics for further computations
    f = open('Estimating_statistics/' + case_definition + temp + ".pkl", "wb")
    pickle.dump(mf_statistics, f)
    f.close()

    print('Statististics were loaded successfully!')

    return mf_statistics


def compute_statistics(fcts, N, fct_sol):
    # inizialize matrix for the solutions of the model
    k = len(fcts)

    # initialize a dictionary with statistics of the functions
    statistics = {'mean': np.mean(fct_sol, axis=0),                         # mean of model functions
                  'sigma': np.std(fct_sol, axis=0)}                         # standard deviation of model functions


    g_vals = (fct_sol - statistics['mean']) ** 2                            # Lemma 3.2 g^(k)(Z)
    param_1 = N ** 2 / ((N-1) * (N**2 - 3*N + 3))
    param_2 = 3* (2*N - 3) / (N**2 - 3*N +3)

    # compute further statistical moments of the model functions
    statistics.update({'delta': param_1 * np.sum(g_vals ** 2, axis=0) - param_2 * statistics['sigma'] ** 4,     # fourth moment of model functions
                       'tau': np.std(g_vals, axis=0),                       # standard deviation of the g^(k)(Z) function
                       'q': np.hstack((np.array([1]), np.zeros(k - 1))),    # covariance of g^(k)(Z) and g^(l)(Z)
                       'rho': np.hstack((np.array([1]), np.zeros(k - 1)))}) # Pearson product-moment correlation coefficient between f^(k)(Z) and f^(l)(Z)


    for model in np.arange(1, k):
        statistics['rho'][model] = np.sum(
            (fct_sol[:, 0] - statistics['mean'][0]) * (fct_sol[:, model] - statistics['mean'][model]), axis=0) / (
                                           (N) * statistics['sigma'][0] * statistics['sigma'][model]) # (N-1)
        statistics['q'][model] = np.dot((g_vals[:, 0] - statistics['sigma'][0] ** 2), (
                g_vals[:, model] - statistics['sigma'][model] ** 2)) / (
                                         (N) * statistics['tau'][0] * statistics['tau'][model]) # (N-1)

    return statistics



######################################################################################################################
# ploting images of model solutions

    def plot_model_solutions(model_solutions, perturbed_solutions, QoIs):
        plt.rcParams.update({'mathtext.fontset': 'stix',
                             'font.family': 'STIXGeneral',
                             'font.size': 24,
                             'legend.frameon': False,
                             'legend.fontsize': 24,
                             'savefig.transparent': True})

        markers = ['s', 'o', 'p']
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        plot_dict = {'Psys': {'y_label': 'P [mmHg]', 'conversion': 1/133.23, 'limits': [119, 135], 'x_label': r'P$_{sys}^{3D}$', 'y_label': r'P$_{sys}^{1D/0D}$'},
                     'PP': {'y_label': 'PP [mmHg]', 'conversion': 1/133.23, 'limits': [38, 60], 'x_label': r'$PP^{3D}}$', 'y_label': r'PP$^{1D/0D}$'},
                     'Delta_R_max': {'y_label': r'\Delta R_{max} [mm]', 'conversion': 1000, 'limits': [0.12, 0.27], 'x_label': r'$\Delta r^{3D}}$', 'y_label': r'$\Delta r^{1D/0D}}$'}}

        figure_dir = '/home/friedees/Documents/mulifidelity-mc/Multifidelity_3D_1D_0D/Figures/'

        for QoI in QoIs:
            perturbed_0D = model_solutions[QoI][:, -1] + alpha_perturb * (np.dot(lstsq[QoI][0], samples.T))
            fig, ax = plt.subplots()
            fig_name = 'Estimating_Statistics_' + QoI + '.svg'
            ax.axline([0, 0], [1, 1], linestyle='dashed', color='tab:grey')
            ax.scatter(model_solutions[QoI][:, 0] * plot_dict[QoI]['conversion'],
                       model_solutions[QoI][:, 1] * plot_dict[QoI]['conversion'], marker=markers[0], s=70)
            ax.scatter(model_solutions[QoI][:, 0] * plot_dict[QoI]['conversion'],
                       model_solutions[QoI][:, 2] * plot_dict[QoI]['conversion'], marker=markers[1], s=70,
                       facecolors="None", edgecolors=colors[1], lw=2)
            ax.scatter(model_solutions[QoI][:, 0] * plot_dict[QoI]['conversion'],
                       perturbed_0D * plot_dict[QoI]['conversion'], marker=markers[2], s=70,
                       facecolors=colors[2])
            plt.subplots_adjust(left=0.2,
                                bottom=0.2,
                                right=0.95,
                                top=0.95,
                                wspace=0.1,
                                hspace=0.1)
            ax.set_xlim(plot_dict[QoI]['limits'])
            ax.set_ylim(plot_dict[QoI]['limits'])
            plt.xlabel(plot_dict[QoI]['x_label'])
            plt.ylabel(plot_dict[QoI]['y_label'])
            plt.locator_params(axis='both', nbins=4)
            plt.savefig(os.path.join(figure_dir, fig_name), bbox_inches='tight')
            plt.show()



