import chaospy_wrapper as cpw
import chaospy as cp
import numpy as np
import pickle
from split_cycles import split_signal_diastole_auto
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
rc('font', size=24)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{sans}')


if __name__ == "__main__":

    QoI_3D_file = 'Simulation_Folder/QoIs_3D_FSI_StatisticsEstimation_150.pkl'

    ####################################################################################################################
    # load 3D-FSI data
    # QoIs dict for 3D-FSI data
    QoI_dict = {'Psys': np.zeros(150),
                'PP': np.zeros(150),
                'Delta_R_max': np.zeros(150)}

    QoI_file = open(QoI_3D_file, 'rb')
    data_dict = pickle.load(QoI_file)

    # extract the last caridac cycle and compute the QoIs
    for sample_idx, sample in enumerate(data_dict.keys()):
        data_dict[sample]['time'] = np.linspace(0, (len(data_dict[sample]['inner_displacement']) - 1) * 0.01,
                                                len(data_dict[sample]['inner_displacement']))
        cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(
            np.array(data_dict[sample]['time']),
            np.array(data_dict[sample]['volume_flow']).squeeze())

        for QoI in data_dict[sample].keys():
            if QoI == 'inner_displacement' or QoI == 'outer_displacement' or QoI == 'time':
                data_dict[sample][QoI] = data_dict[sample][QoI][cycle_indices[-1][0]:cycle_indices[-1][1]]
            elif QoI != 'avg_dispplacement':
                data_dict[sample][QoI] = data_dict[sample][QoI][0][cycle_indices[-1][0]:cycle_indices[-1][1]]

        # compute QoIs for statistics evaluation
        QoI_dict['Psys'][sample_idx] = np.max(np.array(data_dict[sample]['pressure']))
        QoI_dict['PP'][sample_idx] = np.max(data_dict[sample]['pressure']) - np.min(data_dict[sample]['pressure'])
        # QoI_dict['Delta_R_max'][sample_idx] = np.mean([np.max(data_dict['outer_displacement']), np.max(data_dict['inner_displacement'])])
        QoI_dict['Delta_R_max'][sample_idx] = np.max(data_dict[sample]['inner_displacement'])

    ####################################################################################################################

    # definition of QoIs
    QoIs = ['PP', 'Psys', 'Delta_R_max']

    # define output dictionary
    results_dict = {}

    # definition of uncertain parameters
    uncertainties = {'IMT': {'mean': 0.785 * 1e-3, 'lower': 0.785 * 1e-3 * 0.9, 'upper': 0.785 * 1e-3 * 1.1},
                     'E': {'mean': 440043, 'lower': 440043 * 0.9, 'upper': 440043 * 1.1},
                     'R': {'mean': 3.289 * 1e-3, 'lower': 3.289 * 0.9 * 1e-3, 'upper': 3.289 * 1.1 * 1e-3}
                     }

    IMT = cp.Uniform(uncertainties['IMT']['lower'], uncertainties['IMT']['upper'])
    E = cp.Uniform(uncertainties['E']['lower'], uncertainties['E']['upper'])
    R = cp.Uniform(uncertainties['R']['lower'], uncertainties['R']['upper'])

    # number of uncertain parameters
    N = len(uncertainties)

    jpdf = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform())
    #jpdf = cp.J(IMT,E,R)

    for order in range(5):

        # create orthogonal polynomials
        polynomial_order = order + 1
        # update results dict with order of PC
        results_dict.update({polynomial_order: {}})
        basis = cpw.generate_basis(polynomial_order, jpdf)

        # create samples
        sample_scheme = 'S'
        Ns = 2 * len(basis['poly'])
        samples = jpdf.sample(Ns, sample_scheme)


        # alpha for confidence interval
        plotMeanConfidenceAlpha = 5

        for QoI in QoIs:
            # compute polynomial expansion
            polynomial_expansion = cpw.fit_regression(basis, samples, QoI_dict[QoI][:samples.shape[1]])

            # compute statistics
            results_dict[polynomial_order].update({QoI: {}})
            results_dict[polynomial_order][QoI].update({'expectedValue': cpw.E(polynomial_expansion, jpdf),
                                                            'confidenceInterval': cpw.Perc(polynomial_expansion,
                                                                                           [plotMeanConfidenceAlpha / 2.,
                                                                                            100 - plotMeanConfidenceAlpha / 2.],
                                                                                           jpdf),
                                                            'variance': cpw.Var(polynomial_expansion, jpdf),
                                                            'standardDeviation': cpw.Std(polynomial_expansion, jpdf),
                                                            'firstOrderSensitivities': cpw.Sens_m(polynomial_expansion,
                                                                                                  jpdf),
                                                            'totalSensitivities': cpw.Sens_t(polynomial_expansion,
                                                                                             jpdf)})


            colorsPie = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
            labels = ['h', 'E', 'R']
            ind = np.arange(N)  # the x locations for the groups
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(1)
            rects1 = ax.bar(ind, results_dict[polynomial_order][QoI]['firstOrderSensitivities'], width, color=colorsPie, alpha = 0.5)
            rects2 = ax.bar(ind + width, results_dict[polynomial_order][QoI]['totalSensitivities'], width, color=colorsPie, hatch = '.')

            # add some text for labels, title and axes ticks
            ax.set_ylabel('Si')
            ax.set_xticks(ind + width)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlim([-width, N + width])
            ax.set_ylim([0, 1])
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', top='off')
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', right='off')
            plt.title(QoI)
            plt.tight_layout()
            plt.show()

            print(QoI)
            print(results_dict[polynomial_order][QoI]['firstOrderSensitivities'])
            print(results_dict[polynomial_order][QoI]['totalSensitivities'])


    print('Done')