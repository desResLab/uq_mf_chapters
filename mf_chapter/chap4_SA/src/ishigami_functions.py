########################################################################################################################
# Definition of the Ishigami functions of all levels as well as the analytical mean and conditional variances
########################################################################################################################
import numpy as np
import os.path
import pickle
import time


class Ishigami_func_level0():

    def __init__(self, Z, a=5, b=0.1):

        self.Z = Z
        self.a = a
        self.b = b

    def run_simulation(self):
        Y = np.sin(self.Z[0]) + self.a * np.sin(self.Z[1]) ** 2 + self.b * self.Z[2] ** 4 * np.sin(self.Z[0])
        return Y
    # definition of the Ishigami function - high fidelity model
    ########################################################################################################################
    # def ishigami_func_level0(self):
    #     Y = np.sin(self.Z[0]) + self.a * np.sin(self.Z[1]) ** 2 + self.b * self.Z[2] ** 4 * np.sin(self.Z[0])
    #     return Y
    ########################################################################################################################



    # definition of the Ishigami function with reduced fidelity - Level 1
    ########################################################################################################################
    # def ishigami_func_level1(self):
    #     Y = np.sin(self.Z[0]) + 0.95 * self.a * np.sin(self.Z[1]) ** 2 + self.b * self.Z[2] ** 4 * np.sin(self.Z[0])
    #     return Y
    ########################################################################################################################



    # definition of the Ishigami function with reduced fidelity - Level 2
    ########################################################################################################################
    # def ishigami_func_level2(self):
    #     Y = np.sin(self.Z[0]) + 0.6 * self.a * np.sin(self.Z[1]) ** 2 + 9 * self.b * self.Z[2] ** 2 * np.sin(self.Z[0])
    #     return Y
    ########################################################################################################################


    def run_UQSA_case(self):

        Y = self.run_simulation()

        return Y

    def mfmc(self, A, B, C_BA, case_definition, QoIs):


        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 0D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_L0' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_L0' + case_definition + '.pkl'
            Y_ishigami = open(pkl_name, 'rb')
            Y = pickle.load(Y_ishigami)
            print('0D model solutions successfully loaded')

        else:

            # compute 0D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)

        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):

        # compute the solution of the 0D model
        for QoI in QoIs:
            Y['Y_A'][QoI] = self.mfmc_solution_computation_Y_A(A)
            Y['Y_B'][QoI] = self.mfmc_solution_computation_Y_B(B)
            Y['Y_BA'][QoI] = self.mfmc_solution_computation_Y_BA(C_BA)

        return Y

    def mfmc_solution_computation_Y_A(self, A):

        start = time.time()
        print('Computing Y_A Ishigami model')
        self.Z = A.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_B(self, B):

        start = time.time()
        print('Computing Y_B Ishigami model')
        self.Z = B.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA):

        start = time.time()
        print('Computing Y_BA Ishigami model')

        self.Z = C_BA.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        Y = {}
        for QoI in QoIs:
            self.Z = sample_mat.T
            Y[QoI] = self.run_UQSA_case()

        return Y

########################################################################################################################
class Ishigami_func_level1():

    def __init__(self, Z, a=5, b=0.1):
        self.Z = Z
        self.a = a
        self.b = b


    def run_simulation(self):
        Y = np.sin(self.Z[0]) + 0.95 * self.a * np.sin(self.Z[1]) ** 2 + self.b * self.Z[2] ** 4 * np.sin(self.Z[0])
        return Y

    def run_UQSA_case(self):

        Y = self.run_simulation()

        return Y

    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 0D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_L1' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_L1' + case_definition + '.pkl'
            Y_ishigami = open(pkl_name, 'rb')
            Y = pickle.load(Y_ishigami)
            print('0D model solutions successfully loaded')

        else:

            # compute 0D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)

        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):

        # compute the solution of the 0D model
        for QoI in QoIs:
            Y['Y_A'][QoI] = self.mfmc_solution_computation_Y_A(A)
            Y['Y_B'][QoI] = self.mfmc_solution_computation_Y_B(B)
            Y['Y_BA'][QoI] = self.mfmc_solution_computation_Y_BA(C_BA)

        return Y

    def mfmc_solution_computation_Y_A(self, A):

        start = time.time()
        print('Computing Y_A Ishigami model')
        self.Z = A.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_B(self, B):

        start = time.time()
        print('Computing Y_B Ishigami model')
        self.Z = B.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA):

        start = time.time()
        print('Computing Y_BA Ishigami model')

        self.Z = C_BA.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        Y = {}
        for QoI in QoIs:
            self.Z = sample_mat.T
            Y[QoI] = self.run_UQSA_case()

        return Y


class Ishigami_func_level2():

    def __init__(self, Z, a=5, b=0.1):
        self.Z = Z
        self.a = a
        self.b = b

    def run_simulation(self):
        Y = np.sin(self.Z[0]) + 0.6 * self.a * np.sin(self.Z[1]) ** 2 + 9 * self.b * self.Z[2] ** 2 * np.sin(self.Z[0])
        return Y


    def run_UQSA_case(self):

        Y = self.run_simulation()

        return Y

    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 0D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_L2' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_L2' + case_definition + '.pkl'
            Y_ishigami = open(pkl_name, 'rb')
            Y = pickle.load(Y_ishigami)
            print('0D model solutions successfully loaded')

        else:

            # compute model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)

        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):

        # compute the solution of the 0D model
        for QoI in QoIs:
            Y['Y_A'][QoI] = self.mfmc_solution_computation_Y_A(A)
            Y['Y_B'][QoI] = self.mfmc_solution_computation_Y_B(B)
            Y['Y_BA'][QoI] = self.mfmc_solution_computation_Y_BA(C_BA)

        return Y

    def mfmc_solution_computation_Y_A(self, A):

        start = time.time()
        print('Computing Y_A Ishigami model')
        self.Z = A.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_B(self, B):

        start = time.time()
        print('Computing Y_B Ishigami model')
        self.Z = B.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA):

        start = time.time()
        print('Computing Y_BA Ishigami model')

        self.Z = C_BA.T
        Y = self.run_UQSA_case()
        print(time.time() - start)

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        Y = {}
        for QoI in QoIs:
            self.Z = sample_mat.T
            Y[QoI] = self.run_UQSA_case()

        return Y








    # analytic mean and conditional variances for the Ishigami function from Sudret_2006_global Equation 58
    ########################################################################################################################
    def ishigami_analytic(a,b):
        measures = {}
        measures["mean"] = a / 2.0
        D = a ** 2. / 8 + b * np.pi ** 4. / 5 + b ** 2 * np.pi ** 8. / 18 + 1. / 2

        measures["var"] = D
        # Conditional variances
        D1 = b * np.pi ** 4. / 5 + b ** 2 * np.pi ** 8. / 50. + 1. / 2
        D2 = a ** 2 / 8.
        D3 = 0

        D12 = 0
        D13 = b**2. * np.pi**8 * 8/ 225
        D23 = 0
        D123 = 0

        # Main and total sensitivity indices
        measures["sens_m"] = [D1 / D, D2 / D, D3 / D]

        measures["sens_t"] = [(D1 + D12 + D13 + D123) / D,
                              (D2 + D12 + D23 + D123) / D,
                              (D3 + D13 + D23 + D123) / D]
        return measures
########################################################################################################################