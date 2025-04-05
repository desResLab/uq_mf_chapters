########################################################################################################################
# Definition of the models of all levels
########################################################################################################################
import time
import os
from pathlib import Path
import csv
import json
import shutil
import numpy as np
import h5py
import subprocess
import pickle
import glob
import pyvista as pv
import chaospy as cp

import starfish.UtilityLib.moduleXML as mXML
import starfish.NetworkLib.classVascularNetwork as cVN
import starfish.NetworkLib.classBoundaryConditions as cBC
import starfish.SolverLib.class1DflowSolver as c1dFS
import starfish.VascularPolynomialChaosLib.moduleFilePathHandlerVPC as mFPH_VPC
# Manages the file paths of the individual simulations and results withing the starfish working directory
import starfish.VascularPolynomialChaosLib.classUqsaCase as cUqsaCase
# Top level object configuring the UQSA it has the following subcomponents of importance
import starfish.VascularPolynomialChaosLib.classSampleManager as cSM
# Specifies and manages the sampling method to be used
import starfish.VascularPolynomialChaosLib.classRandomInputManager as cRIM
# Handles the random inputs and their dependence structure if specified
import starfish.VascularPolynomialChaosLib.classUqsaMethods as cUQM
# Manages the methods of UQSA to evaulate
import starfish.VascularPolynomialChaosLib.classLocationOfInterestManager as cLOIM
# Tracks the various locations in the network where UQSA should be done
import starfish.VascularPolynomialChaos as vpc

from split_cycles import split_signal_diastole_auto
#from Simulation_Folder.Estimation_Statistics_Simulations.EstimateStatistics_1D import split_signal_diastole_auto
from zeroD_Elements_computation import compute_0D_elements
from svzerodsolver.runnercpp import run_from_config
from postprocessVTK import extract_data
from sim_fold_gen_3DFSI import sim_folder_gen_3DFSI

# conversion factors
unit_mmhg_pa = 133.3
unit_pa_mmhg = unit_mmhg_pa**(-1)
unit_cm_m= 1. / 100.
unit_m_cm = unit_cm_m**(-1)
unit_m2_cm2 = unit_m_cm**2
unit_cm2_m2 = unit_cm_m**2
unit_mm_m = 1./1000.

########################################################################################################################
# definition of the 3D-FSI model
########################################################################################################################
class Model3D:

    def __init__(self, base_folder, solution_folder, QoI_3D_file, samples, idx, QoIs):
        '''
        3D FSI model
        '''
        self.base_file = base_folder
        self.h = samples[0, idx]
        self.r = samples[2, idx]
        self.E = samples[1, idx]
        self.solution_folder = solution_folder
        self.QoI_3D_file = QoI_3D_file
        self.QoIs = QoIs

    def run_simulation(self):
        pass

    def run_UQSA_case(self, samples):
        pass

    def load_data(self, solution_folder):

        samples = sorted(os.listdir(solution_folder))
        # remove samples .pkl file
        samples.remove([s for s in samples if s.endswith('.pkl')][0])

        # samples.remove([s for s in samples if s.endswith('.pkl')][0])

        data_dict = {}

        for sample in samples:
            print(sample)
            idx = sample[-3:]

            data_dict.update({sample: self.load_single_data(idx, solution_folder)})

        return data_dict


    def load_single_data(self, sample_idx, sol_dir):

        path = sol_dir + '/sample_' + str(sample_idx).zfill(3) + '/02-deform-ale_' + str(sample_idx).zfill(
            3) + '/02.2-fsi-ALE_' + str(sample_idx).zfill(3) + '/02.2-solution_' + str(sample_idx).zfill(3)
        # path = '/run/media/friedees/LaCie/MFMC_CCA/00_Data_3DFSI_NHK_Statistics_Estimation/sample_'+ str(sample_idx).zfill(3) + '/02-deform-ale_'+ str(sample_idx).zfill(3) + '/02.2-fsi-ALE_'+ str(sample_idx).zfill(3) + '/02.2-solution_'+ str(sample_idx).zfill(3)
        # filename = 'Statistics_Estimation/data/sample_000/02-deform-ale_000/02.2-fsi-ALE_000/02.2-solution_000/fsi_000_30000.vtu'

        # determine number of time steps to read in
        n_t_pts = len([name for name in os.listdir(path) if name.endswith('.vtu')])

        # temporary set parameters
        step_idx = 0
        n_z_vals = 1

        # initialize solution vectors
        inner_displacement_int = np.zeros(n_t_pts)
        outer_displacement_int = np.zeros(n_t_pts)
        volume_flow = np.zeros([n_z_vals, n_t_pts])
        pressure = np.zeros([n_z_vals, n_t_pts])
        area = np.zeros([n_z_vals, n_t_pts])

        # read in files
        for step_idx, filename in enumerate(glob.glob(os.path.join(path, '*.vtu'))):  # filter txt files only
            with open(os.path.join(os.getcwd(), filename), 'r') as file:
                data = pv.read(filename)

            # extract wall and lumen mesh
            wall_ind = np.nonzero(data['Domain_ID'] == 4)[0]
            wall_grid = data.extract_cells(wall_ind)

            lumen_ind = np.nonzero(data['Domain_ID'] == 2)[0]
            lumen_grid = data.extract_cells(lumen_ind)
            ref_points = lumen_grid.points

            length = lumen_grid.points[:, 2].max() - lumen_grid.points[:, 2].min()

            # Displacement cross sections
            z_displacement = length / 2
            plane_center = np.array([0, 0, length / 2])
            centerline_tangent = [0, 0, 1]
            axial_normal = [0, 0, 1]

            # radial displacements
            csx = wall_grid.slice(normal=axial_normal, origin=(0, 0, z_displacement))
            radial_vector = np.zeros_like(csx.points)
            radial_vector[:, 0] = csx.points[:, 0]
            radial_vector[:, 1] = csx.points[:, 1]
            radial_vector = radial_vector.T / np.linalg.norm(radial_vector, axis=1)
            radial_vector = radial_vector.T  # Transposed for broadcasting normalization
            radial_displacement = np.einsum('ij, ij -> i', csx.point_data['Displacement'], radial_vector)
            csx.point_data.set_array(radial_displacement, 'radial_displacement')
            csx.set_active_scalars('radial_displacement')

            # extract edges from the wall slice
            edges = csx.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                              non_manifold_edges=False)

            region_id = edges.connectivity().cell_data['RegionId']
            e0 = edges.extract_cells(region_id == 0)
            e1 = edges.extract_cells(region_id == 1)

            # integrate displacement for average displacement
            e0_int = e0.integrate_data()
            e1_int = e1.integrate_data()

            # determine which is inner and outer displacement
            if e1_int['Length'] > e0_int['Length']:
                inner_int = e0_int
                outer_int = e1_int
            else:
                inner_int = e1_int
                outer_int = e0_int

            inner_displacement_int[step_idx] = inner_int['radial_displacement'] / inner_int['Length']
            outer_displacement_int[step_idx] = outer_int['radial_displacement'] / outer_int['Length']

            # Deform mesh for proper calculations
            lumen_grid.points = ref_points + lumen_grid['Displacement']

            # 3.2 axial volume flows at inlet, midpoint and outlet
            z_idx = 0
            plane = pv.Plane(plane_center, direction=centerline_tangent)
            csx = lumen_grid.slice(normal=centerline_tangent, origin=plane_center)
            axial_velocity = np.dot(csx['Velocity'], centerline_tangent)

            int_data = csx.integrate_data()
            csx_area = int_data['Area']
            area[z_idx, step_idx] = int_data['Area']

            volume_flow[z_idx, step_idx] = int_data['Velocity'][0][2]
            pressure[z_idx, step_idx] = int_data['Pressure'] / csx_area

            edge = csx.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                             non_manifold_edges=False)
            edge_point = edge.points[0]
            edge_point = edge.points[np.argmax(np.linalg.norm(edge.points[:, 0:2], axis=1))]

        # save results in solution dictionary and convert to SI units
        results_dict = dict(
            area=area.tolist(),
            inner_displacement=((inner_displacement_int - np.min(inner_displacement_int)) / 100).tolist(),
            outer_displacement=((outer_displacement_int - np.min(outer_displacement_int)) / 100).tolist(),
            # avg_displacement=np.mean(((inner_displacement_int - np.min(inner_displacement_int)) /100) - ((outer_displacement_int - np.min(outer_displacement_int)) / 100)).tolist(),
            pressure=(pressure / 10).tolist(),
            volume_flow=(volume_flow * 1e-6).tolist(),
            time=np.linspace(0, (n_t_pts - 1) * 0.01, n_t_pts)
        )

        return results_dict

    def extract_QoI(self, data_dict, samples):

        # QoIs dict for 3D-FSI data
        QoI_dict = {'Psys': np.zeros(samples.shape[1]),
                    'PP': np.zeros(samples.shape[1]),
                    'Delta_R_max': np.zeros(samples.shape[1])}

        for sample_idx, sample in enumerate(data_dict.keys()):
            data_dict[sample]['time'] = np.linspace(0, (len(data_dict[sample]['inner_displacement']) - 1) * 0.01, len(data_dict[sample]['inner_displacement']))
            cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(np.array(data_dict[sample]['time']),
                                                                                          np.array(data_dict[sample]['volume_flow']).squeeze())

            for QoI in data_dict[sample].keys():
                if QoI == 'inner_displacement' or QoI == 'outer_displacement' or QoI == 'time':
                    data_dict[sample][QoI] = data_dict[sample][QoI][cycle_indices[-1][0]:cycle_indices[-1][1]]
                elif QoI != 'avg_dispplacement':
                    data_dict[sample][QoI]= data_dict[sample][QoI][0][cycle_indices[-1][0]:cycle_indices[-1][1]]

            # compute QoIs
            QoI_dict['Psys'][sample_idx] = np.max(np.array(data_dict[sample]['pressure']))
            QoI_dict['PP'][sample_idx] = np.max(data_dict[sample]['pressure']) - np.min(data_dict[sample]['pressure'])
            QoI_dict['Delta_R_max'][sample_idx] = np.max(data_dict[sample]['inner_displacement'])

        return QoI_dict

    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        # definition of solution dictionary
        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 3D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_3D_' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_3D_' + case_definition + '.pkl'
            Y_3D = open(pkl_name, 'rb')
            Y = pickle.load(Y_3D)
            print('3D model solutions successfully loaded')

        else:

            # compute 3D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_3D_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)


        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):

        # check if 3D solution can be loaded for the case or if the simultaion folders need to be generated
        if os.path.isdir('Simulation_Folder/MFMC_3D_FSI/' + case_definition):
            # load the solution of the 3D-FSI model
            Y['Y_A'] = self.mfmc_solution_computation_Y_A(A, QoIs, case_definition)
            Y['Y_B'] = self.mfmc_solution_computation_Y_B(B, QoIs, case_definition)
            Y['Y_BA'] = self.mfmc_solution_computation_Y_BA(C_BA, QoIs, case_definition)

        # generate sample folders
        else:
            sim_folder_gen_3DFSI(case_definition, A * [100, 10, 100], 'A', 0)
            sim_folder_gen_3DFSI(case_definition, B * [100, 10, 100], 'B', 0)  # [100,10,100] conversion from SI to CGS
            # generate sample folders for C_BA matrix
            for dim in range(C_BA.shape[0]):
                sim_folder_gen_3DFSI(case_definition, C_BA[dim, :,:] * [100, 10, 100], 'C', dim)  # [100,10,100] conversion from SI to CGS

            # Return message to run 3D-FSI simulations and exit
            print(
                "Now the 3D-FSI simulations need to be run before proceeding!\nAfter running the 3D-FSI simulations restart the main.")
            quit()

        return Y

    def mfmc_solution_computation_Y_A(self, A, QoIs, case_definition):

        solution_folder = os.path.join(self.solution_folder, 'A_0')
        data_dict = self.load_data(solution_folder)
        Y = self.extract_QoI(data_dict, A.T)

        return Y

    def mfmc_solution_computation_Y_B(self, B, QoIs, case_definition):

        solution_folder = os.path.join(self.solution_folder, 'B_0')
        data_dict = self.load_data(solution_folder)
        Y = self.extract_QoI(data_dict, B.T)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA, QoIs, case_definition):

        Y = {'Psys': np.zeros((C_BA.shape[1], 3)), 'PP': np.zeros((C_BA.shape[1], 3)),
                      'Delta_R_max': np.zeros((C_BA.shape[1], 3))}

        for idx_mat, mat in enumerate(['C_0', 'C_1', 'C_2']):


            solution_folder = os.path.join(self.solution_folder, mat)
            data_dict = self.load_data(solution_folder)
            temp = self.extract_QoI(data_dict, C_BA[idx_mat,:,:].T)
            for QoI in QoIs:
                Y[QoI][:, idx_mat] = temp[QoI]

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        # load solution data from pilot run
        file = 'Estimating_statistics/QoIs_3D_FSI_StatisticsEstimation_150.pkl'
        Y_3D = open(file, 'rb')
        solutions = pickle.load(Y_3D)

        # extract QoIs
        Y = self.extract_QoI(solutions,sample_mat.T)

        return Y


########################################################################################################################
# definition of the 3D-FSI PCE surrogatemodel
########################################################################################################################
class Model3DPCR:

    def __init__(self, base_folder, solution_folder, QoI_3D_file, samples, idx, QoIs, uncertainties: dict):
        '''
        3D FSI model
        '''
        self.model_prefix = 'Y_3D_PCR' # Used in making paths
        self.base_file = base_folder
        self.h = samples[0, idx]
        self.r = samples[2, idx]
        self.E = samples[1, idx]
        self.solution_folder = solution_folder
        self.QoI_3D_file = QoI_3D_file
        self.QoIs = QoIs
        self.pcrs = None
        self.input_names = ['IMT', 'E', 'R']
        self.uncertainties = uncertainties # TODO make a deep copy?
        self.samples_file = 'MFMC_1D_model_statistics_estimate/vascularPolynomialChaos_999/samplingMethodS/samples_S.hdf5'
        self.QoIs_training_dict = {}
        self.initialize_expansion()


    def initialize_expansion(self, order=4):
        samples_hdf5 = h5py.File(self.samples_file)
        samples_array = samples_hdf5['samples'][::]
        samples_hdf5.close()
        n_samples, n_inputs = samples_array.shape

        with open(self.QoI_3D_file, 'rb') as QoI_file:
            data_dict = pickle.load(QoI_file)

        QoI_dict = {'Psys': np.zeros(n_samples),
                    'PP': np.zeros(n_samples),
                    'Delta_R_max': np.zeros(n_samples)}

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
            QoI_dict['PP'][sample_idx] = np.max(data_dict[sample]['pressure']) - np.min(
                data_dict[sample]['pressure'])
            # QoI_dict['Delta_R_max'][sample_idx] = np.mean([np.max(data_dict['outer_displacement']), np.max(data_dict['inner_displacement'])])
            QoI_dict['Delta_R_max'][sample_idx] = np.max(data_dict[sample]['inner_displacement'])

        self.QoIs_training_dict = QoI_dict
        jpdf = cp.Iid(cp.Uniform(), n_inputs)
        poly_base = cp.generate_expansion(order, jpdf)
        self.pcrs = dict()
        for qoi, values in QoI_dict.items():
            pcr = cp.fit_regression(poly_base, samples_array.T, values)
            self.pcrs[qoi] = pcr

    def normalize_samples(self, samples):
        samples_normalized = samples.copy()
        for idx, input_name in enumerate(self.input_names):
            samples_normalized[:, idx] = samples_normalized[:, idx] - self.uncertainties[input_name]['lower']
            samples_normalized[:, idx] = samples_normalized[:, idx]/(self.uncertainties[input_name]['upper'] - self.uncertainties[input_name]['lower'])
        return samples_normalized

    def evaluate(self, samples):
        """ assumes samples are passed with input as last index"""
        samples_norm = self.normalize_samples(samples)
        results = dict()
        for qoi, pcr in self.pcrs.items():
            results[qoi] = pcr(*samples_norm.T) # need to pass one argument for each input
        return results

    def run_UQSA_case(self, samples):
        return self.evaluate(samples)

    def extract_QoI(self, data_dict, samples):

        # QoIs dict for 3D-FSI data
        QoI_dict = {'Psys': np.zeros(samples.shape[1]),
                    'PP': np.zeros(samples.shape[1]),
                    'Delta_R_max': np.zeros(samples.shape[1])}

        for sample_idx, sample in enumerate(data_dict.keys()):
            data_dict[sample]['time'] = np.linspace(0, (len(data_dict[sample]['inner_displacement']) - 1) * 0.01, len(data_dict[sample]['inner_displacement']))
            cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(np.array(data_dict[sample]['time']),
                                                                                          np.array(data_dict[sample]['volume_flow']).squeeze())

            for QoI in data_dict[sample].keys():
                if QoI == 'inner_displacement' or QoI == 'outer_displacement' or QoI == 'time':
                    data_dict[sample][QoI] = data_dict[sample][QoI][cycle_indices[-1][0]:cycle_indices[-1][1]]
                elif QoI != 'avg_dispplacement':
                    data_dict[sample][QoI]= data_dict[sample][QoI][0][cycle_indices[-1][0]:cycle_indices[-1][1]]

            # compute QoIs
            QoI_dict['Psys'][sample_idx] = np.max(np.array(data_dict[sample]['pressure']))
            QoI_dict['PP'][sample_idx] = np.max(data_dict[sample]['pressure']) - np.min(data_dict[sample]['pressure'])
            QoI_dict['Delta_R_max'][sample_idx] = np.max(data_dict[sample]['inner_displacement'])

        return QoI_dict

    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        # definition of solution dictionary
        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 3D simulations for this budget already exists
        if os.path.isfile( f'Simulation_Folder/{self.model_prefix}' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = f'Simulation_Folder/{self.model_prefix}'+ case_definition + '.pkl'
            Y_3D = open(pkl_name, 'rb')
            Y = pickle.load(Y_3D)
            print('3D model solutions successfully loaded')

        else:
            # compute 3D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = f'Simulation_Folder/{self.model_prefix}' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)
        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):
        # TODO modifies Y as a refrence, but also returns...
        Y['Y_A'] = self.mfmc_solution_computation_Y_A(A, QoIs, case_definition)
        Y['Y_B'] = self.mfmc_solution_computation_Y_B(B, QoIs, case_definition)
        Y['Y_BA'] = self.mfmc_solution_computation_Y_BA(C_BA, QoIs, case_definition)
        return Y

    def mfmc_solution_computation_Y_A(self, A, QoIs, case_definition):
        Y = self.evaluate(A)
        return Y

    def mfmc_solution_computation_Y_B(self, B, QoIs, case_definition):
        Y = self.evaluate(B)
        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA, QoIs, case_definition):

        Y = {'Psys': np.zeros((C_BA.shape[1], 3)), 'PP': np.zeros((C_BA.shape[1], 3)),
                      'Delta_R_max': np.zeros((C_BA.shape[1], 3))}

        for idx_mat, mat in enumerate(['C_0', 'C_1', 'C_2']):
            print(f'{idx_mat=}')
            temp = self.evaluate(C_BA[idx_mat,:,:])
            for QoI in QoIs:
                print(f'{temp[QoI].shape} {temp[QoI].dtype=} {Y[QoI].dtype=}')
                Y[QoI][:, idx_mat] = temp[QoI].T

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        # load solution data from pilot run
        file = 'Estimating_statistics/QoIs_3D_FSI_StatisticsEstimation_150.pkl'
        Y_3D = open(file, 'rb')
        solutions = pickle.load(Y_3D)

        # extract QoIs
        Y = self.extract_QoI(solutions,sample_mat.T)

        return Y


########################################################################################################################
# definition of the 1D model
########################################################################################################################

class Model1D:

    def __init__(self, networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs):
        ''' general initialization of the 1D model'''
        self.networkName = networkName
        self.uncertainties = uncertainties
        self.description = description
        self.vesselId = 0
        self.base_dataNumber = base_dataNumber
        self.case_dataNumber = case_dataNumber
        self.QoIs = QoIs
        #model_1D.workingdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Simulation_Folder')


    def generate_network(self):
        ''' generation of a artery network of the 1D model using Starfish'''
        vascularNetwork = cVN.VascularNetwork()
        vesselId = self.vesselId
        vascularNetwork.addVessel(vesselId)
        vessel = vascularNetwork.vessels[vesselId]

        vessel.geometryType = 'cone'
        vessel.length = 0.126
        vessel.radiusDistal = self.uncertainties['R']['mean']
        vessel.radiusProximal = self.uncertainties['R']['mean']
        vessel.N = 5

        vessel.complianceType = 'Laplace2comp'
        vessel.constantCompliance = False
        vessel.externalPressure = 0.0
        vessel.As = 'None'
        vessel.Ps = 9917
        vessel.wallThickness = self.uncertainties['IMT']['mean']
        vessel.poissonRatio = 0.49098
        vessel.youngModulus = self.uncertainties['E']['mean']

        inlet = cBC.CCAInflow()
        inlet.name = 'Flow-CCAInflow'
        inlet.prescribe = 'total'
        inlet.runtimeEvaluation = False
        inlet.freq = 1 / 1.1
        inlet.Npulse = 80

        outlet = cBC.Windkessel3()
        outlet.name = '_Windkessel-3Elements'
        outlet.Rc = 1869700000.0
        outlet.Rtotal = 2118450000
        outlet.C = 1.3546411199999999e-10
        outlet.Z = 248750000.0

        vascularNetwork.boundaryConditions[vessel.Id] = [inlet, outlet]

        vessel.my = 0.00465
        vessel.rho = 1048.5
        vessel.gamma = 2
        vessel.applyGlobalFluid = False

        # Save network
        networkName = 'Statistics_Estimation_1D_model'
        vascularNetwork.name = self.networkName
        vascularNetwork.description = 'Statistics_Estimation_1D_model'
        vascularNetwork.quiet = True # # TODO supress output??
        mXML.writeNetworkToXML(vascularNetwork)

    def run_1D_model(self, dataNumber):
        ''' Run 1D model using Starfish'''
        # Run a specific instance of this network
        vascularNetwork = mXML.loadNetworkFromXML(self.networkName,
                                                  dataNumber='xxx')  # dataNumber 'xxx' indicates the default network for the networkName\n",
        vascularNetwork.dataNumber = dataNumber
        vascularNetwork.description = self.description
        vascularNetwork.totalTime = 6.0
        vascularNetwork.CFL = 0.8
        vascularNetwork.dt = -2
        vascularNetwork.centralVenousPressure = 0.0
        vascularNetwork.automaticGridAdaptation = True
        vascularNetwork.initialsationMethod = 'AutoLinearSystem'
        vascularNetwork.initMeanPressure = 9917
        vascularNetwork.estimateWindkesselCompliance = 'No'
        vascularNetwork.timeSaveEnd = vascularNetwork.totalTime
        flowSolver = c1dFS.FlowSolver(vascularNetwork, quiet=True)
        flowSolver.solve()
        vascularNetwork.saveSolutionData()
        mXML.writeNetworkToXML(vascularNetwork)
        mXML.writeNetworkToXML(vascularNetwork, dataNumber)  # Save specific solution XML
        del flowSolver  # allow python to clean up solver memory

    def run_simulation(self):
        ''' Run 1D model using Starfish'''
        # Run a specific instance of this network
        vascularNetwork = mXML.loadNetworkFromXML(self.networkName,
                                                  dataNumber='xxx')  # dataNumber 'xxx' indicates the default network for the networkName\n",
        vascularNetwork.dataNumber = self.case_dataNumber
        vascularNetwork.description = self.description
        vascularNetwork.totalTime = 6.0
        vascularNetwork.CFL = 0.8
        vascularNetwork.dt = -2
        vascularNetwork.centralVenousPressure = 0.0
        vascularNetwork.automaticGridAdaptation = True
        vascularNetwork.initialsationMethod = 'AutoLinearSystem'
        vascularNetwork.initMeanPressure = 9917
        vascularNetwork.estimateWindkesselCompliance = 'No'
        vascularNetwork.timeSaveEnd = vascularNetwork.totalTime
        flowSolver = c1dFS.FlowSolver(vascularNetwork, quiet=True)
        flowSolver.solve()
        vascularNetwork.saveSolutionData()
        mXML.writeNetworkToXML(vascularNetwork)
        mXML.writeNetworkToXML(vascularNetwork, self.case_dataNumber)  # Save specific solution XML
        del flowSolver  # allow python to clean up solver memory

        data = self.load_data()

        return data


    def load_data(self):
        ''' Loading of 1D model single simulation solution data
        Returns a dictionary with all QoIs for one specific simulation
        '''

        # localization of the solution file
        working_dir = os.path.dirname(os.path.realpath(__file__))
        solution_file = working_dir + '/' + self.networkName + '/SolutionData_' + case_dataNumber + '/'+ self.networkName + '_SolutionData_' + case_dataNumber + '.hdf5'

        # extraction of the solution
        data_dict = {}
        with h5py.File(solution_file, "r") as f:
            data_dict['t'] = np.array(f['VascularNetwork']['simulationTime'])
            data_dict['P'] = np.array(f['vessels']['vessel_0  -  0']['Psol'])
            data_dict['Q'] = np.array(f['vessels']['vessel_0  -  0']['Qsol'])
            data_dict['A'] = np.array(f['vessels']['vessel_0  -  0']['Asol'])

        return data_dict



    def run_UQSA_case(self, MC_samples):
        ''' generation of UQSA cases of the 1D model using Starfish
        for each sample a file will be written to disk
        all UQSA cases will be run and their solution will be written to disk
        '''
        vascularNetwork = mXML.loadNetworkFromXML(self.networkName, self.base_dataNumber)
        vessel = vascularNetwork.vessels[self.vesselId]
        loi_manager = cLOIM.LocationOfInterestManager()
        locationId = 'vessel_0'
        locationName = 'midPoint'
        quantitiesOfInterestToProcess = ['Pressure', 'Flow', 'Area']
        xVal = vessel.length / 2
        confidenceAlpha = 5
        loi_manager.addLocationOfInterest(locationId, locationName,
                                          quantitiesOfInterestToProcess, xVal, confidenceAlpha)

        uqsaCaseFile = mFPH_VPC.getFilePath('uqsaCaseXmlFile', self.networkName, self.case_dataNumber, 'write')  # Creates path
        uqsaCase = cUqsaCase.UqsaCase()
        if False and os.path.exists(uqsaCaseFile):
            uqsaCase.loadXMLFile(uqsaCaseFile)
            print(uqsaCase.uqsaMethods)
        else:
            # 1. specify random variables
            rim = vascularNetwork.randomInputManager = cRIM.RandomInputManager()

            b = (self.uncertainties['IMT']['upper'] - self.uncertainties['IMT']['lower'])
            a = self.uncertainties['IMT']['lower']
            rvType = 'Uniform'
            parameter = 'vessel_0_wallThickness'
            rim.addRandomInput('h', a, b, rvType, parameter)

            b = (self.uncertainties['E']['upper'] - self.uncertainties['E']['lower'])
            a = self.uncertainties['E']['lower']
            rvType = 'Uniform'
            parameter = 'vessel_0_youngModulus'
            rim.addRandomInput('E', a, b, rvType, parameter)

            b = 1
            a = 0
            rvType = 'radius'
            parameter = 'vessel_0_radiusDistal'
            rim.addRandomInput('radiusDistal', a, b, rvType, parameter)

            b = 1
            a = 0
            rvType = 'radius'
            parameter = 'vessel_0_radiusProximal'
            rim.addRandomInput('radiusProximal', a, b, rvType, parameter)

            b = (self.uncertainties['R']['upper'] - self.uncertainties['R']['lower'])
            a = self.uncertainties['R']['lower']
            rvType = 'Uniform'
            parameter = None
            rim.addRandomInput('radius', a, b, rvType, parameter)

            sample_manager = uqsaCase.sampleManager = cSM.SampleManager()
            sample_manager.dependentCase = False
            sample_manager.samplingMethod = 'S'

            # 2. specify quantities of interest
            uqsaCase.locationOfInterestManager = loi_manager = cLOIM.LocationOfInterestManager()
            locationId = 'vessel_0'
            locationName = 'midPoint'
            quantitiesOfInterestToProcess = ['Pressure', 'Flow', 'Area']
            xVal = vessel.length / 2
            confidenceAlpha = 5
            loi_manager.addLocationOfInterest(locationId, locationName,
                                              quantitiesOfInterestToProcess, xVal, confidenceAlpha)

            # 3. specify uqsa methods
            MC = cUQM.UqsaMethodMonteCarlo()
            MC.sensitivityAnalysis = False
            MC.sampleSize = MC_samples

            uqsaCase.uqsaMethods = {'MC': MC}

            # 4.
            uqsaCase.initialize(self.networkName, self.case_dataNumber)
            uqsaCase.simulateEvaluations = True
            uqsaCase.locationOfInterestManager.evaluateSimulationTime = True
            uqsaCase.multiprocessing = True
            # 5. save the configuration file to the starfish database
            configurationFilePath = mFPH_VPC.getFilePath('uqsaCaseXmlFile', self.networkName, self.case_dataNumber, 'write')
            uqsaCase.writeXMLFile(configurationFilePath)

            # copy network file
            vascularNetwork.randomInputManager = rim
            destinationFile = mFPH_VPC.getFilePath('vpcNetworkXmlFile', self.networkName, self.case_dataNumber, 'write')
            mXML.writeNetworkToXML(vascularNetwork, networkXmlFile=destinationFile)

        vpc.run_uqsa_case(uqsaCase, vascularNetwork=vascularNetwork)

    def load_single_data(self, case_dataNumber):
        ''' Loading of 1D model single simulation solution data
        Returns a dictionary with all QoIs for one specific simulation
        '''

        # localization of the solution file
        working_dir = os.path.dirname(os.path.realpath(__file__))
        solution_file = working_dir + '/' + self.networkName + '/SolutionData_' + case_dataNumber + '/'+ self.networkName + '_SolutionData_' + case_dataNumber + '.hdf5'

        # extraction of the solution
        data_dict = {}
        with h5py.File(solution_file, "r") as f:
            data_dict['t'] = np.array(f['VascularNetwork']['simulationTime'])
            data_dict['P'] = np.array(f['vessels']['vessel_0  -  0']['Psol'])
            data_dict['Q'] = np.array(f['vessels']['vessel_0  -  0']['Qsol'])
            data_dict['A'] = np.array(f['vessels']['vessel_0  -  0']['Asol'])

        # extract the last cycle for all variables of the 1D model
        cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(data_dict['t'],
                                                                                      data_dict['Q'][:,0])
        for QoI in data_dict.keys():
            if QoI == 't':
                data_dict[QoI] = data_dict[QoI][cycle_indices[-1][0]:cycle_indices[-1][1]] - data_dict[QoI][cycle_indices[-1][0]]
            else:
                data_dict[QoI] = data_dict[QoI][cycle_indices[-1][0]:cycle_indices[-1][1], 2]

        # compute the QoIs
        QoIs = {'Psys': np.max(data_dict['P']),
                'PP': np.max(data_dict['P']) - np.min(data_dict['P']),
                'Delta_R_max': np.sqrt((np.max(data_dict['A']) / np.pi)) - np.sqrt((np.min(data_dict['A']) / np.pi))}
        #print(QoIs['PP']/133.32)

        return QoIs

    # function which evaluates the solution of the mfmc samples
    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        networkNameUQSA = 'MFMC_1D_model_UQSA_' + case_definition

        # definition of solution dictionary
        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 0D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_1D_' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_1D_' + case_definition + '.pkl'
            Y_1D = open(pkl_name, 'rb')
            Y = pickle.load(Y_1D)
            print('1D model solutions successfully loaded')

        else:

            # compute 1D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition, networkNameUQSA)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_1D_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)

        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition, networkNameUQSA):

        Y['Y_A'] = self.mfmc_solution_computation_Y_A(A, QoIs, case_definition, networkNameUQSA)
        Y['Y_B'] = self.mfmc_solution_computation_Y_B(B, QoIs, case_definition, networkNameUQSA)
        Y['Y_BA'] = self.mfmc_solution_computation_Y_BA(C_BA, QoIs, case_definition, networkNameUQSA)

        return Y

    def mfmc_solution_computation_Y_A(self, A, QoIs, case_definition, networkNameUQSA):

        start = time.time()
        sample_mat = A

        Y = {}
        for QoI in QoIs:
            Y.update({QoI: np.zeros(A.shape[0])})

        for run in range(A.shape[0]):
            uncertainties = {'IMT': {'mean': sample_mat[run, 0]},
                             'E': {'mean': sample_mat[run, 1]},
                             'R': {'mean': sample_mat[run, 2]}}

            networkName = networkNameUQSA
            description = case_definition
            base_dataNumber = str(run).zfill(5)
            case_dataNumber = base_dataNumber

            model = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs)
            model.generate_network()
            model.run_1D_model(base_dataNumber)

        # load data and save data in correct format
        print('Loading Y_A 1D model data')
        for run in range(A.shape[0]):
            case_dataNumber = str(run).zfill(5)
            data = model.load_single_data(case_dataNumber)
            for QoI in QoIs:
                Y[QoI][run] = data[QoI]
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_B(self, B, QoIs, case_definition, networkNameUQSA):
        start = time.time()
        sample_mat = B
        file_num_shift = int(1e4)

        Y = {}
        for QoI in QoIs:
            Y.update({QoI: np.zeros(B.shape[0])})

        for run in range(B.shape[0]):
            uncertainties = {'IMT': {'mean': sample_mat[run, 0]},
                             'E': {'mean': sample_mat[run, 1]},
                             'R': {'mean': sample_mat[run, 2]}}

            networkName = networkNameUQSA
            description = case_definition
            base_dataNumber = str(run + file_num_shift).zfill(5)
            case_dataNumber = base_dataNumber

            model = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs)
            model.generate_network()
            model.run_1D_model(base_dataNumber)

        # load data and save data in correct format
        print('Loading Y_B 1D model data')
        for run in range(B.shape[0]):
            case_dataNumber = str(run + file_num_shift).zfill(5)
            data = model.load_single_data(case_dataNumber)
            for QoI in QoIs:
                Y[QoI][run] = data[QoI]
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA, QoIs, case_definition, networkNameUQSA):

        # computation of Y_BA solutions
        sample_mat = C_BA
        file_num_shift = int(2e4)

        Y = {}
        for QoI in QoIs:
            Y.update({QoI: np.zeros((C_BA.shape[1], int(sample_mat.shape[0])))})

        start = time.time()
        for d in range(sample_mat.shape[0]):
            for run in range(C_BA.shape[1]):
                uncertainties = {'IMT': {'mean': sample_mat[d, run, 0]},
                                 'E': {'mean': sample_mat[d, run, 1]},
                                 'R': {'mean': sample_mat[d, run, 2]}}

                networkName = networkNameUQSA
                description = case_definition
                base_dataNumber = str(int(run + (d + 1) * 1e4 + file_num_shift)).zfill(5)
                case_dataNumber = base_dataNumber

                model = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber,
                                QoIs)
                model.generate_network()
                model.run_1D_model(base_dataNumber)

            # load data and save data in correct format
            print('Loading Y_BA 1D model data')
            for run in range(C_BA.shape[1]):
                case_dataNumber = str(int(run + (d + 1) * 1e4 + file_num_shift)).zfill(5)
                data = model.load_single_data(case_dataNumber)
                for QoI in QoIs:
                    Y[QoI][run, d] = data[QoI]
            print(time.time() - start)

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        Y = {}
        for QoI in QoIs:
            Y.update({QoI: np.zeros(sample_mat.shape[0])})

        for run in range(sample_mat.shape[0]):
            uncertainties = {'IMT': {'mean': sample_mat[run, 0]},
                             'E': {'mean': sample_mat[run, 1]},
                             'R': {'mean': sample_mat[run, 2]}}

            networkName = 'Estimating_statistics'
            description = case_definition
            base_dataNumber = str(run).zfill(5)
            case_dataNumber = base_dataNumber

            model = Model1D(networkName, uncertainties, description, base_dataNumber, case_dataNumber, QoIs)
            model.generate_network()
            model.run_1D_model(base_dataNumber)

        # load data and save data in correct format
        for run in range(sample_mat.shape[0]):
            case_dataNumber = str(run).zfill(5)
            data = model.load_single_data(case_dataNumber)
            for QoI in QoIs:
                Y[QoI][run] = data[QoI]

        return Y

########################################################################################################################
# definition of the 0D model
########################################################################################################################
class Model0D():

    def __init__(self,config, base_file, samples, idx, QoIs, perturb_model, L=0.063, mu=0.00465):
        '''
        0D model which can be run from a configuration file containing all parameters
        uncertain model parameters are set through the sample matrix and an index specifying which parameter triplet to take
        '''
        self.config = config
        self.base_file = base_file
        self.h = samples[0, idx]
        self.r = samples[2, idx]
        self.E = samples[1, idx]
        self.L = L
        self.mu = mu
        self.samples = samples
        self.QoIs = QoIs
        self.perturb_model = perturb_model
        # compute the electrical analogue values of the two vessels
        R, C = compute_0D_elements(self.r, self.E, self.h)
        for vessel in range(len(self.config['vessels'])):
            self.config['vessels'][vessel]['zero_d_element_values']['R_poiseuille'] = R
            self.config['vessels'][vessel]['zero_d_element_values']['C'] = C



    def run_0D_model_config(self):
        ''' Runs 0D model from configuration file and returns the QoIs of the last cycle
        Data is saved only in memory but not written to disk
        '''

        # run simulation from configuration file
        #start = time.time()
        results_df = run_from_config(self.config)
        #print(time.time() - start)
        # extract QoI of last cycle
        QoIs = self.extract_QoI(results_df)
        return QoIs

    def run_simulation(self):
        ''' Runs 0D model from configuration file and returns the QoIs of the last cycle
        Data is saved only in memory but not written to disk
        '''

        # run simulation from configuration file
        results_df = run_from_config(self.config)
        return results_df


    def extract_QoI(self, results_df):
        ''' extracts QoIs (Psys, Pulse pressure, maximal radius change) of a single computation
        Computation of Delta_R_max follows the tube law
        '''

        # extract the data from the data frame
        t = np.array(results_df['time'][:int(len(results_df['time']) / 2)])
        P_mid = np.array(results_df['pressure_in'][int(len(results_df['time']) / 2):])
        Q_in = np.array(results_df['flow_in'][:int(len(results_df['time']) / 2)])

        # extract the last cycle
        cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(t, Q_in)

        # compute the QoIs
        Psys = np.max(P_mid[cycle_indices[-1][0]:cycle_indices[-1][-1]])
        PP = np.max(P_mid[cycle_indices[-1][0]:cycle_indices[-1][-1]]) - np.min(P_mid[cycle_indices[-1][0]:cycle_indices[-1][-1]])
        Delta_R_max = (np.max(P_mid[cycle_indices[-1][0]:cycle_indices[-1][-1]]) - np.min(P_mid[cycle_indices[-1][0]:cycle_indices[-1][-1]])) \
                      * 3 / 4 * (self.r ** 2) / (self.E * self.h)

        return Psys, PP, Delta_R_max

    def run_0D_model(self, case):
        ''' Run 0D model from a case file and write the solution to disk'''
        simulation = case
        solution = case[:-4] + 'csv'
        subprocess.Popen(
            ["/home/friedees/Documents/mulifidelity-mc/svZeroDPlus/Release/svzerodsolver", simulation, solution])

    def generate_UQSA_files(self, samples):
        ''' Generates 0D model files from the base case file
        uncertain parameters are taken from the provided sample matrix
        '''

        # compute the resistance and compliance for all samples
        R, C = compute_0D_elements(samples[2, :], samples[1, :], samples[0, :], self.L, self.mu)

        # generate the UQSA files and write them to disk
        for sample in range(samples.shape[1]):
            # define destination file for the sample simulation
            destination_file = self.base_file[:-5] + '_' + str(sample).zfill(5) + '.json'
            shutil.copyfile(self.base_file, destination_file)

            # update the R and C values in the specific sample simulation
            with open(destination_file, "r+") as jsonFile:
                data = json.load(jsonFile)
                data['vessels'][0]['zero_d_element_values']['C'] = float(C[sample])
                data['vessels'][0]['zero_d_element_values']['R_poiseuille'] = float(R[sample])
                data['vessels'][1]['zero_d_element_values']['C'] = float(C[sample])
                data['vessels'][1]['zero_d_element_values']['R_poiseuille'] = float(R[sample])

                jsonFile.seek(0)  # rewind
                # jsonFile.write(json.dumps(data))
                json.dump(data, jsonFile, indent=4)
                jsonFile.truncate()

    def run_UQSA_case(self, N_cases):
        ''' Runs all 0D model UQSA case files using the run_0D_model_config() function
        simulation solutions are only saved in memory and are not written to disk
        '''

        # initialize dictionary for all QoIs for all model runs
        QoIs = {'Psys': np.zeros(N_cases), 'PP': np.zeros(N_cases),
                'Delta_R_max': np.zeros(N_cases)}

        # run all UQSA cases
        for case in range(N_cases):
            # update values to perform UQSA
            self.h = self.samples[0, case]
            self.r = self.samples[2, case]
            self.E = self.samples[1, case]
            R, C = compute_0D_elements(self.r, self.E, self.h)
            for vessel in range(len(self.config['vessels'])):
                self.config['vessels'][vessel]['zero_d_element_values']['R_poiseuille'] = R
                self.config['vessels'][vessel]['zero_d_element_values']['C'] = C

            # evaluate model and save solution
            QoIs['Psys'][case], QoIs['PP'][case], QoIs['Delta_R_max'][case] = self.run_0D_model_config()

        return QoIs

        # function implementation if the UQSA cases are written in files
        # for case in range(N_cases):
        #     case_file = self.base_file[:-5] + '_' + str(case).zfill(5) + '.json'
        #     self.run_0D_model(case_file)


    def load_data(self, samples):
        ''' Load 0D model data from disk and extract QoIs (Psys, Pulse pressure, maximal radius change)
        for all cases located in the same folder as the base file
        Computation of Delta_R_max follows the tube law
        '''


        # load data
        path = Path(self.base_file)
        solution_dir = path.parent.absolute()
        file_list = list(solution_dir.glob('MFMC_mixed_40s_*' + '.csv'))
        file_list.sort()

        # extract data from csv file
        data_dict = {}
        for idx, file in enumerate(file_list):
            data_dict[idx] = {'allData': {}}
            P_in = np.array([])
            P_out = np.array([])
            Q_out = np.array([])
            Q_in = np.array([])
            t = np.array([])
            with open(file) as csv_file:
                csv_read = csv.reader(csv_file, delimiter=',')
                next(csv_read, None)
                for row in csv_read:
                    t = np.append(t, float(row[1]))
                    P_in = np.append(P_in, float(row[4]))
                    P_out = np.append(P_out, float(row[5]))
                    Q_out = np.append(Q_out, float(row[3]))
                    Q_in = np.append(Q_in, float(row[2]))

            # detach the solutions of the two vessels from another
            if np.all(np.diff(t) > 0) == False:
                data_dict[idx]['allData']['time'] = t[:int(len(t) / 2)]
                data_dict[idx]['allData']['P_in'] = P_in[:int(len(t) / 2)]
                data_dict[idx]['allData']['P_mid'] = P_in[int(len(t) / 2):]
                data_dict[idx]['allData']['P_out'] = P_out[:int(len(t) / 2)]
                data_dict[idx]['allData']['Q_in'] = Q_in[:int(len(t) / 2)]
                data_dict[idx]['allData']['Q_out'] = Q_out[int(len(t) / 2):]
                data_dict[idx]['allData']['Q_mid'] = Q_in[int(len(t) / 2):]

            # extract last cardiac cycle
            cycle_times, cycle_samples, peaks, cycle_indices = split_signal_diastole_auto(
                data_dict[idx]['allData']['time'],
                data_dict[idx]['allData']['Q_in'])
            data_dict[idx].update(
                {'time': data_dict[idx]['allData']['time'][cycle_indices[-1][0]:cycle_indices[-(1)][1]] -
                         data_dict[idx]['allData']['time'][cycle_indices[-1][0]],
                 'Pressure': data_dict[idx]['allData']['P_mid'][
                             cycle_indices[-1][0]:cycle_indices[-1][1]],
                 'Flow': data_dict[idx]['allData']['Q_mid'][cycle_indices[-1][0]:cycle_indices[-1][1]]
                 })

        # compute QoIs
        QoIs = {}
        for idx in data_dict.keys():
            if idx == 0:
                QoIs['Psys'] = np.max(data_dict[idx]['Pressure'])
                QoIs['PP'] = np.max(data_dict[idx]['Pressure']) - np.min(
                    data_dict[idx]['Pressure'])
                QoIs['Delta_R_max'] = (np.max(data_dict[idx]['Pressure']) - np.min(
                    data_dict[idx]['Pressure'])) * 3 / 4 * (samples[2, idx]) ** 2 / (
                                              samples[0, idx] * samples[1, idx])
            else:
                QoIs['Psys'] = np.vstack([QoIs['Psys'], np.max(data_dict[idx]['Pressure'])])
                QoIs['PP'] = np.vstack([QoIs['PP'], np.max(data_dict[idx]['Pressure']) - np.min(
                    data_dict[idx]['Pressure'])])
                QoIs['Delta_R_max'] = np.vstack(
                    [QoIs['Delta_R_max'], (np.max(data_dict[idx]['Pressure']) - np.min(
                        data_dict[idx]['Pressure'])) * 3 / 4 * (samples[2, idx]) ** 2 / (
                             samples[0, idx] * samples[1, idx])])

        # remove additional (,1) dimension
        for QoI in QoIs.keys():
            QoIs[QoI] = np.squeeze(QoIs[QoI])

        return QoIs

    # function which evaluates the solution of the mfmc samples
    def mfmc(self, A, B, C_BA, case_definition, QoIs):

        Y = {'Y_A': {}, 'Y_B': {}, 'Y_BA': {}}

        # check if solution file of 0D simulations for this budget already exists
        if os.path.isfile('Simulation_Folder/Y_0D_' + case_definition + '.pkl'):
            # load already existing solution file for 1D model
            pkl_name = 'Simulation_Folder/Y_0D_' + case_definition + '.pkl'
            Y_0D = open(pkl_name, 'rb')
            Y = pickle.load(Y_0D)
            print('0D model solutions successfully loaded')

        else:

            # compute 0D model solutions
            Y = self.mfmc_solution_computation(A, B, C_BA, Y, QoIs, case_definition)

            # save solution file
            pkl_name = 'Simulation_Folder/Y_0D_' + case_definition + '.pkl'
            with open(pkl_name, 'wb') as f:
                pickle.dump(Y, f)

        # perturb 0D-model solution
        if self.perturb_model == True:
            # perturb the 0D model
            Y = self.perturb_0D_model(Y, QoIs, A, B, C_BA)

        return Y

    def mfmc_solution_computation(self, A, B, C_BA, Y, QoIs, case_definition):

        # compute the solution of the 0D model
        Y['Y_A'] = self.mfmc_solution_computation_Y_A(A)
        Y['Y_B'] = self.mfmc_solution_computation_Y_B(B)
        Y['Y_BA'] = self.mfmc_solution_computation_Y_BA(C_BA, QoIs)

        return Y


    def mfmc_solution_computation_Y_A(self, A):

        start = time.time()
        print('Computing Y_A 0D model')
        self.samples = A.T
        Y = self.run_UQSA_case(np.shape(self.samples)[1])
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_B(self, B):

        print('Computing Y_B 0D model')
        start = time.time()
        self.samples = B.T
        Y = self.run_UQSA_case(np.shape(self.samples)[1])
        print(time.time() - start)

        return Y

    def mfmc_solution_computation_Y_BA(self, C_BA, QoIs):

        Y = {}
        for QoI in QoIs:
            Y.update({QoI: np.zeros((C_BA.shape[1], int(self.samples.shape[0])))})
        for d in range(len(QoIs)):
            start = time.time()
            print('Computing Y_BA 0D model')
            self.samples = C_BA[d, :, :].T
            temp = self.run_UQSA_case(np.shape(self.samples)[1])
            for QoI in QoIs:
                Y[QoI][:, d] = temp[QoI]
            print(time.time() - start)

        return Y

    def perturb_0D_model(self, Y, QoIs, A, B, C_BA, alpha_perturb=5):

        # load least square fit
        lstq_file = "Simulation_Folder/0D_Model_lstsq_perturbation.pkl"
        f = open(lstq_file, "rb")
        lstsq = pickle.load(f)
        #
        for QoI in QoIs:
            Y['Y_A'][QoI] = Y['Y_A'][QoI] + alpha_perturb * (np.dot(lstsq[QoI][0], A[:len(Y['Y_A'][QoI]), :].T))
            Y['Y_B'][QoI] = Y['Y_B'][QoI] + alpha_perturb * (np.dot(lstsq[QoI][0], B[:len(Y['Y_A'][QoI]), :].T))
            temp = np.zeros((Y['Y_A'][QoI].size, 3))
            for dim in range((Y['Y_BA'][QoI].shape[1])):
                temp[:, dim] = Y['Y_BA'][QoI][:, dim] + alpha_perturb * (
                    np.dot(lstsq[QoI][0], C_BA[dim, :len(Y['Y_A'][QoI][:len(Y['Y_A'][QoI])]), :].T))
            Y['Y_BA'][QoI] = temp

        return Y

    def solutions_estimating_statistics(self, sample_mat, QoIs, case_definition):

        self.samples = sample_mat.T
        Y = self.run_UQSA_case(np.shape(self.samples)[1])

        return Y