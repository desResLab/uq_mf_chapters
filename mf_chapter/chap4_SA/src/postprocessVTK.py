import numpy as np
import os
import glob
import pyvista as pv
import matplotlib.pyplot as plt
import pickle


def extract_data(sample_idx, sol_dir):

    path = sol_dir + '/sample_'+ str(sample_idx).zfill(3) + '/02-deform-ale_'+ str(sample_idx).zfill(3) + '/02.2-fsi-ALE_'+ str(sample_idx).zfill(3) + '/02.2-solution_'+ str(sample_idx).zfill(3)
    #path = '/run/media/friedees/LaCie/MFMC_CCA/00_Data_3DFSI_NHK_Statistics_Estimation/sample_'+ str(sample_idx).zfill(3) + '/02-deform-ale_'+ str(sample_idx).zfill(3) + '/02.2-fsi-ALE_'+ str(sample_idx).zfill(3) + '/02.2-solution_'+ str(sample_idx).zfill(3)
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
        inner_displacement=((inner_displacement_int - np.min(inner_displacement_int)) /100).tolist(),
        outer_displacement=((outer_displacement_int - np.min(outer_displacement_int)) / 100).tolist(),
        #avg_displacement=np.mean(((inner_displacement_int - np.min(inner_displacement_int)) /100) - ((outer_displacement_int - np.min(outer_displacement_int)) / 100)).tolist(),
        pressure=(pressure / 10).tolist(),
        volume_flow=(volume_flow * 1e-6).tolist(),
        time = np.linspace(0, (n_t_pts - 1) * 0.01, n_t_pts)
    )

    # plt.figure()
    # plt.plot(results_dict['pressure'][0])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(results_dict['inner_displacement'], linestyle='solid')
    # plt.plot(results_dict['outer_displacement'], linestyle='dashed')
    # plt.show()

    return results_dict

if __name__ == '__main__':

    parent_dir = '/run/media/friedees/LaCie/MFMC_CCA/00_Data_3DFSI_NHK_Statistics_Estimation/'

    # get the list of sample directories
    #samples = ['sample_000', 'sample_001', 'sample_002']
    samples = sorted(os.listdir(parent_dir))

    results = {}

    for sample in samples:
        print(sample)
        idx = sample[-3:]

        results.update({sample: extract_data(idx, parent_dir)})

    f = open("QoIs_3D_FSI.pkl", "wb")
    pickle.dump(results, f)
    f.close()

    plt.figure()
    for sample in results.keys():
        plt.plot(np.array(results[sample]['pressure'][0])/133.32)
    plt.tight_layout()
    plt.show()

    plt.figure()
    for sample in results.keys():
        plt.plot(np.array(results[sample]['inner_displacement'])*1e3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    for sample in results.keys():
        plt.plot(np.array(results[sample]['outer_displacement'])*1e3)
    plt.tight_layout()
    plt.show()





    print('Postprocessing done')