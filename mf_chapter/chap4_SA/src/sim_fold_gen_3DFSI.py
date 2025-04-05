import chaospy as cp
import shutil
import os
from mesh_generation import mesh_gen
import pickle

# dim - dimension of the C matrix the folders are generated for
def sim_folder_gen_3DFSI(case_definition, samples, matrix, dim, path_bf='/home/friedees/Documents/mulifidelity-mc/Multifidelity_1D_0D/Base_simulation_folder_3DFSI/'):
    ####################################################################################################################
    ####################################################################################################################

    # generate folder structure for the three matrices
    os.makedirs('Simulation_Folder/MFMC_3D_FSI/'+ case_definition + '/' + matrix + '_' + str(dim), exist_ok=True)

    f = open('Simulation_Folder/MFMC_3D_FSI/' + case_definition + '/' + matrix + '_' + str(dim) + '/samples_' + matrix + '_' + str(dim) + '.pkl', "wb")
    pickle.dump(samples, f)
    f.close()

    # matrix count for determining the starting number of the samples
    if matrix == 'A':
        mat_count = 0   # 'sample_000' - 'sample_099'
    elif matrix == 'B':
        mat_count = 100 # 'sample_100' - 'sample_199'
    else:
        mat_count = 200 # 'sample_200' - 'sample_299', 'sample_300' - 'sample_399', 'sample_400' - 'sample_499'


    for idx, sample in enumerate(samples):
        # define sample simulation folder structure
        path_sample = 'Simulation_Folder/MFMC_3D_FSI/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3)
        #os.makedirs('sample_' + str(idx+dim*100+mat_count).zfill(3) + '/00-mesh_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/01-rigid_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.1-prestress_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/init_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)
        os.makedirs(path_sample + '/03-postprocessing_' + str(idx+dim*100+mat_count).zfill(3), exist_ok=True)


        # generate mesh for the sample

        mesh_gen(path_bf + '00-mesh_base/mesh_base_file.geo', path_sample, sample, (idx+dim*100+mat_count))

        #path_work = 'MFMC_3DFSI/Budget_'  + '/' + matrix + '/sample_' + str(idx+dim*100+mat_count).zfill(3)

        # generate run file
        # update run_xxx.job file with correct run description
        with open(path_bf + 'run_base.job', 'r') as input_file, open(path_sample + '/run_' +str(idx+dim*100+mat_count).zfill(3) + '.job', 'w') as output_file:
            for line in input_file:
                if line.strip() == '#$ -N sample_xxx':
                    output_file.write('#$ -N sample_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 01-rigid_base/svFSI_01_base.inp':
                    output_file.write('mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 01-rigid_' + str(idx+dim*100+mat_count).zfill(3) + '/svFSI_01_' + str(idx+dim*100+mat_count).zfill(3) +'.inp\n')
                elif line.strip() == 'python3 ../calcMeanPressTrac.py -s solutionDir -o outputDir -m lumenWall -i sampleIdx':
                    output_file.write('python3 ../../../calcMeanPressTrac.py -s ' + '/afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) + '/01-rigid_' + str(idx+dim*100+mat_count).zfill(3) + '/01-solution_' + str(idx+dim*100+mat_count).zfill(3) +
                                      ' -o ' + '/afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) +'/02.1-prestress_' + str(idx+dim*100+mat_count).zfill(3) +
                                      ' -m ' + '/afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) + '/00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/lumen_' + str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_' + str(idx+dim*100+mat_count).zfill(3) + '/lumen_wall_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp' +
                                      ' -i ' + str(idx+dim*100+mat_count).zfill(3))
                elif line.strip() == 'mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 02-deform-ale_base/02.1-prestress_base/svFSI_02.1_base.inp':\
                        output_file.write('mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.1-prestress_' +
                                      str(idx+dim*100+mat_count).zfill(3) +'/svFSI_02.1_' + str(idx+dim*100+mat_count).zfill(3) +'.inp\n')
                elif line.strip() == 'python3 ../copy_initialConditions.py -f /afs/crc.nd.edu/user/f/fschaefe/Private/sample_base/01-rigid_base/01-solution_base/ ' \
                                     '-p /afs/crc.nd.edu/user/f/fschaefe/Private/sample_base/02-deform-ale_base/02.1-prestress_base/02.1-solution_base/ ' \
                                     '-o /afs/crc.nd.edu/user/f/fschaefe/Private/sample_base/02-deform-ale_base/02.2-fsi-ALE_base/init_base/ -i base':
                    output_file.write('python3 ../../../copy_initialConditions.py -f /afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) +'/01-rigid_' + str(idx+dim*100+mat_count).zfill(3) +'/01-solution_' +  str(idx+dim*100+mat_count).zfill(3) +'/ ' \
                                     '-p /afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) +'/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) +'/02.1-prestress_' + str(idx+dim*100+mat_count).zfill(3) +'/02.1-solution_'  + str(idx+dim*100+mat_count).zfill(3) +'/ ' \
                                     '-o /afs/crc.nd.edu/user/f/fschaefe/Private/' + case_definition + '/' + matrix + '_' + str(dim) + '/sample_' + str(idx+dim*100+mat_count).zfill(3) +'/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) +'/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) +'/init_' + str(idx+dim*100+mat_count).zfill(3) +'/ -i ' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 02-deform-ale_base/02.2-fsi_ALE_base/svFSI_02.2_base.inp':
                    output_file.write('mpirun -np 24 /afs/crc.nd.edu/group/tulip/01_code/05_svFSI/svBin/svFSI-build/bin/svFSI 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' +
                                      str(idx+dim*100+mat_count).zfill(3) +'/svFSI_02.2_' + str(idx+dim*100+mat_count).zfill(3) +'.inp\n')
                else:
                    output_file.write(line)


        # fill folder with the respective files

        #####
        # folder 01-rigid
        path_work = path_sample +  '/01-rigid_' + str(idx+dim*100+mat_count).zfill(3)

        # copy inlet flow file
        shutil.copy(path_bf + 'lumen_inlet.flow', path_sample)

        # update svFSI_01_xxx.inp file
        with open(path_bf + '01-rigid_base/svFSI_01_base.inp', 'r') as input_file, open(path_work + '/svFSI_01_' +str(idx+dim*100+mat_count).zfill(3) + '.inp', 'w') as output_file:
            for line in input_file:
                if line.strip() == 'Save results in folder: 01-rigid_xxx/01-solution_xxx':
                    output_file.write('Save results in folder: 01-rigid_'+ str(idx+dim*100+mat_count).zfill(3) +'/01-solution_'+ str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Name prefix of saved VTK files: fluid_only_xxx':
                    output_file.write('Name prefix of saved VTK files: fluid_only_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Mesh file path: 00-mesh_xxx/lumen_xxx/lumen_xxx.vtu':
                    output_file.write('      Mesh file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_inlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_inlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_outlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_outlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_wall_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_wall_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                # elif line.strip() == 'Temporal values file path: 01-rigid_base/lumen_inlet.flow':
                #     output_file.write('      Temporal values file path: 01-rigid_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_inlet.flow\n')
                else:
                    output_file.write(line)

        ####
        # folder 02.1-prestress
        path_work = path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3)  + '/02.1-prestress_' + str(idx+dim*100+mat_count).zfill(3)


        # update svFSI_02.1_xxx.inp file
        with open(path_bf + '02-deform-ale_base/02.1-prestress_base/svFSI_02.1_base.inp', 'r') as input_file, open(
                path_work + '/svFSI_02.1_' + str(idx+dim*100+mat_count).zfill(3) + '.inp', 'w') as output_file:
            for line in input_file:
                if line.strip() == 'Save results in folder: 02-deform-ale_base/02.1-prestress_xxx/02.1-solution_xxx':
                    output_file.write('Save results in folder: 02-deform-ale_'+ str(idx+dim*100+mat_count).zfill(3) +'/02.1-prestress_'+ str(idx+dim*100+mat_count).zfill(3) +'/02.1-solution_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Name prefix of saved VTK files: prestress_xxx':
                    output_file.write('Name prefix of saved VTK files: prestress_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Mesh file path:    00-mesh_xxx/wall_xxx/wall_xxx.vtu':
                    output_file.write(
                        '      Mesh file path: 00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_' + str(idx+dim*100+mat_count).zfill(
                            3) + '/wall_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_inlet_xxx.vtp':
                    output_file.write(
                        '      Face file path: 00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_' + str(idx+dim*100+mat_count).zfill(
                            3) + '/mesh-surfaces_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_inlet_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_outlet_xxx.vtp':
                    output_file.write(
                        '      Face file path: 00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_' + str(idx+dim*100+mat_count).zfill(
                            3) + '/mesh-surfaces_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_outlet_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_inner_xxx.vtp':
                    output_file.write(
                        '      Face file path: 00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_' + str(idx+dim*100+mat_count).zfill(
                            3) + '/mesh-surfaces_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_inner_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_outer_xxx.vtp':
                    output_file.write(
                        '      Face file path: 00-mesh_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_' + str(idx+dim*100+mat_count).zfill(
                            3) + '/mesh-surfaces_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_outer_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp\n')
                elif line.strip() == 'Elasticity modulus: xxx':
                    output_file.write('   Elasticity modulus: ' + str(sample[1]) + '\n')
                elif line.strip() == 'Spatial values file path: 02-deform-ale_base/02.1-prestress_base/rigid_wall_mean_pressure_xxx.vtp':
                    output_file.write(
                        '      Spatial values file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.1-prestress_' + str(idx+dim*100+mat_count).zfill(3) + '/rigid_wall_mean_pressure_' + str(idx+dim*100+mat_count).zfill(3) + '.vtp\n')
                else:
                    output_file.write(line)



        ####
        # folder 02.2-fsi-ALE
        path_work = path_sample + '/02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3)

        # copy inlet flow file
        #shutil.copy(path_bf + 'lumen_inlet.flow', path_work)

        # update svFSI_02.2_xxx.inp file

        with open(path_bf + '02-deform-ale_base/02.2-fsi-ALE_base/svFSI_02.2_base.inp', 'r') as input_file, open(path_work + '/svFSI_02.2_' + str(idx+dim*100+mat_count).zfill(3) + '.inp', 'w') as output_file:
            for line in input_file:
                # lumen mesh
                if line.strip() == 'Save results in folder: 02-deform-ale_base/02.2-fsi-ALE_xxx/02.2-solution_xxx':
                    output_file.write('Save results in folder: 02-deform-ale_'+ str(idx+dim*100+mat_count).zfill(3) +'/02.2-fsi-ALE_'+ str(idx+dim*100+mat_count).zfill(3) + '/02.2-solution_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Name prefix of saved VTK files: fsi_xxx':
                    output_file.write('Name prefix of saved VTK files: fsi_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Mesh file path: 00-mesh_xxx/lumen_xxx/lumen_xxx.vtu':
                    output_file.write('      Mesh file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_inlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_inlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_outlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_outlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/lumen_xxx/mesh-surfaces_xxx/lumen_wall_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/lumen_wall_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Initial velocities file path: 02-deform-ale_base/02.2-fsi-ALE_base/init_base/rigid_wall_flow_xxx.vtu':
                    output_file.write('   Initial velocities file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/init_' + str(idx+dim*100+mat_count).zfill(3) + '/rigid_wall_flow_'+ str(idx+dim*100+mat_count).zfill(3)  +'.vtu\n')
                elif line.strip() == 'Initial pressures file path:  02-deform-ale_base/02.2-fsi-ALE_base/init_base/rigid_wall_flow_xxx.vtu':
                    output_file.write('   Initial pressures file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/init_' + str(idx+dim*100+mat_count).zfill(3) + '/rigid_wall_flow_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')

                # wall mesh
                elif line.strip() == 'Name prefix of saved VTK files: prestress_xxx':
                    output_file.write('Name prefix of saved VTK files: prestress_' + str(idx+dim*100+mat_count).zfill(3) + '\n')
                elif line.strip() == 'Mesh file path: 00-mesh_xxx/wall_xxx/wall_xxx.vtu':
                    output_file.write('      Mesh file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_inlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_inlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_outlet_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_outlet_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_inner_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_inner_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Face file path: 00-mesh_xxx/wall_xxx/mesh-surfaces_xxx/wall_outer_xxx.vtp':
                    output_file.write('      Face file path: 00-mesh_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_'+ str(idx+dim*100+mat_count).zfill(3) +'/mesh-surfaces_'+ str(idx+dim*100+mat_count).zfill(3) +'/wall_outer_'+ str(idx+dim*100+mat_count).zfill(3) +'.vtp\n')
                elif line.strip() == 'Prestress file path: 02-deform-ale_base/02.2-fsi-ALE_base/init_base/wall_prestress_xxx.vtu':
                    output_file.write('   Prestress file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/init_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_prestress_'+ str(idx+dim*100+mat_count).zfill(3)  +'.vtu\n')
                elif line.strip() == 'Initial displacements file path: 02-deform-ale_base/02.2-fsi-ALE_base/init_base/wall_prestress_xxx.vtu':
                    output_file.write('   Initial displacements file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/init_' + str(idx+dim*100+mat_count).zfill(3) + '/wall_prestress_' + str(idx+dim*100+mat_count).zfill(3) + '.vtu\n')

                # update elasticty modulus
                elif line.strip() == 'Elasticity modulus: xxx':
                    output_file.write('      Elasticity modulus: ' + str(sample[1]) + '\n')

                # elif line.strip() == '   Temporal values file path: 02-deform-ale_base/02.2-fsi-ALE_base/lumen_inlet.flow':
                #     output_file.write('      Temporal values file path: 02-deform-ale_' + str(idx+dim*100+mat_count).zfill(3) + '/02.2-fsi-ALE_' + str(idx+dim*100+mat_count).zfill(3) + '/lumen_inlet.flow\n')
                else:
                    output_file.write(line)






    print('Done folder generation')
