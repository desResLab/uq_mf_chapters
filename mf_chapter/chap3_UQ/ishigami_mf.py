import numpy as np
import matplotlib.pyplot as plt
import sys

# generate orcale statistics for the Ishigami problem
def generate_oracle(N_sample):
    
    a = 5.0
    b = 0.1

    Q_vect = np.empty([N_sample,3])
    x = -np.pi + 2.*np.pi*np.random.rand(N_sample,3)

    for j in range(3):
 
      # parametric ishigami   
      # sin(z1) + C1*a*sin(z2)*sin(z2) + C2*b*(z3**alpha)*sin(z1) 
 
      M_ID = j   # [0 HF, 1 LF1, 2 LF2]
    
      if(M_ID == 0):
        C1    = 1.0
        C2    = 1.0
        alpha = 4.0
      if(M_ID == 1):
        C1    = 0.95
        #C2    = 1.0
        C2 = 0.5
        alpha = 4.0
      if(M_ID == 2):
        C1    = 0.6
        C2    = 9.0
        alpha = 2.0    

      for i in range(N_sample):
        # Get samples
        z1 = x[i,0]
        z2 = x[i,1]
        z3 = x[i,2]
        # Add QoI
        Q_vect[i,j] = np.sin(z1) + C1*a*np.sin(z2)*np.sin(z2) + C2*b*(z3**alpha)*np.sin(z1)

    mean     = np.mean(Q_vect[:,0])
    var      = np.var(Q_vect[:,0],ddof=1)
    corr_mat = np.corrcoef(Q_vect,rowvar=False)

    rho_1  = corr_mat[0,1]
    rho_2  = corr_mat[0,2]
    rho_12 = corr_mat[1,2]

    return mean, var, rho_1, rho_2, rho_12

# evaluate the Rsq for ACV-MF
def eval_Rsq_MFMC(r1,r2,rho_1,rho_2,rho_12):
    return (rho_1*rho_1 * (r1-1.)/r1 + rho_2*rho_2 * (r2-r1)/(r2*r1) )


# evaluate the Rsq for ACV-MF
def eval_Rsq_ACVMF(r1,r2,rho_1,rho_2,rho_12):
    
    q1 = (r1-1.)/r1
    q2 = (r2-1.)/r2
    rmin = np.min([r1,r2])
    qm = (rmin-1.)/rmin
    
    Rsq = (q1*q2)/(q1*q2-qm*qm*rho_12*rho_12) * ( q1 * rho_1*rho_1 
                                                + q2 * rho_2*rho_2 
                                                - 2. * qm *rho_1*rho_2*rho_12)
    
    return Rsq


# find minimum solution for ACV-MF (2 models) via simple grid search
def eval_optimal_res_alloc(method,Ctot, w1, w2, var_Q, rho_1, rho_2, rho_12, r1_min, r1_max, r2_min, r2_max, 
                           plot_flag=False, real_all=True):
    
    nx = 498
    ny = 498

    r1_vect = np.linspace(r1_min, r1_max, nx)
    r2_vect = np.linspace(r2_min, r2_max, ny)

    mat_plot = np.ones([nx*ny,5])*10
    
    count = 0
    for i in range(nx):
        for j in range(i+1,ny):
            
            if(method == 'ACV_MF'):
                Rsq = eval_Rsq_ACVMF(r1_vect[i],r2_vect[j],rho_1,rho_2,rho_12)
            elif(method == 'MFMC'):
                Rsq = eval_Rsq_MFMC(r1_vect[i],r2_vect[j],rho_1,rho_2,rho_12)
            else:
                print('ERROR: Invalid Method')
                exit(-1)

            N = Ctot / (1. + r1_vect[i] * w1 + r2_vect[j] * w2)            
            
            if (real_all):
               N  = np.floor(N)
               N1 = np.ceil(r1_vect[i]*N)
               N2 = np.ceil(r2_vect[j]*N)
    
               r1_eff = N1/N
               r2_eff = N2/N
               
               if(method == 'ACV_MF'):
                    Rsq = eval_Rsq_ACVMF(r1_eff,r2_eff,rho_1,rho_2,rho_12)
               elif(method == 'MFMC'):
                    Rsq = eval_Rsq_MFMC(r1_eff,r2_eff,rho_1,rho_2,rho_12)
               else:
                    print('ERROR: Invalid Method')
                    exit(-1)
            
            J = np.log(1.0 + (var_Q/N) * np.abs((1. - Rsq)))
            # print(r1_vect[i],r2_vect[j],N,J)            
            mat_plot[count,:] = [ r1_vect[i], r2_vect[j], N, Rsq, J ]
            count = count + 1
    
    # Get Min objective
    id_min = np.argmin(mat_plot[:,4])
    # print('Grid search min (r1, r2, N, rsq, J):', mat_plot[id_min,:])
    
    if (plot_flag == True):
       plt.figure(figsize=(6,5))
       plt.scatter(mat_plot[:,0],mat_plot[:,1],c=mat_plot[:,4],s=0.5)
       plt.colorbar()
       # Plot min    
       plt.plot(mat_plot[id_min,0],mat_plot[id_min,1],'rx')
       plt.title('Estimator Variance')
       plt.xlabel('r1')
       plt.ylabel('r2')
       plt.tight_layout()
       plt.show()
    
    # ACV MF optimal solution
    Nopt = np.floor( mat_plot[id_min,2])
    N1   = np.ceil(mat_plot[id_min,0]*Nopt)
    N2   = np.ceil(mat_plot[id_min,1]*Nopt)
    
    r1_eff = N1/Nopt
    r2_eff = N2/Nopt
    
    Cost_opt = Nopt + w1*N1 + w2*N2
    
    if(method == 'ACV_MF'):
        Rsq_opt = eval_Rsq_ACVMF(r1_eff,r2_eff,rho_1,rho_2,rho_12)
    elif(method == 'MFMC'):
        Rsq_opt = eval_Rsq_MFMC(r1_eff,r2_eff,rho_1,rho_2,rho_12)
    else:
        print('ERROR: Invalid Method')
        exit(-1)
    
    Jopt = (var_Q/Nopt) * (1. - Rsq_opt)    
    
    return Nopt, N1, N2, Cost_opt, Rsq_opt, Jopt


# MFMC analytical solution (it does not check for optimality!)
def MFMC_optimal( Ctot, w1, w2, var_Q, rho_1, rho_2, rho_12 ):
    
    r1_mfmc = np.sqrt( (rho_1*rho_1 * rho_2*rho_2)/( w1 * ( 1.-rho_1*rho_1 ) ) )
    r2_mfmc = np.sqrt( (rho_2*rho_2)/( w2 * ( 1.-rho_1*rho_1 ) ) )

    N_mfmc = np.floor( Ctot / (1. + r1_mfmc * w1 + r2_mfmc * w2) )

    N1_mfmc = np.ceil(r1_mfmc*N_mfmc)
    N2_mfmc = np.ceil(r2_mfmc*N_mfmc)

    r1_mfmc = N1_mfmc / N_mfmc
    r2_mfmc = N2_mfmc / N_mfmc 

    Cost_mfmc = N_mfmc + w1*N1_mfmc + w2*N2_mfmc

    Rsq_mfmc = (rho_1*rho_1 * (r1_mfmc-1.)/r1_mfmc + rho_2*rho_2 * (r2_mfmc-r1_mfmc)/(r2_mfmc*r1_mfmc) )

    Var_mfmc = var_Q/N_mfmc * (1. - Rsq_mfmc )
    
    return N_mfmc, N1_mfmc, N2_mfmc, Cost_mfmc, Rsq_mfmc, Var_mfmc


# =========
# MAIN CODE
# =========
if __name__ == '__main__':

    np.random.seed(1234)

    ###############################################################################
    ###############################################################################
    # Input
    N_sample  = 100 # number of samples for the oracle statistics
    plot_flag = True  # if True plot the estimator variance solution
    real_all  = True  # if True select the best variance given the solution corresponding to integers for N,N1,N2 
    Ctot      = 100   # total budget in eq simulations
    w1        = 0.1   # cost of LF1 (w.r.t w0=1)
    w2        = 0.01  # cost of LF2 (w.r.t w0=1)
    r1_min    = 2     # min and max values for grid search (r1, r2)
    r1_max    = 500
    r2_min    = 2
    r2_max    = 500
    ###############################################################################

    # Generate oracle statistics
    mean_Q, var_Q, rho_1, rho_2, rho_12 = generate_oracle(N_sample)

    print()
    print()
    print('*********')
    print('PILOT RUN')
    print('*********')
    print('Number of Pilot runs ', N_sample)
    print('mean_Q ', mean_Q)
    print('var_Q', var_Q)
    print('rho_1 ', rho_1)
    print('rho_2 ', rho_2)
    print('rho_12 ', rho_12)

    # MC solution
    MC_var = var_Q/Ctot

    print('***********')
    print('MONTE CARLO')
    print('***********')
    print('MC Var ', MC_var)

    # MFMC optimal solution
    N_mfmc, N1_mfmc, N2_mfmc, Cost_mfmc, Rsq_mfmc, Var_mfmc = MFMC_optimal( Ctot, w1, w2, var_Q, rho_1, rho_2, rho_12 )

    print('************')
    print('MFMC - EXACT') 
    print('************')
    print('N HF:', N_mfmc)
    print('N LF1:', N1_mfmc)
    print('N_LF2:', N2_mfmc)
    print('r1:', N1_mfmc/N_mfmc)
    print('r2:', N2_mfmc/N_mfmc)
    print('ACV MF Est Variance:', Var_mfmc)
    print('Rsq:', Rsq_mfmc)
    print('Cost:', Cost_mfmc)

    # get the solution for ACV-MF
    Nopt, N1, N2, Cost_opt, Rsq_opt, Jopt_mfmc = eval_optimal_res_alloc('MFMC',Ctot, w1, w2, var_Q, rho_1, rho_2, rho_12, r1_min, r1_max, r2_min, r2_max, plot_flag=plot_flag, real_all=real_all)    

    print('**********')
    print('MFMC - OPT') 
    print('**********')
    print('N HF:', Nopt)
    print('N LF1:', N1)
    print('N_LF2:', N2)
    print('r1:', N1/Nopt)
    print('r2:', N2/Nopt)
    print('ACV MF Est Variance:', Jopt_mfmc)
    print('Rsq:', Rsq_opt)
    print('Cost:', Cost_opt)

    # get the solution for ACV-MF
    Nopt, N1, N2, Cost_opt, Rsq_opt, Jopt = eval_optimal_res_alloc('ACV_MF',Ctot, w1, w2, var_Q, rho_1, rho_2, rho_12, r1_min, r1_max, r2_min, r2_max, plot_flag=plot_flag, real_all=real_all)

    print('******')
    print('ACV MF') 
    print('******')
    print('N HF:', Nopt)
    print('N LF1:', N1)
    print('N_LF2:', N2)
    print('r1:', N1/Nopt)
    print('r2:', N2/Nopt)
    print('ACV MF Est Variance:', Jopt)
    print('Rsq:', Rsq_opt)
    print('Cost:', Cost_opt)


    print('************************')
    print('FINAL VARIANCE REDUCTION') 
    print('************************')
    print('gamma ratio (MFMC_EXACT)', Var_mfmc/(var_Q/Ctot))
    print('gamma ratio (MFMC-OPT)', Jopt_mfmc/(var_Q/Ctot))
    print('gamma ratio (ACV MF)', Jopt/(var_Q/Ctot))
    
