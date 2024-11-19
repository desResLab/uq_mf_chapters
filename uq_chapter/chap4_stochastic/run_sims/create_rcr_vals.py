import numpy as np
import matplotlib.pyplot as plt
import sys
#import pyDOE as doe
import scipy.io as sio
from SALib.sample import sobol

method    = 'sobol' #sys.argv[1] # 'lhs' or 'uni'
N         = 2048 #int(sys.argv[2]) # number of samples (max for tutorial: 2**10)
s         = 1 #int(sys.argv[3]) # number of experiments (max for tutorial: 100)

def rescale_samples(x, domain):
    """
    Rescale samples from [0, 1] to the domain specified by the user.
    """
    for i in range(x.shape[1]):
        bd = domain[i]
        x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
    return x

def create_RCR_vals(method, N, s, RC_bounds=[0.5, 0.5, 0.5]):
    Rp_bounds = RC_bounds[0]
    C_bounds  = RC_bounds[1]
    Rd_bounds = RC_bounds[2]

    # load the RCR values
    Rp = 6.8123e2
    C  = 3.6664e-5
    Rd = 3.1013e4

    if method == 'uni':
        Rp = np.random.uniform(Rp-Rp_bounds*Rp, Rp+Rp_bounds*Rp, (s,N))
        C  = np.random.uniform(C-C_bounds*C,    C+C_bounds*C,    (s,N))
        Rd = np.random.uniform(Rd-Rd_bounds*Rd, Rd+Rd_bounds*Rd, (s,N))
    elif method == 'grid':
        Rp = np.linspace(Rp-Rp_bounds*Rp, Rp+Rp_bounds*Rp, N)
        C  = np.linspace(C-C_bounds*C, C+C_bounds*C, N)
        Rd = np.linspace(Rd-Rd_bounds*Rd, Rd+Rd_bounds*Rd, N)
        Rp, C, Rd = np.meshgrid(Rp, C, Rd)
        Rp = Rp.flatten()
        C  = C.flatten()
        Rd = Rd.flatten()
    elif method == 'sobol':
        problem = {'num_vars': 3,
                   'names': ['Rp', 'C', 'Rd'],
                   'bounds': [[Rp-Rp_bounds*Rp, Rp+Rp_bounds*Rp],
                              [C-C_bounds*C, C+C_bounds*C],
                              [Rd-Rd_bounds*Rd, Rd+Rd_bounds*Rd]]
                  }
        samples = sobol.sample(problem, int(N/8))
        Rp = samples[:,0]
        C = samples[:,1]
        Rd = samples[:,2]

    sio.savemat(method+'_RCR_vals.mat', {'Rp': Rp, 'C': C, 'Rd': Rd})

if __name__ == '__main__':
    create_RCR_vals(method, N, s)
