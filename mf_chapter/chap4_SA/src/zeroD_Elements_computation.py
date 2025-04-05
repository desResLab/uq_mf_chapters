import numpy as np

def compute_0D_elements(r, E, h, L=0.063, mu=0.00465):
    """
    computes compliance and resistance value of single vessel
    Inputs:
    r       float       diastolic radius of the vessel in [m]
    E       float       elastic modulus of the vessel wall in [Pa]
    h       float       wall thickness of the vessel wall in [m]
    L       float       length of the vessel in [m]
    mu      float       blood viscosity in [Pa s]

    Outputs:
    R        float      resistance of the vessel in [Pa s/m^3]
    C        float      compliance of the vessel in [m^3 / Pa]

    """


    R = 8 * mu * L / (np.pi * r ** 4)          # resistance value of CCA
    C = 3 * L * np.pi * r ** 3 / (2 *E * h)   # compliance value of CCA
    return R, C