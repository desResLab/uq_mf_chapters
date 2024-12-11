import os
import sys
file_path = '/Users/chloe/Documents/Stanford/ME398_Spring/marsden_uq/uq_mf_chapters/uq_chapter/chap6_inv/circuits/'
sys.path.append(file_path)
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

#----------------------------------------------------------------------------------------#
# Inputs: 
#      para:
#         	Rp_star: range of proximal resistance
#         	Rd_star: range of distal resistance
#         	C_star:  range of capacitance
# Outputs:
#         Y: max(P_p), min(P_p)
def RCR_sim(para, totalCycles, cycleTime, plot=False):

	# decomposition
    #para                     = para.numpy()
    para                     = np.array(para)
    Rp_star, Rd_star, C_star = para[0], para[1], para[2] # in cgs units

	# related constants and data
    Pd  = 4000                   # distal pressure, in cgs units
    t_c = totalCycles*cycleTime  # how many cardiac cycles in total computed
	
	# load proximal flow rate Qp cc/s (cubic centimeter/s)
    Qp            = np.loadtxt(file_path+'aobif_inflow.csv')

	# interpolate by cubic splines, i.e. create a C^2 continuous function based on the given data
    Qp_interp     = CubicSpline(Qp[:,0], Qp[:,1], bc_type='periodic')
	
	# initial condition for Pp
    state0     = [0.0]
    t_range    = [0, t_c]
    dt         = cycleTime / 1000

	# find number of time evaluations
    num_t = round(t_c/dt)  + 1 # +1 for ic

	# define ode system, the RCR model
    def f(t, state):
        return Rp_star * Qp_interp(t,1) + Qp_interp(t)/C_star - \
						 (state - Qp_interp(t) * Rp_star - Pd)/(C_star * Rd_star)

	# evaluation time instances: this setting will give both the ic and end point in time series
    t_eval = np.linspace(0, t_c, num_t)

	# solve the system
    P_p    = solve_ivp(f, t_range, state0, method='RK45', t_eval=t_eval)

	# take the final 1/3 of the data to extract max and min and convert back to mmHg
    y_target = P_p.y[0,round(2*num_t/3):]

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(t_eval, P_p.y[0]/1333)
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        plt.title('Pressure vs. Time')
        plt.show()

    return [y_target.min(), y_target.max(), y_target.mean()]

if __name__ == "__main__":
    print(RCR_sim([1000, 1000, 5e-5]))