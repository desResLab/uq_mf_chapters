import numpy as np

def get_QOI_0D(filename, QOI_name=None):

    # radius of the aorta
    mu = 0.04         # viscosity of blood
    r  = 1.019478928  # radius of the aorta

    vessel_names = ['aorta','left_iliac','right_iliac']

    var_names = np.genfromtxt(filename, dtype = 'U', usecols=0, skip_header=1, delimiter=',')
    time      = np.genfromtxt(filename, usecols=1, skip_header=1, delimiter=',')
    data      = np.genfromtxt(filename, usecols=range(2,6), skip_header=1, delimiter=',')

    num_0d_cycles   = 1
    num_0d_tsteps   = len(np.nonzero(var_names == var_names[0])[0])
    steps_per_cycle = int((num_0d_tsteps+num_0d_cycles-1)/num_0d_cycles)

    zerod_data = {}
    for name in vessel_names:
        idxs = np.nonzero(var_names == name)[0]
        zerod_data[name+':flow_out']     = (data[idxs,1])[-steps_per_cycle:]
        zerod_data[name+':pressure_out'] = (data[idxs,3])[-steps_per_cycle:]

    flow_aorta = zerod_data['aorta:flow_out']
    wss_avg    = 4*mu*np.mean(flow_aorta)/(np.pi*r**3)
    #wss_max   = 4*mu*np.max(flow_aorta)/(np.pi*r**3)
    #wss_min   = 4*mu*np.min(flow_aorta)/(np.pi*r**3)

    return wss_avg