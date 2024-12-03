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
        # pressure at the aortic inlet
        zerod_data[name+':pressure_in'] = (data[idxs,2])[-steps_per_cycle:]

    pressure_max = max(zerod_data['aorta:pressure_in'])
    pressure_min = min(zerod_data['aorta:pressure_in'])

    return pressure_max, pressure_min