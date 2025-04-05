
import numpy as np
from scipy.signal import find_peaks


def split_signal_diastole_auto(time, signal, max_heart_rate=100, prominence=0.3):
    """
    Try to find the end diastole point of heart beats from a continuous signal
    Args:
        time (array): relative times in seconds
        signal (array): sample values
        max_heart_rate (Hz): maximum frequency of cycles
        prominence : prominence required to identify a peak. Specifies a fraction of the range of samples
    """
    sample_rate = 1/(time[1]-time[0])
    _prominence = 0.5*prominence*(np.max(signal) - np.min(signal))
    distance = (60/max_heart_rate)*sample_rate
    peaks, properties = find_peaks(-signal, distance=distance, prominence=_prominence)
    right_bases = signal[properties['right_bases']]
    peak_vals = signal[peaks]
    right_prominences = peak_vals/right_bases
    good_peaks = right_prominences > _prominence
    peaks = peaks[good_peaks]

    cycle_times = []
    cycle_samples = []
    cycle_indices = []

    # if no peaks where detected then the input data represents only one cardiac cycle
    if peaks.size == 0:
        cycle_times = time
        cycle_samples = signal

    for i in range(1, len(peaks)):
        cycle_times.append(time[peaks[i - 1]:peaks[i]])
        cycle_samples.append(signal[peaks[i - 1]:peaks[i]])
        cycle_indices.append([peaks[i - 1], peaks[i]])

    return cycle_times, cycle_samples, peaks, cycle_indices