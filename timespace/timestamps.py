import numpy as np

def find_nearest_timestamps(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
