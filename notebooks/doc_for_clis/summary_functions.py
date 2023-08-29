import numpy as np
from scipy.stats import trim_mean

def trim_mean80(a, axis=0):  # noqa
    return trim_mean(a, proportiontocut=.1, axis=axis)

def entropy(a, axis=0):  # noqa
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])