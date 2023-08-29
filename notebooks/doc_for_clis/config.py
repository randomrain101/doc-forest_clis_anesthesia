#%%
"""
This file defines parameters and functions that are needed to compute coherent markers
and features from the ana2con and the clis dataset.
"""
import sys
import numpy as np
sys.path.append("/home/oliver/doc-forest_clis_ba_ws21/nice")

from nice.markers import (PowerSpectralDensity,
                          PowerSpectralDensitySummary,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          KolmogorovComplexity,
                          PowerSpectralDensityEstimator)

from summary_functions import entropy, trim_mean80

#
# --- configure epochs ---
#

# length of one epoch
chunk_seconds = 8

#
# --- window size ---
#
# configure window size for epoch summarization (rolling mean and std)
window = "15 min"

#
# --- configure marker reduction functions ---
#
reduction_functions = [
    dict(channels_fun = np.mean, epochs_fun = np.mean),
    dict(channels_fun = np.std, epochs_fun = np.mean),
    dict(channels_fun = np.mean, epochs_fun = np.std),
    dict(channels_fun = np.std, epochs_fun = np.std)
]

#
# --- configure markers ---
# 
psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)

base_psd = PowerSpectralDensityEstimator(
    psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,
    psd_params=psds_params, comment='default')

marker_list = [
    PermutationEntropy(tmin=None, tmax=None, backend='c'),

    SymbolicMutualInformation(
        tmin=None, tmax=None, method='weighted', backend='openmp',
        method_params={'nthreads': 'auto', "bypass_csd": True}, comment='weighted'),

    KolmogorovComplexity(tmin=None, tmax=None, backend='openmp',
                         method_params={'nthreads': 'auto'}),

    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=False, comment='delta'),
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=True, comment='deltan'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=False, comment='theta'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=True, comment='thetan'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=False, comment='alpha'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=True, comment='alphan'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=False, comment='beta'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=True, comment='betan'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=False, comment='gamma'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=True, comment='gamman'),

    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                         normalize=True, comment='summary_se'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.5, comment='summary_msf'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.9, comment='summary_sef90'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.95, comment='summary_sef95')
]

def get_reduction_params(epochs_fun=np.mean, channels_fun=np.mean, epochs=None):
    reduction_params = {}
    reduction_params['PowerSpectralDensity'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun},
            {'axis': 'frequency', 'function': np.sum}],
        'picks': {
            'epochs': epochs,
            'channels': None}}

    reduction_params['PowerSpectralDensity/summary_se'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': entropy},
            {'axis': 'channels', 'function': channels_fun},
            {'axis': 'epochs', 'function': np.mean}],
        'picks': {
            'epochs': epochs,
            'channels': None}}

    reduction_params['PowerSpectralDensitySummary'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
            {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs,
            'channels': None}}

    reduction_params['PermutationEntropy'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
            {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs,
            'channels': None}}

    reduction_params['SymbolicMutualInformation'] = {
        'reduction_func':
            [{'axis': 'channels_y', 'function': np.median},
            {'axis': 'channels', 'function': channels_fun},
            {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs,
            'channels_y': None,
            'channels': None}}

    reduction_params['KolmogorovComplexity'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
            {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs,
            'channels': None}}
    return reduction_params
# %%
