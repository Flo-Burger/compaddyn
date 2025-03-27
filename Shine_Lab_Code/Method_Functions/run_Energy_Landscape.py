"""
Energy Landscape Analysis using Kernel Density Estimation.

Author: Florian Burger
Date: 2025-03-26
Version: 1.0

Description:
This module estimates the energy landscape of neural activity by analyzing the
distribution of mean squared displacements (MSD) across time lags. It uses 
Kernel Density Estimation (KDE) with a Gaussian kernel to approximate energy 
values (negative log-likelihoods) over a defined displacement space.

Functions:
- PdistGaussKern: Computes the energy via KDE for one MSD vector.
- run_energy_landscape: Applies this analysis across trials and lags.
"""

import numpy as np
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

def pdist_gauss_kern(dat, ds, bandwidth=1):
    """
    Replicates MATLAB's PdistGaussKern.

    Steps:
    1) Fit Gaussian KDE to input data.
    2) Evaluate the probability density at points `ds`.
    3) Return the negative log-likelihood (interpreted as energy).

    Parameters:
        dat (np.ndarray): 1D data array to fit (e.g., MSD values).
        ds (np.ndarray): Points at which to evaluate the KDE.
        bandwidth (float): Bandwidth for the Gaussian kernel.

    Returns:
        nll (np.ndarray): Negative log-likelihood at each point in `ds`.
    """
    # Fit Gaussian Kernel Density Estimation
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(dat[:, None])  # shape (N,) → (N,1)

    # Evaluate log-probability at specified points
    log_pdf = kde.score_samples(ds[:, None])  # shape: (len(ds),)

    return -log_pdf  # Return negative log-likelihood (energy)

def run_Energy_Landscape(data, ndt=20, ds=None, bandwidth=1, ddof=1):
    
    """
    Computes the energy landscape from 3D time series data.

    Parameters:
        data (np.ndarray): Input data with shape (timepoints, channels, trials)
        ndt (int): Maximum lag (default = 20)
        ds (np.ndarray, optional): Displacement values to evaluate KDE on.
                                   Defaults to np.arange(0, ndt+1, 1)
        bandwidth (float): Bandwidth for KDE (default = 1)
        ddof (int): Degrees of freedom for z-scoring (default = 1)

    Returns:
        nrg_sig (np.ndarray): Energy landscape with shape (ndt, len(ds), n_trials)
    """

    # Default ds range: 0 to ndt (inclusive)
    if ds is None:
        ds = np.arange(0, ndt + 1, 1)

    n_trials = data.shape[2]
    n_ds = len(ds)

    # Preallocate output: (lags x ds-points x trials)
    nrg_sig = np.full((ndt, n_ds, n_trials), np.nan)

    # Loop through each trial
    for trial in range(n_trials):
        # Z-score normalize each channel independently
        zscored = stats.zscore(data[:, :, trial], axis=0, ddof=ddof)

        # Loop through each lag
        for lag in range(1, ndt + 1):
            # Compute mean squared displacement (MSD) for this lag
            msd = np.mean((zscored[lag:] - zscored[:-lag]) ** 2, axis=1)

            # Compute energy via KDE over displacement values
            energy = pdist_gauss_kern(msd, ds, bandwidth)

            # Store results; lag index adjusted for 0-based indexing
            nrg_sig[lag - 1, :, trial] = energy

    return nrg_sig

