"""
FFT-Based Spectral Analysis for Neural Time Series Data.

Author: Florian Burger
Date: 2025-03-26
Version: 1.0

Description:
This module provides two functions for performing Fast Fourier Transform (FFT)
on multivariate time series data (e.g., neural recordings):
- `run_fft_global`: Computes FFT on the global signal (average across areas).
- `run_fft_per_area`: Computes FFT separately for each area and subject.

The input data is assumed to be a 3D NumPy array with dimensions:
    (areas, timepoints, subjects)

"""

import numpy as np


def run_fft_global(data, fs=None):
    """
    Computes the FFT for each subject using the global signal,
    defined as the average across all areas for each subject.

    Parameters:
        data (np.ndarray): 
            Neural time series data with shape (areas, time, subjects).
        fs (float, optional): 
            Sampling frequency in Hz. If not provided, frequencies will be in
            cycles per timepoint.

    Returns:
        freqs (np.ndarray): 
            1D array of frequency bins (length = n_freqs).
        fft_global (np.ndarray): 
            2D array (subjects x n_freqs) of FFT magnitudes.
    """
    n_areas, n_time, n_subjs = data.shape

    # Time step: 1/fs if sampling frequency is provided, otherwise 1
    d = 1 / fs if fs is not None else 1.0

    # Frequency bins based on signal length and time step
    freqs = np.fft.rfftfreq(n_time, d=d)
    n_freqs = len(freqs)

    # Initialize output
    fft_global = np.zeros((n_subjs, n_freqs))

    # Loop through each subject
    for subj in range(n_subjs):
        # Global signal: average across all areas
        subject_signal = np.mean(data[:, :, subj], axis=0)  # shape: (time,)
        
        # Compute FFT and store magnitude
        fft_vals = np.fft.rfft(subject_signal)
        fft_global[subj, :] = np.abs(fft_vals)

    return freqs, fft_global


def run_fft_per_area(data, fs=None):
    """
    Computes the FFT separately for each area and subject.

    Parameters:
        data (np.ndarray): 
            Neural time series data with shape (areas, time, subjects).
        fs (float, optional): 
            Sampling frequency in Hz. If not provided, defaults to 1.

    Returns:
        freqs (np.ndarray): 
            1D array of frequency bins (length = n_freqs).
        fft_area (np.ndarray): 
            3D array of FFT magnitudes (areas x n_freqs x subjects).
    """
    n_areas, n_time, n_subjs = data.shape

    # Time step based on sampling frequency
    d = 1 / fs if fs is not None else 1.0
    freqs = np.fft.rfftfreq(n_time, d=d)
    n_freqs = len(freqs)

    # Initialize output
    fft_area = np.zeros((n_areas, n_freqs, n_subjs))

    # Loop through each area and subject
    for area in range(n_areas):
        for subj in range(n_subjs):
            # Compute FFT for this area and subject
            fft_vals = np.fft.rfft(data[area, :, subj])
            fft_area[area, :, subj] = np.abs(fft_vals)

    return freqs, fft_area

