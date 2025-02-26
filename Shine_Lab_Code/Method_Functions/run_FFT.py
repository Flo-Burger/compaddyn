import numpy as np

def run_fft_global(data, fs=None):
    """
    Computes the FFT for each subject using the global signal,
    defined as the average signal across all areas for that subject.

    Assumes that the input data has shape:
        (areas x time x subjects)
    with the subject dimension as the last axis.

    Parameters:
      data (np.ndarray): 
          Input data with dimensions (areas, time, subjects).
      fs (float, optional): 
          Sampling frequency in Hz. If not provided, defaults to 1 
          (i.e., the frequencies will be in cycles per sample unit).

    Returns:
      freqs (np.ndarray): 
          1D array containing the frequency bins.
      fft_global (np.ndarray): 
          2D array of FFT magnitudes with shape (subjects x n_freqs).
          Each row corresponds to one subject.
    """

    n_areas, n_time, n_subjs = data.shape
    # Determine the time step: if fs is provided, d = 1/fs; otherwise, d = 1.
    d = 1/fs if fs is not None else 1.0
    
    # Compute frequency bins.
    freqs = np.fft.rfftfreq(n_time, d=d)
    n_freqs = len(freqs)
    
    fft_global = np.zeros((n_subjs, n_freqs))
    
    for subj in range(n_subjs):
        # Compute the global signal by averaging across areas for this subject.
        subject_signal = np.mean(data[:, :, subj], axis=0)  # shape: (time,)
        fft_vals = np.fft.rfft(subject_signal)
        fft_global[subj, :] = np.abs(fft_vals)
    
    return freqs, fft_global

def run_fft_per_area(data, fs=None):
    """
    Computes the FFT for each area separately for every subject.

    Assumes that the input data has shape:
        (areas x time x subjects)
    with the subject dimension as the last axis.

    Parameters:
      data (np.ndarray): 
          Input data with dimensions (areas, time, subjects).
      fs (float, optional): 
          Sampling frequency in Hz. If not provided, defaults to 1.

    Returns:
      freqs (np.ndarray): 
          1D array containing the frequency bins.
      fft_area (np.ndarray): 
          3D array of FFT magnitudes with shape (areas x n_freqs x subjects).
          Each slice along the first axis corresponds to an area.
    """
    n_areas, n_time, n_subjs = data.shape
    d = 1/fs if fs is not None else 1.0
    freqs = np.fft.rfftfreq(n_time, d=d)
    n_freqs = len(freqs)
    
    fft_area = np.zeros((n_areas, n_freqs, n_subjs))
    
    for area in range(n_areas):
        for subj in range(n_subjs):
            fft_vals = np.fft.rfft(data[area, :, subj])
            fft_area[area, :, subj] = np.abs(fft_vals)
    
    return freqs, fft_area

