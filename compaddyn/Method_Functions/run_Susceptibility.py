import numpy as np
from scipy.stats import zscore
from scipy.io import loadmat, savemat

def run_Susceptibility(data):
    """
    Compute susceptibility for time x region data.

    Parameters:
    -----------
    data : ndarray
        Time x region matrix

    Returns:
    --------
    sus : float
        Susceptibility value
    """

    sus = np.zeros(data.shape[2])  # Initialize susceptibility array

    for subj in range(data.shape[2]):        
        data_subj = data[:, :, subj]  # Extract time x region for subject
        
        Z = zscore(data_subj, axis=0, ddof=0)  # Z-score across time for each region
        N = data.shape[1]  # Number of regions
        density = np.sum(Z > 0, axis=1) / N  # Fraction of regions above mean at each time point
        sus_subject = (np.mean(density ** 2) - np.mean(density) ** 2) / np.mean(density)  # Normalized variance
        sus[subj] = sus_subject  # Store susceptibility for subject

    return sus


# data = loadmat('/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Shine_Lab_Code/Example_Data/cort_ts1c.mat')

# cort_ts1c = data['cort_ts1c']

# susceptibilities = run_Susceptibility(cort_ts1c)

# print(susceptibilities)
