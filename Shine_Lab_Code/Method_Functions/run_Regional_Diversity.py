import numpy as np
from scipy.io import loadmat, savemat
import numpy as np

def run_Regional_Diversity(data):
    """
    Computes the global regional diversity for each subject by
    calculating the standard deviation of the upper triangle of the 
    subject's functional connectivity (correlation) matrix.

    Parameters:
        data (np.ndarray): 3D array (timepoints, regions, subjects)

    Returns:
        diversity (np.ndarray): 1D array (n_subjects,)
                                One value per subject (overall FC variability)
    """
    T, R, N = data.shape
    diversity = np.zeros(N)

    for subj in range(N):
        subj_data = data[:, :, subj]  # shape: (time x regions)

        # Correlation matrix (regions x regions)
        fc = np.corrcoef(subj_data.T)

        # Remove self-connections
        np.fill_diagonal(fc, 0)

        # Extract upper triangle (excluding diagonal)
        upper_triangle = fc[np.triu_indices(R, k=1)]

        # Compute std of upper triangle
        diversity[subj] = np.std(upper_triangle)

    return diversity  # shape: (n_subjects,)