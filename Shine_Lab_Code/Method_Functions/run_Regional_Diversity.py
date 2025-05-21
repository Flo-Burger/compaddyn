import numpy as np

def run_Regional_Diversity(data, mode="per_region", subject_index=0):
    """
    Computes regional diversity for data shaped (regions, time, subjects).

    Parameters:
        data (np.ndarray): shape (regions, time, subjects)
        mode (str): "per_region", "per_subject", or "individual"
        subject_index (int): used only for mode="individual"

    Returns:
        np.ndarray: shape depends on mode:
            - (n_regions,) if mode="per_region" or "individual"
            - (n_subjects,) if mode="per_subject"
    """
    R, T, N = data.shape
    rd_matrix = np.zeros((R, N))  # (regions x subjects)

    for subj in range(N):
        subj_data = data[:, :, subj].T  # shape: (time x regions)
        fc = np.corrcoef(subj_data.T)   # shape: (regions x regions)
        np.fill_diagonal(fc, np.nan)
        rd_matrix[:, subj] = np.nanstd(fc, axis=1)

    if mode == "per_region":
        return rd_matrix.mean(axis=1)  # (n_regions,)
    elif mode == "per_subject":
        return rd_matrix.mean(axis=0)  # (n_subjects,)
    elif mode == "individual":
        if subject_index < 0 or subject_index >= N:
            raise ValueError(f"Invalid subject_index {subject_index} for N={N}")
        return rd_matrix[:, subject_index]  # (n_regions,)
    else:
        raise ValueError(f"Invalid mode: {mode}")
