import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf as sm_acf

def exponential_decay(x, a, b, c):
    return a - b * np.exp(-c * x)

def run_Timescale(data, mode="per_subject", subject_index=0, region_index=None):
    """
    Compute timescales from (regions, time, subjects) EEG/fMRI data.

    Parameters:
    - data: np.ndarray, shape (regions, time, subjects)
    - mode: "per_subject", "per_region", or "individual"
    - subject_index: index used for mode="individual"
    - region_index: if set, compute timescale for a specific region only

    Returns:
    - np.ndarray:
        - shape (n_subjects,) if mode="per_subject"
        - shape (n_regions,) if mode="per_region"
        - shape (n_regions,) if mode="individual"
    """
    R, T, S = data.shape
    timescales = np.full((R, S), np.nan)  # store all results

    for s in range(S):
        subj_data = data[:, :, s].T  # (time, regions)
        acf_all = np.zeros((T, R))
        for r in range(R):
            acf_r = sm_acf(subj_data[:, r], nlags=T-1, fft=False)
            acf_all[:, r] = acf_r

        if region_index is not None:
            if region_index < 0 or region_index >= R:
                raise ValueError(f"Region index {region_index} out of bounds (0 to {R-1})")
            acf_used = acf_all[:, region_index]
            x, y = _get_decay_fit(acf_used)
            if x is not None:
                try:
                    popt, _ = curve_fit(exponential_decay, x, y, p0=[1, 1, 1], maxfev=10000)
                    timescales[region_index, s] = popt[2]
                except Exception:
                    timescales[region_index, s] = np.nan
        else:
            for r in range(R):
                x, y = _get_decay_fit(acf_all[:, r])
                if x is not None:
                    try:
                        popt, _ = curve_fit(exponential_decay, x, y, p0=[1, 1, 1], maxfev=10000)
                        timescales[r, s] = popt[2]
                    except Exception:
                        timescales[r, s] = np.nan

    # Aggregate based on mode
    if mode == "per_subject":
        return np.nanmean(timescales, axis=0)  # → (subjects,)
    elif mode == "per_region":
        return np.nanmean(timescales, axis=1)  # → (regions,)
    elif mode == "individual":
        if subject_index < 0 or subject_index >= S:
            raise ValueError(f"Invalid subject_index {subject_index} for {S} subjects.")
        return timescales[:, subject_index]     # → (regions,)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
def _get_decay_fit(acf_signal):
    grad_acf = np.gradient(acf_signal)
    positive_grad_indices = np.where(grad_acf >= 0)[0]
    x_max = positive_grad_indices[0] if positive_grad_indices.size > 0 else len(acf_signal)
    x = np.arange(x_max)
    y = acf_signal[:x_max]
    return (x, y) if len(x) >= 3 else (None, None)

from scipy.io import loadmat, savemat
import os

if __name__ == "__main__": 
    mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Shine_Lab_Code/Example_Data/cort_ts1c.mat")
    filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    if len(filtered_keys) != 1:
        raise ValueError(f"Expected one data key/variable in .mat file, but found {len(filtered_keys)}: {filtered_keys}")
    data = mat_data[filtered_keys[0]]

    output_dir = "/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Output"
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    import time
    start_time = time.time()
    timescales = run_Timescale(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")
