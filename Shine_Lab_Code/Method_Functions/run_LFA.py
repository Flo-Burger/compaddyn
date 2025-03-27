"""
Linear Forecastability Analysis (LFA) using SVD and linear dynamics.

Author: Florian Burger
Date: 2025-03-26
Version: 1.0

Description:
This function implements a linear forecasting model in PCA-reduced space, estimating
how well future neural activity can be predicted over different time lags.

It computes two measures:
- Linear Mean Squared Error (LMSE): how well the model predicts each lag.
- Mean Squared Displacement (MSD): how much the representation changes over time.

"""

import numpy as np

def run_LFA(data_ts, n_lag=10, exp_var_lim=99):
    """
    Run Linear Forecastability Analysis (LFA) on multivariate time series data.

    Parameters:
        data_ts (np.ndarray): Neural timeseries data (variables x timepoints x subjects)
        n_lag (int): Maximum lag to predict into the future
        exp_var_lim (float): Variance threshold (%) for selecting PCA components

    Returns:
        lmse (np.ndarray): Linear MSE (time - lag) x lag x subjects
        msd (np.ndarray): Mean squared displacement over time
    """
    n_vars, n_time, n_subjs = data_ts.shape

    # Preallocate output arrays
    lmse = np.zeros((n_time - n_lag, n_lag, n_subjs))
    msd = np.zeros((n_time - n_lag, n_lag, n_subjs))

    # Loop through subjects
    for subj in range(n_subjs):
        subj_ts = data_ts[:, :, subj]

        # Prepare input (X) and output (Y) matrices
        X = subj_ts[:, :-1]
        Y = subj_ts[:, 1:]

        # Step 1: Perform SVD on X
        U, S_full, Vt = np.linalg.svd(X, full_matrices=False)

        # Step 2: Select principal components explaining desired variance
        exp_var = 100.0 * (S_full**2) / np.sum(S_full**2)
        accum_exp_var = np.cumsum(exp_var)
        pcs_mask = accum_exp_var > exp_var_lim

        if np.any(pcs_mask):
            n_pcs = np.where(pcs_mask)[0][0] + 1
        else:
            n_pcs = len(S_full)

        # Reduce to selected principal components
        U = U[:, :n_pcs]
        S = S_full[:n_pcs]
        Vt = Vt[:n_pcs, :]
        S_mat = np.diag(S)

        V = Vt.T
        X_svd = V @ S_mat  # Compressed representation

        # Step 3: Estimate linear dynamics A_tilde
        A_tilde = U.T @ Y @ V @ np.linalg.inv(S_mat)

        # Step 4: Forecast future activity and evaluate LMSE and MSD
        for ss in range(n_time - n_lag):
            # Initialize first point
            X_plus = np.zeros((n_lag + 1, n_pcs))
            X_plus[0, :] = X_svd[ss, :]

            for ll in range(1, n_lag + 1):
                # Predict next point
                X_plus[ll, :] = A_tilde @ X_plus[ll - 1, :]

                # Linear prediction error (LMSE)
                lmse[ss, ll - 1, subj] = np.mean(
                    (X_plus[ll - 1, :] - X_svd[ss + ll - 1, :]) ** 2
                )

                # Displacement from original state (MSD)
                msd[ss, ll - 1, subj] = np.mean(
                    (X_svd[ss, :] - X_svd[ss + ll - 1, :]) ** 2
                )

    return lmse, msd


