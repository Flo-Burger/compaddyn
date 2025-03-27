import numpy as np

def run_LFA_with_DMD(data_ts, n_lag=10, exp_var_lim=99, delta_t=1.0):
    """
    Linear Forecastability Analysis (LFA) with DMD.

    Returns:
        lmse (np.ndarray): Forecast MSE
        msd (np.ndarray): Mean squared displacement
        e_vecs (list): Sorted DMD modes
        lambdas (list): Sorted continuous-time DMD eigenvalues (complex)
    """
    n_vars, n_time, n_subjs = data_ts.shape

    lmse = np.zeros((n_time - n_lag, n_lag, n_subjs))
    msd = np.zeros((n_time - n_lag, n_lag, n_subjs))
    e_vecs = []
    lambdas = []

    for subj in range(n_subjs):
        subj_ts = data_ts[:, :, subj]
        X = subj_ts[:, :-1]
        Y = subj_ts[:, 1:]

        # SVD of X
        U, S_full, Vt = np.linalg.svd(X, full_matrices=False)

        # Truncate based on explained variance
        exp_var = 100 * (S_full ** 2) / np.sum(S_full ** 2)
        accum_exp_var = np.cumsum(exp_var)
        pcs_mask = accum_exp_var > exp_var_lim
        n_pcs = np.where(pcs_mask)[0][0] + 1 if np.any(pcs_mask) else len(S_full)

        U_r = U[:, :n_pcs]
        S_r = np.diag(S_full[:n_pcs])
        V_r = Vt[:n_pcs, :]
        V = V_r.T
        X_svd = V @ S_r  # time x n_pcs

        # Estimate linear propagator
        A_tilde = U_r.T @ Y @ V @ np.linalg.inv(S_r)

        # Eigen-decomposition of A_tilde
        eigvals, eigvecs = np.linalg.eig(A_tilde)

        # Compute continuous-time eigenvalues on principal branch
        lambda_cont = (np.log(np.abs(eigvals)) + 1j * np.angle(eigvals)) / delta_t

        # Compute DMD modes
        mode_matrix = Y @ V @ np.linalg.inv(S_r) @ eigvecs

        # Sort everything by |eigenvalue| (modulus)
        sort_idx = np.argsort(-np.abs(eigvals))  # descending
        eigvecs_sorted = eigvecs[:, sort_idx]
        lambda_cont_sorted = lambda_cont[sort_idx]
        mode_matrix_sorted = mode_matrix[:, sort_idx]

        lambdas.append(lambda_cont_sorted)
        e_vecs.append(mode_matrix_sorted)

        # Forecasting
        for ss in range(n_time - n_lag):
            X_plus = np.zeros((n_lag + 1, n_pcs))
            X_plus[0, :] = X_svd[ss, :]

            for ll in range(1, n_lag + 1):
                X_plus[ll, :] = A_tilde @ X_plus[ll - 1, :]
                lmse[ss, ll - 1, subj] = np.mean((X_plus[ll - 1, :] - X_svd[ss + ll - 1, :]) ** 2)
                msd[ss, ll - 1, subj] = np.mean((X_svd[ss, :] - X_svd[ss + ll - 1, :]) ** 2)

    return lmse, msd, e_vecs, lambdas