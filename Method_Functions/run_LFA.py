import numpy as np

def run_LFA(data_ts, n_lag=10, exp_var_lim=99):
    n_vars, n_time, n_subjs = data_ts.shape
    lmse = np.zeros((n_time - n_lag, n_lag, n_subjs))
    msd = np.zeros((n_time - n_lag, n_lag, n_subjs))

    for subj in range(n_subjs):
        subj_ts = data_ts[:, :, subj]
        X = subj_ts[:, :-1]
        Y = subj_ts[:, 1:]

        U, S_full, Vt = np.linalg.svd(X, full_matrices=False)
        exp_var = 100.0 * (S_full**2) / np.sum(S_full**2)
        accum_exp_var = np.cumsum(exp_var)
        pcs_mask = accum_exp_var > exp_var_lim
        if np.any(pcs_mask):
            n_pcs = np.where(pcs_mask)[0][0] + 1
        else:
            n_pcs = len(S_full)

        U = U[:, :n_pcs]
        S = S_full[:n_pcs]
        Vt = Vt[:n_pcs, :]
        S_mat = np.diag(S)

        V = Vt.T
        X_svd = V @ S_mat

        A_tilde = U.T @ Y @ V @ np.linalg.inv(S_mat)

        for ss in range(n_time - n_lag):
            X_plus = np.zeros((n_lag+1, n_pcs))
            X_plus[0,:] = X_svd[ss,:]
            
            for ll in range(1, n_lag+1):
                X_plus[ll,:] = A_tilde @ X_plus[ll-1,:]

                lmse[ss, ll-1, subj] = np.mean((X_plus[ll-1,:] - X_svd[ss+ll-1,:])**2)
                msd[ss, ll-1, subj] = np.mean((X_svd[ss,:] - X_svd[ss+ll-1,:])**2)

    return lmse, msd

# import numpy as np
# from scipy.io import loadmat

# # Load the data produced by the Python code that created data_for_matlab.mat
# mat_data = loadmat('data_for_matlab.mat')
# data_ts = mat_data['data_ts']
# n_lag = int(mat_data['n_lag'][0][0])
# exp_var_lim = float(mat_data['exp_var_lim'][0][0])

# # Assuming run_LFA is already defined in Python as per the earlier translation
# # run_LFA(data_ts, n_lag, exp_var_lim) should run without changes

# lmse_py, msd_py = run_LFA(data_ts, n_lag, exp_var_lim)

# print("Python LMSE shape:", lmse_py.shape)
# print("Python MSD shape:", msd_py.shape)

# # Print a small portion to compare with MATLAB output
# # print("Sample Python LMSE values:\n", lmse_py[:, :, 0])
# # print("Sample Python MSD values:\n", msd_py[:, :, 0])

# print(msd_py[0, :, 0])


