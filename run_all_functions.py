import os
from scipy.io import loadmat, savemat


# Own functions below
from Method_Functions.run_LFA import run_LFA

def run_all_methods(data_ts, n_lag=10, exp_var_lim=99, output_dir='.'):
    # Run LFA
    # Creating specific folder for LFA
    LFA_result_path = os.path.join(output_dir, "LFA")
    if not os.path.exists(LFA_result_path): 
        os.makedirs(LFA_result_path)

    lmse, msd = run_LFA(data_ts, n_lag, exp_var_lim)
    savemat(os.path.join(LFA_result_path, "lmse_results.mat"), {"lmse": lmse})
    savemat(os.path.join(LFA_result_path, "msd_results.mat"), {"msd": msd})
    
    print("All methods have been run and results saved to:", output_dir)

# Load data from .mat
mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Example_Data/cort_ts1c.mat")

# Remove MATLAB-specific metadata keys (those starting with '__')
filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]

# Ensure there is exactly one key left
if len(filtered_keys) != 1:
    raise ValueError(f"Expected one data key/variable in .mat file, but found {len(filtered_keys)}: {filtered_keys}")

# Extract the only key and assign its values to `data`
data = mat_data[filtered_keys[0]]

n_lag = 3
exp_var_lim = 0.95

# Add "/Output" to any path you have to let it create a new folder for saving results
output_dir = "/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Output"

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

run_all_methods(data_ts= data, n_lag=n_lag, exp_var_lim= exp_var_lim, output_dir= output_dir)
