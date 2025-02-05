import os
from scipy.io import loadmat, savemat

# Own functions below
from Method_Functions import run_LFA
from Method_Functions import run_ICG

def run_all_methods(data, output_dir='.'):

    n_vars, n_time, n_subjs = data.shape

    # Run LFA
    # Creating specific folder for LFA
    print("Running LFA")
    LFA_result_path = os.path.join(output_dir, "LFA")
    if not os.path.exists(LFA_result_path): 
        os.makedirs(LFA_result_path)

    # Parameters for LFA can be changed here: 
    # lmse, msd = run_LFA(data, n_lag = 3, exp_var_lim = 0.95)

    # savemat(os.path.join(LFA_result_path, "lmse_results.mat"), {"lmse": lmse})
    # savemat(os.path.join(LFA_result_path, "msd_results.mat"), {"msd": msd})

    # Run THOI 
    # 
    # THOI_result_path = os.path.join(output_dir, "THOI")
    # if not os.path.exists(THOI_result_path): 
    #     os.makedirs(THOI_result_path)

    # o = run_THOI(data)

    # Run ICG
    # Creating Specific folder for ICG
    print("Running ICG")

    ICG_result_path = os.path.join(output_dir, "ICG")
    os.makedirs(ICG_result_path, exist_ok=True)

    # No parameters for ICG currently
    all_activityICG, all_out_pairs = run_ICG(data)

    # Bit more complicated folder structure since ICG should be run per subject
    for subj in range(n_subjs):
        subj_folder = os.path.join(ICG_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)

        # Save each level of activityICG
        subject_activity = all_activityICG[subj]
        subject_pairs = all_out_pairs[subj]

        for level, activity in enumerate(subject_activity):
            if activity is not None:
                savemat(os.path.join(subj_folder, f"activity_level_{level+1}.mat"), {f"activity_{level+1}": activity})

        for level, pairs in enumerate(subject_pairs):
            if pairs is not None:
                savemat(os.path.join(subj_folder, f"pairs_level_{level+1}.mat"), {f"pairs_{level+1}": pairs})

    print("All methods have been run and results saved to:", output_dir)

# Load data from .mat
mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/fMRI_like_data.mat")

# Remove MATLAB-specific metadata keys (those starting with '__')
filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]

# Ensure there is exactly one key left
if len(filtered_keys) != 1:
    raise ValueError(f"Expected one data key/variable in .mat file, but found {len(filtered_keys)}: {filtered_keys}")

# Extract the only key and assign its values to `data`
data = mat_data[filtered_keys[0]]

# Add "/Output" to any path you have to let it create a new folder for saving results
output_dir = "/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Output"

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

run_all_methods(data = data, output_dir= output_dir)
