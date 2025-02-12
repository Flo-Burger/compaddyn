import os
from scipy.io import loadmat, savemat

# Own functions below
from Method_Functions import run_LFA
from Method_Functions import run_ICG, run_ICG_torch

def run_all_methods(data, output_dir='.', overwrite=True):

    n_vars, n_time, n_subjs = data.shape

    # Run LFA
    print("Running LFA")
    LFA_result_path = os.path.join(output_dir, "LFA")
    os.makedirs(LFA_result_path, exist_ok=True)

    # Check if files already exist
    lmse_path = os.path.join(LFA_result_path, "lmse_results.mat")
    msd_path = os.path.join(LFA_result_path, "msd_results.mat")

    if not overwrite and os.path.exists(lmse_path) and os.path.exists(msd_path):
        print("Skipping LFA, results already exist.")
    else:
        lmse, msd = run_LFA(data, n_lag=3, exp_var_lim=0.95)
        savemat(lmse_path, {"lmse": lmse})
        savemat(msd_path, {"msd": msd})

    # Run ICG
    print("Running ICG")
    ICG_result_path = os.path.join(output_dir, "ICG")
    os.makedirs(ICG_result_path, exist_ok=True)

    all_activityICG, all_out_pairs = run_ICG(data)

    for subj in range(n_subjs):
        subj_folder = os.path.join(ICG_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)

        # Save activityICG results
        subject_activity = all_activityICG[subj]
        subject_pairs = all_out_pairs[subj]

        for level, activity in enumerate(subject_activity):
            activity_path = os.path.join(subj_folder, f"activity_level_{level+1}.mat")
            if not overwrite and os.path.exists(activity_path):
                print(f"Skipping {activity_path}, already exists.")
            elif activity is not None:
                savemat(activity_path, {f"activity_{level+1}": activity})

        for level, pairs in enumerate(subject_pairs):
            pairs_path = os.path.join(subj_folder, f"pairs_level_{level+1}.mat")
            if not overwrite and os.path.exists(pairs_path):
                print(f"Skipping {pairs_path}, already exists.")
            elif pairs is not None:
                savemat(pairs_path, {f"pairs_{level+1}": pairs})

    print("All methods have been run and results saved to:", output_dir)

# # Load data from .mat
# mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/fMRI_like_data.mat")

# # Remove MATLAB-specific metadata keys (those starting with '__')
# filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]

# # Ensure there is exactly one key left
# if len(filtered_keys) != 1:
#     raise ValueError(f"Expected one data key/variable in .mat file, but found {len(filtered_keys)}: {filtered_keys}")

# # Extract the only key and assign its values to `data`
# data = mat_data[filtered_keys[0]]

# # Add "/Output" to any path you have to let it create a new folder for saving results
# output_dir = "/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Output"

# if not os.path.exists(output_dir): 
#     os.makedirs(output_dir)

# import time

# # Start the timer
# start_time = time.time()

# # Run the function
# run_all_methods(data=data, output_dir=output_dir)

# # End the timer
# end_time = time.time()

# # Print elapsed time
# elapsed_time = end_time - start_time
# print(f"Execution Time: {elapsed_time:.4f} seconds")
