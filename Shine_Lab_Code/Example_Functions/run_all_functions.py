import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Own functions below
from Method_Functions import run_LFA
from Method_Functions import run_ICG, run_ICG_torch
from Method_Functions import run_fft_global, run_fft_per_area

def run_all_methods(data, output_dir='.', overwrite=True):

    # Parameters 
    fs = 1 #fs for FFT, change if sample rate is known

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

    # --- Run Global FFT (per subject) ---
    print("Running Global FFT")
    FFT_global_result_path = os.path.join(output_dir, "FFT", "Global")
    os.makedirs(FFT_global_result_path, exist_ok=True)
    freqs, fft_global = run_fft_global(data, fs=fs)
    # Save the frequency axis in the global FFT folder.
    savemat(os.path.join(FFT_global_result_path, "freqs.mat"), {"freqs": freqs})
    
    for subj in range(n_subjs):
        subj_folder = os.path.join(FFT_global_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
        # Save one global FFT file per subject (one row)
        subject_fft = fft_global[subj, :]  # shape: (n_freqs,)
        subject_fft_path = os.path.join(subj_folder, f"{subj+1}_global_fft.mat")
        savemat(subject_fft_path, {"global_fft": subject_fft})
        # Plot the global FFT for this subject
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, subject_fft)
        plt.xlabel("Frequency (Hz)" if fs is not None else "Frequency (cycles/sample)")
        plt.ylabel("Magnitude")
        plt.title(f"Global FFT Spectrum - Subject {subj+1}")
        plt.grid(True)
        plt.savefig(os.path.join(subj_folder, "global_fft.png"))
        plt.close()

    # --- Run Per-Area FFT (per subject) ---
    print("Running Per-Area FFT")
    FFT_area_result_path = os.path.join(output_dir, "FFT", "PerArea")
    os.makedirs(FFT_area_result_path, exist_ok=True)
    freqs_area, fft_area = run_fft_per_area(data, fs=fs)
    # Save the frequency axis in the per-area FFT folder.
    savemat(os.path.join(FFT_area_result_path, "freqs.mat"), {"freqs": freqs_area})
    
    for subj in range(n_subjs):
        subj_folder = os.path.join(FFT_area_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
        # Save one per-area FFT file per subject (matrix: areas x n_freqs)
        subject_fft_area = fft_area[:, :, subj]
        subject_fft_area_path = os.path.join(subj_folder, f"{subj + 1}_per_area_fft.mat")
        savemat(subject_fft_area_path, {"per_area_fft": subject_fft_area})
        # No plotting for per-area FFT to avoid redundancy.
    
    print("All methods have been run and results saved to:", output_dir)

# # Load data from .mat
# mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/tests/cort_ts1c_short.mat")

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
