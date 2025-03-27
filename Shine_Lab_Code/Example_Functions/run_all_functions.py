import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np

# Own functions below (assume these are correctly defined in your module)
from Shine_Lab_Code.Method_Functions import run_LFA
from Shine_Lab_Code.Method_Functions import run_ICG
from Shine_Lab_Code.Method_Functions import run_fft_global, run_fft_per_area
from Shine_Lab_Code.Method_Functions import run_Energy_Landscape  
from Shine_Lab_Code.Method_Functions import run_LFA_with_DMD
from Shine_Lab_Code.Method_Functions import run_Regional_Diversity

def run_all_methods(data, output_dir='.', overwrite=True):
    # Parameters 
    fs = 1  # fs for FFT, change if sample rate is known
    n_vars, n_time, n_subjs = data.shape

    # ----------------- Run LFA -----------------
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

    # ----------------- Run ICG -----------------
    print("Running ICG")
    ICG_result_path = os.path.join(output_dir, "ICG")
    os.makedirs(ICG_result_path, exist_ok=True)

    all_activityICG, all_out_pairs = run_ICG(data)
    for subj in range(n_subjs):
        subj_folder = os.path.join(ICG_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
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

    # ----------------- Run Global FFT (per subject) -----------------
    print("Running Global FFT")
    FFT_global_result_path = os.path.join(output_dir, "FFT", "Global")
    os.makedirs(FFT_global_result_path, exist_ok=True)
    freqs, fft_global = run_fft_global(data, fs=fs)
    savemat(os.path.join(FFT_global_result_path, "freqs.mat"), {"freqs": freqs})
    for subj in range(n_subjs):
        subj_folder = os.path.join(FFT_global_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
        subject_fft = fft_global[subj, :]  # shape: (n_freqs,)
        subject_fft_path = os.path.join(subj_folder, f"{subj+1}_global_fft.mat")
        savemat(subject_fft_path, {"global_fft": subject_fft})
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, subject_fft)
        plt.xlabel("Frequency (Hz)" if fs is not None else "Frequency (cycles/sample)")
        plt.ylabel("Magnitude")
        plt.title(f"Global FFT Spectrum - Subject {subj+1}")
        plt.grid(True)
        plt.savefig(os.path.join(subj_folder, "global_fft.png"))
        plt.close()

    # ----------------- Run Per-Area FFT (per subject) -----------------
    print("Running Per-Area FFT")
    FFT_area_result_path = os.path.join(output_dir, "FFT", "PerArea")
    os.makedirs(FFT_area_result_path, exist_ok=True)
    freqs_area, fft_area = run_fft_per_area(data, fs=fs)
    savemat(os.path.join(FFT_area_result_path, "freqs.mat"), {"freqs": freqs_area})
    for subj in range(n_subjs):
        subj_folder = os.path.join(FFT_area_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
        subject_fft_area = fft_area[:, :, subj]
        subject_fft_area_path = os.path.join(subj_folder, f"{subj + 1}_per_area_fft.mat")
        savemat(subject_fft_area_path, {"per_area_fft": subject_fft_area})

    # ----------------- Run Energy Landscape Analysis -----------------
    print("Running Energy Landscape Analysis")
    EL_result_path = os.path.join(output_dir, "Energy_Landscape")
    os.makedirs(EL_result_path, exist_ok=True)
    nrgSig = run_Energy_Landscape(data, ndt=20)
    savemat(os.path.join(EL_result_path, "energy_landscape.mat"), {"nrgSig": nrgSig})
    for subj in range(n_subjs):
        subj_folder = os.path.join(EL_result_path, f"Subject_{subj+1}")
        os.makedirs(subj_folder, exist_ok=True)
        subject_nrgSig = nrgSig[:, :, subj]
        savemat(os.path.join(subj_folder, f"subject_{subj+1}_energy_landscape.mat"), {"subject_nrgSig": subject_nrgSig})
        plt.figure()
        plt.imshow(subject_nrgSig, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('MSD Divisions')
        plt.ylabel('Lag (dt)')
        plt.title(f'Energy Landscape - Subject {subj+1}')
        plt.savefig(os.path.join(subj_folder, f"subject_{subj+1}_energy_landscape.png"))
        plt.close()
    avg_nrgSig = np.mean(nrgSig, axis=2)
    plt.figure()
    plt.imshow(avg_nrgSig, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('MSD Divisions')
    plt.ylabel('Lag (dt)')
    plt.title('Average Energy Landscape Across Subjects')
    plt.savefig(os.path.join(EL_result_path, "average_energy_landscape.png"))
    plt.close()

    # ----------------- Run LFA with DMD -----------------
    # Now aggregate the outputs similar to the standard LFA:
    print("Running LFA with DMD")
    LFA_DMD_result_path = os.path.join(output_dir, "LFA_with_DMD")
    os.makedirs(LFA_DMD_result_path, exist_ok=True)
    # Use parameters similar to MATLAB: n_lag=15, exp_var_lim=95, delta_t=0.5
    lmse_dmd, msd_dmd, e_vecs_dmd, lambdas_dmd = run_LFA_with_DMD(data, n_lag=15, exp_var_lim=95, delta_t=0.5)
    # Save aggregated outputs similar to LFA
    savemat(os.path.join(LFA_DMD_result_path, "lmse_dmd_results.mat"), {"lmse_dmd": lmse_dmd})
    savemat(os.path.join(LFA_DMD_result_path, "msd_dmd_results.mat"), {"msd_dmd": msd_dmd})

    # For lambdas and e_vecs (lists of arrays), we convert them to object arrays
    # Create object arrays for lambdas_dmd and e_vecs_dmd
    lambdas_obj = np.empty((n_subjs,), dtype=object)
    e_vecs_obj = np.empty((n_subjs,), dtype=object)
    for subj in range(n_subjs):
        lambdas_obj[subj] = lambdas_dmd[subj]
        e_vecs_obj[subj] = e_vecs_dmd[subj]
    
    savemat(os.path.join(LFA_DMD_result_path, "lambdas_dmd_results.mat"), {"lambdas_dmd": lambdas_obj})
    savemat(os.path.join(LFA_DMD_result_path, "e_vecs_dmd_results.mat"), {"e_vecs_dmd": e_vecs_obj})

    # ----------------- Run Regional Diversity -----------------
    print("Running Regional Diversity")
    Regional_Diversity_result_path = os.path.join(output_dir, "Regional_Diversity")
    os.makedirs(Regional_Diversity_result_path, exist_ok=True)
    diversity = run_Regional_Diversity(data)
    savemat(os.path.join(Regional_Diversity_result_path, "regional_diversity.mat"), {"diversity": diversity})
    
    print("All methods have been run and results saved to:", output_dir)

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
    run_all_methods(data=data, output_dir=output_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")
