import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Import analysis functions
from Shine_Lab_Code.Method_Functions import run_LFA
from Shine_Lab_Code.Method_Functions import run_ICG
from Shine_Lab_Code.Method_Functions import run_fft_global, run_fft_per_area
from Shine_Lab_Code.Method_Functions import run_Energy_Landscape  
from Shine_Lab_Code.Method_Functions import run_LFA_with_DMD
from Shine_Lab_Code.Method_Functions import run_Regional_Diversity  # New Regional Diversity function

class Controller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main root until needed

        self.input_file = None
        self.output_dir = None
        
        # Flags for selected analyses
        self.run_lfa = True
        self.run_icg = True
        self.run_fft = True
        self.run_energy = True  
        self.run_lfa_dmd = True   # Flag for LFA with DMD
        self.run_regional_diversity = True  # Flag for Regional Diversity Analysis
        self.sampling_freq = None

    def launch_gui(self):
        self.page_one = PageOneWindow(self)
        self.root.mainloop()

    def show_page_two(self):
        if hasattr(self, 'page_one') and self.page_one:
            self.page_one.destroy()
            self.page_one = None
        self.page_two = PageTwoWindow(self)

    def show_results_page(self, message):
        if hasattr(self, 'page_two') and self.page_two:
            self.page_two.destroy()
            self.page_two = None
        self.results_page = ResultsWindow(self, message)

    def run_analysis(self):
        if not self.input_file or not os.path.isfile(self.input_file):
            messagebox.showerror("Error", "Invalid input file selected.")
            return
        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showerror("Error", "Invalid output directory selected.")
            return

        # Create a "Results" subfolder
        self.output_dir = os.path.join(self.output_dir, "Results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load .mat data
        mat_data = loadmat(self.input_file)
        filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        if len(filtered_keys) != 1:
            raise ValueError(f"Expected one data key, found {len(filtered_keys)}: {filtered_keys}")

        data_key = filtered_keys[0]
        data = mat_data[data_key]  # shape: [areas, time, subjects]
        n_vars, n_time, n_subjs = data.shape

        # --- Run LFA if selected ---
        if self.run_lfa:
            LFA_result_path = os.path.join(self.output_dir, "LFA")
            os.makedirs(LFA_result_path, exist_ok=True)
            lmse, msd = run_LFA(data, n_lag=3, exp_var_lim=0.95)
            savemat(os.path.join(LFA_result_path, "lmse_results.mat"), {"lmse": lmse})
            savemat(os.path.join(LFA_result_path, "msd_results.mat"), {"msd": msd})

        # --- Run ICG if selected ---
        if self.run_icg:
            ICG_result_path = os.path.join(self.output_dir, "ICG")
            os.makedirs(ICG_result_path, exist_ok=True)
            all_activityICG, all_outPairID = run_ICG(data)
            for subj in range(n_subjs):
                subj_folder = os.path.join(ICG_result_path, f"Subject_{subj+1}")
                os.makedirs(subj_folder, exist_ok=True)
                subj_activity = all_activityICG[subj]
                subj_pairs = all_outPairID[subj]
                for lvl, lvl_activity in enumerate(subj_activity):
                    if lvl_activity is not None:
                        savemat(
                            os.path.join(subj_folder, f"activity_level_{lvl+1}.mat"),
                            {f"activity_{lvl+1}": lvl_activity}
                        )
                for lvl, lvl_pairs in enumerate(subj_pairs):
                    if lvl_pairs is not None:
                        savemat(
                            os.path.join(subj_folder, f"pairs_level_{lvl+1}.mat"),
                            {f"pairs_{lvl+1}": lvl_pairs}
                        )

        # --- Run FFT if selected ---
        if self.run_fft:
            fs = self.sampling_freq  # Could be None or a float
            fft_path = os.path.join(self.output_dir, "FFT")
            os.makedirs(fft_path, exist_ok=True)
            # Global FFT
            global_path = os.path.join(fft_path, "Global")
            os.makedirs(global_path, exist_ok=True)
            freqs, fft_global = run_fft_global(data, fs=fs)
            savemat(os.path.join(global_path, "freqs.mat"), {"freqs": freqs})
            for subj in range(n_subjs):
                subj_folder = os.path.join(global_path, f"Subject_{subj+1}")
                os.makedirs(subj_folder, exist_ok=True)
                subj_fft = fft_global[subj, :]
                savemat(os.path.join(subj_folder, "global_fft.mat"), {"global_fft": subj_fft})
                plt.figure()
                plt.plot(freqs, subj_fft)
                plt.xlabel("Frequency (Hz)" if fs else "Frequency (cycles/sample)")
                plt.ylabel("Magnitude")
                plt.title(f"Global FFT - Subject {subj+1}")
                plt.grid(True)
                plt.savefig(os.path.join(subj_folder, "global_fft.png"))
                plt.close()
            # Per-Area FFT
            per_area_path = os.path.join(fft_path, "PerArea")
            os.makedirs(per_area_path, exist_ok=True)
            freqs_area, fft_area = run_fft_per_area(data, fs=fs)
            savemat(os.path.join(per_area_path, "freqs.mat"), {"freqs": freqs_area})
            for subj in range(n_subjs):
                subj_folder = os.path.join(per_area_path, f"Subject_{subj+1}")
                os.makedirs(subj_folder, exist_ok=True)
                subj_fft_area = fft_area[:, :, subj]
                savemat(os.path.join(subj_folder, "per_area_fft.mat"), {"per_area_fft": subj_fft_area})

        # --- Run Energy Landscape Analysis if selected ---
        if self.run_energy:
            nrgSig = run_Energy_Landscape(data, ndt=20, ds=np.arange(0, 21, 1), bandwidth=1, ddof=1)
            EL_result_path = os.path.join(self.output_dir, "Energy_Landscape")
            os.makedirs(EL_result_path, exist_ok=True)
            savemat(os.path.join(EL_result_path, "energy_landscape.mat"), {"nrgSig": nrgSig})
            for subj in range(n_subjs):
                subj_folder = os.path.join(EL_result_path, f"Subject_{subj+1}")
                os.makedirs(subj_folder, exist_ok=True)
                subject_nrgSig = nrgSig[:, :, subj]
                savemat(os.path.join(subj_folder, f"subject_{subj+1}_energy_landscape.mat"),
                        {"subject_nrgSig": subject_nrgSig})
                plt.figure()
                plt.imshow(subject_nrgSig, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.xlabel('MSD Divisions')
                plt.ylabel('Lag (dt)')
                plt.title(f'Energy Landscape - Subject {subj+1}')
                plt.savefig(os.path.join(subj_folder, f"subject_{subj+1}_energy_landscape.png"))
                plt.close()
            avg_nrgSig = np.mean(nrgSig, axis=2)
            savemat(os.path.join(EL_result_path, "average_energy_landscape.mat"), {"avg_nrgSig": avg_nrgSig})
            plt.figure()
            plt.imshow(avg_nrgSig, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.xlabel('MSD Divisions')
            plt.ylabel('Lag (dt)')
            plt.title('Average Energy Landscape Across Subjects')
            plt.savefig(os.path.join(EL_result_path, "average_energy_landscape.png"))
            plt.close()
        
        # --- Run Regional Diversity Analysis if selected ---
        if self.run_regional_diversity:
            RD_result_path = os.path.join(self.output_dir, "Regional_Diversity")
            os.makedirs(RD_result_path, exist_ok=True)
            # Our run_Regional_Diversity function expects data as (time, regions, subjects).
            # Our data is currently (areas, time, subjects); so we transpose the first two dimensions.
            data_transposed = np.transpose(data, (1, 0, 2))
            regional_diversity = run_Regional_Diversity(data_transposed)
            savemat(os.path.join(RD_result_path, "regional_diversity.mat"), {"regional_diversity": regional_diversity})
            plt.figure(figsize=(8, 4))
            subjects = np.arange(1, n_subjs + 1)
            plt.bar(subjects, regional_diversity)
            plt.xlabel("Subject")
            plt.ylabel("Regional Diversity")
            plt.title("Global Regional Diversity Across Subjects")
            plt.savefig(os.path.join(RD_result_path, "regional_diversity.png"))
            plt.close()

        # --- Run LFA with DMD if selected ---
        if self.run_lfa_dmd:
            LFA_DMD_result_path = os.path.join(self.output_dir, "LFA_with_DMD")
            os.makedirs(LFA_DMD_result_path, exist_ok=True)
            # Using parameters similar to MATLAB: n_lag=15, exp_var_lim=95, delta_t=0.5
            lmse_dmd, msd_dmd, e_vecs_dmd, lambdas_dmd = run_LFA_with_DMD(data, n_lag=15, exp_var_lim=95, delta_t=0.5)
            savemat(os.path.join(LFA_DMD_result_path, "lmse_dmd_results.mat"), {"lmse_dmd": lmse_dmd})
            savemat(os.path.join(LFA_DMD_result_path, "msd_dmd_results.mat"), {"msd_dmd": msd_dmd})
            # For lambdas and e_vecs (lists of arrays), create object arrays
            lambdas_obj = np.empty((n_subjs,), dtype=object)
            e_vecs_obj = np.empty((n_subjs,), dtype=object)
            for subj in range(n_subjs):
                lambdas_obj[subj] = lambdas_dmd[subj]
                e_vecs_obj[subj] = e_vecs_dmd[subj]
            savemat(os.path.join(LFA_DMD_result_path, "lambdas_dmd_results.mat"), {"lambdas_dmd": lambdas_obj})
            savemat(os.path.join(LFA_DMD_result_path, "e_vecs_dmd_results.mat"), {"e_vecs_dmd": e_vecs_obj})
            
            # Optionally, create an aggregated plot of DMD eigenvalues across subjects:
            plt.figure(figsize=(8, 4))
            for subj in range(n_subjs):
                eigs = lambdas_dmd[subj]
                plt.scatter(np.real(eigs), np.imag(eigs), label=f"Subj {subj+1}", alpha=0.6)
            plt.xlabel("Real(λ)")
            plt.ylabel("Imag(λ)")
            plt.title("Aggregated DMD Eigenvalues Across Subjects")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(LFA_DMD_result_path, "aggregated_dmd_eigenvalues.png"))
            plt.close()

        # Show results page
        msg = f"Analysis completed!\nResults saved to: {self.output_dir}"
        self.show_results_page(msg)

class PageOneWindow(tk.Toplevel):
    """Page 1: Select input file and output directory."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Step 1: Select Input/Output")
        tk.Label(self, text="Select Input Data and Output Directory", font=("Arial", 14)).pack(pady=10)
        tk.Button(self, text="Select Input MAT File", command=self.select_file).pack(pady=5)
        tk.Button(self, text="Select Output Directory", command=self.select_directory).pack(pady=5)
        self.status_label = tk.Label(self, text="No file or directory selected.")
        self.status_label.pack(pady=10)
        tk.Button(self, text="Next", command=self.controller.show_page_two).pack(pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if path:
            self.controller.input_file = path
            self.update_status()

    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.controller.output_dir = directory
            self.update_status()

    def update_status(self):
        self.status_label.config(text=f"Input: {self.controller.input_file}\nOutput: {self.controller.output_dir}")

class PageTwoWindow(tk.Toplevel):
    """Page 2: Select analysis methods and run."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Step 2: Select Analysis Methods")
        tk.Label(self, text="Select Analysis Methods", font=("Arial", 14)).pack(pady=10)
        self.run_lfa_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run LFA Analysis", variable=self.run_lfa_var).pack(pady=5)
        self.run_icg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run ICG Analysis", variable=self.run_icg_var).pack(pady=5)
        self.run_fft_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run FFT Analysis", variable=self.run_fft_var).pack(pady=5)
        self.run_energy_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run Energy Landscape Analysis", variable=self.run_energy_var).pack(pady=5)
        self.run_lfa_dmd_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run LFA with DMD Analysis", variable=self.run_lfa_dmd_var).pack(pady=5)
        self.run_reg_div_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run Regional Diversity Analysis", variable=self.run_reg_div_var).pack(pady=5)
        tk.Label(self, text="Sampling frequency (Hz), leave blank for 1 sample/unit").pack(pady=5)
        self.fs_entry = tk.Entry(self)
        self.fs_entry.pack()
        tk.Button(self, text="Run Analysis", command=self.on_run_clicked).pack(pady=20)
        tk.Button(self, text="Back", command=self.on_back_clicked).pack(pady=10)

    def on_run_clicked(self):
        self.controller.run_lfa = self.run_lfa_var.get()
        self.controller.run_icg = self.run_icg_var.get()
        self.controller.run_fft = self.run_fft_var.get()
        self.controller.run_energy = self.run_energy_var.get()
        self.controller.run_lfa_dmd = self.run_lfa_dmd_var.get()
        self.controller.run_regional_diversity = self.run_reg_div_var.get()
        fs_text = self.fs_entry.get().strip()
        if fs_text:
            try:
                self.controller.sampling_freq = float(fs_text)
            except ValueError:
                messagebox.showwarning("Warning", "Invalid sampling frequency. Defaulting to 1.0.")
                self.controller.sampling_freq = None
        else:
            self.controller.sampling_freq = None
        self.controller.run_analysis()

    def on_back_clicked(self):
        self.destroy()
        self.controller.page_one = PageOneWindow(self.controller)

class ResultsWindow(tk.Toplevel):
    def __init__(self, controller, message):
        super().__init__()
        self.controller = controller
        self.title("Analysis Results")
        msg_widget = tk.Message(self, text=message, width=400, bg="white", fg="blue")
        msg_widget.pack(expand=True, fill="both", padx=20, pady=20)
        tk.Button(self, text="Back to Start", command=self.on_back_clicked).pack(pady=10)
        tk.Button(self, text="Exit", command=self.controller.root.destroy).pack(pady=10)
        
    def on_back_clicked(self):
        self.destroy()
        self.controller.page_one = PageOneWindow(self.controller)

def run_gui():
    """Launch the GUI."""
    app = Controller()
    app.launch_gui()

if __name__ == "__main__": 
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    run_gui()
