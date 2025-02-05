import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
from scipy.io import loadmat, savemat

# Existing imports
from Method_Functions.run_LFA import run_LFA
from Method_Functions.run_ICG import run_ICG  # <-- NEW IMPORT for your ICG function

class Controller:
    """
    The Controller holds shared data (input_file, output_dir, etc.) and
    creates/destroys the pages (Toplevel windows).
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main root

        self.input_file = None
        self.output_dir = None
        
        # Flags for which analyses to run
        self.run_lfa = True
        self.run_icg = False  # <-- NEW flag for ICG

        # Start with Page One
        self.page_one = PageOneWindow(self)
        self.root.mainloop()

    def show_page_two(self):
        # Destroy PageOneWindow and create PageTwoWindow
        if hasattr(self, 'page_one') and self.page_one is not None:
            self.page_one.destroy()
            self.page_one = None
        self.page_two = PageTwoWindow(self)

    def show_results_page(self, message):
        # Destroy PageTwoWindow and create ResultsWindow
        if hasattr(self, 'page_two') and self.page_two is not None:
            self.page_two.destroy()
            self.page_two = None
        self.results_page = ResultsWindow(self, message)

    def run_analysis(self):
        """
        Perform the analysis logic, using run_LFA and run_ICG (optional),
        then move to results page.
        """
        if not self.input_file or not os.path.isfile(self.input_file):
            messagebox.showerror("Error", "Invalid input file selected.")
            return
        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showerror("Error", "Invalid output directory selected.")
            return

        # Create the main "Results" folder
        self.output_dir = os.path.join(self.output_dir, "Results")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load data from .mat
        mat_data = loadmat(self.input_file)
        filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        if len(filtered_keys) != 1:
            raise ValueError(f"Expected exactly 1 data key, found {len(filtered_keys)}: {filtered_keys}")

        data_key = filtered_keys[0]
        data = mat_data[data_key]  # shape: [variables, time, subjects]

        # ==================
        # 1) Run LFA (if checked)
        # ==================
        if self.run_lfa:

            LFA_result_path = os.path.join(self.output_dir, "LFA")
            if not os.path.exists(LFA_result_path):
                os.makedirs(LFA_result_path)

            lmse, msd = run_LFA(data, n_lag = 3, exp_var_lim = 0.95)

            # Save LMSE and MSD
            lmse_path = os.path.join(LFA_result_path, "lmse_results.mat")
            msd_path = os.path.join(LFA_result_path, "msd_results.mat")
            savemat(lmse_path, {"lmse": lmse})
            savemat(msd_path, {"msd": msd})

        # ==================
        # 2) Run ICG (if checked)
        # ==================
        if self.run_icg:
            # Call your multi-subject ICG function. 
            # It returns: all_activityICG, all_outPairID
            # shape: each is a list of length `n_subjs`
            ICG_result_path = os.path.join(self.output_dir, "ICG")
            if not os.path.exists(ICG_result_path):
                os.makedirs(ICG_result_path)

            all_activityICG, all_outPairID = run_ICG(data)

            # Now save them subject-by-subject
            n_vars, n_time, n_subjs = data.shape
            for subj in range(n_subjs):
                subj_folder = os.path.join(ICG_result_path, f"Subject_{subj+1}")
                os.makedirs(subj_folder, exist_ok=True)

                subj_activity = all_activityICG[subj]
                subj_pairs = all_outPairID[subj]

                # subj_activity and subj_pairs are both lists of length ICGsteps
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

        # Done! Show results
        msg = (
            "Analysis completed successfully!\n"
            f"Results saved to:\n{self.output_dir}"
        )
        self.show_results_page(msg)

class PageOneWindow(tk.Toplevel):
    """
    Step 1: Select input file and output directory.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Step 1: Select Input/Output")

        label = tk.Label(self, text="Select Input Data and Output Directory", font=("Arial", 14))
        label.pack(pady=10, padx=10)

        btn_file = tk.Button(self, text="Select Input MAT File", command=self.select_file)
        btn_file.pack(pady=5)

        btn_dir = tk.Button(self, text="Select Output Directory", command=self.select_directory)
        btn_dir.pack(pady=5)

        self.status_label = tk.Label(self, text="No file or directory selected.")
        self.status_label.pack(pady=10)

        next_btn = tk.Button(self, text="Next", command=self.controller.show_page_two)
        next_btn.pack(pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(
            title="Select Input .mat File",
            filetypes=[("MAT files", "*.mat")]
        )
        if path:
            self.controller.input_file = path
            self.update_status()

    def select_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.controller.output_dir = directory
            self.update_status()

    def update_status(self):
        msg = (
            f"Input File: {self.controller.input_file if self.controller.input_file else 'Not selected'}\n"
            f"Output Dir: {self.controller.output_dir if self.controller.output_dir else 'Not selected'}"
        )
        self.status_label.config(text=msg)


class PageTwoWindow(tk.Toplevel):
    """
    Step 2: Select analysis methods and run analysis.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Step 2: Analysis")

        label = tk.Label(self, text="Select Analysis Methods", font=("Arial", 14))
        label.pack(pady=10, padx=10)

        # Checkbutton for run_LFA
        self.run_lfa_var = tk.BooleanVar(value=True)
        cb_run_lfa = tk.Checkbutton(self, text="Run LFA Analysis", variable=self.run_lfa_var)
        cb_run_lfa.pack(pady=5)

        # Checkbutton for run_ICG
        self.run_icg_var = tk.BooleanVar(value=True)
        cb_run_icg = tk.Checkbutton(self, text="Run ICG Analysis", variable=self.run_icg_var)
        cb_run_icg.pack(pady=5)

        run_btn = tk.Button(self, text="Run Analysis", command=self.on_run_clicked)
        run_btn.pack(pady=20)

        back_btn = tk.Button(self, text="Back", command=self.on_back_clicked)
        back_btn.pack(pady=10)

    def on_run_clicked(self):
        # Update the controller with checkbutton states
        self.controller.run_lfa = self.run_lfa_var.get()
        self.controller.run_icg = self.run_icg_var.get()

        # Execute analysis
        self.controller.run_analysis()

    def on_back_clicked(self):
        # Destroy this window, re-show PageOne
        self.destroy()
        self.controller.page_two = None
        self.controller.page_one = PageOneWindow(self.controller)


class ResultsWindow(tk.Toplevel):
    """
    Results page: Show success message and allow going back or exit.
    """
    def __init__(self, controller, message):
        super().__init__()
        self.controller = controller
        self.title("Analysis Results")

        msg_label = tk.Label(self, text=message, font=("Arial", 10), fg="#00008b")
        msg_label.pack(pady=10)

        back_btn = tk.Button(self, text="Back to Start", command=self.on_back_clicked)
        back_btn.pack(pady=10)

        exit_btn = tk.Button(self, text="Exit", command=self.controller.root.destroy)
        exit_btn.pack(pady=10)

    def on_back_clicked(self):
        # Destroy this window, re-show PageOne
        self.destroy()
        self.controller.results_page = None
        self.controller.page_one = PageOneWindow(self.controller)

if __name__ == "__main__":
    Controller()

