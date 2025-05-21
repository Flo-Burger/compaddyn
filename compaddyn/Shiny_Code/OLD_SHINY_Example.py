from shiny import App, ui, reactive, render, run_app
import os
import webbrowser

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from Shine_Lab_Code.Method_Functions import (
    run_LFA, run_ICG, run_fft_global, run_fft_per_area,
    run_Energy_Landscape, run_LFA_with_DMD, run_Regional_Diversity,
    run_Timescale, run_BrainConnectivityToolbox
)

# --- UI ---
app_ui = ui.page_navbar(
    ui.nav_panel("Run Analysis",
        ui.page_fluid(
            ui.h2("1️⃣ Run Analysis"),

            ui.input_file("input_mat", "Upload .mat file", accept=".mat", multiple=False),

            ui.input_text("output_dir", "Output directory (absolute path)",
                          placeholder="/Users/you/Desktop/Results"),

            ui.input_checkbox_group(
                "analyses", "Select analyses to run",
                choices={
                    "lfa":          "LFA",
                    "icg":          "ICG",
                    "fft":          "FFT",
                    "energy":       "Energy Landscape",
                    "lfa_dmd":      "LFA with DMD",
                    "regional_div": "Regional Diversity",
                    "timescale":    "Timescale",
                    "bct":          "Brain Connectivity Toolbox"
                },
                selected=["lfa","icg","fft","energy","lfa_dmd","regional_div","timescale","bct"]
            ),

            ui.input_numeric("fs", "Sampling frequency (Hz), leave blank for default",
                             value=None, min=0, step=0.1),

            ui.input_action_button("run", "Run Analysis", class_="btn-primary"),

            ui.output_text_verbatim("status")
        )
    ),

    ui.nav_panel("Summary",
        ui.page_fluid(
            ui.h2("2️⃣ Results Summary"),

            ui.output_text("summary_status"),

            ui.h4("Average Energy Landscape"),
            ui.output_image("energy_image"),

            ui.h4("Regional Diversity"),
            ui.output_image("rd_image")
        )
    ),

    ui.nav_panel("Re-run Energy",
        ui.page_fluid(
            ui.h2("3️⃣ Re-run Energy Landscape"),

            ui.input_numeric("ndt", "Number of lags (ndt)", value=20, min=1),
            ui.input_numeric("bandwidth", "Kernel bandwidth", value=1.0, step=0.1, min=0.1),

            ui.input_action_button("rerun_energy", "Re-run Energy Landscape", class_="btn-warning"),

            ui.output_text_verbatim("rerun_status"),

            ui.h4("Updated Energy Landscape"),
            ui.output_image("energy_image_updated")
        )
    )
)

# --- Server ---
def server(input, output, session):
    # remember where the last full-analysis results were saved
    last_results_dir = reactive.Value(None)

    # --- Full analysis ---
    @reactive.Calc
    def do_analysis():
        _ = input.run()

        with reactive.isolate():
            files    = input.input_mat()
            out_dir  = input.output_dir().strip()
            selected = list(input.analyses())
            fs       = input.fs()

        if not files:
            return "⚠️ Please upload a .mat file."
        if not out_dir or not os.path.isabs(out_dir):
            return "⚠️ Enter a valid absolute output directory."

        results_dir = os.path.join(out_dir, "Results")
        os.makedirs(results_dir, exist_ok=True)
        last_results_dir.set(results_dir)

        mat_data = loadmat(files[0]["datapath"])
        keys = [k for k in mat_data if not k.startswith("__")]
        if len(keys) != 1:
            return f"❌ Expected one data variable, found {len(keys)}: {keys}"
        data = mat_data[keys[0]]
        n_areas, n_time, n_subj = data.shape

        # Run all selected methods
        if "lfa" in selected:
            p = os.path.join(results_dir, "LFA"); os.makedirs(p, exist_ok=True)
            lmse, msd = run_LFA(data, n_lag=3, exp_var_lim=0.95)
            savemat(os.path.join(p, "lmse.mat"), {"lmse": lmse})
            savemat(os.path.join(p, "msd.mat"), {"msd": msd})

        if "icg" in selected:
            p = os.path.join(results_dir, "ICG"); os.makedirs(p, exist_ok=True)
            all_act, all_pairs = run_ICG(data)
            for subj in range(n_subj):
                sp = os.path.join(p, f"Subject_{subj+1}"); os.makedirs(sp, exist_ok=True)
                for lvl, act in enumerate(all_act[subj], start=1):
                    if act is not None:
                        savemat(os.path.join(sp, f"activity_lvl{lvl}.mat"), {f"act{lvl}": act})
                for lvl, pairs in enumerate(all_pairs[subj], start=1):
                    if pairs is not None:
                        savemat(os.path.join(sp, f"pairs_lvl{lvl}.mat"), {f"pairs{lvl}": pairs})

        if "fft" in selected:
            p = os.path.join(results_dir, "FFT"); os.makedirs(p, exist_ok=True)
            pg = os.path.join(p, "Global"); os.makedirs(pg, exist_ok=True)
            freqs, fftg = run_fft_global(data, fs=fs)
            savemat(os.path.join(pg, "freqs.mat"), {"freqs": freqs})
            for subj in range(n_subj):
                sp = os.path.join(pg, f"Subject_{subj+1}"); os.makedirs(sp, exist_ok=True)
                savemat(os.path.join(sp, "fft.mat"), {"fft": fftg[subj]})
            pa = os.path.join(p, "PerArea"); os.makedirs(pa, exist_ok=True)
            freqa, ffta = run_fft_per_area(data, fs=fs)
            savemat(os.path.join(pa, "freqs.mat"), {"freqs": freqa})
            for subj in range(n_subj):
                savemat(os.path.join(pa, f"subj_{subj+1}_per_area.mat"),
                        {"per_area": ffta[:, :, subj]})

        if "energy" in selected:
            p = os.path.join(results_dir, "Energy_Landscape"); os.makedirs(p, exist_ok=True)
            nrg = run_Energy_Landscape(data, ndt=20, ds=np.arange(0,21), bandwidth=1, ddof=1)
            savemat(os.path.join(p, "energy.mat"), {"nrgSig": nrg})
            # generate summary plot
            avg = nrg.mean(axis=2)
            plt.figure(figsize=(5,4))
            plt.imshow(avg, aspect="auto", cmap="viridis")
            plt.colorbar()
            plt.title("Avg Energy Landscape")
            plt.savefig(os.path.join(p, "avg_energy.png"))
            plt.close()

        if "regional_div" in selected:
            p = os.path.join(results_dir, "Regional_Diversity"); os.makedirs(p, exist_ok=True)
            rd = run_Regional_Diversity(data.transpose(1,0,2))
            savemat(os.path.join(p, "regional_diversity.mat"), {"rd": rd})
            plt.figure()
            plt.bar(np.arange(1, n_subj+1), rd)
            plt.xlabel("Subject")
            plt.ylabel("Regional Diversity")
            plt.title("Regional Diversity by Subject")
            plt.savefig(os.path.join(p, "rd.png"))
            plt.close()

        if "lfa_dmd" in selected:
            p = os.path.join(results_dir, "LFA_with_DMD"); os.makedirs(p, exist_ok=True)
            lmse_dmd, msd_dmd, evecs, lambdas = run_LFA_with_DMD(data, 15, 95, 0.5)
            savemat(os.path.join(p, "lmse_dmd.mat"), {"lmse": lmse_dmd})
            savemat(os.path.join(p, "msd_dmd.mat"), {"msd": msd_dmd})

        if "timescale" in selected:
            p = os.path.join(results_dir, "Timescale"); os.makedirs(p, exist_ok=True)
            ts = run_Timescale(data)
            savemat(os.path.join(p, "timescales.mat"), {"timescales": ts})

        if "bct" in selected:
            p = os.path.join(results_dir, "BCT_Metrics"); os.makedirs(p, exist_ok=True)
            bct = run_BrainConnectivityToolbox(data.transpose(1,0,2), threshold_prop=0.1, gamma=1.0)
            savemat(os.path.join(p, "bct.mat"), bct)

        return f"✅ Analysis complete!\nResults saved under:\n`{results_dir}`"

    @output
    @render.text
    def status():
        return do_analysis()

    # --- Summary tab outputs ---
    @output
    @render.text
    def summary_status():
        path = last_results_dir.get()
        if not path or not os.path.exists(path):
            return "⚠️ No results yet. Run an analysis first."
        return f"✅ Showing results from:\n{path}"

    @output
    @render.image
    def energy_image():
        path = last_results_dir.get() or ""
        img = os.path.join(path, "Energy_Landscape", "avg_energy.png")
        return {"src": img, "alt": "Avg Energy Landscape"} if os.path.exists(img) else None

    @output
    @render.image
    def rd_image():
        path = last_results_dir.get() or ""
        img = os.path.join(path, "Regional_Diversity", "rd.png")
        return {"src": img, "alt": "Regional Diversity"} if os.path.exists(img) else None

    # --- Re-run Energy tab ---
    @output
    @render.text
    def rerun_status():
        return rerun_energy_landscape()

    @reactive.Calc
    def rerun_energy_landscape():
        _ = input.rerun_energy()

        path = last_results_dir.get()
        if not path or not os.path.exists(path):
            return "⚠️ No results directory found. Run full analysis first."

        files = input.input_mat()
        if not files:
            return "⚠️ Please re-upload your .mat file."

        ndt = input.ndt()
        bw  = input.bandwidth()

        # reload data
        mat_data = loadmat(files[0]["datapath"])
        keys = [k for k in mat_data if not k.startswith("__")]
        if len(keys) != 1:
            return f"❌ Invalid .mat format (found {len(keys)} keys)."
        data = mat_data[keys[0]]

        # re-run Energy Landscape
        p = os.path.join(path, "Energy_Landscape")
        os.makedirs(p, exist_ok=True)
        nrg = run_Energy_Landscape(data, ndt=ndt, ds=np.arange(0, ndt+1), bandwidth=bw, ddof=1)
        savemat(os.path.join(p, "energy.mat"), {"nrgSig": nrg})
        avg = nrg.mean(axis=2)
        plt.figure(figsize=(5,4))
        plt.imshow(avg, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"Avg Energy Landscape (ndt={ndt}, bw={bw})")
        plt.savefig(os.path.join(p, "avg_energy.png"))
        plt.close()

        return f"✅ Recomputed Energy Landscape with ndt={ndt}, bandwidth={bw}"

    @output
    @render.image
    def energy_image_updated():
        path = last_results_dir.get() or ""
        img = os.path.join(path, "Energy_Landscape", "avg_energy.png")
        return {"src": img, "alt": "Updated Energy Landscape"} if os.path.exists(img) else None

# --- Run the app ---
app = App(app_ui, server)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    run_app(app)
