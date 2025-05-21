# app.py

from shiny import App, ui, reactive, render, run_app
import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# ────────────────────────────────────────────────────────────────────────────────
# Your core analysis functions
from Shine_Lab_Code.Method_Functions import (
    run_LFA,
    run_ICG,
    run_fft_global,
    run_fft_per_area,
    run_Energy_Landscape,
    run_LFA_with_DMD,
    run_Regional_Diversity,
    run_Timescale,
    run_BrainConnectivityToolbox
)

# For brain plotting
import nibabel as nib
from nilearn import datasets, plotting

# ─── Atlas registry & plotting helpers ────────────────────────────────────────
_ATLAS_REGISTRY = {
    68:   ("desikan",     datasets.fetch_atlas_surf_destrieux),
    148:  ("destrieux",   datasets.fetch_atlas_surf_destrieux),
    116:  ("aal",         datasets.fetch_atlas_aal),
    100:  ("schaefer100", lambda: datasets.fetch_atlas_schaefer_2018(n_rois=100)),
    400:  ("schaefer400", lambda: datasets.fetch_atlas_schaefer_2018(n_rois=400)),
    1000: ("schaefer1000", lambda: datasets.fetch_atlas_schaefer_2018(n_rois=1000)),
}

def detect_atlas(n_regions):
    try:
        name, fetcher = _ATLAS_REGISTRY[n_regions]
    except KeyError:
        raise ValueError(f"No atlas registered for {n_regions} regions")
    return name, fetcher()

def plot_volume_map(values, atlas, out_png):
    img   = nib.load(atlas.maps)
    lab   = img.get_fdata().astype(int)
    vol   = np.zeros_like(lab, float)
    for i, v in enumerate(values, start=1):
        vol[lab == i] = v
    metric_img = nib.Nifti1Image(vol, img.affine)
    disp = plotting.plot_stat_map(
        metric_img,
        title="Spatial Map",
        display_mode="ortho",
        cmap="viridis",
        colorbar=True
    )
    disp.savefig(out_png); disp.close()

def plot_surf_map(values, atlas, prefix):
    fs = datasets.fetch_surf_fsaverage()
    lh_lab, _, _ = nib.freesurfer.read_annot(atlas["map_left"])
    rh_lab, _, _ = nib.freesurfer.read_annot(atlas["map_right"])
    for hemi, lab in [("left", lh_lab), ("right", rh_lab)]:
        tex = np.zeros_like(lab, float)
        for idx, v in enumerate(values, start=1):
            tex[lab == idx] = v
        fig = plotting.plot_surf_stat_map(
            fs[f"infl_{hemi}"], tex,
            bg_map=fs[f"sulc_{hemi}"],
            hemi=hemi,
            cmap="viridis",
            colorbar=True,
            title=f"Spatial Map ({hemi})"
        )
        fig.savefig(f"{prefix}_{hemi}.png"); fig.close()

def plot_brain(values, prefix):
    name, atlas = detect_atlas(len(values))
    if name in ("aal", "schaefer100", "schaefer400", "schaefer1000"):
        plot_volume_map(values, atlas, prefix + ".png")
    else:
        plot_surf_map(values, atlas, prefix)

# ─── PRE-FETCH ATLAS SHAPE FOR SLIDER RANGES ───────────────────────────────────
try:
    _, _atlas_1000 = detect_atlas(1000)
    img1000        = nib.load(_atlas_1000.maps)
    _x_max, _y_max, _z_max = img1000.shape
except Exception:
    _x_max = _y_max = _z_max = 50

# ─── UI ────────────────────────────────────────────────────────────────────────
app_ui = ui.page_navbar(

    ui.nav_panel("Run Analysis",
        ui.page_fluid(
            ui.h2("Run Analysis"),
            ui.input_file("input_mat",    "Upload .mat file", accept=".mat"),
            ui.input_text("output_dir",   "Output directory", placeholder="~/Results"),
            ui.input_checkbox_group(
                "analyses", "Select analyses",
                choices={
                    "lfa":          "LFA",
                    "icg":          "ICG",
                    "fft":          "FFT",
                    "energy":       "Energy Landscape",
                    "lfa_dmd":      "LFA with DMD",
                    "regional_div": "Global Regional Diversity",
                    "timescale":    "Timescale per region",
                    "bct":          "Brain Connectivity Toolbox (Takes a while to run this one)"
                },
                selected=["lfa","icg","fft","energy",
                          "lfa_dmd","regional_div","timescale","bct"]
            ),
            ui.input_numeric("fs",      "Sampling freq (Hz)", value=None, min=0, step=0.1),
            ui.input_action_button("run", "Run Analysis", class_="btn-primary"),
            ui.output_text_verbatim("status")
        )
    ),

    ui.nav_panel("Summary",
        ui.page_fluid(
            ui.h2("Results Summary"),
            ui.output_text("summary_status"),
            ui.h4("Average Energy Landscape"),
            ui.output_image("energy_image"),
            ui.h4("Global Regional Diversity"),
            ui.output_image("rd_image")
        )
    ),

    ui.nav_panel("Re-run Energy",
        ui.page_fluid(
            ui.h2("Re-run Energy"),
            ui.input_numeric("ndt",       "Number of lags", value=20, min=1),
            ui.input_numeric("bandwidth","Kernel bandwidth", value=1.0, step=0.1, min=0.1),
            ui.input_action_button("rerun_energy", "Re-run Energy", class_="btn-warning"),
            ui.output_text_verbatim("rerun_status"),
            ui.h4("Updated Energy Landscape"),
            ui.output_image("energy_image_updated")
        )
    ),

    ui.nav_panel("Brain plots",
        ui.page_fluid(
            ui.h2("Brain plots"),
            ui.input_file("brain_mat", "Upload .mat (regions×time×subjects)", accept=".mat"),

            ui.input_radio_buttons(
              "brain_metric", "Metric",
               choices={
                 "fft":           "FFT Power",
                 "regional_div":  "Regional Diversity",
                 "timescale":     "Timescale"
               },
               selected="fft"
            ),

            ui.input_action_button("plot_brain", "Generate Brain Map", class_="btn-success"),
            ui.output_image("brain_map"),

            ui.h4("Explore Brain Volume"),
            ui.input_slider("x_slice", "X Slice", min=0, max=_x_max-1, value=_x_max//2),
            ui.input_slider("y_slice", "Y Slice", min=0, max=_y_max-1, value=_y_max//2),
            ui.input_slider("z_slice", "Z Slice", min=0, max=_z_max-1, value=_z_max//2),
            ui.output_image("slice_view")
        )
    )
)

# ─── Server ───────────────────────────────────────────────────────────────────
def server(input, output, session):
    last_results_dir = reactive.Value(None)
    brain_volume     = reactive.Value(None)  # store 3D volume for slice view

    # —— Full analysis ————————————————————————————————————————————————————
    @reactive.Calc
    def do_analysis():
        _ = input.run()
        with reactive.isolate():
            files    = input.input_mat()
            out_dir  = input.output_dir().strip()
            selected = list(input.analyses())
            fs       = input.fs() or 1.0

        if not files:
            return "⚠️ Please upload a .mat file."
        if not out_dir or not os.path.isabs(out_dir):
            return "⚠️ Enter a valid absolute output directory."

        results_dir = os.path.join(out_dir, "Results")
        os.makedirs(results_dir, exist_ok=True)
        last_results_dir.set(results_dir)

        mat = loadmat(files[0]["datapath"])
        key = next(k for k in mat if not k.startswith("__"))
        data = mat[key]  # (regions, time, subjects)
        n_reg, n_time, n_subj = data.shape

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

        return f"✅ Analysis complete!\nResults in:\n`{results_dir}`"

    @output
    @render.text
    def status():
        return do_analysis()

    # — Summary tab —————————————————————————————————————————————
    @output
    @render.text
    def summary_status():
        path = last_results_dir.get() or ""
        if not os.path.exists(path):
            return "⚠️ No results yet."
        return f"✅ Showing results from:\n{path}"

    @output
    @render.image
    def energy_image():
        path = last_results_dir.get() or ""
        img  = os.path.join(path, "Energy_Landscape", "avg_energy.png")
        return {"src": img, "alt": "Avg Energy"} if os.path.exists(img) else None

    @output
    @render.image
    def rd_image():
        path = last_results_dir.get() or ""
        img  = os.path.join(path, "Regional_Diversity", "rd.png")
        return {"src": img, "alt": "Regional Diversity"} if os.path.exists(img) else None

    # — Re-run Energy tab —————————————————————————————————————
    @reactive.Calc
    def rerun_energy_landscape():
        _ = input.rerun_energy()
        path  = last_results_dir.get() or ""
        files = input.input_mat()
        if not path or not files:
            return "⚠️ Run full analysis first."
        mat   = loadmat(files[0]["datapath"])
        key   = next(k for k in mat if not k.startswith("__"))
        data  = mat[key]
        ndt   = input.ndt(); bw = input.bandwidth()
        p     = os.path.join(path, "Energy_Landscape"); os.makedirs(p, exist_ok=True)
        nrg   = run_Energy_Landscape(data, ndt=ndt, ds=np.arange(0, ndt+1),
                                     bandwidth=bw, ddof=1)
        savemat(os.path.join(p, "energy.mat"), { "nrgSig": nrg })
        avg   = nrg.mean(axis=2)
        plt.figure(figsize=(5,4))
        plt.imshow(avg, aspect="auto", cmap="viridis"); plt.colorbar()
        plt.title(f"Energy (ndt={ndt}, bw={bw})")
        plt.savefig(os.path.join(p, "avg_energy.png")); plt.close()
        return f"✅ Recomputed Energy (ndt={ndt}, bw={bw})"

    @output
    @render.text
    def rerun_status():
        return rerun_energy_landscape()

    @output
    @render.image
    def energy_image_updated():
        path = last_results_dir.get() or ""
        img  = os.path.join(path, "Energy_Landscape", "avg_energy.png")
        return {"src": img, "alt": "Updated Energy"} if os.path.exists(img) else None

    # — Brain plots tab (multi-metric) ——————————————————————————————
    @reactive.Calc
    def do_brain_plot():
        _ = input.plot_brain()
        files = input.brain_mat()
        if not files:
            return None

        mat     = loadmat(files[0]["datapath"])
        key     = next(k for k in mat if not k.startswith("__"))
        data3d  = mat[key]                   # (regions, time, subjects)
        n_reg   = data3d.shape[0]
        fs      = input.fs() or 1.0
        metric  = input.brain_metric()

        # compute per-region values
        if metric == "fft":
            _, per_area = run_fft_per_area(data3d, fs=fs)
            region_vals = per_area.mean(axis=(1, 2))
        elif metric == "regional_div":
            region_vals = run_Regional_Diversity(data3d, mode="per_region")
        elif metric == "timescale":
            region_vals = run_Timescale(data3d, mode = "per_region")
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # store volumetric data for slice view if applicable
        name, atlas = detect_atlas(len(region_vals))
        if name in ("aal","schaefer100","schaefer400","schaefer1000"):
            atlas_img = nib.load(atlas.maps)
            lab       = atlas_img.get_fdata().astype(int)
            vol       = np.zeros_like(lab, float)
            for idx, val in enumerate(region_vals, start=1):
                vol[lab == idx] = val
            brain_volume.set(vol)
        else:
            brain_volume.set(None)

        # plot summary map
        base   = last_results_dir.get() or os.getcwd()
        prefix = os.path.join(base, "Brain_plots", metric)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        plot_brain(region_vals, prefix)

        # choose appropriate file name
        if os.path.exists(prefix + ".png"):
            return prefix + ".png"
        elif os.path.exists(prefix + "_left.png"):
            return prefix + "_left.png"
        else:
            return None

    @output
    @render.image
    def brain_map():
        src = do_brain_plot()
        return {"src": src, "alt": "Brain Map"} if src else None

    # ── INTERACTIVE SLICE VIEW (uses stored volume) ─────────────────────────────
    @output
    @render.image
    def slice_view():
        vol = brain_volume.get()
        if vol is None:
            return None

        x, y, z = input.x_slice(), input.y_slice(), input.z_slice()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(vol[x, :, :].T, origin='lower', cmap='viridis')
        axes[0].set_title(f"X = {x}")
        axes[1].imshow(vol[:, y, :].T, origin='lower', cmap='viridis')
        axes[1].set_title(f"Y = {y}")
        axes[2].imshow(vol[:, :, z].T, origin='lower', cmap='viridis')
        axes[2].set_title(f"Z = {z}")
        for ax in axes:
            ax.axis('off')

        out = os.path.join(last_results_dir.get() or os.getcwd(),
                           "Brain_plots", "slice_view.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return {"src": out, "alt": f"Slices X={x},Y={y},Z={z}"}

# ────────────────────────────────────────────────────────────────────────────────
# Launch
app = App(app_ui, server)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    run_app(app)

