# analysis/plot_brain.py

import os
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker

# ─── Atlas registry ─────────────────────────────────────────────────────────────
_ATLAS_REGISTRY = {
    68:  ("desikan",   datasets.fetch_atlas_surf_destrieux),   # FreeSurfer Desikan: 34+34
    148:("destrieux", datasets.fetch_atlas_surf_destrieux),   # Destrieux: 74+74
    116:("aal",       datasets.fetch_atlas_aal),              # AAL: 116 ROIs
    100:("schaefer",  lambda: datasets.fetch_atlas_schaefer_2018("fsaverage5", 100)),
    400:("schaefer",  lambda: datasets.fetch_atlas_schaefer_2018("fsaverage5", 400)),
}

def detect_atlas(n_regions):
    """
    Look up which atlas matches your number of ROIs and fetch it.
    Returns (name, atlas_object).
    """
    try:
        name, fetcher = _ATLAS_REGISTRY[n_regions]
    except KeyError:
        raise ValueError(f"No atlas registered for {n_regions} regions")
    return name, fetcher()

# ─── Volume plotting (e.g. AAL) ────────────────────────────────────────────────
def plot_volume_map(values, atlas, out_png):
    """
    values : array-like, length = #labels in atlas (excluding background)
    atlas  : object with .maps (Nifti1Image) and .labels (list of strings)
    """
    # load parcellation image
    label_img  = atlas.maps              # Nifti1Image
    label_data = label_img.get_fdata().astype(int)
    # build a volume: each voxel ← metric of its parcel
    metric_vol = np.zeros_like(label_data, float)
    for i, val in enumerate(values, start=1):
        metric_vol[label_data == i] = val

    # wrap as Nifti and plot
    metric_img = nib.Nifti1Image(metric_vol, label_img.affine)
    display = plotting.plot_stat_map(
        metric_img,
        title="Regional Diversity",
        display_mode="ortho",
        colorbar=True,
        cmap="viridis",
        threshold=None,
    )
    display.savefig(out_png)
    display.close()

# ─── Surface plotting (e.g. Desikan/DK) ───────────────────────────────────────
def plot_surf_map(values, atlas, out_png_prefix):
    """
    values         : length-68 or 148 vector
    atlas          : return of fetch_atlas_surf_destrieux()
    out_png_prefix : e.g. ".../rd_brain"
    """
    # load fsaverage for plotting
    fs = datasets.fetch_surf_fsaverage()

    # read the annotation files
    lh_labels, _, _ = nib.freesurfer.read_annot(atlas["map_left" ])
    rh_labels, _, _ = nib.freesurfer.read_annot(atlas["map_right"])

    tex_lh = np.zeros_like(lh_labels, float)
    tex_rh = np.zeros_like(rh_labels, float)
    for idx, val in enumerate(values, start=1):
        tex_lh[lh_labels == idx] = val
        tex_rh[rh_labels == idx] = val

    # lateral views for each hemisphere
    for hemi, surf_map, sulc in [
        ("left",  tex_lh, fs["sulc_left" ]), 
        ("right", tex_rh, fs["sulc_right"])
    ]:
        fig = plotting.plot_surf_stat_map(
            fs[f"infl_{hemi}"],    # inflated surface mesh
            surf_map,
            bg_map=sulc,
            hemi=hemi,
            cmap="viridis",
            colorbar=True,
            title=f"Regional Diversity ({hemi.capitalize()})"
        )
        fig.savefig(f"{out_png_prefix}_{hemi}.png")
        fig.close()

# ─── Dispatcher ────────────────────────────────────────────────────────────────
def plot_brain(values, out_prefix):
    """
    values     : 1D array length = #ROIs
    out_prefix : e.g. "/path/to/Results/Regional_Diversity/rd_brain"
    """
    name, atlas = detect_atlas(len(values))
    if name == "aal":
        plot_volume_map(values, atlas, out_prefix + ".png")
    else:
        plot_surf_map(values, atlas, out_prefix)
