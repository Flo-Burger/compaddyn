import numpy as np
from tqdm import tqdm

import numpy as np
import math

def run_ICG(data, apply_threshold = False, correlation_cutoff=0, use_absolute=True):
    """
    Iterative Correlation-Based Grouping (ICG) for multiple subjects,
    with a correlation threshold to prevent merging weakly correlated variables.

    Parameters:
      data (np.ndarray):
          Neuronal timeseries data (neurons x time x subjects)
      correlation_cutoff (float):
          Minimum correlation value required to consider merging two variables.
      use_absolute (bool):
          If True, applies the threshold to |correlation|.
          If False, applies the threshold directly to correlation (e.g., corr > correlation_cutoff).

    Returns:
      all_activityICG (list of list of np.ndarray):
          ICG activity for each level, for each subject.
      all_outPairID (list of list of np.ndarray):
          IDs of original neurons grouped at each level, for each subject.
    """
    n_vars, n_time, n_subjs = data.shape

    all_activityICG = []
    all_outPairID = []

    for subj in range(n_subjs):
        # Extract single subject's data
        allData = np.asarray(data[:, :, subj], dtype=np.float64)
        nData = allData.shape[0]

        # Determine grouping steps
        ICGsteps = int(math.ceil(math.log2(nData)))
        if nData == 2**ICGsteps:
            ICGsteps += 1

        # Initialize
        activityICG = [None] * ICGsteps
        activityICG[0] = allData.copy()

        outPairID = [None] * ICGsteps
        outPairID[0] = np.arange(nData, dtype=np.int64).reshape(-1, 1)

        for ICGlevel in range(1, ICGsteps):
            ICGAct = activityICG[ICGlevel - 1]
            nDataNow = ICGAct.shape[0]

            # 1) Compute correlation matrix
            rho = np.corrcoef(ICGAct)
            np.fill_diagonal(rho, 0)

            # 2) Extract upper triangle
            row_ind, col_ind = np.triu_indices(nDataNow, k=1)
            C = rho[row_ind, col_ind]

            # 3) Apply threshold
            if use_absolute:
                valid_mask = np.abs(C) > correlation_cutoff
            else:
                valid_mask = C > correlation_cutoff

            row_ind = row_ind[valid_mask]
            col_ind = col_ind[valid_mask]
            C = C[valid_mask]

            # If no pairs exceed the threshold, break or keep them unmerged
            if len(C) == 0:
                # You might choose to break or skip merging here
                # For now, let's break this level's merging
                break

            # Sort correlation in descending order
            sorted_idx = np.argsort(-C)
            row_sorted = row_ind[sorted_idx]
            col_sorted = col_ind[sorted_idx]
            # (C_sorted = C[sorted_idx]) # if you want to track sorted correlations

            numPairsTotal = nDataNow // 2
            outdat = np.empty((numPairsTotal, ICGAct.shape[1]), dtype=np.float64)

            outPairID_dim = 2 ** ICGlevel
            outPairID[ICGlevel] = -1 * np.ones((numPairsTotal, outPairID_dim), dtype=np.int64)

            used = np.zeros(nDataNow, dtype=bool)
            pair_count = 0

            prev_ids = outPairID[ICGlevel - 1]
            for idx2 in range(len(row_sorted)):
                r, c = row_sorted[idx2], col_sorted[idx2]
                if used[r] or used[c]:
                    continue

                # Merge the two variables
                outdat[pair_count] = ICGAct[r] + ICGAct[c]

                # Track their original IDs
                merged_ids = np.concatenate([prev_ids[r], prev_ids[c]])
                outPairID[ICGlevel][pair_count] = merged_ids

                used[r] = True
                used[c] = True
                pair_count += 1

                if pair_count >= numPairsTotal:
                    break

            activityICG[ICGlevel] = outdat

        all_activityICG.append(activityICG)
        all_outPairID.append(outPairID)

    return all_activityICG, all_outPairID


import math
import numpy as np
import torch

def custom_triu_indices(n, offset=1, device="cpu"):
    """
    Creates row/col indices for the upper triangle (strictly above diagonal) of an (n x n) matrix.
    Builds them on CPU, then moves to `device`, to avoid MPS not implementing `torch.triu_indices`.
    """
    row = torch.arange(n, device="cpu").view(-1,1).expand(n,n)
    col = torch.arange(n, device="cpu").expand(n,n)
    mask = (col - row) >= offset

    row_indices = row[mask].to(device)
    col_indices = col[mask].to(device)
    return row_indices, col_indices

def run_ICG_torch(
    data,
    correlation_cutoff=0.0,
    use_absolute=True,
    device=None
):
    """
    Optimized Iterative Correlation-Based Grouping (ICG) using PyTorch & NumPy.

    1) Uses NumPy to compute correlation quickly, then converts to PyTorch.
    2) Uses a custom upper-triangle function to avoid MPS issues with `torch.triu_indices`.
    3) Merges variables exceeding a correlation threshold (optional).

    Parameters:
      data (np.ndarray or torch.Tensor):
          Shape: (neurons x time x subjects).
      correlation_cutoff (float, optional):
          Minimum correlation value required for merging (default = 0 -> no threshold).
      use_absolute (bool, optional):
          If True, threshold uses |corr| > correlation_cutoff.
          If False, threshold uses corr > correlation_cutoff.
      device (str or torch.device, optional):
          "cuda", "mps", or "cpu". If None, auto-select:
            - "cuda" if available,
            - else "mps" if available,
            - else "cpu".

    Returns:
      all_activityICG (list of list of torch.Tensor):
          For each subject, a list of Tensors at each ICG level.
      all_outPairID (list of list of torch.Tensor):
          For each subject, a list of ID arrays at each ICG level.
    """
    # 1) Decide on device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Convert data to float32 on chosen device
    if isinstance(data, torch.Tensor):
        data_torch = data.to(device, dtype=torch.float32)
    else:
        data_torch = torch.tensor(data, dtype=torch.float32, device=device)

    n_vars, n_time, n_subjs = data_torch.shape

    all_activityICG = []
    all_outPairID = []

    for subj in range(n_subjs):
        # Extract single subject's data -> shape: (n_vars, n_time)
        subject_data = data_torch[:, :, subj].clone()
        nData = subject_data.shape[0]

        # If data is small, CPU might be faster
        # (uncomment if you want to force CPU for small nData)
        # if nData < 5000 and device != "cpu":
        #     subject_data = subject_data.cpu()
        #     dev_current = "cpu"
        # else:
        #     dev_current = device

        # 2) Determine number of ICG steps
        ICGsteps = int(math.ceil(math.log2(nData)))
        if nData == 2**ICGsteps:
            ICGsteps += 1

        # Prepare storages
        activityICG = [None] * ICGsteps
        activityICG[0] = subject_data.clone()  # level 0 is raw data

        outPairID = [None] * ICGsteps
        outPairID[0] = torch.arange(nData, dtype=torch.long, device=device).unsqueeze(1)

        # 3) Iterate ICG levels
        for ICGlevel in tqdm(range(1, ICGsteps)):
            ICGAct = activityICG[ICGlevel - 1]
            nDataNow = ICGAct.shape[0]

            # ----- A) Correlation via NumPy for speed -----
            # Move to CPU as NumPy arrays
            ICGAct_cpu = ICGAct.cpu().numpy()  # shape: (nDataNow, n_time)
            rho_np = np.corrcoef(ICGAct_cpu)  # shape: (nDataNow, nDataNow)
            np.fill_diagonal(rho_np, 0)

            # Convert back to torch
            rho = torch.tensor(rho_np, device=device, dtype=torch.float32)

            # ----- B) Extract upper triangle indices -----
            row_ind, col_ind = custom_triu_indices(nDataNow, offset=1, device=device)
            C = rho[row_ind, col_ind]

            # ----- C) Apply threshold -----
            if use_absolute:
                valid_mask = torch.abs(C) > correlation_cutoff
            else:
                valid_mask = C > correlation_cutoff

            row_ind = row_ind[valid_mask]
            col_ind = col_ind[valid_mask]
            C = C[valid_mask]

            # If no pairs exceed threshold, stop merging
            if C.numel() == 0:
                break

            # ----- D) Sort correlations in descending order -----
            sorted_idx = torch.argsort(C, descending=True)
            row_sorted = row_ind[sorted_idx]
            col_sorted = col_ind[sorted_idx]

            # Prepare new data for merges
            numPairsTotal = nDataNow // 2
            outdat = torch.empty((numPairsTotal, ICGAct.shape[1]), device=device, dtype=torch.float32)

            outPairID_dim = 2 ** ICGlevel
            outPairID[ICGlevel] = -1 * torch.ones((numPairsTotal, outPairID_dim), device=device, dtype=torch.long)

            used = torch.zeros(nDataNow, dtype=torch.bool, device=device)
            pair_count = 0

            prev_ids = outPairID[ICGlevel - 1]

            # ----- E) Greedy merging -----
            for i_sorted in range(len(row_sorted)):
                r = row_sorted[i_sorted]
                c = col_sorted[i_sorted]
                if used[r] or used[c]:
                    continue

                # Merge data
                outdat[pair_count] = ICGAct[r] + ICGAct[c]

                # Merge original IDs
                merged_ids = torch.cat([prev_ids[r], prev_ids[c]])
                outPairID[ICGlevel][pair_count] = merged_ids

                used[r] = True
                used[c] = True
                pair_count += 1
                if pair_count >= numPairsTotal:
                    break

            # Save merged data at this level
            activityICG[ICGlevel] = outdat

        # Store results for the subject
        all_activityICG.append(activityICG)
        all_outPairID.append(outPairID)

    return all_activityICG, all_outPairID




