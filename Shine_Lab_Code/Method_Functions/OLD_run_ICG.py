# This code is likely outdated, I wrote a faster function but keeping this for now to keep 
# a working function in case parallel doesn't work on other computers (it should)

import numpy as np
import numpy as np
import math

from tqdm import tqdm

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
