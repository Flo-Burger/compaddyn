# Questions about ICG: 
# Run it per person/subject I assume? 
# It merges each variable each time independent of the correlation with another? 
# Means it will merge anyways even if one variable correlates ~0 with all others 
# Does this make sense for a data reduction approach? Or what else do you use ICG for? 

import numpy as np

def run_ICG(data_ts):
    """
    Iterative Correlation-Based Grouping (ICG) for multiple subjects.

    Parameters:
      data_ts (np.ndarray):
          Neuronal timeseries data (neurons x time x subjects)

    Returns:
      all_activityICG (list of list of np.ndarray):
          ICG activity for each level, for each subject.
      all_outPairID (list of list of np.ndarray):
          IDs of original neurons grouped at each level, for each subject.
    """
    # Extract shape information
    n_vars, n_time, n_subjs = data_ts.shape

    # Storage for all subjects
    all_activityICG = []
    all_outPairID = []

    # Loop over subjects
    for subj in range(n_subjs):
        # Extract single subject's data (neurons x time)
        allData = np.asarray(data_ts[:, :, subj], dtype=np.float64)
        nData = allData.shape[0]

        # Determine grouping steps
        ICGsteps = int(np.ceil(np.log2(nData)))
        if nData == 2**ICGsteps:
            ICGsteps += 1

        # Initialize storage for the subject
        activityICG = [None] * ICGsteps
        activityICG[0] = allData.copy()

        outPairID = [None] * ICGsteps
        outPairID[0] = np.arange(nData, dtype=np.int64).reshape(-1, 1)

        for ICGlevel in range(1, ICGsteps):
            ICGAct = activityICG[ICGlevel - 1]
            nDataNow = ICGAct.shape[0]

            # 1) Compute correlation matrix
            rho = np.corrcoef(ICGAct)
            np.fill_diagonal(rho, 0)  # set diagonal to 0

            # 2) Extract upper triangle
            row_ind, col_ind = np.triu_indices(nDataNow, k=1)
            C = rho[row_ind, col_ind]

            # 3) Sort correlation in descending order
            sorted_idx = np.argsort(-C)  # negative for descending
            row_sorted = row_ind[sorted_idx]
            col_sorted = col_ind[sorted_idx]

            # Prepare new data
            numPairsTotal = nDataNow // 2
            outdat = np.empty((numPairsTotal, ICGAct.shape[1]), dtype=np.float64)

            outPairID_dim = 2**ICGlevel
            outPairID[ICGlevel] = -1 * np.ones((numPairsTotal, outPairID_dim), dtype=np.int64)

            used = np.zeros(nDataNow, dtype=bool)
            pair_count = 0

            # 4) Merge pairs
            prev_ids = outPairID[ICGlevel - 1]
            for idx in range(len(row_sorted)):
                r, c = row_sorted[idx], col_sorted[idx]
                if used[r] or used[c]:
                    continue

                # Sum the two variables
                outdat[pair_count] = ICGAct[r] + ICGAct[c]

                # Merge their IDs
                merged_ids = np.concatenate([prev_ids[r], prev_ids[c]])
                outPairID[ICGlevel][pair_count] = merged_ids

                used[r] = used[c] = True
                pair_count += 1

                if pair_count >= numPairsTotal:
                    break

            activityICG[ICGlevel] = outdat

        # Store results for this subject
        all_activityICG.append(activityICG)
        all_outPairID.append(outPairID)

    return all_activityICG, all_outPairID
