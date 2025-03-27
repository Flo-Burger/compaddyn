"""
Parallel Iterative Correlation-Based Grouping (ICG) for multiple subjects.

Author: Florian Burger
Date: 26.03.2025
Version: 1.0

Description:
This script implements a parallelized version of the ICG algorithm to group variables
(e.g., neurons or voxels) based on pairwise correlation, applied independently
to multiple subjects in a 3D time series dataset. Grouping continues iteratively
until no pairs exceed the correlation threshold. Each run returns the activity and
grouping structure for each level in the hierarchy.

"""

import os
import math
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.io import loadmat

def process_subject(subj, data, correlation_cutoff, use_absolute):
    """
    Run the ICG algorithm for a single subject.

    Parameters:
        subj (int): Index of the subject to process.
        data (np.ndarray): Data array (variables x timepoints x subjects).
        correlation_cutoff (float): Minimum correlation value to allow merging.
        use_absolute (bool): If True, apply threshold to |correlation|.

    Returns:
        activity_icg (list of np.ndarray): Merged activity at each ICG level.
        out_pair_id (list of np.ndarray): Corresponding original variable IDs at each level.
    """
    
    n_vars, _, _ = data.shape
    subject_data = np.asarray(data[:, :, subj], dtype=np.float64)
    n_data = subject_data.shape[0]

    # Determine how many levels of grouping are required
    icg_steps = int(math.ceil(math.log2(n_data)))
    if n_data == 2 ** icg_steps:
        icg_steps += 1

    activity_icg = [None] * icg_steps
    activity_icg[0] = subject_data.copy()

    out_pair_id = [None] * icg_steps
    out_pair_id[0] = np.arange(n_data, dtype=np.int64).reshape(-1, 1)

    for level in range(1, icg_steps):
        current_data = activity_icg[level - 1]
        n_data_now = current_data.shape[0]

        # Step 1: Compute pairwise correlations
        rho = np.corrcoef(current_data)
        np.fill_diagonal(rho, 0)  # Remove self-correlation

        # Step 2: Extract upper triangle (unique pairs)
        row_ind, col_ind = np.triu_indices(n_data_now, k=1)
        correlations = rho[row_ind, col_ind]

        # Step 3: Apply threshold
        if use_absolute:
            valid_mask = np.abs(correlations) > correlation_cutoff
        else:
            valid_mask = correlations > correlation_cutoff

        row_ind = row_ind[valid_mask]
        col_ind = col_ind[valid_mask]
        correlations = correlations[valid_mask]

        # If no pairs remain, stop merging
        if len(correlations) == 0:
            break

        # Step 4: Sort pairs by correlation (descending)
        sorted_idx = np.argsort(-correlations)
        row_sorted = row_ind[sorted_idx]
        col_sorted = col_ind[sorted_idx]

        num_pairs_total = n_data_now // 2
        merged_data = np.empty((num_pairs_total, current_data.shape[1]), dtype=np.float64)

        pair_id_dim = 2 ** level
        out_pair_id[level] = -1 * np.ones((num_pairs_total, pair_id_dim), dtype=np.int64)

        used = np.zeros(n_data_now, dtype=bool)
        pair_count = 0
        prev_ids = out_pair_id[level - 1]

        # Step 5: Greedy pairwise merging of unused nodes
        for i in range(len(row_sorted)):
            r, c = row_sorted[i], col_sorted[i]
            if used[r] or used[c]:
                continue

            # Merge time series and track origin
            merged_data[pair_count] = current_data[r] + current_data[c]
            merged_ids = np.concatenate([prev_ids[r], prev_ids[c]])
            out_pair_id[level][pair_count] = merged_ids

            used[r] = True
            used[c] = True
            pair_count += 1

            if pair_count >= num_pairs_total:
                break

        activity_icg[level] = merged_data

    return activity_icg, out_pair_id


def run_ICG(data, correlation_cutoff=0.0, use_absolute=True, n_jobs=-1):
    """
    Runs the ICG algorithm across all subjects in parallel.

    Parameters:
        data (np.ndarray): 3D array (variables x timepoints x subjects).
        correlation_cutoff (float): Minimum correlation required to merge pairs.
        use_absolute (bool): If True, threshold is applied to |correlation|.
        n_jobs (int): Number of parallel jobs (default: -1 = use all cores).

    Returns:
        all_activity_icg (list of list of np.ndarray): ICG results per subject.
        all_out_pair_id (list of list of np.ndarray): ID tracking per subject.
    """
    n_subjs = data.shape[2]

    with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        results = parallel(
            delayed(process_subject)(
                subj, data, correlation_cutoff, use_absolute
            ) for subj in range(n_subjs)
        )

    all_activity_icg, all_out_pair_id = zip(*results)
    return list(all_activity_icg), list(all_out_pair_id)



