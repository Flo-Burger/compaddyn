import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
from bct import threshold_proportional, clustering_coef_wu, participation_coef, efficiency_wei

from scipy.io import loadmat

def run_BrainConnectivityToolbox(data, threshold_prop=0.1, gamma = 1.0):
    """
    Computes brain connectivity metrics using Louvain community detection.

    Parameters:
    - data: np.ndarray of shape (time, region, subject)
    - threshold_prop: float, proportion of top connections to retain

    Returns:
    - metrics: dict with keys:
        - 'efficiency_global': (subject,)
        - 'clustering': (region, subject)
        - 'modularity_Q': (subject,)
        - 'participation_coef': (region, subject)
        - 'community_assignments': (region, subject)
    """
    T, R, S = data.shape

    metrics = {
        'efficiency_global': np.zeros(S),
        'clustering': np.zeros((R, S)),
        'modularity_Q': np.zeros(S),
        'participation_coef': np.zeros((R, S)),
        'community_assignments': np.zeros((R, S), dtype=int)
    }

    for subj in range(S):
        subj_data = data[:, :, subj]

        # Pearson correlation matrix
        fc = np.corrcoef(subj_data.T)
        fc = (fc + fc.T) / 2
        np.fill_diagonal(fc, 0)
        fc = np.nan_to_num(fc)

        # Threshold and small self-loop
        fc_thresh = threshold_proportional(fc, threshold_prop)
        fc_thresh += np.eye(R) * 1e-5

        # Clustering
        metrics['clustering'][:, subj] = clustering_coef_wu(fc_thresh)

        # Global efficiency (must be non-negative)
        fc_nonneg = np.copy(fc_thresh)
        fc_nonneg[fc_nonneg < 0] = 0
        metrics['efficiency_global'][subj] = efficiency_wei(fc_nonneg)

        try:
            G = nx.from_numpy_array(fc_thresh)
            partition = community_louvain.best_partition(G, weight='weight', resolution=gamma)
            Ci = np.array([partition[i] for i in range(R)])
            Q = community_louvain.modularity(partition, G, weight='weight')

        except Exception as e:
            print(f"Subject {subj}: Louvain modularity failed: {e}")
            Ci = np.zeros(R, dtype=int)
            Q = np.nan

        metrics['modularity_Q'][subj] = Q
        metrics['community_assignments'][:, subj] = Ci

        # Participation coefficient
        metrics['participation_coef'][:, subj] = participation_coef(fc_thresh, Ci)

    return metrics

if __name__ == "__main__":
    # Load your .mat file
    mat_data = loadmat("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/Shine_Lab_Code/Example_Data/fMRI_like_data.mat")
    data = mat_data['fmri_data']  # should be time x region x subject

    # Compute BCT metrics
    bct_results = run_BrainConnectivityToolbox(data, threshold_prop=0.1)

    print(bct_results)

    # Example: print global efficiency
    print(bct_results['efficiency_global'])

    # Example: visualize clustering for subject 0
    import matplotlib.pyplot as plt
    plt.bar(range(data.shape[1]), bct_results['clustering'][:, 0])
    plt.title("Clustering Coefficient (Subject 0)")
    plt.xlabel("Region")
    plt.ylabel("Clustering")
    plt.show()
    # Example: visualize modularity for subject 0