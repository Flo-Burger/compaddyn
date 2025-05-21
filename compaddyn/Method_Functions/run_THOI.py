from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
from thoi.heuristics import simulated_annealing, greedy
import numpy as np
import pandas as pd
import torch 

device = torch.device("mps")

def run_THOI(data, top_area_combinations = 10):
    n_vars, n_time, n_subjs = data.shape

    print(f"Number of variables: {n_vars}")
    print(f"Number of time points: {n_time}")

    overall_o_df = pd.DataFrame()
    overall_best_nplets_greedy = pd.DataFrame()
    overall_best_nplets_annealing = pd.DataFrame()

    for subject in range(n_subjs): 
        data_subject = data[:, :100, subject]

        # overall_o = nplets_measures(data_subject)

        best_nplets_greedy, best_scores_greedy = greedy(data_subject, 3, 2, repeat=10, device = "cpu")
        best_nplets_annealing, best_scores_annealing = simulated_annealing(data_subject, 5, repeat=10, device = "cpu")

        print(best_nplets_annealing)

    return None



