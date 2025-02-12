import os
import numpy as np
from scipy.io import loadmat, savemat
from Example_Functions import run_all_methods  

# Define paths
TEST_DATA_PATH = "/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/tests/cort_ts1c_short.mat"  
EXPECTED_RESULTS_PATH = "tests/expected_results/"  

os.makedirs(EXPECTED_RESULTS_PATH, exist_ok=True)

# Load test data
mat_data = loadmat(TEST_DATA_PATH)
filtered_keys = [key for key in mat_data.keys() if not key.startswith('__')]
if len(filtered_keys) != 1:
    raise ValueError(f"Expected one variable in {TEST_DATA_PATH}, found {len(filtered_keys)}: {filtered_keys}")

test_data = mat_data[filtered_keys[0]]

# Run the function on test data
run_all_methods(data=test_data, output_dir=EXPECTED_RESULTS_PATH, overwrite=False)

