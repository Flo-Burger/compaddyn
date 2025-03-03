import os
import unittest
import numpy as np
from scipy.io import loadmat
from ..Example_Functions import run_all_methods  # Adjust if needed

# Get the base directory (where this script is located)
# Define paths relative to the test script location
TEST_DIR = os.path.dirname(__file__)  # This will point to the "tests/" directory

TEST_DATA_PATH = os.path.join(TEST_DIR, "cort_ts1c_short.mat")
EXPECTED_RESULTS_PATH = os.path.join(TEST_DIR, "expected_results")
TEST_OUTPUT_PATH = os.path.join(TEST_DIR, "test_output")
                                
# Tolerance for comparison
ABSOLUTE_TOLERANCE = 1e-5  # Change if needed

class TestRunAllMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Runs all methods once before testing"""
        print("\n[INFO] Running all methods to generate test output...")

        os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)

        mat_data = loadmat(TEST_DATA_PATH)
        test_data = next(val for key, val in mat_data.items() if not key.startswith("__"))

        # Keep overwrite on false always, otherwise you may overwrite original results 
        # which goes against idea of testing. 
        run_all_methods(data=test_data, output_dir=TEST_OUTPUT_PATH, overwrite=True)

    def compare_data_arrays(self, expected_file, test_file):
        """Loads and compares two .mat files by numerical similarity"""

        if not os.path.exists(test_file):
            self.fail(f"Test output file {test_file} not found!")

        expected_data = loadmat(expected_file)
        test_data = loadmat(test_file)

        expected_arrays = [v for k, v in expected_data.items() if not k.startswith("__")]
        test_arrays = [v for k, v in test_data.items() if not k.startswith("__")]

        # Ensure the same number of arrays
        self.assertEqual(len(expected_arrays), len(test_arrays), 
                         f"Mismatch in number of variables in {expected_file} and {test_file}")

        # Compare each array
        for exp, test in zip(expected_arrays, test_arrays):
            exp_flat = np.sort(exp.flatten())
            test_flat = np.sort(test.flatten())

            # Compute Mean Absolute Error (MAE)
            mae = np.mean(np.abs(exp_flat - test_flat))

            self.assertLessEqual(mae, ABSOLUTE_TOLERANCE,
                                 f"MAE {mae:.6f} exceeds tolerance for {expected_file}")

    def test_all_results(self):
        """Automatically finds and compares all .mat files"""
        print("\n[INFO] Checking all result files...")

        for root, _, files in os.walk(EXPECTED_RESULTS_PATH):
            for file in files:
                if file.endswith(".mat"):
                    expected_file = os.path.join(root, file)
                    relative_path = os.path.relpath(expected_file, EXPECTED_RESULTS_PATH)
                    test_file = os.path.join(TEST_OUTPUT_PATH, relative_path)

                    with self.subTest(msg=f"Comparing {relative_path}"):
                        self.compare_data_arrays(expected_file, test_file)

if __name__ == "__main__":
    unittest.main()
