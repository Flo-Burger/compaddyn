# import scipy.io as sio

# mat = sio.loadmat("/Users/22119216/Desktop/icg_matlab_output.mat", struct_as_record=False, squeeze_me=True)
# matlab_activityICG = mat["activityICG"]  # cell array from MATLAB
# matlab_outPairID = mat["outPairID"]      # cell array from MATLAB

# import pickle

# with open("/Users/22119216/Desktop/USYD_RA_2025/Shine_Lab_Combined_Code/icg_python_output.pkl", "rb") as f:
#     py_data = pickle.load(f)

# py_activityICG = py_data["activityICG"]
# py_outPairID = py_data["outPairID"]

# # MATLAB cell array might have shape (ICGsteps, ) if 1D
# mat_num_levels = matlab_activityICG.shape[0]  # e.g. 5
# py_num_levels = len(py_activityICG)          # e.g. 5
# print("MATLAB levels:", mat_num_levels, "Python levels:", py_num_levels)

# for lvl in range(py_num_levels):
#     py_dat = py_activityICG[lvl]
#     mat_dat = matlab_activityICG[lvl]  # might require indexing carefully

#     if py_dat is None or mat_dat is None:
#         print(f"Level {lvl}: One of them is None.")
#         continue

#     # Check shapes
#     py_shape = py_dat.shape
#     mat_shape = mat_dat.shape  # or mat_dat.shape if it's a numeric array

#     print(f"Level {lvl}: Python shape={py_shape}, MATLAB shape={mat_shape}")


# import numpy as np

# tol = 1e-6
# for lvl in range(py_num_levels):
#     py_dat = py_activityICG[lvl]
#     # Might have to cast MATLAB data to np.float64 if it's a different type
#     mat_dat = matlab_activityICG[lvl]

#     if py_dat is not None and mat_dat is not None:
#         # Convert to float64
#         py_arr = np.asarray(py_dat, dtype=np.float64)
#         mat_arr = np.asarray(mat_dat, dtype=np.float64)
        
#         # Check close
#         if py_arr.shape == mat_arr.shape:
#             diff = np.abs(py_arr - mat_arr)
#             max_diff = diff.max()
#             print(f"Level {lvl} max diff: {max_diff}")
#         else:
#             print(f"Level {lvl} shape mismatch: {py_arr.shape} vs {mat_arr.shape}")


