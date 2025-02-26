import numpy as np
from scipy.io import savemat

# Let's simulate 90 ROIs (brain regions), 300 timepoints, 10 subjects
num_rois = 16384
num_timepoints = 16384
num_subjects = 1

# Generate random data (here we use a normal distribution)
# shape: (num_rois, num_timepoints, num_subjects)
fmri_data = np.random.randn(num_rois, num_timepoints, num_subjects)

# Save to a .mat file with the variable name 'fmri_data'
savemat("fMRI_like_data.mat", {"fmri_data": fmri_data})
