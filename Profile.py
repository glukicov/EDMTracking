# Make a profile plot 
# Gleb Lukicov (11 Jan 2020)  

import os, sys
sys.path.append('CommonUtils/')
import CommonUtils as cu
import RUtils as ru

import argparse, math
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib # to set global parameters 





### 
# data, n_bins, dBins = ru.hist2np(file_path="DATA/largeEDM_sample.root", hist_path="AllStationsNoTQ/VertexExtap/t>30/700<p<2400/h_verticalPos")
# data, n_bins, dBins = ru.hist2np(file_path="DATA/test.root", hist_path="h_verticalPos")


## TODO extract pr

# Profile Plot 
# sns.regplot(x=data[0], y=data[1], x_bins=200, fit_reg=None)
# plt.show()


