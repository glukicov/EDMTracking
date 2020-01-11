# Make a profile plot 

import os, sys
sys.path.append('CommonUtils/')
import CommonUtils as cu
import RUtils as ru

import argparse, math
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# freq, edges = ru.root2np(file_path="DATA/largeEDM_sample.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/h_thetay_vs_time_reco_modg2")

# data = ru.root2np(file_path="DATA/largeEDM_sample.root", hist_path="AllStations/TrackFit/t>30/700<p<2400/pValue")
# data, n_bins = ru.root2np(file_path="DATA/largeEDM.root", hist_path="AllStationsNoTQ/TrackFit/t>0/0<p<3600/p")
# data, n_bins = ru.root2np(file_path="DATA/largeEDM_sample.root", hist_path="S0/VertexExtap/t>30/700<p<2400/h_verticalPos")
#data, n_bins = ru.root2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/TrackFit/t>0/0<p<3600/p")

# print(len(data))
# plt, ax, legend = cu.plotHist(data, n_bins=None)
# cu.textL(ax, 0.8, 0.85, str(legend), font_size=14)
# plt.show()

## TODO extract pr


# Profile Plot 
# sns.regplot(x=data[0], y=data[1], x_bins=200, fit_reg=None)
# plt.show()


