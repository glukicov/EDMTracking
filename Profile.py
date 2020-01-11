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


# data, n_bins = ru.root2np(file_path="DATA/largeEDM_sample.root", hist_path="AllStations/TrackFit/t>30/700<p<2400/pValue")
# data, n_bins = ru.root2np(file_path="DATA/largeEDM.root", hist_path="AllStationsNoTQ/TrackFit/t>0/0<p<3600/p")
# data, n_bins = ru.root2np(file_path="DATA/largeEDM_sample.root", hist_path="S0/VertexExtap/t>30/700<p<2400/h_verticalPos")
# data, n_bins = ru.root2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/TrackFit/t>0/0<p<3600/p")
dataX, binsX, dBinX = ru.root2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/radialPos")
dataY, binsY, dBinY = ru.root2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/verticalPos")

# ax, legend = cu.plotHist(dataX, n_bins=None)
# cu.textL(ax, 0.8, 0.85, str(legend), font_size=14)
# plt.show()

# plt, ax, legend = cu.plotHist(dataY, n_bins=None)
# cu.textL(ax, 0.8, 0.85, str(legend), font_size=14)
# plt.show()

# print(binsX, binsY)
# print(len(dataX), len(dataY))

ax, cb, legendX, legendY = cu.plotHist2D(dataX, dataX, n_binsX=binsX, n_binsY=binsX)
cb.set_label(r'$\log_{10}$',fontsize=13)
# cu.textL(ax, 0.8, 0.85, str(legendX), font_size=14)
# cu.textL(ax, 0.8, 0.65, str(legendY), font_size=14)
plt.show()

## TODO extract pr


# Profile Plot 
# sns.regplot(x=data[0], y=data[1], x_bins=200, fit_reg=None)
# plt.show()


