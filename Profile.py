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

#Example of getting some data, bins and bind width from ROOT 1D Histogram
dataX, binsX, dBinX = ru.hist2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/radialPos")
# dataY, binsY, dBinY = ru.hist2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/verticalPos")


# Example of plotting 1D histo from data 
# ax, legend = cu.plotHist(dataX, n_bins=None)
# cu.textL(ax, 0.8, 0.85, str(legend), font_size=14)
# plt.show()


# Example of getting 2D histro from data 
ax, cb, legendX, legendY = cu.plotHist2D(dataX, dataX, n_binsX=binsX, n_binsY=binsX)
cb.set_label("Tracks",fontsize=13)
cu.textL(ax, 0.8, 0.85, str(legendX), font_size=14)
cu.textL(ax, 0.8, 0.65, str(legendY), font_size=14)
plt.savefig("2Dtest.png")


### 

# data, n_bins, dBins = ru.hist2np(file_path="DATA/largeEDM_sample.root", hist_path="AllStationsNoTQ/VertexExtap/t>30/700<p<2400/h_verticalPos")
# data, n_bins, dBins = ru.hist2np(file_path="DATA/test.root", hist_path="h_verticalPos")


## TODO extract pr

# Profile Plot 
# sns.regplot(x=data[0], y=data[1], x_bins=200, fit_reg=None)
# plt.show()


