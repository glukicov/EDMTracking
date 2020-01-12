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



data, n_bins, dBins = ru.hist2D2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>30/700<p<2400/thetay_vs_time_modg2")


# ##Example of getting 2D histro from data 
# ##TODO adjust value to get default scaling better (can now adjust from few)
# jg, legendX, legendY = cu.plotHist2D(data[0], data[1], n_binsX=n_bins[0], n_binsY=n_bins[1], return_cb=False)
# print(legendX, legendY)
# ## cb.set_label("Tracks",fontsize=13)
# ## cu.textL(cbaxes, -15.0, 1.2, "X:\n"+str(legendX), font_size=14)
# ## cu.textL(cbaxes, 0.8, 1.2, "Y:\n"+str(legendY), font_size=14)
# jg.set_axis_labels("x", "y")
# sns.set(font_scale = 4)
# plt.show()
# ## plt.savefig("2Dtest.png")

# Profile Plot 
# make into a function 
# see more options
# https://seaborn.pydata.org/generated/seaborn.regplot.html
sns.regplot(x=data[0], y=data[1], x_bins=10, fit_reg=None)
plt.show()


